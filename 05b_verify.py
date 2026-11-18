"""Módulo 5b - Verificador multimodal com autocorreção (Gemini API ou Ollama).

Uso:
    python 05b_verify.py <workdir> [--backend gemini|ollama] [--verify-mode video|frames]
                                   [--min-score 7.0] [--gemini-model gemini-4-hydrogen]

Fluxo:
1. VLM (Gemini com .mp4 direto ou Ollama com 6 frames) avalia cada corte_XX.mp4 quanto a
   hook, completude do contexto (setup antes da risada/clímax), qualidade visual e título.
2. Cortes reprovados por falta de setup/corte tardio vão ao LLM (gemini-4-argon ou Ollama)
   junto com o feedback e a janela completa para recuar o start_time (1 tentativa por corte).
3. Re-renderiza apenas os cortes corrigidos (reaproveitando webcam.json).
4. Re-verifica apenas os corrigidos e grava verified_clips.json + verify_report.json.
"""
from __future__ import annotations

import argparse
import base64
import importlib
import json
import re
import shutil
import time
from pathlib import Path

import cv2
import requests
from rich.console import Console
from rich.table import Table

import config

console = Console()


# ========================================================= extração de frames
def sample_clip_frames(video_path: Path, n: int = config.VERIFY_SAMPLES) -> list[tuple[float, str]]:
    """Extrai n frames estratégicos (foco no hook inicial + meio + fim) em JPEG base64."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    dur = total / fps if total > 0 else 0.0
    if dur <= 0:
        cap.release()
        return []

    # Pontos fixos no início (0.5s e 2.0s) + distribuição até o fim
    targets = [min(0.5, dur * 0.1), min(2.0, dur * 0.2),
               dur * 0.35, dur * 0.55, dur * 0.75, max(0.0, dur - 1.5)]
    if n != 6:
        targets = [dur * (i + 0.5) / n for i in range(n)]
    times = sorted({round(min(max(0.0, t), max(0.0, dur - 0.1)), 2) for t in targets})

    out: list[tuple[float, str]] = []
    for t in times:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
        ok, frame = cap.read()
        if not ok:
            continue
        # Reduz para 540x960 para economizar tokens/VRAM mantendo legendas e webcam nítidas
        small = cv2.resize(frame, (540, 960), interpolation=cv2.INTER_AREA)
        ok_enc, buf = cv2.imencode(".jpg", small, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
        if ok_enc:
            out.append((t, base64.b64encode(buf.tobytes()).decode("ascii")))
    cap.release()
    return out


def relative_transcript(windows: list[dict], start: float, end: float) -> str:
    """Monta transcrição do corte com timestamps relativos (0.0s .. duração)."""
    lines: list[str] = []
    seen = set()
    for win in windows:
        for seg in win.get("segments", []):
            if seg["end"] < start or seg["start"] > end:
                continue
            key = (round(seg["start"], 1), seg["text"])
            if key in seen or not seg["text"].strip():
                continue
            seen.add(key)
            rs = max(0.0, seg["start"] - start)
            re_ = max(rs, seg["end"] - start)
            lines.append(f"[{rs:.1f}s - {re_:.1f}s] {seg['text'].strip()}")
    return "\n".join(lines) or "(sem fala transcrita)"


def _gemini_retry(fn, attempts: int = 3):
    for i in range(attempts):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001
            if i == attempts - 1:
                raise
            wait = 2 * (i + 1)
            print(f" [erro rede: {type(e).__name__}, tentando em {wait}s]", end="", flush=True)
            time.sleep(wait)


def ask_gemini_video(prompt: str, video_path: Path, model: str) -> dict:
    """Envia o .mp4 curto ao Gemini (inline até 8 MB; Files API resumable acima disso)."""
    from google.genai import types

    client = config.make_gemini_client()
    cfg = types.GenerateContentConfig(response_mime_type="application/json", temperature=0.2)
    size = video_path.stat().st_size
    if size <= 8 * 1024 * 1024:
        part = types.Part.from_bytes(data=video_path.read_bytes(), mime_type="video/mp4")
        resp = _gemini_retry(lambda: client.models.generate_content(model=model, contents=[part, prompt], config=cfg))
        return _parse_json(resp.text or "")

    uploaded = _gemini_retry(lambda: client.files.upload(file=str(video_path)))
    try:
        while getattr(uploaded.state, "name", "") == "PROCESSING":
            time.sleep(1.5)
            uploaded = _gemini_retry(lambda: client.files.get(name=uploaded.name))
        if getattr(uploaded.state, "name", "") == "FAILED":
            raise RuntimeError(f"Falha no processamento do vídeo no Gemini: {video_path.name}")
        resp = _gemini_retry(lambda: client.models.generate_content(model=model, contents=[uploaded, prompt], config=cfg))
        return _parse_json(resp.text or "")
    finally:
        try:
            client.files.delete(name=uploaded.name)
        except Exception:
            pass


def ask_gemini_frames(prompt: str, images_b64: list[str], model: str) -> dict:
    from google.genai import types

    client = config.make_gemini_client()
    parts = [types.Part.from_bytes(data=base64.b64decode(b), mime_type="image/jpeg") for b in images_b64]
    resp = _gemini_retry(lambda: client.models.generate_content(
        model=model,
        contents=[*parts, prompt],
        config=types.GenerateContentConfig(response_mime_type="application/json", temperature=0.2),
    ))
    return _parse_json(resp.text or "")


def _parse_json(text: str) -> dict:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        return json.loads(m.group(0)) if m else {}


# ========================================================= chamada VLM Ollama
def ask_vlm(prompt: str, images_b64: list[str], model: str, url: str) -> dict:
    payload = {
        "model": model,
        "prompt": prompt,
        "images": images_b64,
        "format": "json",
        "stream": False,
        "think": False,
        "keep_alive": "10m",
        "options": {"temperature": 0.2, "num_ctx": 8192},
    }
    r = requests.post(url, json=payload, timeout=600)
    if r.status_code == 404:
        raise RuntimeError(
            f"Modelo de visão '{model}' não encontrado no Ollama. Rode: ollama pull {model}"
        )
    r.raise_for_status()
    text = r.json().get("response", "")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        return json.loads(m.group(0)) if m else {}


def unload_model(model: str, url: str) -> None:
    try:
        requests.post(url, json={"model": model, "keep_alive": 0}, timeout=30)
    except requests.RequestException:
        pass


def build_verify_prompt(clip: dict, frame_times: list[float] | None, transcript: str,
                        topic: str | None) -> str:
    topic_txt = f"Tema da live: {topic}." if topic else "Tema não especificado."
    if frame_times:
        times_txt = ", ".join(f"{t:.1f}s" for t in frame_times)
        media_txt = f"Você está vendo {len(frame_times)} frames em ordem cronológica deste corte nos instantes: [{times_txt}]."
    else:
        media_txt = "Você está assistindo ao vídeo completo (.mp4 com áudio e imagem) deste corte."
    return f"""Você é um editor sênior especialista em retenção e viralização de Shorts/TikTok/Reels.
{topic_txt}
Título proposto: "{clip.get('title', '')}"
Duração do corte: {clip.get('duration', 0):.1f}s.

{media_txt}
Abaixo está a transcrição com timestamps relativos ao início do corte (0.0s):
{transcript}

Avalie com rigor se este corte funciona de forma INDEPENDENTE para quem nunca viu a live:
1. context_complete (bool): O início (0s a 4s) mostra o GATILHO/SETUP do que causa a reação?
   Responda FALSE se o vídeo já começar com risada, grito ou reação sobre algo que aconteceu ANTES do corte começar.
2. webcam_ok (bool): No modo split (com linha verde no meio), o painel superior (acima da legenda)
   mostra APENAS a webcam bem enquadrada com o rosto centralizado?
   Responda FALSE se estiver vazando faixa do jogo no painel superior, se o rosto estiver cortado/descentralizado,
   ou se o painel superior mostrar só jogo sem webcam. (Se o vídeo inteiro for 9:16 sem divisória, responda true).
3. webcam_fix (string): Se webcam_ok=false, escolha UM ajuste:
   - "zoom_in": está vazando jogo nas bordas/embaixo da webcam (precisa dar mais zoom no rosto).
   - "shift_up" / "shift_down" / "shift_left" / "shift_right": rosto descentralizado ou cortado de um lado.
   - "fallback": não existe webcam nessa cena (mudar para tela cheia 9:16).
   - "none": se webcam_ok=true.
4. recoverable (bool): Se ficou ruim por falta de contexto ou webcam_ok=false, dá para salvar? (FALSE se o momento inteiro for sem graça).
5. hook_strength (1 a 10): Os primeiros 3 segundos prendem atenção visual/verbal?
6. context_score (1 a 10): A história faz sentido e mantém bom ritmo? (Cortes longos de 40s–55s são ótimos se tiverem história/diálogo contínuo; pequenas pausas naturais de gameplay de 3s–5s são normais — só reduza bastante a nota se houver 10s+ de tempo morto sem nada acontecendo).
7. climax_payoff (1 a 10): Entrega uma recompensa clara (risada forte, jogada, susto, virada)?
8. visual_quality (1 a 10): Webcam limpa em cima, gameplay embaixo, legenda legível na divisória, sem borda/erro grosseiro?
9. title_match (1 a 10): O título condiz com o que acontece no vídeo?
10. feedback (string curta em PT-BR): Motivo principal da nota ou instrução objetiva do que faltou (ex.: "Vazando jogo abaixo da webcam", "Começa no meio da risada aos 0.5s", "Enrolação/tempo morto no início ou fim, dá para enxugar para ~25s").
(Calibração da escala: 7–8 = corte bom pronto para publicar; 8.5–10 = muito forte/viral; <=6 = realmente fraco ou quebrado.)

Responda SOMENTE com JSON:
{{"context_complete": true, "webcam_ok": true, "webcam_fix": "none", "recoverable": false,
"hook_strength": 8, "context_score": 8, "climax_payoff": 8, "visual_quality": 9, "title_match": 8, "feedback": "..."}}"""


def score_eval(ev: dict) -> float:
    def clamp(k: str) -> float:
        try:
            return max(1.0, min(10.0, float(ev.get(k, 5))))
        except (TypeError, ValueError):
            return 5.0

    score = (
        0.25 * clamp("hook_strength")
        + 0.30 * clamp("context_score")
        + 0.25 * clamp("climax_payoff")
        + 0.10 * clamp("visual_quality")
        + 0.10 * clamp("title_match")
    )
    return round(score, 2)


def evaluate_clip(workdir: Path, clip: dict, windows: list[dict], topic: str | None,
                  vision_model: str, url: str, min_score: float,
                  backend: str = config.VERIFY_BACKEND,
                  verify_mode: str = config.VERIFY_INPUT_MODE,
                  gemini_model: str = config.GEMINI_VERIFY_MODEL) -> dict:
    video_path = workdir / clip["file"]
    transcript = relative_transcript(windows, clip["start_time"], clip["end_time"])

    if backend == "gemini" and verify_mode == "video":
        if not video_path.exists():
            return {**clip, "verify_score": 0.0, "context_complete": False, "recoverable": False,
                    "verify_approved": False, "verify_feedback": "Arquivo mp4 não encontrado"}
        ev = ask_gemini_video(build_verify_prompt(clip, None, transcript, topic),
                              video_path, gemini_model)
    else:
        frames = sample_clip_frames(video_path, config.VERIFY_SAMPLES)
        if not frames:
            return {**clip, "verify_score": 0.0, "context_complete": False, "recoverable": False,
                    "verify_approved": False, "verify_feedback": "Falha ao ler frames do mp4"}
        times = [t for t, _ in frames]
        imgs = [b64 for _, b64 in frames]
        prompt = build_verify_prompt(clip, times, transcript, topic)
        ev = (ask_gemini_frames(prompt, imgs, gemini_model)
              if backend == "gemini" else ask_vlm(prompt, imgs, vision_model, url))
    ctx_ok = bool(ev.get("context_complete", True))
    cam_ok = bool(ev.get("webcam_ok", True))
    cam_fix = str(ev.get("webcam_fix", "none")).strip().lower()
    if cam_fix not in {"zoom_in", "shift_up", "shift_down", "shift_left", "shift_right", "fallback"}:
        cam_fix = "none"
    vscore = score_eval(ev)
    approved = ctx_ok and cam_ok and vscore >= min_score
    return {
        **clip,
        "verify_score": vscore,
        "context_complete": ctx_ok,
        "webcam_ok": cam_ok,
        "webcam_fix": cam_fix,
        "recoverable": bool(ev.get("recoverable", False)),
        "verify_approved": approved,
        "verify_feedback": str(ev.get("feedback", "")).strip(),
        "verify_details": {
            "hook": ev.get("hook_strength"),
            "context": ev.get("context_score"),
            "climax": ev.get("climax_payoff"),
            "visual": ev.get("visual_quality"),
            "title_match": ev.get("title_match"),
        },
    }


# ========================================================= autocorreção (LLM)
def find_source_window(clip: dict, windows: list[dict]) -> dict | None:
    rank = clip.get("window_rank")
    for w in windows:
        if rank is not None and w.get("rank") == rank:
            return w
    # Fallback por maior interseção temporal
    best, best_ov = None, 0.0
    for w in windows:
        ov = min(clip["end_time"], w["end"]) - max(clip["start_time"], w["start"])
        if ov > best_ov:
            best, best_ov = w, ov
    return best


def build_fix_prompt(clip: dict, win: dict, topic: str | None, min_s: int, max_s: int) -> str:
    topic_txt = f"Tema da live: {topic}." if topic else ""
    lines = "\n".join(f"[{s['start']:.1f} - {s['end']:.1f}] {s['text']}" for s in win["segments"])
    return f"""Você é um editor sênior corrigindo um corte que falhou na revisão visual.
{topic_txt}

CORTE ANTERIOR (REPROVADO, nota={clip.get('verify_score', 0):.1f}, context_complete={clip.get('context_complete', True)}):
- start_time={clip['start_time']:.2f}s, end_time={clip['end_time']:.2f}s
- Título: {clip.get('title', '')}
- Motivo da reprovação: {clip.get('verify_feedback', 'Faltou o contexto/setup inicial')}

TRANSCRIÇÃO COMPLETA DA JANELA ({win['start']:.1f}s a {win['end']:.1f}s):
{lines}

TAREFA DE CORREÇÃO:
- Se context_complete=False (faltou contexto): recue start_time em 5s a 20s para incluir o GATILHO/SETUP antes da risada ou reação.
- Se context_complete=True: NÃO corte fora o início/setup que já está funcionando — apenas apare tempo morto explícito na ponta inicial ou final.
- Se teve tempo morto/enrolação sem nada acontecendo: enxugue mantendo só o setup + clímax/reação (só mantenha 36s–55s se todo o trecho tiver conteúdo/diálogo bom).
- Mantenha duração entre {min_s}s e {max_s}s, alinhada a início/fim de frases.
- Se realmente não existir setup/conteúdo para salvar o corte, devolva {{"fixed": false}}.

Responda SOMENTE com JSON:
{{"fixed": true, "start_time": 0.0, "end_time": 0.0, "title": "{clip.get('title', '')}"}}"""


def attempt_fix(clip: dict, win: dict, topic: str | None, ollama_model: str,
                url: str, min_s: int, max_s: int, curator,
                backend: str = config.VERIFY_BACKEND,
                gemini_fix_model: str = config.GEMINI_CURATOR_MODEL) -> dict | None:
    prompt = build_fix_prompt(clip, win, topic, min_s, max_s)
    resp = (curator.ask_gemini_json(prompt, gemini_fix_model, temperature=0.2)
            if backend == "gemini" else curator.ask_ollama(prompt, ollama_model, url))
    if not isinstance(resp, dict) or not resp.get("fixed", True):
        return None
    try:
        s, e = float(resp["start_time"]), float(resp["end_time"])
    except (KeyError, TypeError, ValueError):
        return None
    s, e = max(s, win["start"]), min(e, win["end"])
    if e <= s:
        return None
    snapped = curator.snap_clip(win, s, e, min_s, max_s)
    if not snapped:
        return None
    ns, ne = snapped
    if abs(ns - clip["start_time"]) < 1.5 and abs(ne - clip["end_time"]) < 1.5:
        return None  # não mudou quase nada, não vale re-renderizar
    title = str(resp.get("title") or clip["title"]).strip()[:100]
    return {**clip, "start_time": ns, "end_time": ne, "duration": round(ne - ns, 2), "title": title}


# ============================================================== orquestração
def print_summary(items: list[dict], min_score: float) -> None:
    t = Table(title=f"Verificação Multimodal (nota mínima {min_score:.1f})")
    t.add_column("Corte")
    t.add_column("Nota", justify="right")
    t.add_column("Contexto")
    t.add_column("Status")
    t.add_column("Feedback")
    for it in items:
        st = ("[green]APROVADO[/green]" if it["verify_approved"]
              else "[red]REPROVADO[/red]")
        if it.get("auto_fixed"):
            st += " [cyan](corrigido)[/cyan]"
        ctx = "[green]ok[/green]" if it["context_complete"] else "[red]incompleto[/red]"
        t.add_row(it["file"], f"{it['verify_score']:.1f}", ctx, st, it["verify_feedback"][:75])
    console.print(t)


def apply_webcam_fix(c_info: dict, cam_fix: str, vr) -> dict:
    """Ajusta o recorte da webcam conforme diagnóstico do VLM (zoom, deslocamento ou fallback)."""
    if cam_fix == "fallback" or c_info.get("mode") != "split":
        return {"mode": "fallback", "W": c_info["W"], "H": c_info["H"], "face": None}
    tw, th, tx, ty = c_info["top_crop"]
    W, H = c_info["W"], c_info["H"]
    face = c_info.get("face")
    if cam_fix == "zoom_in":
        nw = max(160, int(tw * 0.78) // 2 * 2)
        nh = max(100, int(nw / vr.TOP_ASPECT) // 2 * 2)
        cx = (face["x"] + face["w"] / 2) if face else (tx + tw / 2)
        cy = (face["y"] + face["h"] / 2) if face else (ty + th / 2)
        nx = min(max(int(cx - nw / 2) // 2 * 2, 0), W - nw)
        ny = min(max(int(cy - 0.45 * nh) // 2 * 2, 0), H - nh)
        return {**c_info, "top_crop": (nw, nh, nx, ny)}
    dx = int(tw * 0.14) // 2 * 2
    dy = int(th * 0.14) // 2 * 2
    if cam_fix == "shift_left":
        tx = max(0, tx - dx)
    elif cam_fix == "shift_right":
        tx = min(W - tw, tx + dx)
    elif cam_fix == "shift_up":
        ty = max(0, ty - dy)
    elif cam_fix == "shift_down":
        ty = min(H - th, ty + dy)
    return {**c_info, "top_crop": (tw, th, tx, ty)}


def run_verify(workdir: Path, vision_model: str, ollama_model: str, url: str,
               min_score: float, min_s: int, max_s: int, topic: str | None,
               backend: str = config.VERIFY_BACKEND,
               verify_mode: str = config.VERIFY_INPUT_MODE,
               gemini_model: str = config.GEMINI_VERIFY_MODEL,
               gemini_fix_model: str = config.GEMINI_CURATOR_MODEL,
               watermark: bool = config.WATERMARK_ENABLED,
               watermark_text: str = config.WATERMARK_TEXT) -> list[dict]:
    rend_p = workdir / config.RENDERED_NAME
    trans_p = workdir / config.TRANSCRIBED_NAME
    for p in (rend_p, trans_p):
        if not p.exists():
            raise SystemExit(f"{p} não encontrado. Rode os módulos anteriores.")

    clips = json.loads(rend_p.read_text(encoding="utf-8"))
    windows = json.loads(trans_p.read_text(encoding="utf-8"))
    if topic is None and (workdir / config.META_NAME).exists():
        topic = json.loads((workdir / config.META_NAME).read_text(encoding="utf-8")).get("topic")
    if not clips:
        print("Nenhum corte renderizado para verificar.")
        return []

    active_vlm = f"{gemini_model} ({verify_mode})" if backend == "gemini" else vision_model
    active_fix = gemini_fix_model if backend == "gemini" else ollama_model

    # 1) Primeira passada com VLM em lote
    console.print(f"[bold][1/4] Verificando {len(clips)} corte(s) com {active_vlm}...[/bold]")
    evaluated: list[dict] = []
    try:
        for i, c in enumerate(clips, 1):
            print(f"  [{i}/{len(clips)}] {c['file']} ({c['duration']:.0f}s)...", end=" ", flush=True)
            try:
                res = evaluate_clip(
                    workdir, c, windows, topic, vision_model, url, min_score,
                    backend=backend, verify_mode=verify_mode, gemini_model=gemini_model,
                )
            except Exception as ex:  # noqa: BLE001 - erro em 1 corte não derruba os demais
                res = {**c, "verify_score": 0.0, "context_complete": False, "webcam_ok": True,
                       "webcam_fix": "none", "recoverable": False, "verify_approved": False,
                       "verify_feedback": f"Erro na API: {type(ex).__name__}"}
            print(f"nota={res['verify_score']:.1f} ctx={'ok' if res['context_complete'] else 'FALTA'} "
                  f"cam={'ok' if res.get('webcam_ok', True) else res.get('webcam_fix', 'none')}")
            evaluated.append(res)
    finally:
        if backend == "ollama":
            unload_model(vision_model, url)

    # 2) Identifica cortes recuperáveis: falta de contexto OU webcam mal enquadrada
    ctx_indices = [
        idx for idx, ev in enumerate(evaluated)
        if not ev["verify_approved"] and (not ev["context_complete"] or ev["recoverable"])
    ]
    cam_indices = {
        idx: ev.get("webcam_fix", "none")
        for idx, ev in enumerate(evaluated)
        if not ev["verify_approved"] and not ev.get("webcam_ok", True) and ev.get("webcam_fix", "none") != "none"
    }
    adjusted: dict[int, dict] = {}
    if ctx_indices:
        curator = importlib.import_module("04_curator")
        console.print(f"[bold][2/4] Tentando recuperar contexto de {len(ctx_indices)} corte(s) com {active_fix}...[/bold]")
        try:
            for idx in ctx_indices:
                ev = evaluated[idx]
                win = find_source_window(ev, windows)
                if not win:
                    continue
                new_clip = attempt_fix(
                    ev, win, topic, ollama_model, url, min_s, max_s, curator,
                    backend=backend, gemini_fix_model=gemini_fix_model,
                )
                if new_clip:
                    print(f"  {ev['file']}: {ev['start_time']:.1f}-{ev['end_time']:.1f}s -> "
                          f"{new_clip['start_time']:.1f}-{new_clip['end_time']:.1f}s")
                    adjusted[idx] = new_clip
                else:
                    print(f"  {ev['file']}: tempo não alterado.")
        finally:
            if backend == "ollama":
                unload_model(ollama_model, url)
    else:
        console.print("[dim][2/4] Nenhum corte precisou de ajuste de tempo.[/dim]")

    for idx in cam_indices:
        adjusted.setdefault(idx, evaluated[idx])

    # 3) Re-renderiza apenas os cortes corrigidos (tempo e/ou webcam)
    if adjusted:
        vr = importlib.import_module("05_vision_render")
        video = workdir / config.VIDEO_NAME
        info = vr.detect_webcam(video, workdir)
        finder = vr.FaceFinder()
        fonts_rel = None
        if config.FONTS_DIR.exists() and any(config.FONTS_DIR.iterdir()):
            shutil.copytree(config.FONTS_DIR, workdir / "fonts", dirs_exist_ok=True)
            fonts_rel = "fonts"
        console.print(f"[bold][3/4] Re-renderizando {len(adjusted)} corte(s) ajustado(s)...[/bold]")
        backups: dict[int, tuple[dict, bytes, bytes, bytes]] = {}
        for idx, nc in adjusted.items():
            stem = f"corte_{nc['index']:02d}"
            backups[idx] = (
                evaluated[idx],
                (workdir / f"{stem}.mp4").read_bytes(),
                (workdir / f"{stem}.ass").read_bytes(),
                (workdir / f"{stem}.txt").read_bytes(),
            )
            c_info = vr.clip_webcam_info(video, nc, info, finder)
            if idx in cam_indices:
                c_info = apply_webcam_fix(c_info, cam_indices[idx], vr)
                print(f"  {nc['file']}: ajuste de webcam -> {cam_indices[idx]}")
            words = vr.clip_words(windows, nc["start_time"], nc["end_time"])
            vr.render_clip(workdir, nc, nc["index"], c_info, words, fonts_rel, watermark, watermark_text)
            vr.write_txt(workdir / f"{stem}.txt", nc)

        # 4) Re-verifica apenas os corrigidos com o VLM
        console.print(f"[bold][4/4] Re-verificando {len(adjusted)} corte(s) com {active_vlm}...[/bold]")
        try:
            for idx, nc in adjusted.items():
                res = evaluate_clip(
                    workdir, nc, windows, topic, vision_model, url, min_score,
                    backend=backend, verify_mode=verify_mode, gemini_model=gemini_model,
                )
                old_ev, b_mp4, b_ass, b_txt = backups[idx]
                if res["verify_score"] < old_ev["verify_score"]:
                    # Rollback: correção piorou o corte -> restaura versão original
                    stem = f"corte_{nc['index']:02d}"
                    (workdir / f"{stem}.mp4").write_bytes(b_mp4)
                    (workdir / f"{stem}.ass").write_bytes(b_ass)
                    (workdir / f"{stem}.txt").write_bytes(b_txt)
                    print(f"  {res['file']}: correção piorou ({old_ev['verify_score']:.1f} -> "
                          f"{res['verify_score']:.1f}), restaurado original.")
                else:
                    res["auto_fixed"] = True
                    evaluated[idx] = res
                    print(f"  {res['file']}: nova nota={res['verify_score']:.1f} "
                          f"ctx={'ok' if res['context_complete'] else 'FALTA'} "
                          f"cam={'ok' if res.get('webcam_ok', True) else 'RUIM'}")
        finally:
            if backend == "ollama":
                unload_model(vision_model, url)

        # Atualiza rendered_clips.json com os novos tempos dos cortes corrigidos
        updated_rendered = [{k: v for k, v in ev.items() if not k.startswith("verify_")
                             and k not in {"context_complete", "webcam_ok", "webcam_fix", "recoverable", "auto_fixed"}}
                            for ev in evaluated]
        rend_p.write_text(json.dumps(updated_rendered, ensure_ascii=False, indent=1), encoding="utf-8")

    print_summary(evaluated, min_score)
    (workdir / config.VERIFY_REPORT_NAME).write_text(
        json.dumps(evaluated, ensure_ascii=False, indent=1), encoding="utf-8")

    approved = [ev for ev in evaluated if ev["verify_approved"]]
    approved.sort(key=lambda c: (-c["verify_score"], -c.get("virality_score", 0)))
    (workdir / config.VERIFIED_NAME).write_text(
        json.dumps(approved, ensure_ascii=False, indent=1), encoding="utf-8")
    console.print(f"\n{len(approved)}/{len(evaluated)} corte(s) aprovado(s) -> {workdir / config.VERIFIED_NAME}")
    return approved


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("workdir", type=Path)
    ap.add_argument("--backend", choices=["gemini", "ollama"], default=config.VERIFY_BACKEND)
    ap.add_argument("--verify-mode", choices=["video", "frames"], default=config.VERIFY_INPUT_MODE)
    ap.add_argument("--gemini-model", default=config.GEMINI_VERIFY_MODEL)
    ap.add_argument("--gemini-fix-model", default=config.GEMINI_CURATOR_MODEL)
    ap.add_argument("--vision-model", default=config.OLLAMA_VISION_MODEL)
    ap.add_argument("--min-score", type=float, default=config.VERIFY_MIN_SCORE)
    ap.add_argument("--ollama-url", default=config.OLLAMA_URL)
    ap.add_argument("--watermark", action="store_true", default=config.WATERMARK_ENABLED)
    ap.add_argument("--watermark-text", default=config.WATERMARK_TEXT)
    config.add_common_args(ap)
    args = ap.parse_args()
    run_verify(args.workdir, args.vision_model, args.ollama_model, args.ollama_url,
               args.min_score, args.min_clip, args.max_clip, args.topic,
               backend=args.backend, verify_mode=args.verify_mode,
               gemini_model=args.gemini_model, gemini_fix_model=args.gemini_fix_model,
               watermark=args.watermark, watermark_text=args.watermark_text)


if __name__ == "__main__":
    main()

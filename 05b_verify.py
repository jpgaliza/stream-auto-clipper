"""Módulo 5b - Verificador multimodal com autocorreção (Ollama Vision + LLM).

Uso:
    python 05b_verify.py <workdir> [--min-score 7.0] [--vision-model qwen2.5vl:7b]

Fluxo em lote (respeita 8 GB de VRAM):
1. VLM (qwen2.5vl:7b) assiste frames-chave + transcrição de cada corte_XX.mp4 e avalia
   hook, completude do contexto (setup antes da risada/clímax), qualidade visual e título.
   Descarrega da VRAM.
2. Cortes reprovados por falta de setup/corte tardio vão em lote ao LLM (qwen3.5:9b) junto
   com o feedback do VLM e a janela completa de candidates_transcribed.json para recuar o
   start_time (1 tentativa por corte). Descarrega da VRAM.
3. Re-renderiza apenas os cortes corrigidos (reaproveitando webcam.json).
4. Re-verifica apenas os corrigidos com o VLM e grava verified_clips.json + verify_report.json.
"""
from __future__ import annotations

import argparse
import base64
import importlib
import json
import re
import shutil
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


def build_verify_prompt(clip: dict, frame_times: list[float], transcript: str,
                        topic: str | None) -> str:
    topic_txt = f"Tema da live: {topic}." if topic else "Tema não especificado."
    times_txt = ", ".join(f"{t:.1f}s" for t in frame_times)
    return f"""Você é um editor sênior especialista em retenção e viralização de Shorts/TikTok/Reels.
{topic_txt}
Título proposto: "{clip.get('title', '')}"
Duração do corte: {clip.get('duration', 0):.1f}s.

Você está vendo {len(frame_times)} frames em ordem cronológica deste corte nos instantes: [{times_txt}].
Abaixo está a transcrição com timestamps relativos ao início do corte (0.0s):
{transcript}

Avalie com rigor se este corte funciona de forma INDEPENDENTE para quem nunca viu a live:
1. context_complete (bool): O início (0s a 4s) mostra o GATILHO/SETUP do que causa a reação?
   Responda FALSE se o vídeo já começar com risada, grito ou reação sobre algo que aconteceu ANTES do corte começar.
2. recoverable (bool): Se ficou ruim ou sem contexto, daria para salvar voltando 5s–20s antes na live para pegar o setup? (FALSE se o momento inteiro for fraco/sem graça).
3. hook_strength (1 a 10): Os primeiros 3 segundos prendem atenção visual/verbal?
4. context_score (1 a 10): A história faz sentido do começo ao fim sem parecer cortada?
5. climax_payoff (1 a 10): Entrega uma recompensa clara (risada forte, jogada, susto, virada)?
6. visual_quality (1 a 10): Webcam limpa em cima, gameplay embaixo, legenda legível na divisória, sem borda/erro grosseiro?
7. title_match (1 a 10): O título condiz com o que acontece no vídeo?
8. feedback (string curta em PT-BR): Motivo principal da nota ou instrução objetiva do que faltou (ex.: "Começa no meio da risada aos 0.5s sem mostrar o lance anterior").

Responda SOMENTE com JSON:
{{"context_complete": true, "recoverable": false, "hook_strength": 8, "context_score": 8,
"climax_payoff": 8, "visual_quality": 9, "title_match": 8, "feedback": "..."}}"""


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
                  vision_model: str, url: str, min_score: float) -> dict:
    frames = sample_clip_frames(workdir / clip["file"], config.VERIFY_SAMPLES)
    if not frames:
        return {**clip, "verify_score": 0.0, "context_complete": False, "recoverable": False,
                "verify_approved": False, "verify_feedback": "Falha ao ler frames do mp4"}
    times = [t for t, _ in frames]
    imgs = [b64 for _, b64 in frames]
    transcript = relative_transcript(windows, clip["start_time"], clip["end_time"])
    ev = ask_vlm(build_verify_prompt(clip, times, transcript, topic), imgs, vision_model, url)
    ctx_ok = bool(ev.get("context_complete", True))
    vscore = score_eval(ev)
    approved = ctx_ok and vscore >= min_score
    return {
        **clip,
        "verify_score": vscore,
        "context_complete": ctx_ok,
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

CORTE ANTERIOR (REPROVADO):
- start_time={clip['start_time']:.2f}s, end_time={clip['end_time']:.2f}s
- Título: {clip.get('title', '')}
- Motivo da reprovação: {clip.get('verify_feedback', 'Faltou o contexto/setup inicial')}

TRANSCRIÇÃO COMPLETA DA JANELA ({win['start']:.1f}s a {win['end']:.1f}s):
{lines}

TAREFA DE CORREÇÃO:
- Ajuste start_time e end_time dentro desta janela para que o corte inclua o GATILHO/SETUP antes da
  risada ou reação (normalmente recuando o start_time em 5s a 25s).
- Mantenha duração entre {min_s}s e {max_s}s, alinhada a início/fim de frases.
- Se realmente não existir setup suficiente na janela para salvar o corte, devolva {{"fixed": false}}.

Responda SOMENTE com JSON:
{{"fixed": true, "start_time": 0.0, "end_time": 0.0, "title": "{clip.get('title', '')}"}}"""


def attempt_fix(clip: dict, win: dict, topic: str | None, ollama_model: str,
                url: str, min_s: int, max_s: int, curator) -> dict | None:
    resp = curator.ask_ollama(build_fix_prompt(clip, win, topic, min_s, max_s), ollama_model, url)
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


def run_verify(workdir: Path, vision_model: str, ollama_model: str, url: str,
               min_score: float, min_s: int, max_s: int, topic: str | None) -> list[dict]:
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

    # 1) Primeira passada com VLM em lote
    console.print(f"[bold][1/4] Verificando {len(clips)} corte(s) com {vision_model}...[/bold]")
    evaluated: list[dict] = []
    try:
        for i, c in enumerate(clips, 1):
            print(f"  [{i}/{len(clips)}] {c['file']} ({c['duration']:.0f}s)...", end=" ", flush=True)
            res = evaluate_clip(workdir, c, windows, topic, vision_model, url, min_score)
            print(f"nota={res['verify_score']:.1f} ctx={'ok' if res['context_complete'] else 'FALTA'}")
            evaluated.append(res)
    finally:
        unload_model(vision_model, url)

    # 2) Identifica cortes recuperáveis (sem contexto ou recuperáveis com nota baixa)
    fix_indices = [
        idx for idx, ev in enumerate(evaluated)
        if not ev["verify_approved"] and (not ev["context_complete"] or ev["recoverable"])
    ]
    fixed_clips: list[tuple[int, dict]] = []
    if fix_indices:
        curator = importlib.import_module("04_curator")
        console.print(f"[bold][2/4] Tentando recuperar {len(fix_indices)} corte(s) com {ollama_model}...[/bold]")
        try:
            for idx in fix_indices:
                ev = evaluated[idx]
                win = find_source_window(ev, windows)
                if not win:
                    continue
                new_clip = attempt_fix(ev, win, topic, ollama_model, url, min_s, max_s, curator)
                if new_clip:
                    print(f"  {ev['file']}: {ev['start_time']:.1f}-{ev['end_time']:.1f}s -> "
                          f"{new_clip['start_time']:.1f}-{new_clip['end_time']:.1f}s")
                    fixed_clips.append((idx, new_clip))
                else:
                    print(f"  {ev['file']}: não recuperável na janela.")
        finally:
            unload_model(ollama_model, url)
    else:
        console.print("[dim][2/4] Nenhum corte precisou de autocorreção.[/dim]")

    # 3) Re-renderiza apenas os cortes corrigidos
    if fixed_clips:
        vr = importlib.import_module("05_vision_render")
        info = vr.detect_webcam(workdir / config.VIDEO_NAME, workdir)
        fonts_rel = None
        if config.FONTS_DIR.exists() and any(config.FONTS_DIR.iterdir()):
            shutil.copytree(config.FONTS_DIR, workdir / "fonts", dirs_exist_ok=True)
            fonts_rel = "fonts"
        console.print(f"[bold][3/4] Re-renderizando {len(fixed_clips)} corte(s) ajustado(s)...[/bold]")
        for _, nc in fixed_clips:
            words = vr.clip_words(windows, nc["start_time"], nc["end_time"])
            vr.render_clip(workdir, nc, nc["index"], info, words, fonts_rel)
            vr.write_txt(workdir / f"corte_{nc['index']:02d}.txt", nc)

        # 4) Re-verifica apenas os corrigidos com o VLM
        console.print(f"[bold][4/4] Re-verificando {len(fixed_clips)} corte(s) com {vision_model}...[/bold]")
        try:
            for idx, nc in fixed_clips:
                res = evaluate_clip(workdir, nc, windows, topic, vision_model, url, min_score)
                res["auto_fixed"] = True
                evaluated[idx] = res
                print(f"  {res['file']}: nova nota={res['verify_score']:.1f} "
                      f"ctx={'ok' if res['context_complete'] else 'FALTA'}")
        finally:
            unload_model(vision_model, url)

        # Atualiza rendered_clips.json com os novos tempos dos cortes corrigidos
        updated_rendered = [{k: v for k, v in ev.items() if not k.startswith("verify_")
                             and k not in {"context_complete", "recoverable", "auto_fixed"}}
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
    ap.add_argument("--vision-model", default=config.OLLAMA_VISION_MODEL)
    ap.add_argument("--min-score", type=float, default=config.VERIFY_MIN_SCORE)
    ap.add_argument("--ollama-url", default=config.OLLAMA_URL)
    config.add_common_args(ap)
    args = ap.parse_args()
    run_verify(args.workdir, args.vision_model, args.ollama_model, args.ollama_url,
               args.min_score, args.min_clip, args.max_clip, args.topic)


if __name__ == "__main__":
    main()

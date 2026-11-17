"""Módulo 4 - Curadoria por LLM (Ollama).

Uso:
    python 04_curator.py <workdir> [--topic "..."]

Lê candidates_transcribed.json, pergunta ao Ollama (format=json) quais cortes
valem a pena, ajusta start/end para fronteiras de frase e grava approved_clips.json
ordenado por virality_score.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import requests

import config

SENTENCE_END = re.compile(r"[.!?…]+[\"')\]]*$")


# ------------------------------------------------------------------- prompt
def fmt_ts(t: float) -> str:
    return f"{t:.1f}"


def chat_reaction_summary(chat: list[dict], start: float, end: float) -> str:
    """Resume em quais segundos desta janela o chat explodiu em risadas/reações."""
    if not chat:
        return ""
    pat = re.compile("|".join(config.REACTION_PATTERNS), re.IGNORECASE)
    buckets: dict[int, int] = {}
    for m in chat:
        t = m["timestamp"]
        if start <= t <= end and pat.search(m["message"]):
            b = int(t // 5) * 5
            buckets[b] = buckets.get(b, 0) + 1
    top = sorted(buckets.items(), key=lambda kv: -kv[1])[:4]
    top = [(t, n) for t, n in top if n >= 2]
    if not top:
        return ""
    pts = ", ".join(f"{t}s ({n} reações)" for t, n in sorted(top))
    return f"Explosões de risadas/reações no chat nesta janela: {pts}.\n"


def build_prompt(win: dict, topic: str | None, min_s: int, max_s: int,
                 chat_summary: str = "") -> str:
    topic_line = (f"Tema/contexto da live informado pelo usuário: {topic}."
                  if topic else "Nenhum tema específico informado; julgue pelo conteúdo.")
    lines = "\n".join(
        f"[{fmt_ts(s['start'])} - {fmt_ts(s['end'])}] {s['text']}" for s in win["segments"]
    )
    return f"""Você é um editor sênior de Shorts/TikTok/Reels que corta momentos de lives.
{topic_line}

Abaixo está a transcrição (português) de um trecho de live, com timestamps em segundos
do vídeo original. Sinais do radar (0 a 1): energia de áudio={win['audio_score']:.2f},
atividade do chat={win['chat_score']:.2f}; pico principal em {fmt_ts(win['peak_time'])}s.
{chat_summary}Trecho: {fmt_ts(win['start'])}s a {fmt_ts(win['end'])}s.

TRANSCRIÇÃO:
{lines}

TAREFA:
- Selecione APENAS momentos realmente engraçados, tensos, surpreendentes ou de alta retenção.
  Dê ênfase especial também a risadas fortes, crises de riso e momentos cômicos (não apenas tensão).
  Se nada prestar, devolva uma lista vazia. Não force cortes.
- Cada corte deve ter narrativa completa: contexto/gatilho (setup), desenvolvimento e reação/clímax.
- REGRA DO DELAY DE CHAT E RISADA (MUITO IMPORTANTE):
  O `peak_time` ({fmt_ts(win['peak_time'])}s) e as explosões de "kkkkk"/risadas no chat acontecem
  5s a 20s DEPOIS do momento que gerou a graça!
  NUNCA comece o corte em cima do `peak_time` ou já no meio da risada: o `start_time` DEVE começar
  pelo menos 10s a 20s ANTES da risada/pico para incluir a jogada, fala ou situação que provocou tudo.
- Duração elástica entre {min_s}s e {max_s}s. Comece onde o contexto começa e termine logo após o
  clímax/reação acabar. NÃO estique um momento curto (ex.: 18s) só para alcançar 50s. Priorize
  cortes completos de {config.TARGET_MIN_SECONDS}s a {config.TARGET_MAX_SECONDS}s quando houver narrativa.
- start_time e end_time devem ser timestamps em segundos do vídeo original, dentro do trecho,
  alinhados a início/fim de frases (nunca no meio de uma palavra).
- title: curto, magnético, honesto (sem clickbait mentiroso). description: 1-2 frases.
- hashtags: lista de 3 a 6, incluindo "#shorts".
- virality_score: inteiro de 1 a 10. reason: por que esse corte retém a audiência.

Responda SOMENTE com JSON neste formato:
{{"clips": [{{"start_time": 0.0, "end_time": 0.0, "title": "", "description": "",
"hashtags": ["#shorts"], "virality_score": 7, "reason": ""}}]}}"""


# ------------------------------------------------------------------- ollama
def ask_ollama(prompt: str, model: str, url: str) -> dict:
    payload = {
        "model": model,
        "prompt": prompt,
        "format": "json",
        "stream": False,
        "think": False,  # sem cadeia de raciocínio: só o JSON
        "keep_alive": "10m",
        "options": {"temperature": 0.3, "num_ctx": config.OLLAMA_NUM_CTX},
    }
    r = requests.post(url, json=payload, timeout=600)
    r.raise_for_status()
    text = r.json().get("response", "")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        return json.loads(m.group(0)) if m else {"clips": []}


def unload_model(model: str, url: str) -> None:
    try:
        requests.post(url, json={"model": model, "keep_alive": 0}, timeout=30)
    except requests.RequestException:
        pass


# ------------------------------------------------------- snap às frases
def boundaries(win: dict) -> tuple[list[float], list[float]]:
    """Candidatos a início (começo de frase) e fim (fim de frase)."""
    starts, ends = [], []
    for seg in win["segments"]:
        words = seg.get("words") or []
        if not words:
            starts.append(seg["start"])
            ends.append(seg["end"])
            continue
        starts.append(words[0]["start"])
        ends.append(words[-1]["end"])
        for i, w in enumerate(words):
            if i > 0 and SENTENCE_END.search(words[i - 1]["word"].strip()):
                starts.append(w["start"])
            if i < len(words) - 1 and SENTENCE_END.search(w["word"].strip()):
                ends.append(w["end"])
    return sorted(set(starts)), sorted(set(ends))


def nearest(vals: list[float], t: float, tol: float) -> float | None:
    if not vals:
        return None
    best = min(vals, key=lambda v: abs(v - t))
    return best if abs(best - t) <= tol else None


def snap_clip(win: dict, start: float, end: float, min_s: int, max_s: int) -> tuple[float, float] | None:
    starts, ends = boundaries(win)
    tol = config.SNAP_TOLERANCE_SECONDS
    s = nearest(starts, start, tol)
    e = nearest(ends, end, tol)
    all_words = [w for seg in win["segments"] for w in seg.get("words", [])]
    if s is None and all_words:   # fallback: palavra mais próxima (nunca no meio dela)
        s = min((w["start"] for w in all_words), key=lambda v: abs(v - start))
    if e is None and all_words:
        e = min((w["end"] for w in all_words), key=lambda v: abs(v - end))
    s = start if s is None else s
    e = end if e is None else e
    # Duração acima do limite: recua para o fim de frase mais próximo que caiba
    if e - s > max_s:
        fits = [v for v in ends if s < v <= s + max_s]
        if not fits:
            return None
        e = max(fits)
    # Pequena folga para não engolir consoantes
    s = max(win["start"], s - 0.15)
    e = min(win["end"], e + 0.25)
    if e - s > max_s:
        e = s + max_s
    if e - s < min_s:
        return None
    return round(s, 2), round(e, 2)


# ------------------------------------------------------------- validação
def clean_clip(raw: dict, win: dict, min_s: int, max_s: int) -> dict | None:
    try:
        s, e = float(raw["start_time"]), float(raw["end_time"])
        score = max(1, min(10, int(round(float(raw.get("virality_score", 0))))))
    except (KeyError, TypeError, ValueError):
        return None
    s, e = max(s, win["start"]), min(e, win["end"])
    if e <= s:
        return None
    snapped = snap_clip(win, s, e, min_s, max_s)
    if not snapped:
        return None
    s, e = snapped
    tags = [("#" + str(t).lstrip("#").replace(" ", "")) for t in raw.get("hashtags", []) if str(t).strip()]
    if "#shorts" not in [t.lower() for t in tags]:
        tags.append("#shorts")
    return {
        "start_time": s,
        "end_time": e,
        "duration": round(e - s, 2),
        "title": str(raw.get("title", "")).strip()[:100],
        "description": str(raw.get("description", "")).strip(),
        "hashtags": tags,
        "virality_score": score,
        "reason": str(raw.get("reason", "")).strip(),
        "window_rank": win.get("rank"),
        "radar_score": win.get("score"),
    }


def dedupe(clips: list[dict]) -> list[dict]:
    """Remove cortes que se sobrepõem >50% a outro de nota maior."""
    kept: list[dict] = []
    for c in sorted(clips, key=lambda c: (-c["virality_score"], -(c["radar_score"] or 0))):
        dup = False
        for k in kept:
            ov = min(c["end_time"], k["end_time"]) - max(c["start_time"], k["start_time"])
            if ov > 0.5 * min(c["duration"], k["duration"]):
                dup = True
                break
        if not dup:
            kept.append(c)
    return kept


# -------------------------------------------------------------------- main
def curate(workdir: Path, topic: str | None, model: str, url: str,
           min_s: int, max_s: int) -> list[dict]:
    src = workdir / config.TRANSCRIBED_NAME
    if not src.exists():
        raise SystemExit(f"{src} não encontrado. Rode 03_transcribe.py primeiro.")
    windows = json.loads(src.read_text(encoding="utf-8"))

    if topic is None:
        meta = workdir / config.META_NAME
        if meta.exists():
            topic = json.loads(meta.read_text(encoding="utf-8")).get("topic")

    chat: list[dict] = []
    chat_p = workdir / config.CHAT_NAME
    if chat_p.exists():
        try:
            chat = json.loads(chat_p.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            pass

    approved: list[dict] = []
    try:
        for n, win in enumerate(windows, 1):
            if not win.get("segments"):
                print(f"[{n}/{len(windows)}] sem fala, pulando")
                continue
            print(f"[{n}/{len(windows)}] {win['start']:.0f}s-{win['end']:.0f}s ...", end=" ")
            try:
                csum = chat_reaction_summary(chat, win["start"], win["end"])
                resp = ask_ollama(build_prompt(win, topic, min_s, max_s, csum), model, url)
            except (requests.RequestException, json.JSONDecodeError) as ex:
                print(f"erro ({type(ex).__name__}); pulando")
                continue
            raw_clips = resp.get("clips", []) if isinstance(resp, dict) else []
            good = [c for c in (clean_clip(r, win, min_s, max_s) for r in raw_clips) if c]
            print(f"{len(good)} corte(s)")
            approved.extend(good)
    finally:
        unload_model(model, url)

    approved = dedupe(approved)
    approved.sort(key=lambda c: (-c["virality_score"], -(c["radar_score"] or 0)))
    for i, c in enumerate(approved, 1):
        c["rank"] = i
    return approved


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("workdir", type=Path)
    ap.add_argument("--ollama-url", default=config.OLLAMA_URL)
    config.add_common_args(ap)
    args = ap.parse_args()

    clips = curate(args.workdir, args.topic, args.ollama_model, args.ollama_url,
                   args.min_clip, args.max_clip)
    out = args.workdir / config.APPROVED_NAME
    out.write_text(json.dumps(clips, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n{len(clips)} cortes aprovados -> {out}")
    for c in clips[:10]:
        print(f"  #{c['rank']} [{c['virality_score']}/10] {c['start_time']:.1f}-{c['end_time']:.1f}s "
              f"({c['duration']:.0f}s) {c['title']}")


if __name__ == "__main__":
    main()

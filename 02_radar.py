"""Módulo 2 - Radar de momentos.

Uso:
    python 02_radar.py <workdir>

Lê audio.wav (+ chat_normalized.json se houver) e escreve candidates.json
com as top 40 janelas candidatas (-45s/+20s em torno de cada pico, mescladas).
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import librosa
import numpy as np
from scipy.signal import find_peaks

import config


# --------------------------------------------------------------- normalização
def robust_normalize(x: np.ndarray) -> np.ndarray:
    """Normaliza em [0,1] usando mediana/MAD (robusto a outliers) + clip."""
    if x.size == 0:
        return x
    med = np.median(x)
    mad = np.median(np.abs(x - med)) * 1.4826
    if mad < 1e-9:
        mad = x.std() or 1.0
    z = np.clip((x - med) / mad, 0, 6)  # só interessa o que está acima do normal
    return z / 6.0


# --------------------------------------------------------------------- áudio
def audio_scores(wav: Path, win: float) -> tuple[np.ndarray, float]:
    y, sr = librosa.load(str(wav), sr=config.AUDIO_SAMPLE_RATE, mono=True)
    hop = int(win * sr)
    rms = librosa.feature.rms(y=y, frame_length=hop, hop_length=hop, center=False)[0]
    return rms.astype(np.float64), len(y) / sr


# ---------------------------------------------------------------------- chat
def load_chat(workdir: Path) -> list[dict]:
    p = workdir / config.CHAT_NAME
    if not p.exists():
        return []
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []


def chat_scores(chat: list[dict], n_windows: int, win: float,
                extra_patterns: list[str] | None = None) -> np.ndarray:
    """MPS por janela multiplicado por (1 + proporção de mensagens de reação)."""
    pat = re.compile("|".join(config.REACTION_PATTERNS + (extra_patterns or [])),
                     re.IGNORECASE)
    counts = np.zeros(n_windows)
    hits = np.zeros(n_windows)
    for m in chat:
        i = int(m["timestamp"] // win)
        if 0 <= i < n_windows:
            counts[i] += 1
            if pat.search(m["message"]):
                hits[i] += 1
    mps = counts / win
    reaction_ratio = np.divide(hits, counts, out=np.zeros_like(hits), where=counts > 0)
    multiplier = 1.0 + 2.0 * reaction_ratio  # 1x (neutro) até 3x (tudo reação)
    return mps * multiplier


# ------------------------------------------------------------------- janelas
def merge_windows(wins: list[dict]) -> list[dict]:
    wins = sorted(wins, key=lambda w: w["start"])
    merged: list[dict] = []
    for w in wins:
        # Mescla sobreposições, mas limita o tamanho para não encadear o VOD inteiro
        if (merged and w["start"] <= merged[-1]["end"]
                and max(merged[-1]["end"], w["end"]) - merged[-1]["start"]
                <= config.RADAR_MAX_MERGED_SECONDS):
            last = merged[-1]
            last["end"] = max(last["end"], w["end"])
            last["peaks"].extend(w["peaks"])
            if w["score"] > last["score"]:
                last["score"], last["peak_time"] = w["score"], w["peak_time"]
            for k in ("audio_score", "chat_score"):
                last[k] = max(last[k], w[k])
        else:
            merged.append({**w, "peaks": list(w["peaks"])})
    return merged


def find_candidates(combined: np.ndarray, audio_n: np.ndarray, chat_n: np.ndarray,
                    duration: float, win: float, top_n: int) -> list[dict]:
    peaks, _ = find_peaks(combined, distance=config.RADAR_MIN_PEAK_DISTANCE_WINDOWS)
    if peaks.size == 0:
        return []
    # Sobre-amostra picos: a mesclagem reduz a contagem final.
    order = peaks[np.argsort(combined[peaks])[::-1]][: top_n * 3]
    raw = []
    for i in order:
        t = (i + 0.5) * win
        raw.append({
            "start": max(0.0, t - config.RADAR_PRE_ROLL),
            "end": min(duration, t + config.RADAR_POST_ROLL),
            "peak_time": float(t),
            "score": float(combined[i]),
            "audio_score": float(audio_n[i]),
            "chat_score": float(chat_n[i]),
            "peaks": [float(t)],
        })
    merged = merge_windows(raw)
    merged.sort(key=lambda w: w["score"], reverse=True)
    out = merged[:top_n]
    for rank, w in enumerate(out, 1):
        w["rank"] = rank
        w["duration"] = round(w["end"] - w["start"], 2)
        for k in ("start", "end", "peak_time"):
            w[k] = round(w[k], 2)
        for k in ("score", "audio_score", "chat_score"):
            w[k] = round(w[k], 4)
        w["peaks"] = sorted(round(p, 2) for p in w["peaks"])
    return out


def run_radar(workdir: Path, top_n: int = config.RADAR_TOP_N,
              extra_patterns: list[str] | None = None) -> list[dict]:
    win = config.RADAR_WINDOW_SECONDS
    wav = workdir / config.AUDIO_NAME
    if not wav.exists():
        raise SystemExit(f"{wav} não encontrado. Rode 01_ingest.py primeiro.")

    print("[1/3] Energia RMS do áudio...")
    rms, duration = audio_scores(wav, win)
    audio_n = robust_normalize(rms)

    print("[2/3] Densidade do chat...")
    chat = load_chat(workdir)
    if chat:
        chat_n = robust_normalize(chat_scores(chat, len(rms), win, extra_patterns))
        wa, wc = config.RADAR_AUDIO_WEIGHT, config.RADAR_CHAT_WEIGHT
        print(f"  {len(chat)} mensagens (pesos: áudio {wa}, chat {wc}).")
    else:
        chat_n = np.zeros_like(audio_n)
        wa, wc = 1.0, 0.0
        print("  Sem chat: usando apenas áudio.")

    combined = wa * audio_n + wc * chat_n
    # Suaviza levemente (3 janelas) para favorecer momentos sustentados
    combined = np.convolve(combined, np.array([0.25, 0.5, 0.25]), mode="same")

    print("[3/3] Selecionando picos e janelas de contexto...")
    cands = find_candidates(combined, audio_n, chat_n, duration, win, top_n)
    out = workdir / config.CANDIDATES_NAME
    out.write_text(json.dumps(cands, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"  {len(cands)} candidatos -> {out}")
    return cands


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("workdir", type=Path, help="Pasta gerada pelo 01_ingest.py")
    ap.add_argument("--top-n", type=int, default=config.RADAR_TOP_N)
    ap.add_argument("--reaction-regex", action="append", default=None,
                    help="Regex extra de reação do chat (repetível), ex.: 'LUL|catJAM'")
    args = ap.parse_args()
    cands = run_radar(args.workdir, args.top_n, args.reaction_regex)
    for c in cands[:10]:
        print(f"  #{c['rank']:>2} {c['start']:>8.1f}s-{c['end']:>8.1f}s "
              f"score={c['score']:.3f} (a={c['audio_score']:.2f} c={c['chat_score']:.2f})")


if __name__ == "__main__":
    main()

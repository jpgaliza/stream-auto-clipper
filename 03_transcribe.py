"""Módulo 3 - Transcrição seletiva (faster-whisper).

Uso:
    python 03_transcribe.py <workdir>

Transcreve SOMENTE as janelas de candidates.json (nunca o VOD inteiro), com
timestamps por palavra em tempo absoluto do vídeo. Ao final libera toda a VRAM
para que o Ollama possa carregar o LLM em seguida.
"""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import numpy as np
import soundfile as sf

import config


def _expose_cuda_libs() -> None:
    """ctranslate2 (faster-whisper) procura cublas64_12.dll/cudnn no PATH. No Windows
    essas DLLs vêm junto do PyTorch (torch/lib) ou dos pacotes nvidia-*-cu12, mas
    não estão no PATH: registra os diretórios antes de carregar o faster-whisper."""
    import importlib.util
    import os
    import sys

    dirs: list[Path] = []
    spec = importlib.util.find_spec("torch")
    if spec and spec.submodule_search_locations:
        dirs.append(Path(list(spec.submodule_search_locations)[0]) / "lib")
    nv = importlib.util.find_spec("nvidia")
    for base in (nv.submodule_search_locations if nv and nv.submodule_search_locations else []):
        dirs += [p / "bin" for p in Path(base).iterdir() if (p / "bin").is_dir()]
        dirs += [p / "lib" for p in Path(base).iterdir() if (p / "lib").is_dir()]
    for d in dirs:
        if d.is_dir():
            os.environ["PATH"] = str(d) + os.pathsep + os.environ.get("PATH", "")
            if sys.platform == "win32":
                os.add_dll_directory(str(d))
    try:
        import torch  # noqa: F401  carrega as DLLs CUDA no processo
    except ImportError:
        pass


_expose_cuda_libs()


def load_window_audio(wav: Path, start: float, end: float) -> np.ndarray:
    """Lê apenas o trecho [start, end] do wav (sem carregar o arquivo todo)."""
    with sf.SoundFile(str(wav)) as f:
        sr = f.samplerate
        f.seek(int(start * sr))
        data = f.read(int((end - start) * sr), dtype="float32")
    if data.ndim > 1:
        data = data.mean(axis=1)
    return data


def release_gpu(model) -> None:
    """Remove o modelo da memória e devolve a VRAM ao driver."""
    del model
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            free, total = torch.cuda.mem_get_info()
            print(f"  VRAM livre após limpeza: {free / 2**30:.1f}/{total / 2**30:.1f} GiB")
    except ImportError:
        pass


def transcribe_windows(workdir: Path, model_name: str, device: str,
                       compute_type: str, language: str) -> list[dict]:
    cand_path = workdir / config.CANDIDATES_NAME
    wav = workdir / config.AUDIO_NAME
    if not cand_path.exists():
        raise SystemExit(f"{cand_path} não encontrado. Rode 02_radar.py primeiro.")
    candidates = json.loads(cand_path.read_text(encoding="utf-8"))
    if not candidates:
        print("Nenhum candidato para transcrever.")
        return []

    from faster_whisper import WhisperModel

    print(f"Carregando Whisper {model_name} ({device}, {compute_type})...")
    model = WhisperModel(model_name, device=device, compute_type=compute_type)
    results: list[dict] = []
    try:
        for n, cand in enumerate(candidates, 1):
            start, end = cand["start"], cand["end"]
            print(f"[{n}/{len(candidates)}] {start:.0f}s-{end:.0f}s")
            audio = load_window_audio(wav, start, end)
            # segments é um gerador: consumir ANTES de descartar o modelo
            segments, _info = model.transcribe(
                audio,
                language=language,
                word_timestamps=True,
                vad_filter=True,
                condition_on_previous_text=False,
                beam_size=5,
            )
            segs = []
            for s in segments:
                words = [{
                    "start": round(start + w.start, 3),
                    "end": round(start + w.end, 3),
                    "word": w.word,
                    "probability": round(float(w.probability), 3),
                } for w in (s.words or [])]
                segs.append({
                    "start": round(start + s.start, 3),
                    "end": round(start + s.end, 3),
                    "text": s.text.strip(),
                    "words": words,
                })
            results.append({**cand, "segments": segs,
                            "text": " ".join(s["text"] for s in segs)})
    finally:
        print("Liberando VRAM...")
        release_gpu(model)
        model = None
        gc.collect()
    return results


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("workdir", type=Path)
    ap.add_argument("--model", default=config.WHISPER_MODEL)
    ap.add_argument("--device", default=config.WHISPER_DEVICE)
    ap.add_argument("--compute-type", default=config.WHISPER_COMPUTE_TYPE)
    ap.add_argument("--language", default=config.WHISPER_LANGUAGE)
    args = ap.parse_args()

    results = transcribe_windows(args.workdir, args.model, args.device,
                                 args.compute_type, args.language)
    out = args.workdir / config.TRANSCRIBED_NAME
    out.write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{len(results)} janelas transcritas -> {out}")


if __name__ == "__main__":
    main()

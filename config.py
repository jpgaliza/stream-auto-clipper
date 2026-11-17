"""Configuração central do stream-auto-clipper.

Todos os módulos importam daqui. Nada é específico de um jogo/tema:
o contexto vem do argumento opcional --topic.
"""
from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

# Console do Windows (cp1252) quebra com emojis nos títulos: força UTF-8 na saída.
for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")

ROOT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT_DIR / "output"

# ---------- Vídeo vertical ----------
OUT_WIDTH, OUT_HEIGHT = 1080, 1920
TOP_RATIO, BOTTOM_RATIO = 0.35, 0.65  # webcam / conteúdo principal
TOP_HEIGHT = round(OUT_HEIGHT * TOP_RATIO)          # 672
BOTTOM_HEIGHT = OUT_HEIGHT - TOP_HEIGHT             # 1248

# ---------- Modelos ----------
OLLAMA_MODEL = "qwen3.5:9b"
OLLAMA_HOST = "http://localhost:11434"
WHISPER_MODEL = "large-v3"

# ---------- Duração dos clipes (s) ----------
MIN_CLIP_SECONDS = 15
MAX_CLIP_SECONDS = 58

# ---------- Ingestão ----------
DOWNLOAD_FORMAT = (
    "bv*[height<=1080][fps<=60]+ba/bv*[height<=1080]+ba/b[height<=1080]/b"
)
AUDIO_SAMPLE_RATE = 16_000

# ---------- Radar ----------
RADAR_WINDOW_SECONDS = 2.0
RADAR_PRE_ROLL = 45.0    # segundos de contexto antes do pico
RADAR_POST_ROLL = 20.0   # segundos de contexto depois do pico
RADAR_TOP_N = 40
RADAR_MAX_MERGED_SECONDS = 180.0  # teto de uma janela após mesclar sobreposições
RADAR_AUDIO_WEIGHT = 0.6
RADAR_CHAT_WEIGHT = 0.4
# Distância mínima entre picos (em janelas) para evitar picos redundantes
RADAR_MIN_PEAK_DISTANCE_WINDOWS = 10

# Padrões de reação (risadas, hype, pedidos de clip, emotes comuns).
# Multilíngue e independente de nicho; estenda via --reaction-regex se quiser.
REACTION_PATTERNS = [
    r"k{3,}", r"(?:ha){2,}h?", r"(?:ja){2,}", r"(?:he){2,}", r"rs{2,}",
    r"lo+l+", r"lm+a+o+", r"\bclip\w*\b", r"\bpog\w*\b", r"kappa\w*",
    r"omegalul", r"\bkek\w*\b", r"\bkkk\w*\b", r"\bwtf\b", r"\bomg\b",
    r"\bgg\b", r"\binsane\b", r"\bcrazy\b", r"\bnooo+\b", r"\bmano\b",
    r"\bvish\b", r"[\U0001F600-\U0001F64F\U0001F923\U0001F480\U0001F525]",
]


@dataclass
class Settings:
    """Parâmetros resolvidos em tempo de execução (CLI + defaults)."""
    topic: str | None = None
    out_width: int = OUT_WIDTH
    out_height: int = OUT_HEIGHT
    top_ratio: float = TOP_RATIO
    ollama_model: str = OLLAMA_MODEL
    whisper_model: str = WHISPER_MODEL
    min_clip: int = MIN_CLIP_SECONDS
    max_clip: int = MAX_CLIP_SECONDS
    extra: dict = field(default_factory=dict)


def add_common_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Argumentos compartilhados por todos os módulos da pipeline."""
    parser.add_argument(
        "--topic",
        default=None,
        help="Tema/jogo opcional (ex.: 'Minecraft speedrun', 'just chatting') "
        "usado depois para contextualizar a curadoria por LLM.",
    )
    parser.add_argument("--workdir", type=Path, default=None,
                        help="Pasta de trabalho (default: output/<slug-do-video>).")
    parser.add_argument("--ollama-model", default=OLLAMA_MODEL)
    parser.add_argument("--whisper-model", default=WHISPER_MODEL)
    parser.add_argument("--min-clip", type=int, default=MIN_CLIP_SECONDS)
    parser.add_argument("--max-clip", type=int, default=MAX_CLIP_SECONDS)
    return parser


def settings_from_args(args: argparse.Namespace) -> Settings:
    return Settings(
        topic=args.topic,
        ollama_model=args.ollama_model,
        whisper_model=args.whisper_model,
        min_clip=args.min_clip,
        max_clip=args.max_clip,
    )


def slugify(text: str, max_len: int = 60) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", text).strip("-").lower()
    return (slug[:max_len] or "video").strip("-")


# Nomes de artefatos padronizados entre módulos
VIDEO_NAME = "video.mp4"
AUDIO_NAME = "audio.wav"
CHAT_NAME = "chat_normalized.json"
CANDIDATES_NAME = "candidates.json"
META_NAME = "meta.json"

# ---------- Transcrição / Curadoria ----------
WHISPER_LANGUAGE = "pt"
WHISPER_DEVICE = "cuda"
WHISPER_COMPUTE_TYPE = "float16"   # large-v3 fp16 ~ 3-4 GB VRAM
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_NUM_CTX = 8192
TARGET_MIN_SECONDS = 35            # faixa preferida (narrativa completa)
TARGET_MAX_SECONDS = 55
SNAP_TOLERANCE_SECONDS = 3.0       # quanto o corte pode andar para fechar frase
TRANSCRIBED_NAME = "candidates_transcribed.json"
APPROVED_NAME = "approved_clips.json"

# ---------- Render ----------
ACCENT_HEX = "#10B981"             # verde esmeralda (divisória + palavra ativa)
DIVIDER_PX = 4
SUB_FONT = "Montserrat ExtraBold"
SUB_FONT_SIZE = 70
SUB_Y = TOP_HEIGHT                 # linha divisória (672)
SUB_Y_FALLBACK = 1344              # sem webcam não há divisória: terço inferior
WEBCAM_SAMPLES = 15
WEBCAM_MIN_FRAMES = 5              # frames mínimos com o rosto p/ aceitar o cluster
WEBCAM_BOX_FACTOR = 4.2            # largura estimada da webcam = rosto * fator
WEBCAM_INNER_MARGIN = 0.15         # zoom interno p/ cortar molduras/alertas
FACE_MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/face_detector/"
                  "blaze_face_short_range/float16/latest/blaze_face_short_range.tflite")
MODELS_DIR = ROOT_DIR / "models"
FONTS_DIR = ROOT_DIR / "fonts"     # coloque Montserrat-ExtraBold.ttf aqui
RENDERED_NAME = "rendered_clips.json"
VERIFIED_NAME = "verified_clips.json"
VERIFY_REPORT_NAME = "verify_report.json"
UPLOADS_NAME = "uploads.json"
DEFAULT_MAX_CLIPS = 15

# ---------- Verificador Multimodal ----------
OLLAMA_VISION_MODEL = "qwen2.5vl:7b"
VERIFY_MIN_SCORE = 7.0
VERIFY_SAMPLES = 6

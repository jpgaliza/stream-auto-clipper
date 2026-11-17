"""Limpeza de artefatos em output/.

Uso:
    python clean.py                       # lista o espaço usado por live (nada é apagado)
    python clean.py --downloads           # apaga vídeo/áudio baixados (mantém cortes e metadados)
    python clean.py --clips               # apaga cortes renderizados (corte_XX.*, rendered_clips.json)
    python clean.py --cache               # apaga intermediários (webcam.json, logs, fonts, .ass)
    python clean.py --all                 # apaga a pasta inteira de cada live
    python clean.py --downloads --slug minha-live     # só uma live
    python clean.py --all --dry-run       # mostra o que seria apagado
    python clean.py --all --yes           # sem pedir confirmação

Nunca toca em client_secrets.json, token.json, tiktok_cookies.txt, models/ ou fonts/ da raiz.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import config

DOWNLOADS = ["video.*", "audio.wav", "*.info.json", "*.live_chat.json", "*.part", "*.ytdl"]
CLIPS = ["corte_*.mp4", "corte_*.txt", "corte_*.ass", config.RENDERED_NAME, config.UPLOADS_NAME]
CACHE = ["webcam.json", "*.ass", "logs", "fonts"]


def size(p: Path) -> int:
    if p.is_file():
        return p.stat().st_size
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())


def human(n: float) -> str:
    for u in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {u}"
        n /= 1024
    return f"{n:.1f} TB"


def matches(folder: Path, patterns: list[str]) -> list[Path]:
    seen: dict[Path, None] = {}
    for pat in patterns:
        for p in folder.glob(pat):
            seen[p] = None
    return list(seen)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--downloads", action="store_true")
    ap.add_argument("--clips", action="store_true")
    ap.add_argument("--cache", action="store_true")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--slug", help="limita a uma pasta de output/")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--yes", action="store_true", help="não pedir confirmação")
    args = ap.parse_args()

    if not config.OUTPUT_DIR.exists():
        print("output/ não existe. Nada a limpar.")
        return
    folders = sorted(p for p in config.OUTPUT_DIR.iterdir() if p.is_dir())
    if args.slug:
        folders = [f for f in folders if f.name == args.slug]
        if not folders:
            raise SystemExit(f"Pasta output/{args.slug} não encontrada.")

    if not (args.downloads or args.clips or args.cache or args.all):
        total = 0
        for f in folders:
            s = size(f)
            total += s
            dl = sum(size(p) for p in matches(f, DOWNLOADS))
            cl = sum(size(p) for p in matches(f, CLIPS))
            print(f"{f.name:<40} total {human(s):>9} | downloads {human(dl):>9} | cortes {human(cl):>9}")
        print(f"\nTotal em output/: {human(total)}")
        print("Use --downloads, --clips, --cache ou --all para apagar (veja --help).")
        return

    targets: list[Path] = []
    for f in folders:
        if args.all:
            targets.append(f)
            continue
        pats = (DOWNLOADS if args.downloads else []) + (CLIPS if args.clips else []) \
            + (CACHE if args.cache else [])
        targets += matches(f, pats)

    if not targets:
        print("Nada para apagar.")
        return
    total = sum(size(p) for p in targets if p.exists())
    for p in targets:
        print(f"  {'[dry-run] ' if args.dry_run else ''}{p.relative_to(config.ROOT_DIR)}  ({human(size(p))})")
    print(f"\n{len(targets)} item(ns), {human(total)} liberados.")
    if args.dry_run:
        return
    if not args.yes and input("Apagar? [s/N] ").strip().lower() not in {"s", "sim", "y", "yes"}:
        print("Cancelado.")
        return
    for p in targets:
        try:
            shutil.rmtree(p) if p.is_dir() else p.unlink(missing_ok=True)
        except PermissionError:
            print(f"  ! em uso (download em andamento?), pulado: {p.name}")
    print("Pronto.")


if __name__ == "__main__":
    main()

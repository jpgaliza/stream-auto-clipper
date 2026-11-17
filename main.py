"""Orquestrador do stream-auto-clipper.

Exemplo:
    python main.py --url "URL" --topic "GTA VI" --max-clips 15 --upload-all

Cada etapa roda em um subprocesso próprio: ao terminar o 03 (Whisper) o processo
morre e a VRAM é devolvida por completo antes de o Ollama carregar o LLM no 04.
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import subprocess
import sys
import time
from collections import deque
from pathlib import Path

from rich.console import Console
from rich.markup import escape
from rich.progress import (BarColumn, Progress, SpinnerColumn, TaskProgressColumn,
                           TextColumn, TimeElapsedColumn)
from rich.prompt import Confirm, IntPrompt, Prompt
from rich.table import Table

import config

console = Console()
ROOT = config.ROOT_DIR

STAGES = [
    (1, "Ingestão", "01_ingest.py"),
    (2, "Radar", "02_radar.py"),
    (3, "Transcrição", "03_transcribe.py"),
    (4, "Curadoria", "04_curator.py"),
    (5, "Render", "05_vision_render.py"),
    (5.5, "Verificação", "05b_verify.py"),
    (6, "Publicação", "06_publisher.py"),
]

PLATFORMS = [("youtube", "YouTube Shorts", "private", "upload_yt"),
             ("tiktok", "TikTok", "only_you", "upload_tiktok")]
UPLOADS: dict[str, str] = {}  # plataforma -> resto da linha "RESULT" do publisher


def fmt(sec: float) -> str:
    m, s = divmod(int(sec), 60)
    return f"{m}m{s:02d}s" if m else f"{s}s"


def stage_cmd(num: int, script: str, a: argparse.Namespace, workdir: Path) -> list[str]:
    py = [sys.executable, "-u", str(ROOT / script)]
    if num == 1:
        cmd = py + [a.url, "--workdir", str(workdir)]
        if a.topic:
            cmd += ["--topic", a.topic]
        if a.cookies_from_browser:
            cmd += ["--cookies-from-browser", a.cookies_from_browser]
        if a.live_minutes:
            cmd += ["--live-minutes", str(a.live_minutes)]
        return cmd
    if num == 4:
        cmd = py + [str(workdir), "--ollama-model", a.ollama_model,
                    "--min-clip", str(a.min_clip), "--max-clip", str(a.max_clip)]
        return cmd + (["--topic", a.topic] if a.topic else [])
    if num == 3:
        return py + [str(workdir), "--model", a.whisper_model, "--device", a.whisper_device,
                     "--compute-type", a.whisper_compute_type]
    if num == 5:
        return py + [str(workdir), "--max-clips", str(a.max_clips)]
    if num == 5.5:
        cmd = py + [str(workdir), "--vision-model", a.vision_model,
                    "--min-score", str(a.min_verify_score),
                    "--ollama-model", a.ollama_model,
                    "--min-clip", str(a.min_clip), "--max-clip", str(a.max_clip)]
        return cmd + (["--topic", a.topic] if a.topic else [])
    if num == 6:
        cmd = py + [str(workdir), "--tiktok-browser", a.tiktok_browser]
        if a.upload_yt:
            cmd.append("--upload-yt")
        if a.upload_tiktok:
            cmd.append("--upload-tiktok")
        if a.tiktok_headless:
            cmd.append("--tiktok-headless")
        return cmd
    return py + [str(workdir)]


def run_stage(cmd: list[str], progress: Progress, task_id, log_path: Path) -> int:
    tail: deque[str] = deque(maxlen=15)
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, encoding="utf-8", errors="replace",
                                env={**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"})
        for line in proc.stdout:  # type: ignore[union-attr]
            line = line.rstrip()
            log.write(line + "\n")
            if line.startswith("RESULT "):
                parts = line.split(" ", 2)
                UPLOADS[parts[1]] = parts[2] if len(parts) > 2 else ""
            if line.strip():
                tail.append(line)
                progress.update(task_id, detail=escape(line.strip()[:70]))
        code = proc.wait()
    # O publisher sai com 2 se alguma plataforma falhou: o detalhe vai na tabela de envio.
    if code != 0 and not UPLOADS:
        console.print(f"\n[red]Falhou (código {code}). Últimas linhas:[/red]")
        for l in tail:
            console.print(f"  {l}", markup=False, highlight=False)
    return code


def upload_failed_count(plat: str) -> int | None:
    line = UPLOADS.get(plat)
    if line is None:
        return None
    head = line.split(" error=", 1)[0]
    kv = dict(x.split("=", 1) for x in head.split() if "=" in x)
    return int(kv.get("failed", 0))


def upload_status(plat: str) -> str:
    line = UPLOADS.get(plat)
    if line is None:
        return "[red]não executado / erro (veja logs)[/red]"
    err = line.split(" error=", 1)[1] if " error=" in line else ""
    head = line.split(" error=", 1)[0]
    kv = dict(x.split("=", 1) for x in head.split() if "=" in x)
    failed = int(kv.get("failed", 0))
    color = "red" if failed else "green"
    status = (f"[{color}]enviados={kv.get('ok', 0)} falhas={failed} "
              f"já enviados={kv.get('skipped', 0)}[/{color}]")
    if err:
        status += f"\n[dim]{escape(err)}[/dim]"
    return status


def interactive_fill(a: argparse.Namespace) -> None:
    """Quando chamado sem --url, pergunta os parâmetros principais no terminal."""
    console.rule("[bold green]stream-auto-clipper (modo interativo)")
    while not a.url:
        raw = Prompt.ask("[bold]Cole a URL da live/VOD ou caminho do .mp4[/bold]").strip().strip("\"'")
        if raw:
            a.url = raw
    if not a.topic:
        topic = Prompt.ask("Tema/jogo (Enter para pular)", default="").strip()
        a.topic = topic or None
    a.max_clips = IntPrompt.ask("Máximo de cortes", default=a.max_clips)
    if not a.live_minutes and not Path(a.url).expanduser().is_file():
        if Confirm.ask("A live ainda está ao vivo agora?", default=False):
            a.live_minutes = float(IntPrompt.ask("Quantos minutos capturar ao vivo?", default=30))
    if not a.verify:
        a.verify = Confirm.ask("Ativar verificador multimodal (VLM)?", default=False)
    if not (a.upload_yt or a.upload_tiktok or a.upload_all or a.upload_draft):
        up = Prompt.ask("Enviar rascunhos após render?", choices=["n", "yt", "tiktok", "all"], default="n")
        if up == "yt":
            a.upload_yt = True
        elif up == "tiktok":
            a.upload_tiktok = True
        elif up == "all":
            a.upload_all = True
    console.print()


def main() -> None:
    ap = argparse.ArgumentParser(description="Transforma VODs em Shorts verticais, 100% local.")
    ap.add_argument("--url", default=None, help="URL (YouTube/Twitch/Kick) ou .mp4 local (omitir abre modo interativo)")
    ap.add_argument("--max-clips", type=int, default=config.DEFAULT_MAX_CLIPS,
                    help="máximo de cortes renderizados/enviados (default 15)")
    ap.add_argument("--upload-yt", action="store_true", help="envia ao YouTube Shorts (private)")
    ap.add_argument("--upload-tiktok", action="store_true", help="envia ao TikTok (only_you)")
    ap.add_argument("--upload-all", action="store_true", help="YouTube e TikTok em sequência")
    ap.add_argument("--upload-draft", action="store_true", help="(legado) equivale a --upload-yt")
    ap.add_argument("--tiktok-browser", default="chrome",
                    choices=["chrome", "safari", "chromium", "edge", "firefox"])
    ap.add_argument("--tiktok-headless", action="store_true")
    ap.add_argument("--whisper-device", default=config.WHISPER_DEVICE,
                    help="cuda (padrão) ou cpu (Mac/sem GPU NVIDIA)")
    ap.add_argument("--whisper-compute-type", default=config.WHISPER_COMPUTE_TYPE,
                    help="float16 (GPU) ou int8 (CPU)")
    ap.add_argument("--verify", action="store_true",
                    help="executa o verificador multimodal (VLM + autocorreção) após o render")
    ap.add_argument("--vision-model", default=config.OLLAMA_VISION_MODEL)
    ap.add_argument("--min-verify-score", type=float, default=config.VERIFY_MIN_SCORE)
    ap.add_argument("--cookies-from-browser", default=None, metavar="BROWSER")
    ap.add_argument("--live-minutes", type=float, default=None,
                    help="captura N minutos de uma live ainda ao vivo (+ chat em tempo real)")
    ap.add_argument("--start-at", type=int, default=1, choices=range(1, 7),
                    help="retoma a partir da etapa N (1-6) reutilizando artefatos existentes")
    config.add_common_args(ap)
    a = ap.parse_args()
    if not a.url:
        interactive_fill(a)
    if a.upload_all:
        a.upload_yt = a.upload_tiktok = True
    if a.upload_draft:
        a.upload_yt = True
    do_upload = a.upload_yt or a.upload_tiktok

    ingest = importlib.import_module("01_ingest")
    workdir = (a.workdir or ingest.default_workdir(a.url)).resolve()
    workdir.mkdir(parents=True, exist_ok=True)
    (workdir / "logs").mkdir(exist_ok=True)

    targets = [label for _, label, _, attr in PLATFORMS if getattr(a, attr)]
    console.rule("[bold green]stream-auto-clipper")
    console.print(f"Fonte: {a.url}\nPasta: {workdir}\nTema: {a.topic or '(nenhum)'}\n"
                  f"Upload: {', '.join(targets) or '(desativado)'}\n")

    timings: list[tuple[str, float, str]] = []
    t_all = time.time()
    active = [
        s for s in STAGES
        if s[0] >= a.start_at
        and (s[0] != 5.5 or a.verify)
        and (s[0] != 6 or do_upload)
    ]
    columns = [SpinnerColumn(), TextColumn("[bold]{task.description:<12}"),
               BarColumn(), TaskProgressColumn(), TimeElapsedColumn(),
               TextColumn("[dim]{task.fields[detail]}")]
    failed_stage = False
    with Progress(*columns, console=console) as progress:
        overall = progress.add_task("Total", total=len(active), detail="")
        for num, name, script in active:
            label = f"{'5b' if num == 5.5 else int(num)}. {name}"
            tid = progress.add_task(label, total=None, detail="iniciando...")
            t0 = time.time()
            log_key = "05b" if num == 5.5 else f"{int(num):02d}"
            code = run_stage(stage_cmd(num, script, a, workdir), progress, tid,
                             workdir / "logs" / f"{log_key}_{Path(script).stem}.log")
            dt = time.time() - t0
            # Etapa 6: falhas parciais de upload são reportadas na tabela de envio
            ok = code == 0 or (num == 6 and bool(UPLOADS))
            progress.update(tid, total=1, completed=1, detail="ok" if ok else "ERRO")
            progress.advance(overall)
            timings.append((label, dt, ("ok" if code == 0 else "parcial") if ok else "erro"))
            if not ok:
                failed_stage = True
                break

    table = Table(title="Resumo")
    table.add_column("Etapa")
    table.add_column("Tempo", justify="right")
    table.add_column("Status")
    for name, dt, st in timings:
        table.add_row(name, fmt(dt), {"ok": "[green]ok[/green]", "parcial": "[yellow]falhas de envio[/yellow]"}.get(st, "[red]erro[/red]"))
    table.add_row("[bold]Total", f"[bold]{fmt(time.time() - t_all)}", "")
    console.print(table)

    upload_failed = False
    if do_upload and any(n.startswith("6.") for n, _, _ in timings):
        up = Table(title="Envio (privado / rascunho para revisão)")
        for col in ("Plataforma", "Visibilidade", "Status"):
            up.add_column(col)
        for plat, label, vis, attr in PLATFORMS:
            if getattr(a, attr):
                up.add_row(label, vis, upload_status(plat))
                if upload_failed_count(plat) != 0:
                    upload_failed = True
        console.print(up)

    rep_p = workdir / config.VERIFY_REPORT_NAME
    if rep_p.exists() and any(n.startswith(("5b.", "6.")) for n, _, _ in timings):
        rep = json.loads(rep_p.read_text(encoding="utf-8"))
        blocked = [c for c in rep if not c.get("verify_approved")]
        if blocked:
            bt = Table(title=f"Bloqueados na verificação — não enviados ({len(blocked)}/{len(rep)})")
            bt.add_column("Arquivo")
            bt.add_column("Nota", justify="right")
            bt.add_column("Título")
            bt.add_column("Motivo")
            for b in blocked:
                bt.add_row(b["file"], f"{b.get('verify_score', 0):.1f}",
                           b.get("title", "")[:40], b.get("verify_feedback", "")[:70])
            console.print(bt)
        else:
            console.print(f"[green]Verificação: todos os {len(rep)} cortes aprovados.[/green]")

    if failed_stage:
        console.print(f"\nLogs em: {workdir / 'logs'}")
        sys.exit(1)
    console.print(f"\nCortes em: [bold]{workdir}[/bold]")
    if upload_failed:
        sys.exit(2)


if __name__ == "__main__":
    main()

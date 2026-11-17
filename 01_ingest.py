"""Módulo 1 - Ingestão.

Uso:
    python 01_ingest.py <URL|arquivo.mp4> [--cookies-from-browser chrome] [--topic "..."]

Gera em <workdir>/: video.mp4, audio.wav (mono 16kHz), chat_normalized.json, meta.json
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import config


# ---------------------------------------------------------------- utilidades
def run(cmd: list[str], check: bool = True) -> subprocess.CompletedProcess:
    print("  $", " ".join(str(c) for c in cmd))
    return subprocess.run(cmd, check=check)


def detect_platform(source: str) -> str:
    if Path(source).expanduser().is_file():
        return "local"
    host = (urlparse(source).netloc or "").lower()
    if "youtube.com" in host or "youtu.be" in host:
        return "youtube"
    if "twitch.tv" in host:
        return "twitch"
    if "kick.com" in host:
        return "kick"
    return "other"


def ytdlp_base(cookies_browser: str | None) -> list[str]:
    cmd = [sys.executable, "-m", "yt_dlp"]
    if cookies_browser:
        cmd += ["--cookies-from-browser", cookies_browser]
    return cmd


def require_ffmpeg() -> None:
    if not shutil.which("ffmpeg"):
        sys.exit("ffmpeg não encontrado no PATH. Instale-o (ex.: winget install ffmpeg).")


# ------------------------------------------------------------------ download
def download_video(url: str, workdir: Path, cookies: str | None) -> Path:
    out_tpl = str(workdir / "video.%(ext)s")
    cmd = ytdlp_base(cookies) + [
        "-f", config.DOWNLOAD_FORMAT,
        "-S", "res:1080,fps:60",
        "--merge-output-format", "mp4",
        "--no-playlist",
        "--write-info-json",
        "-o", out_tpl,
        url,
    ]
    run(cmd)
    video = workdir / config.VIDEO_NAME
    if not video.exists():
        cands = [p for p in workdir.glob("video.*")
                 if p.suffix in {".mp4", ".mkv", ".webm"}]
        if not cands:
            sys.exit("Download falhou: nenhum arquivo de vídeo gerado.")
        video = cands[0]
    return video


def extract_audio(video: Path, wav: Path) -> None:
    run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(video),
         "-vn", "-ac", "1", "-ar", str(config.AUDIO_SAMPLE_RATE),
         "-c:a", "pcm_s16le", str(wav)])


# ---------------------------------------------------------------------- chat
def _runs_to_text(runs: list[dict]) -> str:
    parts = []
    for r in runs:
        if "text" in r:
            parts.append(r["text"])
        elif "emoji" in r:
            e = r["emoji"]
            shortcuts = e.get("shortcuts") or []
            parts.append(shortcuts[0] if shortcuts else e.get("emojiId", ""))
    return "".join(parts).strip()


def parse_youtube_live_chat(path: Path) -> list[dict]:
    """Normaliza o .live_chat.json (JSON lines) gerado pelo yt-dlp."""
    out: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                action = obj["replayChatItemAction"]
                offset_ms = float(action.get("videoOffsetTimeMsec", 0))
                for a in action.get("actions", []):
                    item = a.get("addChatItemAction", {}).get("item", {})
                    renderer = (item.get("liveChatTextMessageRenderer")
                                or item.get("liveChatPaidMessageRenderer"))
                    if not renderer:
                        continue
                    msg = _runs_to_text(renderer.get("message", {}).get("runs", []))
                    if not msg:
                        continue
                    out.append({
                        "timestamp": offset_ms / 1000.0,
                        "user": renderer.get("authorName", {}).get("simpleText", ""),
                        "message": msg,
                    })
            except (KeyError, ValueError, TypeError):
                continue
    return out


def download_youtube_chat(url: str, workdir: Path, cookies: str | None) -> list[dict]:
    cmd = ytdlp_base(cookies) + [
        "--skip-download", "--write-subs", "--sub-langs", "live_chat",
        "--no-playlist", "-o", str(workdir / "video.%(ext)s"), url,
    ]
    run(cmd, check=False)  # falha de chat nunca deve derrubar a pipeline
    files = list(workdir.glob("*.live_chat.json"))
    if not files:
        return []
    return parse_youtube_live_chat(files[0])


def _clean_kick_content(text: str) -> str:
    # Converte [emote:12345:NomeDoEmote] em NomeDoEmote para o radar reconhecer reações
    return re.sub(r"\[emote:\d+:([^\]]+)\]", r"\1", text).strip()


def _parse_iso_utc(ts: str) -> float:
    return datetime.fromisoformat(ts.replace("Z", "+00:00")).astimezone(timezone.utc).timestamp()


def _kick_fetch_chunk(channel_id: int, start_epoch: float, c_start: float, c_end: float) -> list[tuple[str, dict]]:
    from curl_cffi import requests

    s = requests.Session(impersonate="chrome")
    cursor = int(c_end * 1_000_000)
    out: list[tuple[str, dict]] = []
    for _ in range(400):  # teto de segurança por bloco de 15 min
        data = None
        for attempt in range(3):
            try:
                r = s.get(f"https://kick.com/api/v2/channels/{channel_id}/messages?cursor={cursor}", timeout=15)
                if r.status_code == 200:
                    data = r.json().get("data") or {}
                    break
            except Exception:  # noqa: BLE001
                pass
            time.sleep(0.5 * (attempt + 1))
        if not data:
            break
        msgs = data.get("messages") or []
        if not msgs:
            break
        oldest = c_end
        for m in msgs:
            ts_str = m.get("created_at")
            content = _clean_kick_content(str(m.get("content") or ""))
            if not ts_str or not content:
                continue
            msg_epoch = _parse_iso_utc(ts_str)
            oldest = min(oldest, msg_epoch)
            if c_start <= msg_epoch <= c_end:
                out.append((str(m.get("id", f"{msg_epoch}-{content[:16]}")), {
                    "timestamp": round(msg_epoch - start_epoch, 3),
                    "user": (m.get("sender") or {}).get("username", ""),
                    "message": content,
                }))
        next_cur = data.get("cursor")
        if oldest < c_start or not next_cur or int(next_cur) >= cursor:
            break
        cursor = int(next_cur)
    return out


def download_kick_chat(url: str, duration_hint: float | None = None) -> list[dict]:
    """Coletor dedicado de chat replay para VODs da Kick via API + curl_cffi."""
    m = re.search(r"([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})", url, re.I)
    if not m:
        print("  ! URL da Kick sem UUID de VOD; pulando chat.")
        return []
    try:
        from curl_cffi import requests
    except ImportError:
        print("  ! curl_cffi não instalado; pulando chat da Kick.")
        return []

    uuid = m.group(1)
    try:
        s = requests.Session(impersonate="chrome")
        resp = s.get(f"https://kick.com/api/v1/video/{uuid}", timeout=15)
        if resp.status_code != 200:
            print(f"  ! Kick API retornou HTTP {resp.status_code}; pulando chat.")
            return []
        meta = resp.json()
        ls = meta.get("livestream") or {}
        channel_id = ls.get("channel_id")
        start_str = ls.get("start_time")
        duration_s = float(ls.get("duration") or 0) / 1000.0 or (duration_hint or 0.0)
        if not channel_id or not start_str or duration_s <= 0:
            print("  ! Metadados da live Kick incompletos; pulando chat.")
            return []
        start_epoch = _parse_iso_utc(start_str)
    except Exception as e:  # noqa: BLE001
        print(f"  ! Erro ao consultar VOD na Kick ({type(e).__name__}: {e}).")
        return []

    chunk_s = 900.0  # blocos de 15 min em paralelo
    chunks = [(start_epoch + t, min(start_epoch + t + chunk_s, start_epoch + duration_s))
              for t in range(0, int(duration_s) + 1, int(chunk_s))]
    seen: dict[str, dict] = {}
    with ThreadPoolExecutor(max_workers=6) as pool:
        futs = [pool.submit(_kick_fetch_chunk, int(channel_id), start_epoch, cs, ce) for cs, ce in chunks]
        for fut in as_completed(futs):
            try:
                for mid, item in fut.result():
                    if item["timestamp"] >= 0:
                        seen[mid] = item
            except Exception:  # noqa: BLE001
                pass
    return list(seen.values())


def download_chat_downloader(url: str) -> list[dict]:
    """Twitch via chat_downloader (VOD com replay)."""
    try:
        from chat_downloader import ChatDownloader
    except ImportError:
        print("  ! chat_downloader não instalado; pulando chat.")
        return []
    out: list[dict] = []
    try:
        chat = ChatDownloader().get_chat(url, message_groups=["messages"])
        for m in chat:
            ts = m.get("time_in_seconds")
            msg = m.get("message")
            if ts is None or not msg:
                continue
            out.append({
                "timestamp": float(ts),
                "user": (m.get("author") or {}).get("name", ""),
                "message": str(msg),
            })
    except Exception as e:  # noqa: BLE001 - chat é opcional
        print(f"  ! chat indisponível ({type(e).__name__}: {e}).")
    return [m for m in out if m["timestamp"] >= 0]


def get_chat(platform: str, source: str, workdir: Path, cookies: str | None,
             duration_hint: float | None = None) -> list[dict]:
    if platform == "youtube":
        msgs = download_youtube_chat(source, workdir, cookies)
    elif platform == "twitch":
        msgs = download_chat_downloader(source)
    elif platform == "kick":
        msgs = download_kick_chat(source, duration_hint)
    elif platform == "local":
        # Aceita sidecar opcional: <video>.live_chat.json ao lado do arquivo
        p = Path(source).expanduser()
        side = p.with_name(p.stem + ".live_chat.json")
        msgs = parse_youtube_live_chat(side) if side.exists() else []
    else:
        msgs = []
    msgs.sort(key=lambda m: m["timestamp"])
    return msgs


# ------------------------------------------------------------------ captura ao vivo
def _kick_live_chat(url: str, start_epoch: float, stop: threading.Event, out: list[dict]) -> None:
    slug = urlparse(url).path.strip("/").split("/")[0]
    if not slug:
        return
    try:
        from curl_cffi import requests
        s = requests.Session(impersonate="chrome")
        r = s.get(f"https://kick.com/api/v2/channels/{slug}", timeout=15)
        if r.status_code != 200:
            return
        cid = r.json().get("id") or (r.json().get("chatroom") or {}).get("chatable_id")
        if not cid:
            return
    except Exception:  # noqa: BLE001
        return

    seen: set[str] = set()
    while not stop.is_set():
        try:
            mr = s.get(f"https://kick.com/api/v2/channels/{cid}/messages", timeout=10)
            if mr.status_code == 200:
                for m in reversed(mr.json().get("data", {}).get("messages") or []):
                    mid = str(m.get("id", ""))
                    if mid in seen:
                        continue
                    seen.add(mid)
                    ts_str = m.get("created_at")
                    content = _clean_kick_content(str(m.get("content") or ""))
                    if not ts_str or not content:
                        continue
                    dt = _parse_iso_utc(ts_str) - start_epoch
                    if dt >= -2.0:
                        out.append({
                            "timestamp": round(max(0.0, dt), 3),
                            "user": (m.get("sender") or {}).get("username", ""),
                            "message": content,
                        })
        except Exception:  # noqa: BLE001
            pass
        stop.wait(2.5)


def _cd_live_chat(url: str, t0_mono: float, stop: threading.Event, out: list[dict]) -> None:
    try:
        from chat_downloader import ChatDownloader
        chat = ChatDownloader().get_chat(url, message_groups=["messages"])
        for m in chat:
            if stop.is_set():
                break
            msg = m.get("message")
            if not msg:
                continue
            out.append({
                "timestamp": round(max(0.0, time.monotonic() - t0_mono), 3),
                "user": (m.get("author") or {}).get("name", ""),
                "message": str(msg),
            })
    except Exception:  # noqa: BLE001
        pass


def record_live(platform: str, url: str, workdir: Path, minutes: float,
                cookies: str | None) -> tuple[Path, list[dict]]:
    """Grava `minutes` minutos de uma transmissão ao vivo + chat em paralelo."""
    seconds = max(5.0, minutes * 60.0)
    cmd = ytdlp_base(cookies) + ["-g", "-f", config.DOWNLOAD_FORMAT, "--no-playlist", url]
    print("  $", " ".join(str(c) for c in cmd))
    res = subprocess.run(cmd, capture_output=True, text=True, check=True)
    stream_urls = [u.strip() for u in res.stdout.splitlines() if u.strip()]
    if not stream_urls:
        sys.exit("yt-dlp não retornou URL de stream ao vivo.")

    video = workdir / config.VIDEO_NAME
    ff_cmd = ["ffmpeg", "-y", "-loglevel", "error"]
    for u in stream_urls[:2]:
        ff_cmd += ["-reconnect", "1", "-reconnect_streamed", "1", "-i", u]
    ff_cmd += ["-t", f"{seconds:.1f}", "-c:v", "copy", "-c:a", "aac",
               "-movflags", "+faststart", str(video)]

    chat_out: list[dict] = []
    stop = threading.Event()
    t0_mono, start_epoch = time.monotonic(), time.time()
    target = _kick_live_chat if platform == "kick" else _cd_live_chat
    t_args = (url, start_epoch if platform == "kick" else t0_mono, stop, chat_out)
    chat_thread = threading.Thread(target=target, args=t_args, daemon=True)
    chat_thread.start()

    print(f"  Gravando ao vivo por {minutes:.1f} min ({seconds:.0f}s)...")
    try:
        subprocess.run(ff_cmd, check=True, timeout=seconds + 90)
    finally:
        stop.set()
        chat_thread.join(timeout=5)
    chat_out.sort(key=lambda m: m["timestamp"])
    return video, chat_out


# ---------------------------------------------------------------------- main
def default_workdir(source: str) -> Path:
    platform = detect_platform(source)
    if platform == "local":
        name = Path(source).stem
    else:
        u = urlparse(source)
        name = u.path.strip("/") or u.netloc
        if u.query and "v=" in u.query:  # youtube watch?v=ID
            name += "-" + u.query.split("v=")[1].split("&")[0]
    return config.OUTPUT_DIR / config.slugify(name)


def ingest(source: str, workdir: Path | None, cookies: str | None,
           live_minutes: float | None = None) -> Path:
    require_ffmpeg()
    platform = detect_platform(source)
    print(f"[1/4] Fonte detectada: {platform}" + (f" (ao vivo {live_minutes:g} min)" if live_minutes else ""))

    if workdir is None:
        workdir = default_workdir(source)
    workdir.mkdir(parents=True, exist_ok=True)

    live_chat: list[dict] | None = None
    print("[2/4] Vídeo")
    video = workdir / config.VIDEO_NAME
    if platform == "local":
        src = Path(source).expanduser().resolve()
        if src != video.resolve():
            shutil.copy2(src, video)
    elif live_minutes:
        video, live_chat = record_live(platform, source, workdir, live_minutes, cookies)
    else:
        video = download_video(source, workdir, cookies)

    print("[3/4] Áudio mono 16kHz")
    wav = workdir / config.AUDIO_NAME
    extract_audio(video, wav)
    try:
        import soundfile as sf
        duration_hint = sf.info(str(wav)).duration
    except Exception:  # noqa: BLE001
        duration_hint = None

    print("[4/4] Chat")
    chat = live_chat if live_chat is not None else get_chat(platform, source, workdir, cookies, duration_hint)
    (workdir / config.CHAT_NAME).write_text(
        json.dumps(chat, ensure_ascii=False, indent=1), encoding="utf-8")
    if chat:
        print(f"  {len(chat)} mensagens normalizadas.")
    else:
        print("  Sem chat disponível — o radar usará apenas áudio.")

    return workdir


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source", help="URL (YouTube/Twitch/Kick) ou caminho de .mp4 local")
    ap.add_argument("--cookies-from-browser", default=None, metavar="BROWSER",
                    help="ex.: chrome, firefox, edge (evita bloqueios/anti-bot)")
    ap.add_argument("--live-minutes", type=float, default=None,
                    help="captura N minutos de uma live em andamento (+ chat em tempo real)")
    config.add_common_args(ap)
    args = ap.parse_args()
    settings = config.settings_from_args(args)

    t0 = time.time()
    workdir = ingest(args.source, args.workdir, args.cookies_from_browser, args.live_minutes)
    (workdir / config.META_NAME).write_text(json.dumps({
        "source": args.source,
        "topic": settings.topic,
        "platform": detect_platform(args.source),
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nPronto em {time.time() - t0:.0f}s -> {workdir}")


if __name__ == "__main__":
    main()

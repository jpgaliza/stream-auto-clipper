"""Módulo 6 - Publicação (YouTube Shorts + TikTok), sempre como privado/rascunho.

Uso:
    python 06_publisher.py <workdir> --upload-yt --upload-tiktok     (ou --upload-all)

YouTube : YouTube Data API v3, client_secrets.json + token.json (cache), privacyStatus="private".
TikTok  : biblioteca tiktok-uploader, cookies em tiktok_cookies.txt (formato Netscape),
          visibility="only_you" (somente você) para revisão antes de publicar.
Sem nenhuma flag de upload nada é enviado. Estado em uploads.json (não reenvia o que já subiu).
Ao final imprime linhas `RESULT <plataforma> ok=N failed=N skipped=N` lidas pelo main.py.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import config

YT_SCOPES = ["https://www.googleapis.com/auth/youtube.upload"]
YT_SECRETS = config.ROOT_DIR / "client_secrets.json"
YT_TOKEN = config.ROOT_DIR / "token.json"
TIKTOK_COOKIES = config.ROOT_DIR / "tiktok_cookies.txt"
TIKTOK_CAPTION_LIMIT = 2200
PLATFORMS = ("youtube", "tiktok")


# ================================================================= YouTube
def get_youtube_service():
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow
    from googleapiclient.discovery import build

    creds = None
    if YT_TOKEN.exists():
        creds = Credentials.from_authorized_user_file(str(YT_TOKEN), YT_SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            if not YT_SECRETS.exists():
                raise RuntimeError(f"{YT_SECRETS.name} não encontrado (OAuth 'Desktop app' do Google Cloud)")
            creds = InstalledAppFlow.from_client_secrets_file(str(YT_SECRETS), YT_SCOPES).run_local_server(port=0)
        YT_TOKEN.write_text(creds.to_json(), encoding="utf-8")
    return build("youtube", "v3", credentials=creds)


def upload_youtube_clip(service, path: Path, clip: dict) -> str:
    from googleapiclient.http import MediaFileUpload

    tags = [h.lstrip("#") for h in clip["hashtags"]]
    description = f"{clip['description']}\n\n{' '.join(clip['hashtags'])}"
    body = {
        "snippet": {"title": clip["title"][:100] or path.stem, "description": description,
                    "tags": tags, "categoryId": "24"},  # 24 = Entertainment
        "status": {"privacyStatus": "private", "selfDeclaredMadeForKids": False},
    }
    media = MediaFileUpload(str(path), mimetype="video/mp4", chunksize=8 * 1024 * 1024, resumable=True)
    req = service.videos().insert(part="snippet,status", body=body, media_body=media)
    resp = None
    while resp is None:
        status, resp = req.next_chunk()
        if status:
            print(f"    {int(status.progress() * 100)}%", end="\r")
    return resp["id"]


# ================================================================== TikTok
def tiktok_caption(clip: dict) -> str:
    # Remove #shorts (YouTube) e garante tags de descoberta do TikTok
    tags = [h for h in clip["hashtags"] if h.lower() != "#shorts"]
    lower = {h.lower() for h in tags}
    for extra in ("#fy", "#fyp", "#cortes"):
        if extra not in lower:
            tags.append(extra)
            lower.add(extra)
    caption = f"{clip['title']} {' '.join(tags)}".strip()
    if len(caption) > TIKTOK_CAPTION_LIMIT:
        caption = caption[:TIKTOK_CAPTION_LIMIT].rsplit(" ", 1)[0]
    return caption


def upload_tiktok_clip(path: Path, clip: dict, browser: str, headless: bool) -> None:
    """Roda cada upload do TikTok em subprocesso isolado para evitar conflito de
    loop asyncio do Playwright entre chamadas sequenciais."""
    import subprocess

    payload = json.dumps({
        "filename": str(path),
        "description": tiktok_caption(clip),
        "cookies": str(TIKTOK_COOKIES),
        "visibility": "only_you",
        "browser": browser,
        "headless": headless,
    })
    code = (
        "import json, sys; from tiktok_uploader.upload import upload_video; "
        "kw = json.loads(sys.stdin.read()); "
        "failed = upload_video(**kw); "
        "sys.exit(1 if failed else 0)"
    )
    res = subprocess.run([sys.executable, "-c", code], input=payload, text=True)
    if res.returncode != 0:
        raise RuntimeError("tiktok-uploader reportou falha no envio")


# =================================================================== estado
def load_state(path: Path) -> dict:
    if not path.exists():
        return {p: {} for p in PLATFORMS}
    data = json.loads(path.read_text(encoding="utf-8"))
    if any(k.endswith(".mp4") for k in data):  # formato antigo (só YouTube)
        data = {"youtube": data}
    for p in PLATFORMS:
        data.setdefault(p, {})
    return data


def publish(workdir: Path, platforms: set[str], tiktok_browser: str = "chrome",
            tiktok_headless: bool = False) -> dict[str, dict]:
    verified = workdir / config.VERIFIED_NAME
    rendered = workdir / config.RENDERED_NAME
    manifest = verified if verified.exists() else rendered
    if not manifest.exists():
        raise SystemExit(f"{rendered} não encontrado. Rode 05_vision_render.py primeiro.")
    clips = json.loads(manifest.read_text(encoding="utf-8"))
    if verified.exists():
        print(f"Usando {len(clips)} corte(s) aprovado(s) em {verified.name}.")
        rep_p = workdir / config.VERIFY_REPORT_NAME
        if rep_p.exists():
            blocked = [c for c in json.loads(rep_p.read_text(encoding="utf-8")) if not c.get("verify_approved")]
            if blocked:
                print(f"Bloqueados na verificação ({len(blocked)}):")
                for b in blocked:
                    print(f"  - {b['file']} (nota {b.get('verify_score', 0):.1f}): {b.get('verify_feedback', '')}")
    state_p = workdir / config.UPLOADS_NAME
    state = load_state(state_p)
    results = {p: {"ok": 0, "failed": 0, "skipped": 0, "error": ""} for p in platforms}

    def save() -> None:
        state_p.write_text(json.dumps(state, indent=1), encoding="utf-8")

    # ---- YouTube
    if "youtube" in platforms:
        print("== YouTube Shorts (private) ==")
        r = results["youtube"]
        try:
            service = get_youtube_service()
        except Exception as e:  # noqa: BLE001
            r["error"] = str(e)
            r["failed"] = len([c for c in clips if c["file"] not in state["youtube"]])
            print(f"  ! autenticação falhou: {e}")
            service = None
        for clip in clips if service else []:
            if clip["file"] in state["youtube"]:
                r["skipped"] += 1
                print(f"  {clip['file']} já enviado, pulando.")
                continue
            print(f"  Enviando {clip['file']} - {clip['title']}")
            try:
                vid = upload_youtube_clip(service, workdir / clip["file"], clip)
            except Exception as e:  # noqa: BLE001 - não perder os demais uploads
                r["failed"] += 1
                r["error"] = str(e)[:200]
                print(f"    ! falhou: {e}")
                continue
            state["youtube"][clip["file"]] = vid
            save()
            r["ok"] += 1
            print(f"    OK: https://studio.youtube.com/video/{vid}/edit")

    # ---- TikTok
    if "tiktok" in platforms:
        print("== TikTok (only_you) ==")
        r = results["tiktok"]
        pending = [c for c in clips if c["file"] not in state["tiktok"]]
        r["skipped"] = len(clips) - len(pending)
        if not TIKTOK_COOKIES.exists():
            r["failed"] = len(pending)
            r["error"] = f"{TIKTOK_COOKIES.name} não encontrado (exporte os cookies do TikTok em formato Netscape)"
            print(f"  ! {r['error']}")
            pending = []
        try:
            import tiktok_uploader  # noqa: F401
        except ImportError:
            if pending:
                r["failed"] = len(pending)
                r["error"] = "tiktok-uploader não instalado (pip install tiktok-uploader; playwright install)"
                print(f"  ! {r['error']}")
                pending = []
        for n, clip in enumerate(pending):
            print(f"  Enviando {clip['file']} - {clip['title']}")
            try:
                upload_tiktok_clip(workdir / clip["file"], clip, tiktok_browser, tiktok_headless)
            except Exception as e:  # noqa: BLE001
                r["failed"] += 1
                r["error"] = str(e)[:200]
                print(f"    ! falhou: {e}")
                continue
            state["tiktok"][clip["file"]] = "only_you"
            save()
            r["ok"] += 1
            print("    OK (visível só para você)")
            if n < len(pending) - 1:
                time.sleep(5)  # respiro entre uploads para evitar rate limit

    for p, r in results.items():
        print(f"RESULT {p} ok={r['ok']} failed={r['failed']} skipped={r['skipped']}"
              + (f" error={r['error']}" if r["error"] else ""))
    return results


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("workdir", type=Path)
    ap.add_argument("--upload-yt", action="store_true", help="envia ao YouTube (private)")
    ap.add_argument("--upload-tiktok", action="store_true", help="envia ao TikTok (only_you)")
    ap.add_argument("--upload-all", action="store_true", help="YouTube e TikTok em sequência")
    ap.add_argument("--upload-draft", action="store_true", help="(legado) equivale a --upload-yt")
    ap.add_argument("--tiktok-browser", default="chrome",
                    choices=["chrome", "safari", "chromium", "edge", "firefox"])
    ap.add_argument("--tiktok-headless", action="store_true",
                    help="navegador oculto (o TikTok costuma bloquear; padrão: visível)")
    args = ap.parse_args()

    platforms = set()
    if args.upload_all:
        platforms |= set(PLATFORMS)
    if args.upload_yt or args.upload_draft:
        platforms.add("youtube")
    if args.upload_tiktok:
        platforms.add("tiktok")
    if not platforms:
        print("Nenhuma flag de upload informada. Nada enviado.")
        return
    results = publish(args.workdir, platforms, args.tiktok_browser, args.tiktok_headless)
    if any(r["failed"] for r in results.values()):
        sys.exit(2)


if __name__ == "__main__":
    main()

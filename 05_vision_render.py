"""Módulo 5 - Visão + render vertical.

Uso:
    python 05_vision_render.py <workdir> [--max-clips 15]

1. Detecta a webcam (rosto do streamer) em 15 frames e calcula o recorte do topo.
2. Gera legendas karaokê .ass (pysubs2) por corte.
3. Renderiza com FFmpeg (h264_nvenc): topo 35% webcam, base 65% conteúdo,
   divisória de 4px esmeralda e legenda queimada.
Saídas: corte_XX.mp4, corte_XX.txt, rendered_clips.json
"""
from __future__ import annotations

import argparse
import json
import shutil
import statistics
import subprocess
import urllib.request
from pathlib import Path

import cv2
import numpy as np
import pysubs2

import config

TOP_ASPECT = config.OUT_WIDTH / config.TOP_HEIGHT
BOTTOM_ASPECT = config.OUT_WIDTH / config.BOTTOM_HEIGHT


# ================================================================== visão
class FaceFinder:
    """MediaPipe Face Detection (API tasks ou legada) com fallback Haar do OpenCV."""

    def __init__(self) -> None:
        self.kind = "haar"
        self._det = None
        try:
            import mediapipe as mp
            if hasattr(mp, "solutions"):  # API legada
                self._det = mp.solutions.face_detection.FaceDetection(
                    model_selection=1, min_detection_confidence=0.6)
                self.kind = "mp-legacy"
            else:
                from mediapipe.tasks import python as mp_python
                from mediapipe.tasks.python import vision
                model = config.MODELS_DIR / "blaze_face_short_range.tflite"
                if not model.exists():
                    config.MODELS_DIR.mkdir(exist_ok=True)
                    print("  Baixando modelo de face do MediaPipe...")
                    urllib.request.urlretrieve(config.FACE_MODEL_URL, model)
                opts = vision.FaceDetectorOptions(
                    base_options=mp_python.BaseOptions(model_asset_path=str(model)),
                    min_detection_confidence=0.6)
                self._det = vision.FaceDetector.create_from_options(opts)
                self._mp = mp
                self.kind = "mp-tasks"
        except Exception as e:  # noqa: BLE001
            print(f"  ! MediaPipe indisponível ({type(e).__name__}); usando Haar cascade.")
            self._det = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    def _detect(self, bgr: np.ndarray) -> list[tuple[int, int, int, int]]:
        h, w = bgr.shape[:2]
        if self.kind == "mp-legacy":
            res = self._det.process(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            out = []
            for d in res.detections or []:
                b = d.location_data.relative_bounding_box
                out.append((int(b.xmin * w), int(b.ymin * h), int(b.width * w), int(b.height * h)))
            return out
        if self.kind == "mp-tasks":
            rgb = np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            img = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
            return [(d.bounding_box.origin_x, d.bounding_box.origin_y,
                     d.bounding_box.width, d.bounding_box.height)
                    for d in self._det.detect(img).detections]
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        return [tuple(map(int, r)) for r in
                self._det.detectMultiScale(gray, 1.1, 6, minSize=(24, 24))]

    def faces(self, frame: np.ndarray) -> list[tuple[int, int, int, int]]:
        """Detecta no frame inteiro e em ladrilhos 2x2 e 4x4 sobrepostos:
        webcams em cantos são pequenas demais para o detector no frame cheio."""
        H, W = frame.shape[:2]
        found = list(self._detect(frame))
        for div in (2, 4):
            tw, th = W // div, H // div
            for ty in range(0, H - th + 1, th // 2):
                for tx in range(0, W - tw + 1, tw // 2):
                    for (x, y, w, h) in self._detect(frame[ty:ty + th, tx:tx + tw]):
                        found.append((x + tx, y + ty, w, h))
        return [(x, y, w, h) for x, y, w, h in found if w >= 20 and h >= 20]


def sample_frames(video: Path, n: int) -> tuple[list[np.ndarray], tuple[int, int]]:
    cap = cv2.VideoCapture(str(video))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames = []
    for i in range(n):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(total * (i + 0.5) / n))
        ok, fr = cap.read()
        if ok:
            frames.append(fr)
    cap.release()
    return frames, (W, H)


def cluster_faces(per_frame: list[list[tuple]]) -> dict | None:
    """Cluster predominante de (x,y,w,h): agrupa por proximidade de centro."""
    clusters: list[dict] = []
    for fi, dets in enumerate(per_frame):
        for (x, y, w, h) in dets:
            c = (x + w / 2, y + h / 2)
            for cl in clusters:
                if (abs(c[0] - cl["c"][0]) < max(w, cl["w"]) * 0.75
                        and abs(c[1] - cl["c"][1]) < max(h, cl["h"]) * 0.75):
                    cl["items"].append((x, y, w, h, fi))
                    cl["c"] = (statistics.median(i[0] + i[2] / 2 for i in cl["items"]),
                               statistics.median(i[1] + i[3] / 2 for i in cl["items"]))
                    cl["w"] = statistics.median(i[2] for i in cl["items"])
                    cl["h"] = statistics.median(i[3] for i in cl["items"])
                    break
            else:
                clusters.append({"c": c, "w": w, "h": h, "items": [(x, y, w, h, fi)]})
    if not clusters:
        return None
    best = max(clusters, key=lambda cl: len({i[4] for i in cl["items"]}))
    n_frames = len({i[4] for i in best["items"]})
    if n_frames < config.WEBCAM_MIN_FRAMES:
        return None
    its = best["items"]
    return {"x": statistics.median(i[0] for i in its), "y": statistics.median(i[1] for i in its),
            "w": statistics.median(i[2] for i in its), "h": statistics.median(i[3] for i in its),
            "frames": n_frames}


def even(v: float) -> int:
    return int(v) // 2 * 2


def _find_edge(grad_1d: np.ndarray, offset: int, from_end: bool = False) -> int | None:
    """Encontra uma borda estática forte (ex.: moldura da webcam) em um perfil 1D."""
    if grad_1d.size < 5:
        return None
    med = float(np.median(grad_1d))
    thresh = max(35.0, 4.0 * med)
    peaks = np.where(grad_1d >= thresh)[0]
    if peaks.size == 0:
        return None
    # Pega a borda mais próxima do rosto (último pico antes do rosto, ou primeiro depois)
    idx = int(peaks[-1] if from_end else peaks[0])
    return offset + idx


def _find_webcam_bounds(frames: list[np.ndarray], face: dict, W: int, H: int) -> tuple[int, int, int, int]:
    """Usa a mediana temporal dos frames para achar as bordas estáticas da webcam
    ao redor do rosto, evitando que o recorte invada a gameplay sem espremer o rosto."""
    fcx, fcy = face["x"] + face["w"] / 2, face["y"] + face["h"] / 2
    fw, fh = face["w"], face["h"]
    med = np.median([cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames], axis=0).astype(np.float32)
    gy = np.abs(cv2.Sobel(med, cv2.CV_32F, 0, 1, ksize=3))
    gx = np.abs(cv2.Sobel(med, cv2.CV_32F, 1, 0, ksize=3))

    x_span0, x_span1 = max(0, int(fcx - fw)), min(W, int(fcx + fw))
    y_span0, y_span1 = max(0, int(fcy - 0.5 * fh)), min(H, int(fcy + 0.5 * fh))

    # Começa a buscar longe o bastante do rosto para nunca confundir cabelo/óculos/ombro com borda
    top_lo = max(0, int(fcy - 2.8 * fh))
    top_hi = max(top_lo, int(fcy - 0.85 * fh))
    by1 = _find_edge(gy[top_lo:top_hi, x_span0:x_span1].mean(axis=1), top_lo, from_end=True)

    bot_lo = min(H, int(fcy + 0.85 * fh))
    bot_hi = min(H, int(fcy + 3.0 * fh))
    by2 = _find_edge(gy[bot_lo:bot_hi, x_span0:x_span1].mean(axis=1), bot_lo, from_end=False)

    left_lo = max(0, int(fcx - 3.8 * fw))
    left_hi = max(left_lo, int(fcx - 1.15 * fw))
    bx1 = _find_edge(gx[y_span0:y_span1, left_lo:left_hi].mean(axis=0), left_lo, from_end=True)

    right_lo = min(W, int(fcx + 1.15 * fw))
    right_hi = min(W, int(fcx + 3.8 * fw))
    bx2 = _find_edge(gx[y_span0:y_span1, right_lo:right_hi].mean(axis=0), right_lo, from_end=False)

    # Desconta a espessura da borda/moldura quando detectada
    pad = max(6, int(0.08 * fw))
    bx1 = (bx1 + pad) if bx1 is not None else 0
    by1 = (by1 + pad) if by1 is not None else 0
    bx2 = (bx2 - pad) if bx2 is not None else W
    by2 = (by2 - pad) if by2 is not None else H
    # Garante tamanho mínimo plausível de webcam para nunca espremer o rosto
    if bx2 - bx1 < fw * 3.2:
        bx1, bx2 = 0, W
    if by2 - by1 < fh * 2.0:
        by1, by2 = 0, H
    return bx1, by1, bx2, by2


def webcam_crop(face: dict, W: int, H: int,
                bounds: tuple[int, int, int, int] | None = None) -> tuple[int, int, int, int]:
    """Recorte fixo com a proporção do painel superior (≈16:9), centrado no rosto,
    limitado às bordas detectadas da webcam para nunca vazar gameplay."""
    bx1, by1, bx2, by2 = bounds or (0, 0, W, H)
    bw, bh = bx2 - bx1, by2 - by1

    cw = min(float(bw), face["w"] * config.WEBCAM_BOX_FACTOR * (1 - config.WEBCAM_INNER_MARGIN))
    ch = cw / TOP_ASPECT
    if ch > bh:
        ch = float(bh)
        cw = ch * TOP_ASPECT
    if cw > bw:
        cw = float(bw)
        ch = cw / TOP_ASPECT

    fcx, fcy = face["x"] + face["w"] / 2, face["y"] + face["h"] / 2
    x = min(max(fcx - cw / 2, bx1), bx2 - cw)
    y = min(max(fcy - 0.45 * ch, by1), by2 - ch)
    return even(cw), even(ch), even(x), even(y)


def center_crop(W: int, H: int, aspect: float) -> tuple[int, int, int, int]:
    cw, ch = H * aspect, H
    if cw > W:
        cw, ch = W, W / aspect
    return even(cw), even(ch), even((W - cw) / 2), even((H - ch) / 2)


def detect_webcam(video: Path, workdir: Path) -> dict:
    cache = workdir / "webcam.json"
    if cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))
    print(f"Detectando webcam em {config.WEBCAM_SAMPLES} frames...")
    frames, (W, H) = sample_frames(video, config.WEBCAM_SAMPLES)
    finder = FaceFinder()
    per_frame = [finder.faces(f) for f in frames]
    face = cluster_faces(per_frame)
    if face:
        bounds = _find_webcam_bounds(frames, face, W, H)
        info = {"mode": "split", "W": W, "H": H, "face": face, "bounds": bounds,
                "top_crop": webcam_crop(face, W, H, bounds)}
        print(f"  Webcam encontrada ({finder.kind}): rosto {face['w']:.0f}px em "
              f"{face['frames']}/{len(frames)} frames.")
    else:
        info = {"mode": "fallback", "W": W, "H": H, "face": None}
        print("  Webcam não detectada de forma consistente -> fallback 9:16 centralizado.")
    cache.write_text(json.dumps(info, indent=1), encoding="utf-8")
    return info


def clip_webcam_info(video: Path, clip: dict, global_info: dict, finder: FaceFinder) -> dict:
    """Confere 3 frames do próprio corte: se o streamer escondeu a webcam (ex.: tela de
    loading/cutscene), usa fallback 9:16 nesse corte; se a webcam mudou de posição ou
    estava deslocada, recentraliza no rosto detectado neste corte."""
    if global_info["mode"] != "split":
        return global_info
    s, e = clip["start_time"], clip["end_time"]
    times = [s + 1.0, (s + e) / 2.0, max(s + 0.5, e - 1.0)]
    cap = cv2.VideoCapture(str(video))
    frames = []
    for t in times:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
        ok, fr = cap.read()
        if ok:
            frames.append(fr)
    cap.release()
    if not frames:
        return global_info

    per = [finder.faces(f) for f in frames]
    gf = global_info["face"]
    g_cx, g_cy = gf["x"] + gf["w"] / 2, gf["y"] + gf["h"] / 2
    # Procura rosto próximo ao cluster global primeiro
    near = []
    for dets in per:
        for (x, y, w, h) in dets:
            cx, cy = x + w / 2, y + h / 2
            if abs(cx - g_cx) < gf["w"] * 2.0 and abs(cy - g_cy) < gf["h"] * 2.0:
                near.append((x, y, w, h))
    if not near:
        # Webcam oculta ou cena sem webcam neste corte -> fallback 9:16 limpo
        return {"mode": "fallback", "W": global_info["W"], "H": global_info["H"], "face": None}

    face = {
        "x": statistics.median(i[0] for i in near),
        "y": statistics.median(i[1] for i in near),
        "w": statistics.median(i[2] for i in near),
        "h": statistics.median(i[3] for i in near),
    }
    bounds = global_info.get("bounds")
    return {
        "mode": "split",
        "W": global_info["W"],
        "H": global_info["H"],
        "face": face,
        "top_crop": webcam_crop(face, global_info["W"], global_info["H"], bounds),
    }


# ============================================================= legendas .ass
def clip_words(windows: list[dict], start: float, end: float) -> list[dict]:
    seen, out = set(), []
    for win in windows:
        for seg in win["segments"]:
            for w in seg.get("words", []):
                if w["start"] >= start - 0.05 and w["end"] <= end + 0.05:
                    key = (round(w["start"], 2), w["word"])
                    if key not in seen and w["word"].strip():
                        seen.add(key)
                        out.append(w)
    return sorted(out, key=lambda w: w["start"])


def group_words(words: list[dict]) -> list[list[dict]]:
    groups: list[list[dict]] = []
    cur: list[dict] = []
    for w in words:
        txt = w["word"].strip()
        if cur:
            prev = cur[-1]["word"].strip()
            chars = sum(len(x["word"].strip()) for x in cur) + len(txt) + len(cur)
            if (len(cur) >= 3 or chars > 18 or prev[-1:] in ".!?…,;:"
                    or w["start"] - cur[-1]["end"] > 0.7):
                groups.append(cur)
                cur = []
        cur.append(w)
    if cur:
        groups.append(cur)
    return groups


def build_ass(words: list[dict], clip_start: float, clip_end: float, y: int, path: Path) -> None:
    accent = config.ACCENT_HEX.lstrip("#")
    r, g, b = (int(accent[i:i + 2], 16) for i in (0, 2, 4))
    ass_accent = f"&H{b:02X}{g:02X}{r:02X}&"  # ASS usa BGR

    subs = pysubs2.SSAFile()
    subs.info["PlayResX"] = str(config.OUT_WIDTH)
    subs.info["PlayResY"] = str(config.OUT_HEIGHT)
    subs.info["WrapStyle"] = "0"
    subs.styles["Default"] = pysubs2.SSAStyle(
        fontname=config.SUB_FONT, fontsize=config.SUB_FONT_SIZE, bold=False,
        primarycolor=pysubs2.Color(255, 255, 255, 0),
        secondarycolor=pysubs2.Color(255, 255, 255, 0),
        outlinecolor=pysubs2.Color(17, 17, 17, 0),
        backcolor=pysubs2.Color(0, 0, 0, 110),   # sombra suave
        outline=2, shadow=4, alignment=5, borderstyle=1,
        marginl=40, marginr=40, marginv=0)

    groups = group_words(words)
    for gi, grp in enumerate(groups):
        grp_end_limit = (groups[gi + 1][0]["start"] if gi + 1 < len(groups)
                         else min(grp[-1]["end"] + 0.15, clip_end))
        for i, w in enumerate(grp):
            t0 = w["start"]
            t1 = grp[i + 1]["start"] if i + 1 < len(grp) else max(w["end"], min(grp_end_limit, w["end"] + 0.15))
            parts = []
            for j, x in enumerate(grp):
                token = x["word"].strip().upper()
                if j == i:
                    parts.append(f"{{\\c{ass_accent}\\t(0,90,\\fscx112\\fscy112)}}{token}{{\\r}}")
                else:
                    parts.append(token)
            text = f"{{\\an5\\pos({config.OUT_WIDTH // 2},{y})}}" + " ".join(parts)
            subs.append(pysubs2.SSAEvent(
                start=pysubs2.make_time(s=max(0.0, t0 - clip_start)),
                end=pysubs2.make_time(s=max(0.01, t1 - clip_start)),
                text=text, style="Default"))
    subs.save(str(path))


# =================================================================== render
def build_filter(info: dict, ass_name: str, fonts_rel: str | None) -> str:
    W, H = info["W"], info["H"]
    ass = f"ass={ass_name}" + (f":fontsdir={fonts_rel}" if fonts_rel else "")
    if info["mode"] == "split":
        tw, th, tx, ty = info["top_crop"]
        bw, bh, bx, by = center_crop(W, H, BOTTOM_ASPECT)
        return (
            f"[0:v]split=2[t][g];"
            f"[t]crop={tw}:{th}:{tx}:{ty},scale={config.OUT_WIDTH}:{config.TOP_HEIGHT}:flags=lanczos[top];"
            f"[g]crop={bw}:{bh}:{bx}:{by},scale={config.OUT_WIDTH}:{config.BOTTOM_HEIGHT}:flags=lanczos[bot];"
            f"[top][bot]vstack=inputs=2,"
            f"drawbox=x=0:y={config.TOP_HEIGHT - config.DIVIDER_PX // 2}:w={config.OUT_WIDTH}"
            f":h={config.DIVIDER_PX}:color=0x{config.ACCENT_HEX.lstrip('#')}:t=fill,"
            f"setsar=1,{ass}[v]")
    cw, ch, cx, cy = center_crop(W, H, config.OUT_WIDTH / config.OUT_HEIGHT)
    return (f"[0:v]crop={cw}:{ch}:{cx}:{cy},scale={config.OUT_WIDTH}:{config.OUT_HEIGHT}:flags=lanczos,"
            f"setsar=1,{ass}[v]")


def render_clip(workdir: Path, clip: dict, idx: int, info: dict,
                words: list[dict], fonts_rel: str | None) -> Path:
    name = f"corte_{idx:02d}"
    ass_name = f"{name}.ass"
    y = config.SUB_Y if info["mode"] == "split" else config.SUB_Y_FALLBACK
    build_ass(words, clip["start_time"], clip["end_time"], y, workdir / ass_name)

    out = workdir / f"{name}.mp4"
    flt = build_filter(info, ass_name, fonts_rel)
    base = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-ss", f"{clip['start_time']:.3f}", "-t", f"{clip['end_time'] - clip['start_time']:.3f}",
            "-i", config.VIDEO_NAME, "-filter_complex", flt,
            "-map", "[v]", "-map", "0:a:0?"]
    audio = ["-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", out.name]
    nvenc = base + ["-c:v", "h264_nvenc", "-preset", "p5", "-rc", "vbr", "-cq", "21",
                    "-b:v", "0", "-pix_fmt", "yuv420p"] + audio
    cpu = base + ["-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
                  "-pix_fmt", "yuv420p"] + audio
    # cwd=workdir mantém caminhos relativos (evita escapar "C:" no filtro ass)
    if subprocess.run(nvenc, cwd=workdir).returncode != 0:
        print("  ! NVENC falhou; tentando libx264 (CPU).")
        subprocess.run(cpu, cwd=workdir, check=True)
    return out


def write_txt(path: Path, clip: dict) -> None:
    path.write_text(
        f"{clip['title']}\n\n{clip['description']}\n\n{' '.join(clip['hashtags'])}\n",
        encoding="utf-8")


def run_render(workdir: Path, max_clips: int) -> list[dict]:
    approved_p = workdir / config.APPROVED_NAME
    trans_p = workdir / config.TRANSCRIBED_NAME
    for p in (approved_p, trans_p):
        if not p.exists():
            raise SystemExit(f"{p} não encontrado. Rode os módulos anteriores.")
    clips = json.loads(approved_p.read_text(encoding="utf-8"))[:max_clips]
    windows = json.loads(trans_p.read_text(encoding="utf-8"))
    if not clips:
        print("Nenhum corte aprovado.")
        return []

    video = workdir / config.VIDEO_NAME
    info = detect_webcam(video, workdir)

    fonts_rel = None
    if config.FONTS_DIR.exists() and any(config.FONTS_DIR.iterdir()):
        shutil.copytree(config.FONTS_DIR, workdir / "fonts", dirs_exist_ok=True)
        fonts_rel = "fonts"
    else:
        print(f"  Aviso: sem fontes em {config.FONTS_DIR}. Instale '{config.SUB_FONT}' no "
              "Windows ou coloque o .ttf nessa pasta, senão o libass usa uma fonte padrão.")

    rendered = []
    finder = FaceFinder()
    for i, clip in enumerate(clips, 1):
        c_info = clip_webcam_info(video, clip, info, finder)
        mode_tag = "split" if c_info["mode"] == "split" else "9:16(sem webcam)"
        print(f"[{i}/{len(clips)}] corte_{i:02d} ({clip['duration']:.0f}s, {mode_tag}) {clip['title']}")
        words = clip_words(windows, clip["start_time"], clip["end_time"])
        out = render_clip(workdir, clip, i, c_info, words, fonts_rel)
        write_txt(workdir / f"corte_{i:02d}.txt", clip)
        rendered.append({**clip, "file": out.name, "index": i})
    (workdir / config.RENDERED_NAME).write_text(
        json.dumps(rendered, ensure_ascii=False, indent=1), encoding="utf-8")
    return rendered


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("workdir", type=Path)
    ap.add_argument("--max-clips", type=int, default=config.DEFAULT_MAX_CLIPS)
    args = ap.parse_args()
    out = run_render(args.workdir, args.max_clips)
    print(f"{len(out)} cortes renderizados em {args.workdir}")


if __name__ == "__main__":
    main()

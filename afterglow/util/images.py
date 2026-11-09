"""Image loading and lightweight analysis helpers (ARCHITECTURE.md §5 ②③⑥).

Angles follow LUMEN's convention: a positive angle turns the picture counter-clockwise.
Saliency centres are normalised (cx, cy) in 0-1.
"""

from __future__ import annotations

import io
import logging
import math
import os
import subprocess
import tempfile
import threading
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

log = logging.getLogger(__name__)

FFMPEG_TIMEOUT = 120

# Straighten estimation (same method as LUMEN's straighten.py: LSD segments vote for the levelling angle).
LSD_EDGE = 1200
SEARCH_DEG = 20.0
SPREAD_DEG = 0.35
SUPPORT_RATIO = 0.3

try:  # HEIC support is optional
    from pillow_heif import register_heif_opener

    register_heif_opener()
except Exception:  # noqa: BLE001 - optional dependency, absence is fine
    pass


# --------------------------------------------------------------------------- loading / saving
def _fit(img: Image.Image, long_edge: int) -> Image.Image:
    orig = img.size
    if long_edge and max(img.size) > long_edge:
        img = img.copy()
        img.thumbnail((long_edge, long_edge), Image.Resampling.LANCZOS)
    img = img.convert("RGB")
    img.info["orig_size"] = orig
    return img


def _load_raw(path: Path, long_edge: int) -> Image.Image:
    import rawpy

    with rawpy.imread(str(path)) as raw:
        try:
            thumb = raw.extract_thumb()
        except Exception as exc:  # noqa: BLE001 - rawpy raises several unrelated error types
            log.debug("无内嵌预览 %s: %s", path, exc)
            thumb = None
        if thumb is not None:
            img = None
            if thumb.format == rawpy.ThumbFormat.JPEG:
                img = ImageOps.exif_transpose(Image.open(io.BytesIO(thumb.data)))
            elif thumb.format == rawpy.ThumbFormat.BITMAP:
                img = Image.fromarray(thumb.data)
            if img is not None and max(img.size) >= min(long_edge, 1024):
                return _fit(img, long_edge)
        rgb = raw.postprocess(half_size=True, use_camera_wb=True, output_bps=8)
    return _fit(Image.fromarray(rgb), long_edge)


def _load_video_frame(path: Path, long_edge: int) -> Image.Image | None:
    with tempfile.TemporaryDirectory(prefix="afterglow-frame-") as tmp:
        out = Path(tmp) / "frame.jpg"
        for seek in ("1", "0"):
            cmd = ["ffmpeg", "-y", "-v", "error", "-ss", seek, "-i", str(path), "-frames:v", "1", str(out)]
            try:
                proc = subprocess.run(cmd, check=False, capture_output=True, text=True, timeout=FFMPEG_TIMEOUT)
            except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
                log.warning("ffmpeg 取帧失败 %s: %s", path, exc)
                return None
            if proc.returncode == 0 and out.exists() and out.stat().st_size > 0:
                with Image.open(out) as im:
                    im.load()
                    return _fit(im, long_edge)
    log.warning("视频取帧失败: %s", path)
    return None


def load_preview(path: str | Path, kind: str, long_edge: int = 2560) -> Image.Image | None:
    """RGB preview whose long edge is at most `long_edge`; `img.info['orig_size']` holds the source size.

    RAW -> embedded thumbnail (fallback: half-size demosaic); jpeg/heic/png -> Pillow with EXIF orientation;
    video -> ffmpeg frame at 1 s (None when unavailable).  Other failures raise.
    """
    p = Path(path)
    if kind == "video":
        return _load_video_frame(p, long_edge)
    if kind == "raw":
        return _load_raw(p, long_edge)
    with Image.open(p) as im:
        im.load()
        return _fit(ImageOps.exif_transpose(im), long_edge)


def save_jpeg(img: Image.Image, path: str | Path, quality: int = 90) -> Path:
    """Write `img` as JPEG atomically (temp file + rename), creating parent directories."""
    dst = Path(path)
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(f".{dst.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    img.convert("RGB").save(tmp, format="JPEG", quality=quality, optimize=True)
    os.replace(tmp, dst)
    return dst


def to_rgb_array(img: Image.Image) -> np.ndarray:
    return np.ascontiguousarray(np.asarray(img.convert("RGB"), dtype=np.uint8))


def from_rgb_array(rgb: np.ndarray) -> Image.Image:
    return Image.fromarray(np.ascontiguousarray(rgb, dtype=np.uint8), "RGB")


def jpeg_bytes(img: Image.Image, quality: int = 85) -> bytes:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


# --------------------------------------------------------------------------- quality metrics
def _grey(rgb: np.ndarray) -> np.ndarray:
    if rgb.ndim == 2:
        return rgb
    return cv2.cvtColor(np.ascontiguousarray(rgb[..., :3]), cv2.COLOR_RGB2GRAY)


def laplacian_sharpness(rgb: np.ndarray) -> float:
    """Sharpness 0-100: log-scaled variance of the Laplacian on a grey copy with long edge <= 1600.

    ~0 for heavily blurred pictures, ~70-100 for crisp ones.
    """
    grey = _grey(rgb)
    h, w = grey.shape
    scale = 1600 / max(h, w)
    if scale < 1:
        grey = cv2.resize(grey, (max(1, round(w * scale)), max(1, round(h * scale))), interpolation=cv2.INTER_AREA)
    var = float(cv2.Laplacian(grey, cv2.CV_64F).var())
    return float(np.clip(100.0 * math.log10(1.0 + var) / math.log10(1.0 + 3000.0), 0.0, 100.0))


def exposure_score(rgb: np.ndarray) -> float:
    """1 - (share of pixels below 2 + share above 253) on the luminance histogram, in 0-1."""
    grey = _grey(rgb)
    total = grey.size
    if total == 0:
        return 0.0
    low = float(np.count_nonzero(grey < 2)) / total
    high = float(np.count_nonzero(grey > 253)) / total
    return float(np.clip(1.0 - (low + high), 0.0, 1.0))


# --------------------------------------------------------------------------- straightening
def _segments(rgb: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
    grey = _grey(rgb)
    h, w = grey.shape
    scale = LSD_EDGE / max(h, w)
    if scale < 1:
        grey = cv2.resize(grey, (max(1, round(w * scale)), max(1, round(h * scale))), interpolation=cv2.INTER_AREA)
    detector = cv2.createLineSegmentDetector()
    lines = detector.detect(np.ascontiguousarray(grey))[0]
    segs = np.zeros((0, 4), np.float32) if lines is None else lines.reshape(-1, 4)
    return segs, (int(grey.shape[0]), int(grey.shape[1]))


def _votes(lines: np.ndarray, shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    h, w = shape
    long_edge = max(h, w)
    angles: list[float] = []
    weights: list[float] = []
    for x0, y0, x1, y1 in lines:
        dx, dy = float(x1 - x0), float(y1 - y0)
        length = math.hypot(dx, dy)
        if length < 0.04 * long_edge:
            continue
        if dx < 0:
            dx, dy = -dx, -dy
        slope = math.degrees(math.atan2(dy, dx))
        if abs(slope) <= SEARCH_DEG:
            angles.append(slope)  # rotating counter-clockwise by the slope levels the segment
            weights.append(length * (2.0 if length > 0.25 * w else 1.0))
        elif abs(slope) >= 90 - SEARCH_DEG:
            if dy < 0:
                dx, dy = -dx, -dy
            angles.append(math.degrees(math.atan2(-dx, dy)))
            middle = abs((x0 + x1) / 2 - w / 2) / (w / 2)
            weights.append(length * 0.8 * (1 - 0.6 * min(1.0, middle)))
    return np.asarray(angles, np.float64), np.asarray(weights, np.float64)


def straighten_estimate(rgb: np.ndarray) -> tuple[float | None, float]:
    """(angle in degrees, confidence 0-1) that levels the picture, or (None, support) without agreeing lines.

    Near-horizontal and near-vertical LSD segments within +-20 degrees vote, weighted by length;
    votes are accumulated with a Gaussian kernel, the strongest peak wins and is refined by the
    weighted mean of nearby votes.  confidence = agreeing line length / total voting line length.
    """
    lines, shape = _segments(rgb)
    angles, weights = _votes(lines, shape)
    if not len(angles):
        return None, 0.0
    grid = np.arange(-SEARCH_DEG, SEARCH_DEG + 1e-9, 0.05)
    score = (weights[None, :] * np.exp(-((grid[:, None] - angles[None, :]) ** 2) / (2 * SPREAD_DEG**2))).sum(axis=1)
    peak = float(grid[int(np.argmax(score))])
    near = np.abs(angles - peak) <= 2 * SPREAD_DEG
    support = float(weights[near].sum())
    if support < SUPPORT_RATIO * max(shape):
        return None, round(support / max(shape), 3)
    angle = float(np.average(angles[near], weights=weights[near]))
    return round(angle, 2), round(min(1.0, support / float(weights.sum())), 3)


def _max_inner_rect(w: float, h: float, angle_rad: float) -> tuple[float, float]:
    """Width/height of the largest axis-aligned rectangle inside a w x h rectangle rotated by `angle_rad`."""
    sin_a, cos_a = abs(math.sin(angle_rad)), abs(math.cos(angle_rad))
    long_side, short_side = (w, h) if w >= h else (h, w)
    if short_side <= 2.0 * sin_a * cos_a * long_side or abs(sin_a - cos_a) < 1e-10:
        x = 0.5 * short_side
        wr, hr = (x / sin_a, x / cos_a) if w >= h else (x / cos_a, x / sin_a)
    else:
        cos_2a = cos_a * cos_a - sin_a * sin_a
        wr = (w * cos_a - h * sin_a) / cos_2a
        hr = (h * cos_a - w * sin_a) / cos_2a
    return wr, hr


def rotate_and_crop(rgb: np.ndarray, angle: float) -> np.ndarray:
    """Rotate counter-clockwise by `angle` degrees and crop to the largest axis-aligned inner rectangle."""
    if abs(angle) < 1e-6:
        return rgb
    h, w = rgb.shape[:2]
    matrix = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)
    rotated = cv2.warpAffine(
        np.ascontiguousarray(rgb), matrix, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT
    )
    wr, hr = _max_inner_rect(w, h, math.radians(angle))
    cw = max(1, min(w, int(math.floor(wr))))
    ch = max(1, min(h, int(math.floor(hr))))
    x0 = (w - cw) // 2
    y0 = (h - ch) // 2
    return np.ascontiguousarray(rotated[y0 : y0 + ch, x0 : x0 + cw])


# --------------------------------------------------------------------------- saliency / cropping
def saliency_center(rgb: np.ndarray) -> tuple[float, float]:
    """Normalised (cx, cy) of the most salient region (spectral residual); geometric centre without OpenCV contrib."""
    try:
        detector = cv2.saliency.StaticSaliencySpectralResidual_create()
    except AttributeError:
        return 0.5, 0.5
    h, w = rgb.shape[:2]
    scale = 512 / max(h, w)
    small = cv2.resize(rgb, (max(1, round(w * scale)), max(1, round(h * scale))), interpolation=cv2.INTER_AREA) \
        if scale < 1 else rgb
    try:
        ok, sal = detector.computeSaliency(cv2.cvtColor(np.ascontiguousarray(small), cv2.COLOR_RGB2BGR))
    except cv2.error:
        return 0.5, 0.5
    if not ok or sal is None:
        return 0.5, 0.5
    weights = np.asarray(sal, np.float64) ** 2
    total = float(weights.sum())
    if total <= 0:
        return 0.5, 0.5
    sh, sw = weights.shape
    ys, xs = np.mgrid[0:sh, 0:sw]
    cx = float((weights * (xs + 0.5)).sum() / total) / sw
    cy = float((weights * (ys + 0.5)).sum() / total) / sh
    return min(1.0, max(0.0, cx)), min(1.0, max(0.0, cy))


def parse_ratio(ratio: str) -> tuple[int, int]:
    a, b = ratio.split(":", 1)
    rw, rh = int(a), int(b)
    if rw <= 0 or rh <= 0:
        raise ValueError(f"无效的裁切比例: {ratio}")
    return rw, rh


def crop_ratio(rgb: np.ndarray, ratio: str, center: tuple[float, float] = (0.5, 0.5)) -> np.ndarray:
    """Largest `ratio` ("W:H") rectangle centred as close to `center` (normalised) as the frame allows."""
    rw, rh = parse_ratio(ratio)
    h, w = rgb.shape[:2]
    if w * rh >= h * rw:  # image is wider than the target ratio
        ch, cw = h, int(h * rw / rh)
    else:
        cw, ch = w, int(w * rh / rw)
    cw, ch = max(1, min(w, cw)), max(1, min(h, ch))
    x0 = int(round(center[0] * w - cw / 2))
    y0 = int(round(center[1] * h - ch / 2))
    x0 = min(max(0, x0), w - cw)
    y0 = min(max(0, y0), h - ch)
    return np.ascontiguousarray(rgb[y0 : y0 + ch, x0 : x0 + cw])

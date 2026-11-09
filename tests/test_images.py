import io
import subprocess
from types import SimpleNamespace

import numpy as np
from PIL import Image

from afterglow.util import images as im
from tests.conftest import make_sharp, make_tilted


def arr(path):
    with Image.open(path) as i:
        return im.to_rgb_array(i)


def test_sharpness_separates_sharp_from_blurred(images):
    sharp = im.laplacian_sharpness(arr(images.sharp))
    blurred = im.laplacian_sharpness(arr(images.blurred))
    assert 70 <= sharp <= 100
    assert blurred < 15
    assert 0 <= blurred < sharp <= 100


def test_exposure_score(images):
    dark, bright, normal = (im.exposure_score(arr(p)) for p in (images.dark, images.bright, images.normal))
    assert dark < 0.1 and bright < 0.1
    assert normal > 0.95


def test_straighten_estimate_finds_three_degrees(images):
    angle, confidence = im.straighten_estimate(arr(images.tilted))
    assert angle is not None and abs(angle - 3.0) < 0.5
    assert confidence > 0.6


def test_straighten_estimate_on_level_and_featureless():
    angle, conf = im.straighten_estimate(to_array(make_tilted(0.0)))
    assert angle is not None and abs(angle) < 0.3
    assert im.straighten_estimate(np.full((300, 400, 3), 128, np.uint8)) == (None, 0.0)


def to_array(img):
    return im.to_rgb_array(img)


def test_rotate_and_crop_levels_the_horizon(images):
    rgb = arr(images.tilted)
    fixed = im.rotate_and_crop(rgb, 3.0)
    assert fixed.shape[0] < rgb.shape[0] and fixed.shape[1] < rgb.shape[1]
    # no black borders survive the inner-rectangle crop
    assert fixed[:3].min() > 0 and fixed[-3:].min() > 0 and fixed[:, :3].min() > 0 and fixed[:, -3:].min() > 0
    angle, conf = im.straighten_estimate(fixed)
    assert angle is not None and abs(angle) < 0.5
    assert im.rotate_and_crop(rgb, 0.0) is rgb


def test_saliency_center_in_unit_square(images):
    cx, cy = im.saliency_center(arr(images.sharp))
    assert 0 <= cx <= 1 and 0 <= cy <= 1


def test_crop_ratio_dimensions_and_centre():
    rgb = np.zeros((800, 1200, 3), np.uint8)
    rgb[:, 900:] = 255
    out = im.crop_ratio(rgb, "4:5", (0.9, 0.5))
    assert out.shape[:2] == (800, 640)  # full height, 4:5 wide
    assert out[:, -1].min() == 255  # window clamped against the right edge
    wide = im.crop_ratio(rgb, "16:9")
    assert wide.shape[1] == 1200 and abs(wide.shape[0] - 675) <= 1
    same = im.crop_ratio(rgb, "3:2")
    assert same.shape[:2] == (800, 1200)


def test_load_preview_jpeg_resizes_and_transposes(tmp_path):
    ex = Image.Exif()
    ex[0x0112] = 6  # rotate 90 deg clockwise to view
    src = tmp_path / "p.jpg"
    Image.new("RGB", (400, 200), (255, 0, 0)).save(src, exif=ex)
    img = im.load_preview(src, "jpeg", long_edge=100)
    assert img.size == (50, 100)
    assert img.info["orig_size"] == (200, 400)


def test_save_jpeg_atomic(tmp_path):
    out = im.save_jpeg(make_sharp(1, (100, 80)), tmp_path / "a" / "b.jpg", quality=80)
    assert out.exists() and Image.open(out).size == (100, 80)
    assert not list((tmp_path / "a").glob(".*tmp"))


def test_load_preview_raw_uses_embedded_thumbnail(tmp_path, monkeypatch):
    import rawpy

    buf = io.BytesIO()
    Image.new("RGB", (2000, 1500), (10, 200, 30)).save(buf, format="JPEG")

    class FakeRaw:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def extract_thumb(self):
            return SimpleNamespace(format=rawpy.ThumbFormat.JPEG, data=buf.getvalue())

        def postprocess(self, **kw):  # pragma: no cover - must not be needed
            raise AssertionError("thumbnail should be used")

    monkeypatch.setattr(rawpy, "imread", lambda p: FakeRaw())
    img = im.load_preview(tmp_path / "x.cr3", "raw", long_edge=1000)
    assert img.size == (1000, 750) and img.info["orig_size"] == (2000, 1500)


def test_load_preview_raw_falls_back_to_postprocess(tmp_path, monkeypatch):
    import rawpy

    class FakeRaw:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def extract_thumb(self):
            raise rawpy.LibRawNoThumbnailError("none")

        def postprocess(self, **kw):
            assert kw["half_size"] is True
            return np.full((300, 400, 3), 99, np.uint8)

    monkeypatch.setattr(rawpy, "imread", lambda p: FakeRaw())
    img = im.load_preview(tmp_path / "x.nef", "raw", long_edge=2560)
    assert img.size == (400, 300)


def test_load_preview_video_frame_and_failure(tmp_path, monkeypatch):
    def ok(cmd, **kw):
        assert "shell" not in kw
        Image.new("RGB", (640, 360), (5, 5, 5)).save(cmd[-1])
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(im.subprocess, "run", ok)
    img = im.load_preview(tmp_path / "v.mp4", "video")
    assert img is not None and img.size == (640, 360)

    monkeypatch.setattr(im.subprocess, "run", lambda cmd, **kw: subprocess.CompletedProcess(cmd, 1, "", "bad"))
    assert im.load_preview(tmp_path / "v.mp4", "video") is None

    def missing(cmd, **kw):
        raise FileNotFoundError("ffmpeg")

    monkeypatch.setattr(im.subprocess, "run", missing)
    assert im.load_preview(tmp_path / "v.mp4", "video") is None


def test_burst_fixture_exif_is_written(images):
    with Image.open(images.bursts[1]) as i:
        ifd = i.getexif().get_ifd(0x8769)
    assert ifd[0x9003] == "2024:05:12 13:05:00" and ifd[0x9291] == "500"

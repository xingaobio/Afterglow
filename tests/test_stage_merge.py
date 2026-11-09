import numpy as np
from PIL import Image

from afterglow.stages.merge import STAGE, Merge, merge_focus, merge_hdr
from tests.conftest import add_file, fake_ctx, make_sharp, tree_hashes


def exposure_variants(base: Image.Image, factors):
    arr = np.asarray(base, np.float32)
    return [Image.fromarray(np.clip(arr * f, 0, 255).astype(np.uint8)) for f in factors]


class FakeLumen:
    def __init__(self):
        self.exports = []

    def open(self, path):
        return {}

    def auto_tone(self, path):
        return {}

    def export(self, path, output, quality, long_edge, overwrite=True):
        self.exports.append((path, output))
        make_sharp(1, (400, 300)).save(output)
        return {}


class FakeLib:
    available = True

    def merge_images(self, images, kind):
        self.seen = (len(images), kind)
        return images[0]


def hdr_library(tmp_path, with_export=True, **kw):
    ctx = fake_ctx(tmp_path, **kw)
    recs = []
    for i, img in enumerate(exposure_variants(make_sharp(2, (400, 300)), (0.4, 1.0, 1.8))):
        p = ctx.config.paths.source / f"h{i}.jpg"
        img.save(p)
        rec = add_file(ctx, f"h{i}.jpg", preview_path=str(p), captured_at=f"2024-05-12T10:00:0{i}.000Z")
        recs.append(rec)
        if with_export:
            out = ctx.config.paths.output / "exports" / f"h{i}.jpg"
            out.parent.mkdir(parents=True, exist_ok=True)
            img.save(out)
            ctx.state.add_output("export", str(out), file_id=rec.id)
    other = add_file(ctx, "solo.jpg")
    ctx.state.create_group("event", [r.id for r in recs], name="傍晚/海边")
    gid = ctx.state.create_group("hdr", [r.id for r in recs])
    return ctx, recs, other, gid


def test_registry():
    assert STAGE is Merge and not Merge.per_file and Merge.uses_gpu


def test_opencv_hdr_fallback_writes_merged_output(tmp_path):
    ctx, recs, other, gid = hdr_library(tmp_path)
    before = tree_hashes(ctx.config.paths.source)
    assert Merge().run(ctx, []) == []
    out = ctx.config.paths.output / "merged" / "傍晚海边" / "hdr_h0.jpg"
    assert out.exists() and Image.open(out).size == (400, 300)
    outs = ctx.state.outputs(group_id=gid)
    assert [o["kind"] for o in outs] == ["hdr"] and outs[0]["params"]["engine"] == "opencv"
    assert all(ctx.state.step(r.id, "merge") == "done" for r in recs)
    assert ctx.state.step(other.id, "merge") == "skipped"
    assert tree_hashes(ctx.config.paths.source) == before

    Merge().run(ctx, [])  # already merged: nothing new
    assert len(ctx.state.outputs(group_id=gid)) == 1


def test_lumen_lib_used_when_available(tmp_path):
    lib = FakeLib()
    ctx, recs, other, gid = hdr_library(tmp_path, lumen_lib=lib)
    Merge().run(ctx, [])
    assert lib.seen == (3, "hdr")
    assert ctx.state.outputs(group_id=gid)[0]["params"]["engine"] == "lumen"


def test_missing_exports_are_exported_on_the_fly_to_cache(tmp_path):
    lumen = FakeLumen()
    ctx, recs, other, gid = hdr_library(tmp_path, with_export=False)
    ctx.lumen = lumen
    Merge().run(ctx, [])
    assert len(lumen.exports) == 3
    assert all(str(ctx.config.paths.cache / "merge") in out for _, out in lumen.exports)
    assert ctx.state.outputs(group_id=gid)


def test_falls_back_to_preview_when_lumen_fails(tmp_path):
    class Broken(FakeLumen):
        def open(self, path):
            raise RuntimeError("no lumen")

    ctx, recs, other, gid = hdr_library(tmp_path, with_export=False)
    ctx.lumen = Broken()
    Merge().run(ctx, [])
    assert ctx.state.outputs(group_id=gid)


def test_failed_group_marks_members_failed(tmp_path):
    ctx = fake_ctx(tmp_path)
    recs = [add_file(ctx, f"p{i}.jpg") for i in range(3)]  # no previews, no lumen -> cannot merge
    ctx.lumen = type("L", (), {"open": lambda s, p: (_ for _ in ()).throw(RuntimeError("down"))})()
    ctx.state.create_group("pano", [r.id for r in recs])
    Merge().run(ctx, [])
    assert all(ctx.state.step(r.id, "merge") == "failed" for r in recs)


def test_focus_stack_prefers_sharp_regions():
    sharp = np.asarray(make_sharp(5, (300, 200)), np.uint8)
    import cv2

    blurred = cv2.GaussianBlur(sharp, (0, 0), 6)
    left_sharp = blurred.copy()
    left_sharp[:, :150] = sharp[:, :150]
    right_sharp = blurred.copy()
    right_sharp[:, 150:] = sharp[:, 150:]
    fused = merge_focus([left_sharp, right_sharp])
    assert fused.shape == sharp.shape

    def lap(a):
        return float(cv2.Laplacian(cv2.cvtColor(a, cv2.COLOR_RGB2GRAY), cv2.CV_64F).var())

    assert lap(fused) > lap(left_sharp) * 1.2 and lap(fused) > lap(right_sharp) * 1.2


def test_mertens_returns_uint8_same_size():
    imgs = [np.asarray(i, np.uint8) for i in exposure_variants(make_sharp(2, (200, 150)), (0.5, 1.0, 1.5))]
    out = merge_hdr(imgs)
    assert out.dtype == np.uint8 and out.shape == imgs[0].shape

from types import SimpleNamespace

import pytest

from afterglow.stages.cull import STAGE, Cull, quality_score
from tests.conftest import add_file, fake_ctx


class FakeLumen:
    def __init__(self, photos=None, bursts=None, fail=False):
        self.photos, self.bursts, self.fail, self.calls = photos or [], bursts or [], fail, []

    def analyze(self, paths):
        self.calls.append(list(paths))
        if self.fail:
            raise RuntimeError("lumen down")
        wanted = set(paths)
        return {"count": len(paths), "photos": [p for p in self.photos if p["path"] in wanted],
                "bursts": [b for b in self.bursts if all(x in wanted for x in b["photos"])]}


def make_library(tmp_path, images, lumen):
    ctx = fake_ctx(tmp_path, lumen=lumen)
    recs = SimpleNamespace(
        sharp=add_file(ctx, "sharp.jpg", preview_path=str(images.sharp), width=1200, height=800),
        blurred=add_file(ctx, "blurred.jpg", preview_path=str(images.blurred), width=1200, height=800),
        tiny=add_file(ctx, "tiny.jpg", preview_path=str(images.normal), width=500, height=400),
        video=add_file(ctx, "clip.mp4", width=1920, height=1080),
    )
    return ctx, recs


def run_cull(ctx, batch=None):
    stage = Cull()
    batch = batch if batch is not None else ctx.state.pending("cull", 100)
    outcomes = stage.run(ctx, batch)
    ctx.state.set_steps(outcomes, "cull")
    return outcomes


def test_registry_and_formula():
    assert STAGE is Cull and Cull.name == "cull" and Cull.per_file
    assert quality_score(100, 1, 0, 0) == 100
    assert quality_score(50, 0.5, 2, 1) == pytest.approx(0.6 * 50 + 12.5)  # eyes closed -> no face bonus
    assert quality_score(0, 0, 0, 0) == 15


def test_fallback_when_lumen_unavailable(tmp_path, images):
    ctx, r = make_library(tmp_path, images, FakeLumen(fail=True))
    paired = add_file(ctx, "sharp_pair.jpg", preview_path=str(images.sharp), width=1200, height=800, pair_id=r.sharp.id)
    dup = add_file(ctx, "dup.jpg", preview_path=str(images.sharp), width=1200, height=800)
    ctx.state.set_analysis(dup.id, reject_reason="duplicate", pick=0)

    outcomes = run_cull(ctx)
    by = {o.file_id: o for o in outcomes}
    assert len(outcomes) == 6
    assert by[r.video.id].status == "skipped" and by[paired.id].status == "skipped" and by[dup.id].status == "skipped"

    a = ctx.state.analysis(r.sharp.id)
    assert a["pick"] == 1 and a["reject_reason"] is None and a["sharpness"] >= 70
    assert 0.9 <= a["exposure_score"] <= 1.0 and a["quality"] > 60
    b = ctx.state.analysis(r.blurred.id)
    assert b["pick"] == 0 and b["reject_reason"] == "blurry" and b["blurry"] == 1
    t = ctx.state.analysis(r.tiny.id)
    assert t["pick"] == 0 and t["reject_reason"] == "tiny"
    assert ctx.state.analysis(dup.id)["reject_reason"] == "duplicate"  # untouched
    assert ctx.state.analysis(r.video.id) is None


def test_lumen_values_eyes_closed_and_missing_photo_falls_back(tmp_path, images):
    lumen = FakeLumen(photos=[
        {"path": "", "sharpness": 90.0, "faces": 2, "eyes_closed": 1, "blurry": False},
    ])
    ctx, r = make_library(tmp_path, images, lumen)
    lumen.photos[0]["path"] = r.sharp.path
    run_cull(ctx)
    a = ctx.state.analysis(r.sharp.id)
    assert a["sharpness"] == 90.0 and a["faces"] == 2 and a["eyes_closed"] == 1
    assert a["pick"] == 0 and a["reject_reason"] == "eyes_closed"
    assert ctx.state.analysis(r.blurred.id)["reject_reason"] == "blurry"  # local fallback for unlisted photo
    assert len(lumen.calls) == 1 and set(lumen.calls[0]) == {r.sharp.path, r.blurred.path, r.tiny.path}


def test_eyes_closed_allowed_when_configured(tmp_path, images):
    from afterglow.config import Cull as CullCfg

    lumen = FakeLumen(photos=[{"path": "", "sharpness": 90.0, "faces": 1, "eyes_closed": 1}])
    ctx, r = make_library(tmp_path, images, lumen)
    ctx.config = type(ctx.config)(**{**ctx.config.__dict__, "cull": CullCfg(reject_eyes_closed=False)})
    lumen.photos[0]["path"] = r.sharp.path
    run_cull(ctx)
    assert ctx.state.analysis(r.sharp.id)["pick"] == 1


def burst_library(tmp_path, images, lumen_factory):
    ctx = fake_ctx(tmp_path)
    recs = []
    for i, p in enumerate(images.bursts):
        recs.append(add_file(
            ctx, p.name, preview_path=str(p), width=1200, height=800, camera="TestCam",
            captured_at=f"2024-05-12T13:05:{i // 2:02d}.{500 if i % 2 else 0:03d}Z",
        ))
    ctx.lumen = lumen_factory(recs)
    return ctx, recs


def test_lumen_bursts_choose_lumen_best(tmp_path, images):
    def factory(recs):
        paths = [r.path for r in recs]
        photos = [{"path": p, "sharpness": 80.0, "faces": 0, "eyes_closed": 0, "blurry": False} for p in paths]
        return FakeLumen(photos=photos, bursts=[{"group": 1, "photos": paths, "best": paths[2]}])

    ctx, recs = burst_library(tmp_path, images, factory)
    run_cull(ctx)
    groups = ctx.state.groups("burst")
    assert len(groups) == 1 and groups[0]["members"] == [r.id for r in recs]
    best = [r for r in recs if ctx.state.analysis(r.id)["burst_best"]]
    assert [r.id for r in best] == [recs[2].id]
    for r in recs:
        a = ctx.state.analysis(r.id)
        assert a["burst_id"] == groups[0]["id"]
        if r is not recs[2]:
            assert a["reject_reason"] == "burst_alt" and a["pick"] == 0
    assert ctx.state.analysis(recs[2].id)["pick"] == 1


def test_gap_bursts_when_lumen_has_none(tmp_path, images):
    ctx, recs = burst_library(tmp_path, images, lambda recs: FakeLumen(fail=True))
    lone = add_file(ctx, "lone.jpg", preview_path=str(images.sharp), width=1200, height=800, camera="TestCam",
                    captured_at="2024-05-12T20:00:00.000Z")
    run_cull(ctx)
    groups = ctx.state.groups("burst")
    assert len(groups) == 1 and len(groups[0]["members"]) == 5
    kept = [r for r in recs if ctx.state.analysis(r.id)["burst_best"]]
    assert len(kept) == 1
    assert sum(1 for r in recs if ctx.state.analysis(r.id)["reject_reason"] == "burst_alt") == 4
    assert ctx.state.analysis(lone.id)["burst_id"] is None and ctx.state.analysis(lone.id)["pick"] == 1


def test_gap_bursts_respect_camera_and_gap(tmp_path, images):
    ctx, recs = burst_library(tmp_path, images, lambda recs: FakeLumen(fail=True))
    ctx.state.files[recs[2].id].camera = "OtherCam"
    run_cull(ctx)
    groups = ctx.state.groups("burst")
    assert [len(g["members"]) for g in groups] == [4]  # the other body stands alone
    assert ctx.state.analysis(recs[2].id)["burst_id"] is None


def test_keep_per_burst(tmp_path, images):
    from afterglow.config import Cull as CullCfg

    ctx, recs = burst_library(tmp_path, images, lambda recs: FakeLumen(fail=True))
    ctx.config = type(ctx.config)(**{**ctx.config.__dict__, "cull": CullCfg(keep_per_burst=2)})
    run_cull(ctx)
    assert sum(ctx.state.analysis(r.id)["burst_best"] for r in recs) == 2

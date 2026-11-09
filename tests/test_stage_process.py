import dataclasses
import shutil
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from afterglow.config import Lumen, Process
from afterglow.stages.process import STAGE
from afterglow.stages.process import Process as ProcessStage
from afterglow.util import images as im
from tests.conftest import add_file, fake_ctx, make_tilted, tree_hashes


class FakeLumen:
    def __init__(self, export_image: Path, fail_on: str | None = None):
        self.export_image = export_image
        self.fail_on = fail_on
        self.calls: list[tuple] = []

    def open(self, path):
        self.calls.append(("open", path))
        return {}

    def auto_tone(self, path):
        self.calls.append(("auto_tone", path))
        if self.fail_on and self.fail_on in path:
            raise RuntimeError("edit.auto failed")
        return {}

    def apply_preset(self, path, name, amount=100):
        self.calls.append(("preset", path, name))
        return {}

    def export(self, path, output, quality, long_edge, overwrite=True):
        self.calls.append(("export", path, output, quality, long_edge))
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(self.export_image, output)
        return {}


class FakeLib:
    available = True

    def __init__(self, angle=None, confidence=0.0):
        self.angle, self.confidence = angle, confidence
        self.denoised = self.upscaled = 0

    def straighten(self, rgb):
        return self.angle, self.confidence

    def denoise(self, rgb, amount=70):
        self.denoised += 1
        return rgb

    def super_resolution(self, rgb, scale=2):
        self.upscaled += 1
        return np.repeat(np.repeat(rgb, scale, axis=0), scale, axis=1)


def setup(tmp_path, images, export_image=None, **overrides):
    lumen = FakeLumen(export_image or images.tilted)
    ctx = fake_ctx(tmp_path, **overrides)  # a `lumen=Lumen(...)` override is the config section
    ctx.lumen = lumen
    src = ctx.config.paths.source
    (src / "trip").mkdir()
    shutil.copyfile(images.sharp, src / "trip" / "IMG_1.dng")  # stand-in RAW; LUMEN is faked
    rec = add_file(ctx, "trip/IMG_1.dng", width=6000, height=4000, iso=200)
    ctx.state.set_analysis(rec.id, pick=1)
    return ctx, lumen, rec


def run(ctx, batch=None):
    stage = ProcessStage()
    batch = batch if batch is not None else ctx.state.pending("process", 100)
    outcomes = stage.run(ctx, batch)
    ctx.state.set_steps(outcomes, "process")
    return outcomes


def kinds(ctx, rec):
    return [o["kind"] for o in ctx.state.outputs(file_id=rec.id)]


def test_registry():
    assert STAGE is ProcessStage and ProcessStage.uses_gpu and ProcessStage.name == "process"


def test_export_straighten_and_crops(tmp_path, images):
    ctx, lumen, rec = setup(tmp_path, images)
    before = tree_hashes(ctx.config.paths.source)
    outcomes = run(ctx)
    assert [o.status for o in outcomes] == ["done"]

    export = ctx.config.paths.output / "exports" / "trip" / "IMG_1.jpg"
    assert export.exists()
    assert [c[0] for c in lumen.calls] == ["open", "auto_tone", "export"]
    assert lumen.calls[-1][3] == 92 and lumen.calls[-1][4] is None

    outs = ctx.state.outputs(file_id=rec.id)
    assert [o["kind"] for o in outs] == ["export", "straighten", "crop", "crop", "crop"]
    straighten = outs[1]
    assert straighten["path"] == str(export)
    assert abs(straighten["params"]["angle"] - 3.0) < 0.5 and straighten["params"]["confidence"] >= 0.6

    with Image.open(export) as fixed:
        fixed_rgb = im.to_rgb_array(fixed)
    assert fixed_rgb.shape[:2] < (800, 1200)  # rotated + cropped, overwritten in place
    angle, _ = im.straighten_estimate(fixed_rgb)
    assert angle is not None and abs(angle) < 0.5

    crops = {Path(o["path"]).name: o for o in outs if o["kind"] == "crop"}
    assert set(crops) == {"IMG_1_3x2.jpg", "IMG_1_4x5.jpg", "IMG_1_16x9.jpg"}
    for name, ratio in (("IMG_1_3x2.jpg", 3 / 2), ("IMG_1_4x5.jpg", 4 / 5), ("IMG_1_16x9.jpg", 16 / 9)):
        with Image.open(ctx.config.paths.output / "crops" / "trip" / name) as c:
            assert abs(c.size[0] / c.size[1] - ratio) < 0.02
    assert tree_hashes(ctx.config.paths.source) == before


def test_level_image_is_not_rotated(tmp_path, images, tmp_path_factory):
    level = tmp_path / "level.jpg"
    make_tilted(0.0).save(level)
    ctx, lumen, rec = setup(tmp_path, images, export_image=level)
    run(ctx)
    assert "straighten" not in kinds(ctx, rec)
    with Image.open(ctx.config.paths.output / "exports" / "trip" / "IMG_1.jpg") as img:
        assert img.size == (1200, 800)


def test_straighten_respects_limits_and_uses_lumen_lib(tmp_path, images):
    ctx, lumen, rec = setup(tmp_path, images, lumen_lib=FakeLib(angle=9.0, confidence=0.9))
    run(ctx)
    assert "straighten" not in kinds(ctx, rec)  # beyond straighten_max_degrees

    ctx, lumen, rec = setup(tmp_path / "b", images, lumen_lib=FakeLib(angle=2.0, confidence=0.3))
    run(ctx)
    assert "straighten" not in kinds(ctx, rec)  # below straighten_min_confidence

    ctx, lumen, rec = setup(tmp_path / "c", images, lumen_lib=FakeLib(angle=2.0, confidence=0.9))
    run(ctx)
    out = [o for o in ctx.state.outputs(file_id=rec.id) if o["kind"] == "straighten"]
    assert out and out[0]["params"] == {"angle": 2.0, "confidence": 0.9}


def test_straighten_and_crops_can_be_disabled(tmp_path, images):
    ctx, lumen, rec = setup(tmp_path, images, process=Process(straighten=False, crops=()))
    run(ctx)
    assert kinds(ctx, rec) == ["export"]


def test_style_preset_from_description(tmp_path, images):
    ctx, lumen, rec = setup(tmp_path, images, process=Process(
        style_presets={"landscape": "Vivid Land", "portrait": "", "night": "", "default": ""}))
    ctx.state.set_description(rec.id, caption="x", scene="landscape", style_preset="landscape")
    run(ctx)
    assert ("preset", rec.path, "Vivid Land") in lumen.calls
    assert [c[0] for c in lumen.calls] == ["open", "auto_tone", "preset", "export"]

    ctx, lumen, rec = setup(tmp_path / "n", images)
    ctx.state.set_description(rec.id, caption="x", scene="night", style_preset="night")
    run(ctx)
    assert not [c for c in lumen.calls if c[0] == "preset"]  # empty preset name -> auto only


def test_enhance_auto_for_high_iso_and_small_images(tmp_path, images):
    lib = FakeLib()
    ctx, lumen, rec = setup(tmp_path, images, lumen_lib=lib, process=Process(straighten=False, crops=()))
    ctx.state.files[rec.id].iso = 6400
    run(ctx)
    out = [o for o in ctx.state.outputs(file_id=rec.id) if o["kind"] == "enhance"]
    assert len(out) == 1 and out[0]["params"]["steps"] == ["denoise"]
    assert lib.denoised == 1 and lib.upscaled == 0
    assert Path(out[0]["path"]).name == "IMG_1_dn.jpg"

    lib = FakeLib()
    ctx, lumen, rec = setup(tmp_path / "s", images, lumen_lib=lib, process=Process(straighten=False, crops=()))
    ctx.state.files[rec.id].width, ctx.state.files[rec.id].height = 2000, 1500
    run(ctx)
    out = [o for o in ctx.state.outputs(file_id=rec.id) if o["kind"] == "enhance"]
    assert Path(out[0]["path"]).name == "IMG_1_x2.jpg" and lib.upscaled == 1
    with Image.open(out[0]["path"]) as big:
        assert big.size == (2400, 1600)

    ctx, lumen, rec = setup(tmp_path / "n", images, lumen_lib=FakeLib(), process=Process(straighten=False, crops=()))
    run(ctx)  # ISO 200, 6000 px: nothing to enhance
    assert "enhance" not in kinds(ctx, rec)


def test_enhance_without_lumen_lib_is_skipped_not_failed(tmp_path, images):
    ctx, lumen, rec = setup(tmp_path, images, process=Process(straighten=False, crops=()),
                            lumen=dataclasses.replace(Lumen(), enhance="all"))
    outcomes = run(ctx)
    assert outcomes[0].status == "done" and outcomes[0].detail["enhance"] == "skipped"
    assert "enhance" not in kinds(ctx, rec)


def test_enhance_off(tmp_path, images):
    lib = FakeLib()
    ctx, lumen, rec = setup(tmp_path, images, lumen_lib=lib, lumen=Lumen(enhance="off"))
    ctx.state.files[rec.id].iso = 12800
    run(ctx)
    assert lib.denoised == 0


def test_scope_and_pairing_filters(tmp_path, images):
    ctx, lumen, raw = setup(tmp_path, images)
    src = ctx.config.paths.source
    for name in ("pair.jpg", "solo.jpg", "reject.dng"):
        shutil.copyfile(images.sharp, src / name)
    paired = add_file(ctx, "pair.jpg", pair_id=raw.id)
    solo = add_file(ctx, "solo.jpg")
    reject = add_file(ctx, "reject.dng")
    ctx.state.set_analysis(solo.id, pick=1)
    ctx.state.set_analysis(reject.id, pick=0)
    video = add_file(ctx, "clip.mp4")
    outcomes = {o.file_id: o for o in run(ctx)}
    assert outcomes[raw.id].status == "done"
    assert outcomes[paired.id].status == "skipped"
    assert outcomes[solo.id].status == "skipped"  # JPEG only when scope = all
    assert outcomes[reject.id].status == "skipped"  # not a pick
    assert outcomes[video.id].status == "skipped"

    ctx2, lumen2, raw2 = setup(tmp_path / "all", images, process=Process(scope="all", straighten=False, crops=()))
    shutil.copyfile(images.sharp, ctx2.config.paths.source / "solo.jpg")
    solo2 = add_file(ctx2, "solo.jpg")
    reject2 = add_file(ctx2, "reject.dng")
    ctx2.state.set_analysis(reject2.id, pick=0)
    shutil.copyfile(images.sharp, ctx2.config.paths.source / "reject.dng")
    outcomes = {o.file_id: o for o in run(ctx2)}
    assert outcomes[solo2.id].status == "done" and outcomes[reject2.id].status == "done"


def test_lumen_failure_only_fails_that_file(tmp_path, images):
    ctx, lumen, good = setup(tmp_path, images)
    lumen.fail_on = "bad"
    shutil.copyfile(images.sharp, ctx.config.paths.source / "bad.dng")
    bad = add_file(ctx, "bad.dng")
    ctx.state.set_analysis(bad.id, pick=1)
    by = {o.file_id: o for o in run(ctx)}
    assert by[bad.id].status == "failed" and "edit.auto failed" in by[bad.id].error
    assert by[good.id].status == "done"
    assert kinds(ctx, bad) == []


def test_export_not_written_is_failure(tmp_path, images):
    ctx, lumen, rec = setup(tmp_path, images)
    lumen.export = lambda *a, **k: {}
    assert run(ctx)[0].status == "failed"


@pytest.mark.parametrize("fmt, name", [("jpg", "IMG_1.jpg"), ("jpeg", "IMG_1.jpg")])
def test_export_extension(tmp_path, images, fmt, name):
    ctx, lumen, rec = setup(tmp_path, images, lumen=Lumen(export_format=fmt, enhance="off"))
    run(ctx)
    assert (ctx.config.paths.output / "exports" / "trip" / name).exists()


class GradeLumen(FakeLumen):
    current = {"exposure": 0.5, "contrast": 10, "tint": 3}

    def get_edits(self, path):
        if getattr(self, "state_fail", False):
            raise RuntimeError("no state")
        return {"path": path, "state": {"adjustments": dict(self.current)}}

    def render_preview(self, path, size=1024):
        self.calls.append(("render", path, size))
        return b"\xff\xd8rendered"

    def apply_edits(self, path, changes, amount=100, label=""):
        self.calls.append(("edits", path, changes, amount, label))
        if getattr(self, "edits_fail", False):
            raise RuntimeError("apply_edits failed")
        return {}


class FakeLLM:
    def __init__(self, result=None, error=None):
        self.result, self.error, self.calls = result, error, []

    def grade(self, image_jpeg, scene, caption, instruction, language="zh"):
        self.calls.append((image_jpeg, scene, caption, instruction, language))
        if self.error:
            raise self.error
        return self.result


def grade_setup(tmp_path, images, llm, scene="landscape", **overrides):
    ctx, _, rec = setup(tmp_path, images, process=Process(straighten=False, crops=()), **overrides)
    ctx.lumen = GradeLumen(images.tilted)
    ctx.llm = llm
    ctx.state.files[rec.id].preview_path = str(images.sharp)
    ctx.state.set_description(rec.id, caption="海边日落", scene=scene, style_preset="landscape")
    return ctx, ctx.lumen, rec


def export_params(ctx, rec):
    return [o for o in ctx.state.outputs(file_id=rec.id) if o["kind"] == "export"][0]["params"]


def test_grade_mode_auto_never_calls_llm(tmp_path, images):
    from afterglow.config import Grade

    llm = FakeLLM({"adjustments": {"contrast": 10}, "reason": "r"})
    ctx, lumen, rec = grade_setup(tmp_path, images, llm, grade=Grade(mode="auto"))
    assert run(ctx)[0].status == "done"
    assert llm.calls == []
    assert [c[0] for c in lumen.calls] == ["open", "auto_tone", "export"]
    assert export_params(ctx, rec)["grade"] is None


def test_grade_applied_between_auto_tone_and_export(tmp_path, images):
    from afterglow.config import Grade

    llm = FakeLLM({"adjustments": {"exposure": 0.3, "contrast": 12, "tint": 0, "bogus": 5}, "reason": "提亮"})
    ctx, lumen, rec = grade_setup(tmp_path, images, llm, grade=Grade(mode="llm", strength=70))
    assert run(ctx)[0].status == "done"
    assert [c[0] for c in lumen.calls] == ["open", "auto_tone", "render", "edits", "export"]
    edits = lumen.calls[3]
    # final values = LUMEN's current values + 70 % of the proposed deltas (edit.apply takes final values)
    assert edits[2] == {"adjustments": {"exposure": 0.71, "contrast": 18}}
    assert edits[3] == 100 and edits[4] == "afterglow grade"
    assert len(llm.calls) == 1
    image, scene, caption, instruction, language = llm.calls[0]
    assert image == b"\xff\xd8rendered" and scene == "landscape" and caption == "海边日落"
    assert instruction == ctx.config.grade.instruction and language == ctx.config.llm.language
    assert export_params(ctx, rec)["grade"] == {
        "adjustments": {"exposure": 0.3, "contrast": 12}, "applied": {"exposure": 0.71, "contrast": 18},
        "reason": "提亮", "strength": 70,
    }


def test_grade_needs_current_state_and_clamps_to_slider_range(tmp_path, images):
    llm = FakeLLM({"adjustments": {"contrast": 40}, "reason": "r"})
    ctx, lumen, rec = grade_setup(tmp_path, images, llm)
    lumen.state_fail = True
    assert run(ctx)[0].status == "done"
    assert "edits" not in [c[0] for c in lumen.calls] and export_params(ctx, rec)["grade"] is None

    from afterglow.config import Grade

    ctx, lumen, rec = grade_setup(tmp_path / "c", images, llm, grade=Grade(mode="llm", strength=100))
    lumen.current = {"contrast": 90}
    assert run(ctx)[0].status == "done"
    assert [c for c in lumen.calls if c[0] == "edits"][0][2] == {"adjustments": {"contrast": 100}}


def test_grade_failure_still_exports(tmp_path, images):
    from afterglow.integrations.llm import LLMError

    llm = FakeLLM(error=LLMError("down"))
    ctx, lumen, rec = grade_setup(tmp_path, images, llm)
    outcomes = run(ctx)
    assert outcomes[0].status == "done"
    assert [c[0] for c in lumen.calls] == ["open", "auto_tone", "render", "export"]
    assert (ctx.config.paths.output / "exports" / "trip" / "IMG_1.jpg").exists()
    assert export_params(ctx, rec)["grade"] is None


def test_apply_edits_failure_still_exports(tmp_path, images):
    llm = FakeLLM({"adjustments": {"contrast": 10}, "reason": "r"})
    ctx, lumen, rec = grade_setup(tmp_path, images, llm)
    lumen.edits_fail = True
    assert run(ctx)[0].status == "done"
    assert [c[0] for c in lumen.calls] == ["open", "auto_tone", "render", "edits", "export"]
    assert export_params(ctx, rec)["grade"] is None


def test_grade_skipped_for_other_scene_or_empty_result(tmp_path, images):
    llm = FakeLLM({"adjustments": {"contrast": 10}, "reason": "r"})
    ctx, lumen, rec = grade_setup(tmp_path, images, llm, scene="document")
    assert run(ctx)[0].status == "done"
    assert llm.calls == [] and "edits" not in [c[0] for c in lumen.calls]
    assert export_params(ctx, rec)["grade"] is None

    llm = FakeLLM({"adjustments": {}, "reason": "fine"})
    ctx, lumen, rec = grade_setup(tmp_path / "e", images, llm)
    run(ctx)
    assert len(llm.calls) == 1 and "edits" not in [c[0] for c in lumen.calls]


def test_grade_skipped_without_description_or_grade_method(tmp_path, images):
    llm = FakeLLM({"adjustments": {"contrast": 10}, "reason": "r"})
    ctx, lumen, rec = grade_setup(tmp_path, images, llm)
    ctx.state._desc.pop(rec.id)
    run(ctx)
    assert llm.calls == []

    ctx, lumen, rec = grade_setup(tmp_path / "n", images, object())  # llm without grade()
    assert run(ctx)[0].status == "done"
    assert export_params(ctx, rec)["grade"] is None

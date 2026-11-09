import dataclasses
import io
import threading
import time

from PIL import Image

from afterglow.config import LLM
from afterglow.stages.understand import STAGE, Understand, sanitize_name
from tests.conftest import add_file, fake_ctx


class FakeLLM:
    def __init__(self, answers=None, delay=0.0):
        self.answers = answers or {}
        self.describe_calls = []
        self.name_calls = []
        self.delay = delay
        self.active = 0
        self.max_active = 0
        self.lock = threading.Lock()
        self.name_result = "苏格兰高地徒步"

    def describe(self, image_jpeg, language="zh"):
        with self.lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        try:
            time.sleep(self.delay)
            size = Image.open(io.BytesIO(image_jpeg)).size
            self.describe_calls.append(size)
            queue = self.answers.get("queue")
            if queue:
                item = queue.pop(0)
                if isinstance(item, Exception):
                    raise item
                return item
            return {"caption": "山间的湖泊", "scene": "landscape", "tags": ["湖", "山", "湖"], "people_count": 0,
                    "quality_notes": "", "style": "landscape"}
        finally:
            with self.lock:
                self.active -= 1

    def name_event(self, captions, date, place, language="zh"):
        self.name_calls.append((list(captions), date, place, language))
        return self.name_result


def library(tmp_path, images, llm, n_pick=3, n_other=1, **cfg):
    llm_cfg = dataclasses.replace(LLM(), max_image_edge=300, concurrency=2, **cfg)
    ctx = fake_ctx(tmp_path, llm=llm, brain_llm=None)
    ctx.config = dataclasses.replace(ctx.config, llm=llm_cfg)
    picks = [add_file(ctx, f"p{i}.jpg", preview_path=str(images.sharp), captured_at=f"2024-05-12T10:0{i}:00.000Z")
             for i in range(n_pick)]
    others = [add_file(ctx, f"o{i}.jpg", preview_path=str(images.blurred), captured_at=f"2024-05-12T11:0{i}:00.000Z")
              for i in range(n_other)]
    for r in picks:
        ctx.state.set_analysis(r.id, pick=1)
    for r in others:
        ctx.state.set_analysis(r.id, pick=0, reject_reason="blurry")
    ctx.state.create_group("event", [r.id for r in picks + others])
    return ctx, picks, others


def run(ctx):
    stage = Understand()
    batch = stage.select(ctx, 100)
    outcomes = stage.run(ctx, batch)
    ctx.state.set_steps(outcomes, "understand")
    return batch, outcomes


def test_registry():
    assert STAGE is Understand and Understand.uses_gpu and Understand.name == "understand"


def test_tags_picks_only_and_resizes(tmp_path, images):
    llm = FakeLLM()
    ctx, picks, others = library(tmp_path, images, llm)
    batch, outcomes = run(ctx)
    by = {o.file_id: o.status for o in outcomes}
    assert [by[r.id] for r in picks] == ["done"] * 3 and by[others[0].id] == "skipped"
    assert len(llm.describe_calls) == 3
    assert all(max(size) <= 300 for size in llm.describe_calls)  # max_image_edge applied before base64
    assert ctx.state.tags(picks[0].id) == [("山", "vlm", None), ("湖", "vlm", None)]
    d = ctx.state.description(picks[0].id)
    assert d["caption"] == "山间的湖泊" and d["scene"] == "landscape" and d["style_preset"] == "landscape"
    assert d["model"] == "qwen3-vl-4b"
    assert ctx.state.description(others[0].id) is None


def test_tag_scope_all(tmp_path, images):
    llm = FakeLLM()
    ctx, picks, others = library(tmp_path, images, llm, tag_scope="all")
    run(ctx)
    assert len(llm.describe_calls) == 4
    assert ctx.state.description(others[0].id) is not None


def test_normalises_unknown_enums_and_limits_tags(tmp_path, images):
    llm = FakeLLM({"queue": [{"caption": "x" * 200, "scene": "weird", "style": "vivid",
                              "tags": [f"t{i}" for i in range(20)]}]})
    ctx, picks, _ = library(tmp_path, images, llm, n_pick=1, n_other=0)
    run(ctx)
    d = ctx.state.description(picks[0].id)
    assert d["scene"] == "other" and d["style_preset"] == "default" and len(d["caption"]) <= 80
    assert len(ctx.state.tags(picks[0].id)) == 8


def test_retry_once_then_fail(tmp_path, images):
    llm = FakeLLM({"queue": [ValueError("bad json"), {"caption": "ok", "scene": "other", "tags": []},
                             ValueError("bad"), ValueError("still bad")]})
    ctx, picks, _ = library(tmp_path, images, llm, n_pick=2, n_other=0)
    _, outcomes = run(ctx)
    statuses = sorted(o.status for o in outcomes)
    assert statuses == ["done", "failed"]
    failed = next(o for o in outcomes if o.status == "failed")
    assert "ValueError" in failed.error


def test_missing_preview_fails_that_file_only(tmp_path, images):
    llm = FakeLLM()
    ctx, picks, _ = library(tmp_path, images, llm, n_pick=2, n_other=0)
    ctx.state.files[picks[0].id].preview_path = str(tmp_path / "gone.jpg")
    _, outcomes = run(ctx)
    by = {o.file_id: o for o in outcomes}
    assert by[picks[0].id].status == "failed" and by[picks[1].id].status == "done"


def test_concurrency_limit(tmp_path, images):
    llm = FakeLLM(delay=0.05)
    ctx, *_ = library(tmp_path, images, llm, n_pick=6, n_other=0)
    run(ctx)
    assert llm.max_active == 2


def test_event_named_after_all_picks_described(tmp_path, images):
    llm = FakeLLM()
    brain = FakeLLM()
    brain.name_result = "苏格兰/高地:徒步旅行很长很长的名字"
    ctx, picks, _ = library(tmp_path, images, llm)
    ctx.brain_llm = brain
    run(ctx)
    group = ctx.state.groups("event")[0]
    assert group["name"] == "苏格兰高地徒步旅行很长很长"[:12] and len(group["name"]) <= 12
    assert not set(group["name"]) & set('/\\:*?"<>|')
    captions, date, place, language = brain.name_calls[0]
    assert captions == ["山间的湖泊"] * 3 and date == "2024-05-12" and place is None and language == "zh"
    assert llm.name_calls == []  # the brain model, not the vision model, names events


def test_event_naming_waits_for_remaining_picks(tmp_path, images):
    llm = FakeLLM()
    ctx, picks, _ = library(tmp_path, images, llm, n_pick=4, n_other=0)
    stage = Understand()
    first = stage.select(ctx, 2)
    outcomes = stage.run(ctx, first)
    ctx.state.set_steps(outcomes, "understand")
    assert ctx.state.groups("event")[0]["name"] is None  # two picks still undescribed
    second = stage.select(ctx, 2)
    outcomes = stage.run(ctx, second)
    ctx.state.set_steps(outcomes, "understand")
    assert ctx.state.groups("event")[0]["name"] == "苏格兰高地徒步"
    assert len(llm.name_calls) == 1


def test_event_naming_retried_from_select_when_nothing_pending(tmp_path, images):
    llm = FakeLLM()
    ctx, picks, _ = library(tmp_path, images, llm)
    llm.name_result = ""
    run(ctx)
    assert ctx.state.groups("event")[0]["name"] is None
    llm.name_result = "周末郊游"
    assert Understand().select(ctx, 10) == []
    assert ctx.state.groups("event")[0]["name"] == "周末郊游"


def test_sanitize_name():
    assert sanitize_name('a/b\\c:d*e?f"g<h>i|j') == "abcdefghij"
    assert sanitize_name("  ..x.. ") == "x"
    assert sanitize_name("一二三四五六七八九十一二三四") == "一二三四五六七八九十一二"

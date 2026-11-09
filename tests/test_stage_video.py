import json
import subprocess
from pathlib import Path

from PIL import Image

from afterglow.config import Video as VideoCfg
from afterglow.stages import video as video_mod
from afterglow.stages.video import STAGE, Video
from tests.conftest import add_file, fake_ctx, make_sharp, tree_hashes

ENCODERS = """Encoders:
 V..... = Video
 ------
 V....D libx264              libx264 H.264 / AVC / MPEG-4 AVC
 V....D h264_nvenc           NVIDIA NVENC H.264 encoder (codec h264)
 A....D aac                  AAC (Advanced Audio Coding)
"""
PROBE = {
    "format": {"duration": "10.0", "tags": {"creation_time": "2024-06-01T08:30:00.000000Z"}},
    "streams": [{"codec_type": "audio"}, {"codec_type": "video", "width": 3840, "height": 2160}],
}


class FakeFFmpeg:
    def __init__(self, encoders=ENCODERS, probe=None, fail_encoders=()):
        self.encoders, self.probe = encoders, PROBE if probe is None else probe
        self.fail_encoders = set(fail_encoders)
        self.calls: list[list[str]] = []

    def __call__(self, cmd, **kw):
        assert kw.get("check") is False and not kw.get("shell")
        self.calls.append(list(cmd))
        tool = cmd[0]
        if tool == "ffprobe":
            if self.probe is False:
                return subprocess.CompletedProcess(cmd, 1, "", "invalid data")
            return subprocess.CompletedProcess(cmd, 0, json.dumps(self.probe), "")
        if "-encoders" in cmd:
            return subprocess.CompletedProcess(cmd, 0, self.encoders, "")
        if "-frames:v" in cmd:
            Path(cmd[-1]).parent.mkdir(parents=True, exist_ok=True)
            make_sharp(3, (320, 180)).save(cmd[-1])
            return subprocess.CompletedProcess(cmd, 0, "", "")
        encoder = cmd[cmd.index("-c:v") + 1]
        if encoder in self.fail_encoders:
            return subprocess.CompletedProcess(cmd, 1, "", "Cannot load libcuda.so.1")
        Path(cmd[-1]).write_bytes(b"proxy")
        return subprocess.CompletedProcess(cmd, 0, "", "")

    def encodes(self):
        return [c for c in self.calls if c[0] == "ffmpeg" and "-c:v" in c]


class FakeLLM:
    def __init__(self):
        self.calls = 0

    def describe(self, image_jpeg, language="zh"):
        self.calls += 1
        return {"caption": f"海边视频{self.calls}", "scene": "landscape", "tags": ["海", f"t{self.calls}"]}


def setup(tmp_path, monkeypatch, ffmpeg=None, n=1, **cfg):
    ffmpeg = ffmpeg or FakeFFmpeg()
    monkeypatch.setattr(video_mod.subprocess, "run", ffmpeg)
    ctx = fake_ctx(tmp_path, llm=FakeLLM(), video=VideoCfg(**cfg) if cfg else VideoCfg())
    vids = []
    for i in range(n):
        (ctx.config.paths.source / "clips").mkdir(exist_ok=True)
        (ctx.config.paths.source / "clips" / f"clip{i}.mp4").write_bytes(b"fake mp4 " * 10)
        vids.append(add_file(ctx, f"clips/clip{i}.mp4", sha256=f"{i:02x}" * 32, captured_at="2024-05-01T00:00:00.000Z",
                             captured_src="mtime"))
    return ctx, ffmpeg, vids


def run(ctx, stage=None, batch=None):
    stage = stage or Video()
    batch = batch if batch is not None else ctx.state.pending("video", 100)
    outcomes = stage.run(ctx, batch)
    ctx.state.set_steps(outcomes, "video")
    return outcomes


def test_registry():
    assert STAGE is Video and Video.uses_gpu and Video.per_file and Video.name == "video"


def test_probe_keyframes_preview_and_nvenc_proxy(tmp_path, monkeypatch):
    ctx, ff, (vid,) = setup(tmp_path, monkeypatch)
    before = tree_hashes(ctx.config.paths.source)
    img = add_file(ctx, "photo.jpg")
    outcomes = {o.file_id: o for o in run(ctx)}
    assert outcomes[vid.id].status == "done" and outcomes[img.id].status == "skipped"

    rec = ctx.state.get_file(vid.id)
    assert rec.duration == 10.0 and (rec.width, rec.height) == (3840, 2160)
    assert rec.captured_at == "2024-06-01T08:30:00.000Z" and rec.captured_src == "exif"
    assert rec.preview_path and Image.open(rec.preview_path).size == (320, 180)

    frames = sorted((ctx.config.paths.cache / "keyframes" / str(vid.id)).glob("kf_*.jpg"))
    assert len(frames) == 5
    seeks = [float(c[c.index("-ss") + 1]) for c in ff.calls if "-frames:v" in c]
    assert seeks == [1.0, 3.0, 5.0, 7.0, 9.0]  # evenly spread over the clip

    enc = ff.encodes()
    assert len(enc) == 1
    cmd = enc[0]
    assert cmd[cmd.index("-c:v") + 1] == "h264_nvenc"
    assert cmd[cmd.index("-preset") + 1] == "p4" and cmd[cmd.index("-cq") + 1] == "28"
    assert cmd[cmd.index("-c:a") + 1] == "aac" and cmd[cmd.index("-b:a") + 1] == "128k"
    assert "min(1080,ih)" in cmd[cmd.index("-vf") + 1]
    proxy = ctx.config.paths.output / "proxies" / "clips" / "clip0_proxy.mp4"
    assert proxy.read_bytes() == b"proxy" and cmd[-1] != str(proxy)  # written to a temp name first
    assert [o["kind"] for o in ctx.state.outputs(file_id=vid.id)] == ["proxy"]
    assert not list(proxy.parent.glob(".*"))

    # keyframes described by the vision model: union of tags, first caption
    assert ctx.llm.calls == 5
    assert {t[0] for t in ctx.state.tags(vid.id)} == {"海", "t1", "t2", "t3", "t4", "t5"}
    assert ctx.state.description(vid.id)["caption"] == "海边视频1"
    assert tree_hashes(ctx.config.paths.source) == before


def test_libx264_fallback_and_encoder_probe_cached(tmp_path, monkeypatch):
    no_nvenc = ENCODERS.replace("h264_nvenc", "h264_vaapi")
    ctx, ff, vids = setup(tmp_path, monkeypatch, FakeFFmpeg(encoders=no_nvenc), n=2)
    run(ctx)
    enc = ff.encodes()
    assert len(enc) == 2
    for cmd in enc:
        assert cmd[cmd.index("-c:v") + 1] == "libx264" and cmd[cmd.index("-crf") + 1] == "23"
        assert "-cq" not in cmd
    assert len([c for c in ff.calls if "-encoders" in c]) == 1  # detected once per stage instance


def test_nvenc_runtime_failure_retries_with_libx264(tmp_path, monkeypatch):
    ctx, ff, (vid,) = setup(tmp_path, monkeypatch, FakeFFmpeg(fail_encoders={"h264_nvenc"}))
    out = run(ctx)
    assert out[0].status == "done"
    assert [c[c.index("-c:v") + 1] for c in ff.encodes()] == ["h264_nvenc", "libx264"]


def test_existing_proxy_is_not_re_encoded(tmp_path, monkeypatch):
    ctx, ff, (vid,) = setup(tmp_path, monkeypatch)
    proxy = ctx.config.paths.output / "proxies" / "clips" / "clip0_proxy.mp4"
    proxy.parent.mkdir(parents=True)
    proxy.write_bytes(b"done earlier")
    run(ctx)
    assert ff.encodes() == [] and proxy.read_bytes() == b"done earlier"


def test_probe_failure_marks_file_failed(tmp_path, monkeypatch):
    ctx, ff, vids = setup(tmp_path, monkeypatch, FakeFFmpeg(probe=False), n=2)
    out = run(ctx)
    assert [o.status for o in out] == ["failed", "failed"] and "ffprobe" in out[0].error


def test_keyframes_zero_skips_frames_and_llm(tmp_path, monkeypatch):
    ctx, ff, (vid,) = setup(tmp_path, monkeypatch, keyframes=0)
    out = run(ctx)
    assert out[0].status == "done" and ctx.llm.calls == 0
    assert not [c for c in ff.calls if "-frames:v" in c]


class FakeLumen:
    def __init__(self):
        self.calls = []

    def timelapse(self, paths, output, fmt="h264", fps=24):
        self.calls.append((list(paths), output, fmt))
        Path(output).write_bytes(b"timelapse")
        return {}


def test_timelapse_group_rendered_once(tmp_path, monkeypatch):
    ctx, ff, (vid,) = setup(tmp_path, monkeypatch)
    ctx.lumen = FakeLumen()
    frames = []
    for i in range(3):
        make_sharp(i).save(ctx.config.paths.source / f"tl{i}.jpg")
        frames.append(add_file(ctx, f"tl{i}.jpg", captured_at=f"2024-05-12T10:00:0{i}.000Z"))
    event = ctx.state.create_group("event", [f.id for f in frames], name="日落延时")
    gid = ctx.state.create_group("timelapse", [f.id for f in frames])
    stage = Video()
    run(ctx, stage)
    assert len(ctx.lumen.calls) == 1
    paths, output, fmt = ctx.lumen.calls[0]
    assert paths == [f.path for f in frames] and fmt == "h264"
    assert output == str(ctx.config.paths.output / "timelapse" / "日落延时" / "timelapse_tl0.mp4")
    outs = ctx.state.outputs(group_id=gid)
    assert [o["kind"] for o in outs] == ["timelapse"] and outs[0]["path"] == output
    assert event

    run(ctx, stage, batch=[])  # same run id: not rendered again
    run(ctx, Video(), batch=[])  # new stage instance (new run): output exists -> skipped
    assert len(ctx.lumen.calls) == 1


def test_timelapse_failure_is_logged_not_raised(tmp_path, monkeypatch):
    ctx, ff, _ = setup(tmp_path, monkeypatch)
    ctx.lumen = FakeLumen()
    ctx.lumen.timelapse = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("lumen error"))
    frames = [add_file(ctx, f"t{i}.jpg", captured_at=f"2024-05-12T10:00:0{i}.000Z") for i in range(3)]
    ctx.state.create_group("timelapse", [f.id for f in frames])
    out = run(ctx)
    assert [o.status for o in out] == ["done"] + ["skipped"] * 3
    assert ctx.state.outputs(kind="timelapse") == []

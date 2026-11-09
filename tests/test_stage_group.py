from datetime import UTC, datetime, timedelta

from afterglow.config import Events, Video
from afterglow.stages.group import SCATTERED, STAGE, Group
from tests.conftest import add_file, fake_ctx


def stamp(base: datetime, seconds: float) -> str:
    dt = base + timedelta(seconds=seconds)
    return dt.strftime("%Y-%m-%dT%H:%M:%S.") + f"{dt.microsecond // 1000:03d}Z"


T0 = datetime(2024, 5, 12, 10, 0, 0, tzinfo=UTC)


def seq(ctx, prefix, offsets, base=T0, **fields):
    """Add one file per offset (seconds from base); per-file field lists are indexed by position."""
    recs = []
    for i, off in enumerate(offsets):
        extra = {k: (v[i] if isinstance(v, list) else v) for k, v in fields.items()}
        recs.append(add_file(ctx, f"{prefix}_{i:03d}.jpg", captured_at=stamp(base, off), camera="C1", **extra))
    return recs


def run_group(ctx):
    outcomes = Group().run(ctx, [])
    assert outcomes == []
    return {k: ctx.state.groups(k) for k in ("event", "hdr", "focus", "pano", "timelapse")}


def ids(recs):
    return [r.id for r in recs]


def test_registry():
    assert STAGE is Group and not Group.per_file and Group.name == "group"


def test_events_split_on_time_gap_gps_and_scatter(tmp_path):
    ctx = fake_ctx(tmp_path, events=Events(gap_hours=4.0, gps_km=5.0, min_photos=3))
    day1 = seq(ctx, "d1", [i * 60 for i in range(10)])
    day2 = seq(ctx, "d2", [86400 + i * 60 for i in range(5)])
    # same afternoon, two cities 500 km apart, 30 minutes between them
    edi = seq(ctx, "edi", [3 * 86400 + i * 60 for i in range(5)], gps_lat=55.95, gps_lon=-3.19)
    ldn = seq(ctx, "ldn", [3 * 86400 + 2400 + i * 60 for i in range(4)], gps_lat=51.50, gps_lon=-0.12)
    strays = [seq(ctx, f"s{k}", [(8 + 2 * k) * 86400])[0] for k in range(3)]  # three lone shots in the same month
    video = add_file(ctx, "v.mp4", captured_at=stamp(T0, 120))  # lands in day1
    paired = add_file(ctx, "d1_000_pair.jpg", captured_at=stamp(T0, 0), pair_id=day1[0].id)
    dup = add_file(ctx, "dup.jpg", captured_at=stamp(T0, 30))
    ctx.state.set_analysis(dup.id, reject_reason="duplicate")

    groups = run_group(ctx)["event"]
    members = [g["members"] for g in groups]
    assert len(groups) == 5
    assert set(ids(day1) + [video.id]) in [set(m) for m in members]
    assert ids(day2) in members and ids(edi) in members and ids(ldn) in members
    scattered = [g for g in groups if g["name"] == SCATTERED]
    assert len(scattered) == 1 and set(scattered[0]["members"]) == set(ids(strays))
    assert scattered[0]["meta"]["scattered"] is True
    flat = {f for m in members for f in m}
    assert paired.id not in flat and dup.id not in flat
    big = next(g for g in groups if g["name"] is None and edi[0].id in g["members"])
    assert big["meta"]["gps_center"] == [55.95, -3.19] and big["meta"]["cameras"] == ["C1"]


def test_group_marks_steps_and_is_idempotent(tmp_path):
    ctx = fake_ctx(tmp_path)
    recs = seq(ctx, "a", [i * 10 for i in range(6)])
    run_group(ctx)
    assert all(ctx.state.step(r.id, "group") == "done" for r in recs)
    before = len(ctx.state.groups("event"))
    run_group(ctx)
    assert len(ctx.state.groups("event")) == before == 1


def test_hdr_by_exposure_bias(tmp_path):
    ctx = fake_ctx(tmp_path)
    hdr = seq(ctx, "h", [0, 1, 2], exposure_bias=[-2.0, 0.0, 2.0], aperture=8.0, iso=100)
    flat = seq(ctx, "f", [600, 601, 602], exposure_bias=[0.0, 0.0, 0.0])
    narrow = seq(ctx, "n", [1200, 1201, 1202], exposure_bias=[-0.3, 0.0, 0.3])
    groups = run_group(ctx)
    assert [g["members"] for g in groups["hdr"]] == [ids(hdr)]
    assert groups["hdr"][0]["meta"]["mode"] == "bias"
    claimed = {f for g in groups["hdr"] for f in g["members"]}
    assert not claimed & set(ids(flat) + ids(narrow))


def test_hdr_by_shutter_doubling_with_locked_aperture(tmp_path):
    ctx = fake_ctx(tmp_path)
    ok = seq(ctx, "s", [0, 1, 2, 3], shutter=[1 / 500, 1 / 250, 1 / 125, 1 / 60], aperture=8.0, iso=100)
    changed_iso = seq(ctx, "t", [500, 501, 502], shutter=[1 / 500, 1 / 125, 1 / 30], aperture=8.0, iso=[100, 400, 100])
    groups = run_group(ctx)
    assert [g["members"] for g in groups["hdr"]] == [ids(ok)]
    assert groups["hdr"][0]["meta"]["mode"] == "shutter"
    assert not set(ids(changed_iso)) & set(groups["hdr"][0]["members"])


def test_hdr_requires_same_body_and_short_gaps(tmp_path):
    ctx = fake_ctx(tmp_path)
    a = seq(ctx, "g", [0, 1, 5], exposure_bias=[-2.0, 0.0, 2.0])  # 4 s gap > 3 s
    assert run_group(ctx)["hdr"] == []
    assert a


def test_focus_stack_by_focus_distance(tmp_path):
    ctx = fake_ctx(tmp_path)
    stack = seq(ctx, "fs", [0, 2, 4, 6, 8], focus_distance=[0.5, 0.6, 0.8, 1.2, 2.0], shutter=0.01, aperture=4.0)
    groups = run_group(ctx)
    assert [g["members"] for g in groups["focus"]] == [ids(stack)]
    assert groups["focus"][0]["meta"]["mode"] == "focus_distance"


def test_focus_stack_dhash_fallback(tmp_path):
    ctx = fake_ctx(tmp_path)
    stack = seq(ctx, "fd", [0, 3, 6, 9], dhash="00ff00ff00ff00ff", shutter=0.01, aperture=4.0)
    groups = run_group(ctx)
    assert [g["members"] for g in groups["focus"]] == [ids(stack)]
    assert groups["focus"][0]["meta"]["mode"] == "dhash"
    assert groups["pano"] == []


def test_focus_needs_four_frames(tmp_path):
    ctx = fake_ctx(tmp_path)
    seq(ctx, "fd", [0, 3, 6], focus_distance=[0.5, 0.6, 0.8])
    assert run_group(ctx)["focus"] == []


def test_panorama_overlapping_frames_with_locked_exposure(tmp_path):
    ctx = fake_ctx(tmp_path)
    hashes = ["0000000000000000", "000000000000ffff", "00000000ffffffff", "0000ffffffffffff"]  # 16 bits apart
    pano = seq(ctx, "p", [0, 3, 6, 9], dhash=hashes, shutter=0.005, aperture=8.0, iso=100)
    groups = run_group(ctx)
    assert [g["members"] for g in groups["pano"]] == [ids(pano)]
    assert groups["focus"] == [] and groups["hdr"] == []


def test_panorama_rejects_identical_or_unrelated_frames(tmp_path):
    ctx = fake_ctx(tmp_path)
    seq(ctx, "same", [0, 3, 6], dhash="0000000000000000")
    seq(ctx, "far", [1000, 1003, 1006], dhash=["0000000000000000", "ffffffffffffffff", "0000000000000000"])
    assert run_group(ctx)["pano"] == []


def test_panorama_exposure_must_be_locked(tmp_path):
    ctx = fake_ctx(tmp_path)
    hashes = ["0000000000000000", "000000000000ffff", "00000000ffffffff"]
    seq(ctx, "p", [0, 3, 6], dhash=hashes, shutter=[0.005, 0.01, 0.02], aperture=8.0)
    assert run_group(ctx)["pano"] == []


def test_timelapse_regular_intervals(tmp_path):
    ctx = fake_ctx(tmp_path, video=Video(timelapse_min_frames=60))
    frames = seq(ctx, "tl", [i * 5.0 + (0.1 if i % 2 else 0.0) for i in range(70)], dhash="00ff00ff00ff00ff")
    irregular = seq(ctx, "ir", [10_000 + i * i * 3 for i in range(70)])
    groups = run_group(ctx)
    assert len(groups["timelapse"]) == 1
    assert groups["timelapse"][0]["members"] == ids(frames)
    assert abs(groups["timelapse"][0]["meta"]["interval_mean"] - 5.0) < 0.2
    assert groups["focus"] == []
    assert not set(ids(irregular)) & set(groups["timelapse"][0]["members"])


def test_timelapse_too_short(tmp_path):
    ctx = fake_ctx(tmp_path, video=Video(timelapse_min_frames=60))
    seq(ctx, "tl", [i * 20.0 for i in range(40)])
    assert run_group(ctx)["timelapse"] == []

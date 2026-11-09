import os
import xml.etree.ElementTree as ET
from datetime import UTC, datetime, timedelta
from pathlib import Path

from afterglow.config import Immich, Organize
from afterglow.stages.publish import STAGE, Publish, build_xmp, rating_for
from tests.conftest import add_file, fake_ctx, make_sharp, tree_hashes

NS = {
    "x": "adobe:ns:meta/", "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
    "xmp": "http://ns.adobe.com/xap/1.0/", "dc": "http://purl.org/dc/elements/1.1/",
}


def make_library(tmp_path, **overrides):
    ctx = fake_ctx(tmp_path, **overrides)
    src = ctx.config.paths.source
    for d in ("trip", "other", "x", "y"):
        (src / d).mkdir()
    for i, rel in enumerate(["trip/a.jpg", "trip/b.dng", "trip/b.jpg", "trip/c.jpg", "trip/d.jpg", "x/IMG_1.jpg",
                             "y/IMG_1.jpg", "other/stray.jpg"]):
        make_sharp(i).save(src / rel, format="JPEG")
    (src / "trip" / "v.mp4").write_bytes(b"video-bytes")
    when = "2024-05-12T10:00:00.000Z"
    f = {}
    f["a"] = add_file(ctx, "trip/a.jpg", sha256="aa" * 32, captured_at=when)
    f["b"] = add_file(ctx, "trip/b.dng", sha256="bb" * 32, captured_at=when)
    f["bj"] = add_file(ctx, "trip/b.jpg", sha256="cc" * 32, captured_at=when, pair_id=f["b"].id)
    f["c"] = add_file(ctx, "trip/c.jpg", sha256="dd" * 32, captured_at=when)
    f["d"] = add_file(ctx, "trip/d.jpg", sha256="ee" * 32, captured_at="2030-01-01T00:00:00.000Z")
    f["v"] = add_file(ctx, "trip/v.mp4", sha256="ff" * 32, captured_at=when)
    f["x"] = add_file(ctx, "x/IMG_1.jpg", sha256="11" * 32, captured_at=when)
    f["y"] = add_file(ctx, "y/IMG_1.jpg", sha256="22" * 32, captured_at=when)
    f["s"] = add_file(ctx, "other/stray.jpg", sha256="33" * 32, captured_at="2023-12-24T09:00:00.000Z")
    st = ctx.state
    st.set_analysis(f["a"].id, pick=1, quality=90.0, sharpness=80.0)
    st.set_analysis(f["b"].id, pick=1, quality=60.0, burst_best=1)
    st.set_analysis(f["c"].id, pick=0, reject_reason="blurry", quality=10.0)
    st.set_analysis(f["d"].id, pick=0, reject_reason="duplicate")
    st.set_analysis(f["x"].id, pick=1, quality=50.0)
    st.set_analysis(f["y"].id, pick=1, quality=50.0)
    st.set_analysis(f["s"].id, pick=1, quality=50.0)
    st.add_tags(f["a"].id, [("湖", "vlm", None), ("山 & 水", "vlm", None)])
    st.set_description(f["a"].id, caption='湖边的 <清晨> "雾"', scene="landscape", style_preset="landscape")
    export = ctx.config.paths.output / "exports" / "trip" / "b.jpg"
    export.parent.mkdir(parents=True)
    make_sharp(50).save(export)
    st.add_output("export", str(export), file_id=f["b"].id)
    members = [f[k].id for k in ("a", "b", "c", "d", "v", "x", "y")]
    st.create_group("event", members, name="苏格兰徒步")
    st.create_group("event", [f["s"].id], name="零散")
    st.create_group("duplicate", [f["a"].id, f["d"].id], meta={"original": f["a"].id})
    return ctx, f


def run(ctx):
    outcomes = Publish().run(ctx, [])
    assert outcomes == []
    return outcomes


def lib(ctx, *parts):
    return ctx.config.paths.organized.joinpath(*parts)


def test_registry():
    assert STAGE is Publish and not Publish.per_file and not Publish.uses_gpu


def test_rating_rules():
    assert rating_for({"pick": 1}) == 3
    assert rating_for({"pick": 1, "burst_best": 1}) == 4
    assert rating_for({"pick": 1, "burst_best": 1, "quality": 85}) == 5
    assert rating_for({"pick": 0, "reject_reason": "blurry", "quality": 99}) == 0
    assert rating_for({}) == 0


def test_xmp_namespaces_and_escaping():
    text = build_xmp({"pick": 1, "burst_best": 1}, ["湖", "山 & 水"], '<b>"雾"</b>')
    root = ET.fromstring(text.split("?>", 1)[1].rsplit("<?xpacket", 1)[0])
    assert root.tag == f"{{{NS['x']}}}xmpmeta"
    desc = root.find("rdf:RDF/rdf:Description", NS)
    assert desc.get(f"{{{NS['xmp']}}}Rating") == "4"
    assert desc.get(f"{{{NS['xmp']}}}Label") is None
    assert [li.text for li in desc.findall("dc:subject/rdf:Bag/rdf:li", NS)] == ["湖", "山 & 水"]
    assert desc.find("dc:description/rdf:Alt/rdf:li", NS).text == '<b>"雾"</b>'
    rejected = build_xmp({"reject_reason": "blurry"}, [], None)
    assert 'xmp:Label="Rejected"' in rejected and 'xmp:Rating="0"' in rejected and "dc:subject" not in rejected


def test_organized_tree_xmp_and_source_untouched(tmp_path):
    ctx, f = make_library(tmp_path)
    before = tree_hashes(ctx.config.paths.source)
    run(ctx)
    src = ctx.config.paths.source
    ev = "2024/2024-05 苏格兰徒步"

    # picks go to the layout, RAW with its paired JPEG and the process export
    for name, target in (("a.jpg", src / "trip/a.jpg"), ("b.dng", src / "trip/b.dng"), ("b.jpg", src / "trip/b.jpg"),
                         ("v.mp4", src / "trip/v.mp4")):
        link = lib(ctx, ev, name)
        assert link.is_symlink() and Path(os.readlink(link)) == target
    export_link = lib(ctx, ev, "b_edit.jpg")
    assert export_link.is_symlink() and Path(os.readlink(export_link)) == ctx.config.paths.output / "exports/trip/b.jpg"

    # rejects and duplicates are separated
    assert lib(ctx, "_culled", ev, "c.jpg").is_symlink()
    assert lib(ctx, "_duplicates", ev, "d.jpg").is_symlink()
    assert not lib(ctx, ev, "c.jpg").exists() and not lib(ctx, ev, "d.jpg").exists()

    # same name from two folders -> the second gets the sha prefix
    first, second = lib(ctx, ev, "IMG_1.jpg"), lib(ctx, ev, "IMG_1_222222.jpg")
    assert first.is_symlink() and second.is_symlink()
    assert Path(os.readlink(first)) == src / "x/IMG_1.jpg" and Path(os.readlink(second)) == src / "y/IMG_1.jpg"

    # lone photo falls into 零散 of its own month
    assert lib(ctx, "2023", "2023-12 零散", "stray.jpg").is_symlink()

    # all actions went through the SafeFS double and none touched the source
    assert {c[0] for c in ctx.fs.calls} <= {"mkdir", "symlink", "write_text"}
    assert all(not c[-1].startswith(str(src)) for c in ctx.fs.calls)
    assert tree_hashes(src) == before

    # XMP sidecars live under output/xmp, not next to the originals
    xmp = ctx.config.paths.output / "xmp" / "trip" / "a.xmp"
    root = ET.parse(xmp).getroot()
    desc = root.find("rdf:RDF/rdf:Description", NS)
    assert desc.get(f"{{{NS['xmp']}}}Rating") == "5"
    assert desc.find("dc:description/rdf:Alt/rdf:li", NS).text == '湖边的 <清晨> "雾"'
    assert sorted(li.text for li in desc.findall("dc:subject/rdf:Bag/rdf:li", NS)) == ["山 & 水", "湖"]
    rejected = ET.parse(ctx.config.paths.output / "xmp" / "trip" / "c.xmp").getroot()
    assert rejected.find("rdf:RDF/rdf:Description", NS).get(f"{{{NS['xmp']}}}Label") == "Rejected"
    assert not list(src.rglob("*.xmp"))
    assert not (ctx.config.paths.output / "xmp" / "trip" / "b.jpg.xmp").exists()
    assert [o["kind"] for o in ctx.state.outputs(file_id=f["a"].id, kind="xmp")] == ["xmp"]

    # every file has a publish step
    assert all(ctx.state.step(rec.id, "publish") == "done" for rec in f.values())


def test_publish_is_idempotent(tmp_path):
    ctx, f = make_library(tmp_path)
    run(ctx)
    first_calls = len(ctx.fs.calls)
    assert first_calls > 0
    ctx.fs.calls.clear()
    run(ctx)
    assert ctx.fs.calls == []  # nothing to link or rewrite


def test_copy_and_hardlink_modes(tmp_path):
    ctx, f = make_library(tmp_path, organize=Organize(mode="copy"))
    run(ctx)
    copied = lib(ctx, "2024/2024-05 苏格兰徒步", "a.jpg")
    assert copied.exists() and not copied.is_symlink()
    assert copied.read_bytes() == (ctx.config.paths.source / "trip/a.jpg").read_bytes()
    assert ctx.fs.ops("copy")

    ctx, f = make_library(tmp_path / "h", organize=Organize(mode="hardlink"))
    run(ctx)
    assert ctx.fs.ops("hardlink")
    assert lib(ctx, "2024/2024-05 苏格兰徒步", "a.jpg").samefile(ctx.config.paths.source / "trip/a.jpg")


def test_move_mode_downgrades_without_fresh_backup(tmp_path):
    ctx, f = make_library(tmp_path, organize=Organize(mode="move"))
    run(ctx)
    assert not ctx.fs.ops("move") and ctx.fs.ops("symlink")
    assert (ctx.config.paths.source / "trip/a.jpg").exists()

    ctx, f = make_library(tmp_path / "fresh", organize=Organize(mode="move"))
    ctx.state.set("last_backup_ok", (datetime.now(UTC) - timedelta(hours=2)).strftime("%Y-%m-%dT%H:%M:%SZ"))
    run(ctx)
    assert ctx.fs.ops("move")
    assert not (ctx.config.paths.source / "trip/a.jpg").exists()
    assert lib(ctx, "2024/2024-05 苏格兰徒步", "a.jpg").exists()

    ctx, f = make_library(tmp_path / "stale", organize=Organize(mode="move"))
    ctx.state.set("last_backup_ok", (datetime.now(UTC) - timedelta(hours=30)).strftime("%Y-%m-%dT%H:%M:%SZ"))
    run(ctx)
    assert not ctx.fs.ops("move")


def test_custom_layout_and_dirs(tmp_path):
    organize = Organize(
        mode="symlink", layout="{event}/{year}{month:02d}{day:02d}_{basename}", culled_dir="rejects",
        duplicates_dir="dupes",
    )
    ctx, f = make_library(tmp_path, organize=organize)
    run(ctx)
    assert lib(ctx, "苏格兰徒步", "20240512_a.jpg").is_symlink()
    assert lib(ctx, "rejects", "苏格兰徒步", "20240512_c.jpg").is_symlink()
    assert lib(ctx, "dupes", "苏格兰徒步", "20240512_d.jpg").is_symlink()


def test_link_failure_marks_only_that_file(tmp_path):
    ctx, f = make_library(tmp_path)
    original = ctx.fs.symlink

    def flaky(target, link):
        if str(link).endswith("c.jpg"):
            raise PermissionError("denied")
        original(target, link)

    ctx.fs.symlink = flaky
    run(ctx)
    assert ctx.state.step(f["c"].id, "publish") == "failed"
    assert ctx.state.step(f["a"].id, "publish") == "done"


# ----------------------------------------------------------------------------- Immich
class FakeImmich:
    def __init__(self, ctx, known: bool = True):
        self.ctx = ctx
        self.calls: list[tuple] = []
        self.libs: list[dict] = []
        self.known = known

    def libraries(self):
        self.calls.append(("libraries",))
        return self.libs

    def create_external_library(self, name, import_paths, owner_id=None):
        self.calls.append(("create_library", name, tuple(import_paths)))
        return "lib-1"

    def scan_library(self, library_id):
        self.calls.append(("scan", library_id))

    def wait_idle(self, names=("library",), timeout=1800):
        self.calls.append(("wait_idle", timeout))
        return True

    def asset_by_path(self, original_path):
        self.calls.append(("asset_by_path", original_path))
        # as inside the Immich container: organized tree mounted at /mnt/afterglow
        if original_path.startswith("/mnt/afterglow/") and self.known:
            return {"id": "asset:" + original_path, "exifInfo": {"description": "old"}}
        return None

    def _undone(self, action):
        assert self.ctx.undo.records and self.ctx.undo.records[-1]["action"] == action, "mutation not logged first"

    def create_album(self, name, asset_ids):
        self.calls.append(("create_album", name, tuple(asset_ids)))
        return "album-" + name

    def add_to_album(self, album_id, asset_ids):
        self._undone("immich_add_assets")
        self.calls.append(("add_to_album", album_id, tuple(asset_ids)))

    def update_asset(self, asset_id, description=None, rating=None):
        self._undone("immich_description")
        self.calls.append(("update_asset", asset_id, description, rating))
        return {}

    def upsert_tag(self, name):
        self.calls.append(("upsert_tag", name))
        return "tag:" + name

    def tag_assets(self, tag_id, asset_ids):
        self._undone("immich_tag")
        self.calls.append(("tag_assets", tag_id, tuple(asset_ids)))


def test_immich_skipped_when_disabled(tmp_path):
    ctx, f = make_library(tmp_path)
    assert ctx.immich is None
    run(ctx)
    assert ctx.undo.records == [] and ctx.state.get("immich_library_id") is None


def test_immich_sync_albums_descriptions_tags_with_undo(tmp_path):
    ctx, f = make_library(tmp_path, immich=Immich(api_key="k"))
    ctx.immich = FakeImmich(ctx)
    run(ctx)
    im = ctx.immich
    names = [c[0] for c in im.calls]
    assert names.index("create_library") < names.index("scan") < names.index("wait_idle") < names.index("create_album")
    assert ("create_library", "Afterglow", (str(ctx.config.paths.organized),)) in im.calls
    assert ctx.state.get("immich_library_id") == "lib-1"

    albums = {c[1]: c[2] for c in im.calls if c[0] == "create_album"}
    assert "2024-05 苏格兰徒步" in albums and "2023-12 零散" in albums
    event_assets = albums["2024-05 苏格兰徒步"]
    assert "asset:/mnt/afterglow/2024/2024-05 苏格兰徒步/a.jpg" in event_assets
    assert not any("_culled" in a or "_duplicates" in a for a in event_assets)
    assert any(r["action"] == "immich_album" and r["meta"]["album_id"] == "album-2024-05 苏格兰徒步"
               for r in ctx.undo.records)

    updates = {c[1]: c for c in im.calls if c[0] == "update_asset"}
    a_asset = "asset:/mnt/afterglow/2024/2024-05 苏格兰徒步/a.jpg"
    assert updates[a_asset][2:] == ('湖边的 <清晨> "雾"', 5)
    desc_record = next(r for r in ctx.undo.records if r["action"] == "immich_description"
                       and r["meta"]["asset_id"] == a_asset)
    assert desc_record["meta"]["previous"] == "old"

    tagged = {c[1]: c[2] for c in im.calls if c[0] == "tag_assets"}
    assert tagged["tag:Afterglow/湖"] == (a_asset,) and "tag:Afterglow/山 & 水" in tagged
    assert any(r["action"] == "immich_tag" and r["meta"]["tag_id"] == "tag:Afterglow/湖" for r in ctx.undo.records)

    # a second run changes nothing in Immich
    n_before = len([c for c in im.calls if c[0] in ("create_album", "update_asset", "tag_assets", "add_to_album")])
    run(ctx)
    n_after = len([c for c in im.calls if c[0] in ("create_album", "update_asset", "tag_assets", "add_to_album")])
    assert n_before == n_after
    assert len([c for c in im.calls if c[0] == "create_library"]) == 1


def test_immich_reuses_existing_library_and_tolerates_unindexed_assets(tmp_path):
    ctx, f = make_library(tmp_path, immich=Immich(api_key="k"))
    im = FakeImmich(ctx, known=False)
    im.libs = [{"id": "existing", "name": "Afterglow"}]
    ctx.immich = im
    run(ctx)
    assert ctx.state.get("immich_library_id") == "existing"
    assert not [c for c in im.calls if c[0] in ("create_library", "create_album", "update_asset")]
    assert ctx.state.step(f["a"].id, "publish") == "done"


def test_immich_failure_does_not_break_tree(tmp_path):
    ctx, f = make_library(tmp_path, immich=Immich(api_key="k"))
    im = FakeImmich(ctx)
    im.scan_library = lambda library_id: (_ for _ in ()).throw(RuntimeError("immich down"))
    ctx.immich = im
    run(ctx)
    assert lib(ctx, "2024/2024-05 苏格兰徒步", "a.jpg").is_symlink()
    assert ctx.state.step(f["a"].id, "publish") == "done"


def test_event_name_missing_falls_back_to_scattered(tmp_path):
    ctx = fake_ctx(tmp_path)
    make_sharp(1).save(ctx.config.paths.source / "z.jpg")
    rec = add_file(ctx, "z.jpg", sha256="44" * 32, captured_at="2022-03-04T05:06:07.000Z")
    ctx.state.set_analysis(rec.id, pick=1)
    run(ctx)
    assert lib(ctx, "2022", "2022-03 零散", "z.jpg").is_symlink()


def test_copy_picks_copies_keepers_and_links_the_rest(tmp_path):
    """organize.mode = copy_picks: picks become real files on the SSD, rejects stay symlinks."""
    ctx = fake_ctx(tmp_path, organize=Organize(mode="copy_picks"))
    src = ctx.config.paths.source
    keep, cull = src / "keep.jpg", src / "cull.jpg"
    keep.write_bytes(b"keep")
    cull.write_bytes(b"cull")
    when = "2024-05-12T10:00:00+00:00"
    k = add_file(ctx, "keep.jpg", captured_at=when, sha256="a" * 64)
    c = add_file(ctx, "cull.jpg", captured_at=when, sha256="b" * 64)
    ctx.state.set_analysis(k.id, pick=1, quality=80.0)
    ctx.state.set_analysis(c.id, pick=0, reject_reason="blurry")
    STAGE().run(ctx, [])
    placed = {p.name: p for p in ctx.config.paths.organized.rglob("*.jpg")}
    assert placed["keep.jpg"].is_file() and not placed["keep.jpg"].is_symlink()
    assert placed["keep.jpg"].read_bytes() == b"keep"
    assert placed["cull.jpg"].is_symlink()
    assert keep.read_bytes() == b"keep" and cull.read_bytes() == b"cull"

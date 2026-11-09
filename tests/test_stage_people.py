from __future__ import annotations

import dataclasses

from afterglow.config import Immich, People
from afterglow.stages.people import STAGE
from afterglow.stages.people import People as PeopleStage
from tests.conftest import FakeState, add_file, fake_ctx


class GroupState(FakeState):
    def add_group_members(self, group_id: int, member_ids: list[int]) -> None:
        g = self._groups[group_id - 1]
        g["members"].extend(m for m in member_ids if m not in g["members"])


class FakeImmich:
    def __init__(self, persons, assets, events):
        self.persons, self.assets, self.events = persons, assets, events
        self.calls: list[tuple] = []

    def people(self, with_hidden=False):
        self.calls.append(("people",))
        return self.persons

    def person_assets(self, person_id):
        self.calls.append(("person_assets", person_id))
        return self.assets.get(person_id, [])

    def create_album(self, name, asset_ids):
        self.calls.append(("create_album", name, list(asset_ids)))
        self.events.append("create")
        return "ALB-" + name

    def add_to_album(self, album_id, asset_ids):
        self.calls.append(("add_to_album", album_id, list(asset_ids)))
        self.events.append("add")


class OrderUndo:
    def __init__(self, events):
        self.events, self.records = events, []

    def record(self, action, src=None, dst=None, meta=None):
        self.events.append("undo:" + action)
        self.records.append((action, meta))
        return len(self.records)


def setup(tmp_path, n=6, people_cfg=None, persons=None, immich=True):
    events: list[str] = []
    ctx = fake_ctx(
        tmp_path, state=GroupState(), people=people_cfg or People(min_photos=3, best_per_person=3),
        immich=Immich(api_key="k"),
    )
    ctx.undo = OrderUndo(events)
    files = []
    for i in range(n):
        rec = add_file(ctx, f"img{i}.jpg")
        ctx.state.set_analysis(rec.id, pick=1, quality=10.0 * i)
        files.append(rec)
    assets = {"p1": [{"id": f"A{i}", "originalPath": f.path} for i, f in enumerate(files)]}
    assets["p1"].append({"id": "X", "originalPath": "/elsewhere/none.jpg"})
    ps = persons if persons is not None else [{"id": "p1", "name": "Ann"}]
    fake = FakeImmich(ps, assets, events)
    ctx.immich = fake if immich else None
    return ctx, fake, files, events


def test_stage_attrs():
    assert STAGE is PeopleStage
    assert (PeopleStage.name, PeopleStage.requires, PeopleStage.per_file, PeopleStage.uses_gpu) == (
        "people", ("publish",), False, False)


def test_album_best_by_quality_and_capped(tmp_path):
    ctx, fake, files, events = setup(tmp_path)
    assert PeopleStage().run(ctx, []) == []
    creates = [c for c in fake.calls if c[0] == "create_album"]
    assert creates == [("create_album", "人物 · Ann", ["A5", "A4", "A3"])]
    assert events == ["create", "undo:immich_album"]  # id only exists after creation, as in publish
    assert ctx.state.groups("person")[0]["name"] == "Ann"
    assert ctx.state.groups("person")[0]["meta"] == {"immich_person_id": "p1"}
    assert ctx.state.groups("person")[0]["members"] == [files[5].id, files[4].id, files[3].id]


def test_best_per_person_zero_means_all(tmp_path):
    ctx, fake, _, _ = setup(tmp_path, people_cfg=People(min_photos=3, best_per_person=0))
    PeopleStage().run(ctx, [])
    assert len(fake.calls[-1][2]) == 6


def test_non_picks_excluded(tmp_path):
    ctx, fake, files, _ = setup(tmp_path)
    ctx.state.set_analysis(files[5].id, pick=0)
    PeopleStage().run(ctx, [])
    assert [c for c in fake.calls if c[0] == "create_album"][0][2] == ["A4", "A3", "A2"]


def test_min_photos(tmp_path):
    ctx, fake, _, _ = setup(tmp_path, people_cfg=People(min_photos=7))
    PeopleStage().run(ctx, [])  # 6 mapped files (the unmapped asset is ignored)
    assert not [c for c in fake.calls if c[0] == "create_album"]
    ctx2, fake2, _, _ = setup(tmp_path / "b", people_cfg=People(min_photos=6, best_per_person=2))
    PeopleStage().run(ctx2, [])
    assert [c for c in fake2.calls if c[0] == "create_album"]


def test_unnamed(tmp_path):
    persons = [{"id": "p1", "name": ""}]
    ctx, fake, _, _ = setup(tmp_path, persons=persons)
    PeopleStage().run(ctx, [])
    assert not [c for c in fake.calls if c[0] == "create_album"]
    ctx2, fake2, _, _ = setup(
        tmp_path / "b", persons=persons, people_cfg=People(min_photos=3, best_per_person=3, include_unnamed=True)
    )
    PeopleStage().run(ctx2, [])
    assert [c for c in fake2.calls if c[0] == "create_album"][0][1] == "人物 · 未命名 1"


def test_idempotent_and_incremental(tmp_path):
    ctx, fake, files, events = setup(tmp_path)
    stage = PeopleStage()
    stage.run(ctx, [])
    n = len(fake.calls)
    stage.run(ctx, [])
    assert [c[0] for c in fake.calls[n:]] == ["people", "person_assets"]  # no writes
    # a better new photo displaces A3
    new = add_file(ctx, "img9.jpg")
    ctx.state.set_analysis(new.id, pick=1, quality=99.0)
    fake.assets["p1"].append({"id": "A9", "originalPath": new.path})
    events.clear()
    stage.run(ctx, [])
    assert len([c for c in fake.calls if c[0] == "create_album"]) == 1
    adds = [c for c in fake.calls if c[0] == "add_to_album"]
    assert adds == [("add_to_album", "ALB-人物 · Ann", ["A9"])]
    assert events == ["undo:immich_add_assets", "add"]  # undo recorded before the mutation
    assert ctx.undo.records[-1][1] == {"album_id": "ALB-人物 · Ann", "asset_ids": ["A9"]}
    assert new.id in ctx.state.groups("person")[0]["members"]
    assert len(ctx.state.groups("person")) == 1


def test_disabled_or_no_immich(tmp_path):
    ctx, fake, _, _ = setup(tmp_path, people_cfg=People(enabled=False))
    assert PeopleStage().run(ctx, []) == [] and fake.calls == []
    ctx2, fake2, _, _ = setup(tmp_path / "b", immich=False)
    assert PeopleStage().run(ctx2, []) == []
    ctx3, fake3, _, _ = setup(tmp_path / "c")
    ctx3.config = dataclasses.replace(ctx3.config, immich=Immich(api_key=""))
    assert PeopleStage().run(ctx3, []) == [] and fake3.calls == []


def test_failure_of_one_person_does_not_stop_others(tmp_path):
    ctx, fake, _, _ = setup(tmp_path, persons=[{"id": "bad", "name": "Bob"}, {"id": "p1", "name": "Ann"}])

    orig = fake.person_assets

    def boom(pid):
        if pid == "bad":
            raise RuntimeError("x")
        return orig(pid)

    fake.person_assets = boom
    assert PeopleStage().run(ctx, []) == []
    assert [c for c in fake.calls if c[0] == "create_album"]


def test_zero_people(tmp_path):
    ctx, fake, _, _ = setup(tmp_path, persons=[])
    assert PeopleStage().run(ctx, []) == []
    assert fake.calls == [("people",)]

from __future__ import annotations

from pathlib import Path

import pytest

from afterglow import config

EXAMPLE = Path(__file__).resolve().parent.parent / "config" / "afterglow.example.toml"


def test_example_loads_and_matches_defaults() -> None:
    cfg = config.load(EXAMPLE)
    default = config.Config()
    for section in ("organize", "cull", "events", "llm", "lumen", "process", "video", "immich", "backup", "gpu", "run"):
        assert getattr(cfg, section) == getattr(default, section), section
    assert cfg.paths.include_ext == default.paths.include_ext
    assert cfg.paths.source == Path("/media/xin/PHOTOS")
    assert cfg.paths.backup == Path("/media/xin/BACKUP")
    assert cfg.path == EXAMPLE


def test_example_has_every_key() -> None:
    import tomllib
    from dataclasses import fields

    data = tomllib.loads(EXAMPLE.read_text(encoding="utf-8"))
    default = config.Config()
    for f in fields(default):
        if f.name == "path":
            continue
        section = getattr(default, f.name)
        assert f.name in data, f.name
        for sf in fields(section):
            assert sf.name in data[f.name], f"{f.name}.{sf.name}"


def test_missing_file_is_config_error(tmp_path: Path) -> None:
    with pytest.raises(config.ConfigError):
        config.load(tmp_path / "nope.toml")


def test_unknown_key_and_section(tmp_path: Path) -> None:
    with pytest.raises(config.ConfigError):
        config.from_dict({"paths": {"bogus": 1}})
    with pytest.raises(config.ConfigError):
        config.from_dict({"bogus": {}})


def test_validate_move_requires_backup(tmp_path: Path) -> None:
    cfg = config.from_dict({"paths": {"source": str(tmp_path)}, "organize": {"mode": "move"}})
    problems = cfg.validate()
    assert any("backup" in p for p in problems)
    ok = config.from_dict(
        {"paths": {"source": str(tmp_path), "backup": str(tmp_path)}, "organize": {"mode": "move"}}
    )
    assert ok.validate() == []


def test_validate_source_and_bad_values(tmp_path: Path) -> None:
    cfg = config.from_dict({"paths": {"source": str(tmp_path / "missing")}, "cull": {}, "llm": {"tag_scope": "x"}})
    problems = cfg.validate()
    assert any("source" in p for p in problems)
    assert any("tag_scope" in p for p in problems)
    assert not any("source" in p for p in cfg.validate(require_source=False))


def test_kind_of() -> None:
    assert config.kind_of("CR3") == "raw"
    assert config.kind_of(".jpg") == "jpeg"
    assert config.kind_of("mov") == "video"
    assert config.kind_of("txt") == "other"

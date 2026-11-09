import dataclasses
import shutil
import subprocess
from pathlib import Path

import pytest

from afterglow.config import Backup as BackupCfg
from afterglow.stages import backup as backup_mod
from afterglow.stages.backup import STAGE, Backup
from tests.conftest import add_file, fake_ctx, tree_hashes


def make(tmp_path, verify="size", with_backup=True):
    ctx = fake_ctx(tmp_path)
    src = ctx.config.paths.source
    (src / "d").mkdir()
    (src / "d" / "a.jpg").write_bytes(b"a" * 100)
    (src / "b.cr3").write_bytes(b"b" * 200)
    (src / "lost+found").mkdir()
    (src / "lost+found" / "junk").write_bytes(b"x")
    rec = add_file(ctx, "d/a.jpg")
    paths = dataclasses.replace(ctx.config.paths, backup=tmp_path / "backup" if with_backup else None)
    ctx.config = dataclasses.replace(ctx.config, paths=paths, backup=BackupCfg(verify=verify))
    return ctx, rec


def fake_rsync(calls, corrupt=False, rc=0):
    def run(cmd, **kw):
        assert kw.get("check") is False and not kw.get("shell")
        calls.append(list(cmd))
        if rc:
            return subprocess.CompletedProcess(cmd, rc, "", "boom")
        src, dst = cmd[-2].rstrip("/"), cmd[-1].rstrip("/")
        shutil.copytree(src, dst, dirs_exist_ok=True, ignore=shutil.ignore_patterns("lost+found"))
        if corrupt:
            next(iter(Path(dst).rglob("a.jpg"))).write_bytes(b"short")
        return subprocess.CompletedProcess(cmd, 0, "", "")

    return run


def test_registry():
    assert STAGE is Backup and not Backup.per_file and Backup.name == "backup"


def test_no_backup_path_skips(tmp_path):
    ctx, rec = make(tmp_path, with_backup=False)
    assert Backup().run(ctx, []) == []
    assert ctx.state.step(rec.id, "backup") == "skipped"
    assert ctx.state.get("last_backup_ok") is None


@pytest.mark.parametrize("verify", ["size", "sha256"])
def test_rsync_and_verify_sets_kv(tmp_path, monkeypatch, verify):
    ctx, rec = make(tmp_path, verify=verify)
    calls = []
    monkeypatch.setattr(backup_mod.subprocess, "run", fake_rsync(calls))
    before = tree_hashes(ctx.config.paths.source)
    Backup().run(ctx, [])
    cmd = calls[0]
    assert cmd[0] == "rsync" and "-a" in cmd and "--delete" not in cmd and "lost+found" in cmd
    assert cmd[-2] == f"{ctx.config.paths.source}/" and cmd[-1] == f"{tmp_path / 'backup' / 'source'}/"
    assert ctx.state.get("last_backup_ok")
    assert ctx.state.step(rec.id, "backup") == "done"
    assert tree_hashes(ctx.config.paths.source) == before


def test_verification_failure_raises_and_leaves_kv_unset(tmp_path, monkeypatch):
    ctx, rec = make(tmp_path)
    monkeypatch.setattr(backup_mod.subprocess, "run", fake_rsync([], corrupt=True))
    with pytest.raises(RuntimeError, match="校验"):
        Backup().run(ctx, [])
    assert ctx.state.get("last_backup_ok") is None


def test_rsync_error_and_vanished_files(tmp_path, monkeypatch):
    ctx, rec = make(tmp_path)
    monkeypatch.setattr(backup_mod.subprocess, "run", fake_rsync([], rc=23))
    with pytest.raises(RuntimeError, match="rsync"):
        Backup().run(ctx, [])
    monkeypatch.setattr(backup_mod.subprocess, "run", fake_rsync([], rc=0))
    Backup().run(ctx, [])
    assert ctx.state.get("last_backup_ok")


def test_dry_run_does_not_call_rsync(tmp_path, monkeypatch):
    ctx, rec = make(tmp_path)
    ctx.dry_run = True
    calls = []
    monkeypatch.setattr(backup_mod.subprocess, "run", fake_rsync(calls))
    Backup().run(ctx, [])
    assert calls == []

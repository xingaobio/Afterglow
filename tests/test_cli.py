from __future__ import annotations

from pathlib import Path

import pytest

from afterglow import cli


def write_cfg(tmp_path: Path, extra: str = "", source: bool = True) -> Path:
    src = tmp_path / "source"
    if source:
        src.mkdir(exist_ok=True)
    p = tmp_path / "afterglow.toml"
    p.write_text(
        f"""
[paths]
source = "{src}"
work = "{tmp_path / 'work'}"
output = "{tmp_path / 'out'}"
organized = "{tmp_path / 'lib'}"
backup = ""

[lumen]
source_dir = "{tmp_path / 'nolumen'}"
python = "{tmp_path / 'nolumen' / 'python'}"
models_dir = "{tmp_path / 'models'}"
library_db = "{tmp_path / 'lib.db'}"

[llm]
server = "http://127.0.0.1:9/v1"
timeout_seconds = 1

[gpu]
lock_file = "{tmp_path / 'gpu.lock'}"
{extra}
""",
        encoding="utf-8",
    )
    return p


def test_missing_config_exits_1(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["--config", str(tmp_path / "none.toml"), "status"]) == 1
    assert "配置" in capsys.readouterr().err


def test_invalid_config_exits_1(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    cfg = write_cfg(tmp_path, source=False)
    assert cli.main(["--config", str(cfg), "run"]) == 1
    assert "paths.source" in capsys.readouterr().err


def test_bad_args_exit_1() -> None:
    assert cli.main(["frobnicate"]) == 1
    assert cli.main(["retry"]) == 1  # --stage required


def test_status_empty_db(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    cfg = write_cfg(tmp_path)
    assert cli.main(["status", "--config", str(cfg)]) == 0
    out = capsys.readouterr().out
    assert "publish" in out and "index" in out


def test_run_with_empty_registry_and_undo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from afterglow import pipeline

    monkeypatch.setattr(pipeline, "REGISTRY", {})
    monkeypatch.setattr(pipeline, "load_stages", lambda: {})
    cfg = write_cfg(tmp_path)
    assert cli.main(["--config", str(cfg), "run", "--stages", "index,cull", "--batch", "10"]) == 0
    assert cli.main(["--config", str(cfg), "run", "--dry-run"]) == 0
    assert cli.main(["--config", str(cfg), "run", "--stages", "bogus"]) == 1
    assert cli.main(["--config", str(cfg), "undo", "--last", "--dry-run"]) == 0
    assert cli.main(["--config", str(cfg), "undo", "--all"]) == 0
    assert cli.main(["--config", str(cfg), "retry", "--stage", "index"]) == 0


def test_doctor_reports_missing_required_tool(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = write_cfg(tmp_path)
    monkeypatch.setattr(cli.shutil, "which", lambda name: None)
    assert cli.main(["doctor", "--config", str(cfg)]) == 3


def test_doctor_ok_with_fake_tools(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import subprocess

    cfg = write_cfg(tmp_path)
    monkeypatch.setattr(cli.shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(
        cli.subprocess, "run",
        lambda argv, **kw: subprocess.CompletedProcess(argv, 0, stdout="1.0\n", stderr=""),
    )
    assert cli.main(["doctor", "--config", str(cfg)]) == 0


def test_setup_creates_dirs_and_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = write_cfg(tmp_path)
    monkeypatch.setattr(cli.shutil, "which", lambda name: None)
    assert cli.main(["setup", "--config", str(cfg)]) == 3  # tools missing in the sandbox, dirs still made
    for d in ("work", "out", "lib"):
        assert (tmp_path / d).is_dir()
    assert (tmp_path / "work" / "afterglow.db").is_file()


def test_bench_missing_script_or_passthrough(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[list[str]] = []

    class Done:
        returncode = 0

    monkeypatch.setattr(cli.subprocess, "run", lambda argv, **kw: (seen.append(argv), Done())[1])
    script = Path(cli.__file__).resolve().parent.parent / "scripts" / "bench_models.py"
    code = cli.main(["bench", "--sample", "7"])
    if script.is_file():
        assert code == 0 and seen[0][-2:] == ["--sample", "7"]
    else:
        assert code == 3

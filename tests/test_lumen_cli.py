from __future__ import annotations

import sys
from pathlib import Path

import pytest

from afterglow.config import Lumen
from afterglow.integrations.lumen import LumenCLI, LumenError, LumenLib

FAKE = Path(__file__).parent / "fake_lumen_mcp.py"


def make_cli(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fail_after: int = 0, restart_every: int = 200
) -> LumenCLI:
    cfg = Lumen(
        python=Path(sys.executable),
        source_dir=Path(__file__).parent,
        models_dir=tmp_path / "models",
        library_db=tmp_path / "lib.db",
        restart_every=restart_every,
    )
    monkeypatch.setenv("FAKE_LUMEN_FAIL_AFTER", str(fail_after))
    cli = LumenCLI(cfg, timeout=30)
    monkeypatch.setattr(cli, "_command", lambda: [sys.executable, str(FAKE)])
    return cli


def test_handshake_and_analyze(tmp_path, monkeypatch):
    with make_cli(tmp_path, monkeypatch) as cli:
        result = cli.analyze(["/a/1.cr3", "/a/2.cr3", "/a/3.cr3"])
        assert "analyze_photos" in cli.tools and "run_command" in cli.tools
        assert result["count"] == 3
        assert result["bursts"][0]["best"] == "/a/1.cr3"
        again = cli.analyze(["/a/1.cr3"])
        assert again["photos"][0]["sharpness"] == result["photos"][0]["sharpness"]


def test_env_passed(tmp_path, monkeypatch):
    cli = make_cli(tmp_path, monkeypatch)
    env = cli._env()
    assert env["LUMEN_MODELS"] == str(tmp_path / "models")
    assert env["LUMEN_LIBRARY"] == str(tmp_path / "lib.db")
    assert env["LUMEN_COMPUTE"] == "auto" and env["ORT_DISABLE_TELEMETRY"] == "1"
    assert LumenCLI(Lumen())._command()[1:] == ["/opt/lumen-raw/main.py", "--cli", "mcp", "--headless"]


def test_export_writes_file(tmp_path, monkeypatch):
    out = tmp_path / "out.jpg"
    with make_cli(tmp_path, monkeypatch) as cli:
        cli.open("/a/1.cr3")
        cli.auto_tone("/a/1.cr3")
        cli.apply_preset("/a/1.cr3", "Vivid")
        res = cli.export("/a/1.cr3", str(out), 92, None)
        assert res["output"] == str(out)
        assert out.stat().st_size > 0
        batch = cli.export_batch(["/a/2.cr3"], str(tmp_path), "jpg", 90, 2000)
        assert (tmp_path / "2-Lumen.jpg").exists() and batch["failed"] == []
        assert cli.write_xmp(["/a/1.cr3"])["errors"] == []
        assert cli.timelapse(["/a/1.cr3", "/a/2.cr3"], str(tmp_path / "t.mp4"))["frames"] == 2


def test_is_error_raises(tmp_path, monkeypatch):
    with make_cli(tmp_path, monkeypatch) as cli:
        with pytest.raises(LumenError, match="unknown command"):
            cli.run_command("no.such", {})
        with pytest.raises(LumenError, match="folder not found"):
            cli.export("/a/1.cr3", str(tmp_path / "missing" / "x.jpg"), 90, None)
        assert cli.info("/a/1.cr3")["tool"] == "photo_info"  # still usable after errors


def test_auto_restart_after_process_dies(tmp_path, monkeypatch):
    cli = make_cli(tmp_path, monkeypatch, fail_after=2)
    try:
        cli.open("/a/1.cr3")
        cli.auto_tone("/a/1.cr3")  # fake exits right after this answer
        first = cli._proc
        assert first is not None
        first.wait(timeout=10)
        gen = cli.generation
        assert cli.info("/a/1.cr3")["tool"] == "photo_info"
        assert cli.generation == gen + 1 and cli._proc is not first
    finally:
        cli.close()


def test_restart_every(tmp_path, monkeypatch):
    with make_cli(tmp_path, monkeypatch, restart_every=2) as cli:
        cli.open("/a")
        cli.open("/a")
        assert cli.generation == 1
        cli.open("/a")
        assert cli.generation == 2
        with cli.sequence():
            for _ in range(4):
                cli.open("/a")
        assert cli.generation == 2


def test_close(tmp_path, monkeypatch):
    cli = make_cli(tmp_path, monkeypatch)
    cli.open("/a")
    proc = cli._proc
    assert proc is not None and proc.poll() is None
    cli.close()
    assert proc.poll() is not None
    cli.close()  # idempotent
    assert cli.open("/a")["name"] == "a"  # restarts lazily
    cli.close()


def test_missing_binary(tmp_path, monkeypatch):
    cli = make_cli(tmp_path, monkeypatch)
    monkeypatch.setattr(cli, "_command", lambda: [str(tmp_path / "nope")])
    with pytest.raises(LumenError):
        cli.open("/a")


def test_lumen_lib_unavailable(tmp_path):
    lib = LumenLib(tmp_path)
    assert lib.available is False
    with pytest.raises(LumenError):
        lib.straighten(None)


def test_lumen_lib_wraps_modules(tmp_path, monkeypatch):
    import numpy as np

    pkg = tmp_path / "lumen"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    (pkg / "culling.py").write_text("def analyse(rgb):\n    return {'sharpness': 50.0}\n")
    (pkg / "straighten.py").write_text("def estimate(rgb):\n    return 1.5, 0.9\n")
    (pkg / "restoration.py").write_text(
        "def denoise(rgb, amount=70):\n    return rgb, 'x'\n\ndef super_resolution(rgb, scale=2):\n"
        "    import numpy as np\n    return np.repeat(np.repeat(rgb, scale, 0), scale, 1), 'x'\n"
    )
    (pkg / "merge.py").write_text(
        "SEEN = []\n\ndef merge_images(images, kind, reference=0):\n"
        "    SEEN.append((kind, reference, images[0].dtype))\n"
        "    return images[0], {}\n"
    )
    for name in [m for m in sys.modules if m == "lumen" or m.startswith("lumen.")]:
        monkeypatch.delitem(sys.modules, name)
    monkeypatch.setattr(sys, "path", list(sys.path))
    lib = LumenLib(tmp_path)
    assert lib.available
    img = np.full((4, 6, 3), 128, np.uint8)
    assert lib.analyse(img)["sharpness"] == 50.0
    assert lib.straighten(img) == (1.5, 0.9)
    assert lib.denoise(img).dtype == np.uint8 and lib.denoise(img).max() == 128
    assert lib.super_resolution(img, 2).shape == (8, 12, 3)
    out = lib.merge_images([img, img], "pano")
    assert out.dtype == np.uint8 and out.shape == img.shape
    from lumen import merge  # type: ignore[import-not-found]

    assert merge.SEEN == [("panorama", 0, np.float32)]
    for name in [m for m in sys.modules if m == "lumen" or m.startswith("lumen.")]:
        monkeypatch.delitem(sys.modules, name)


def test_apply_edits_passes_amount_and_label(tmp_path, monkeypatch):
    changes = {"adjustments": {"exposure": 0.3, "contrast": 10}}
    with make_cli(tmp_path, monkeypatch) as cli:
        default = cli.apply_edits("/a/1.cr3", changes)
        assert default["amount"] == 100 and default["changes"] == changes and default["label"] is None
        res = cli.apply_edits("/a/1.cr3", changes, amount=60, label="afterglow grade")
        assert res["amount"] == 60 and res["label"] == "afterglow grade" and res["changes"] == changes
        assert cli.apply_edits("/a/1.cr3", changes, amount=999)["amount"] == 150
        assert cli.apply_edits("/a/1.cr3", changes, amount=-5)["amount"] == 0


def test_get_edits_and_render_preview(tmp_path, monkeypatch):
    cli = make_cli(tmp_path, monkeypatch)
    try:
        assert cli.get_edits("/p/a.cr3")["state"]["adjustments"]["exposure"] == 0.5
        data = cli.render_preview("/p/a.cr3", 4096)
        assert data[:2] == b"\xff\xd8"
    finally:
        cli.close()

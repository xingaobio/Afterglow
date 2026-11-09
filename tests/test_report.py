from __future__ import annotations

import logging
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from afterglow import config as config_mod
from afterglow import report
from afterglow.config import Config
from afterglow.context import Context
from afterglow.state import State


def make_config(tmp_path: Path, **report_kw: Any) -> Config:
    src = tmp_path / "source"
    src.mkdir(parents=True, exist_ok=True)
    p = tmp_path / "afterglow.toml"
    p.write_text(
        f'[paths]\nsource = "{src}"\nwork = "{tmp_path / "work"}"\noutput = "{tmp_path / "out"}"\n'
        f'organized = "{tmp_path / "lib"}"\nbackup = ""\n',
        encoding="utf-8",
    )
    cfg = config_mod.load(p)
    return replace(cfg, report=replace(cfg.report, **report_kw))


def make_ctx(tmp_path: Path, **report_kw: Any) -> Context:
    cfg = make_config(tmp_path, **report_kw)
    return Context(config=cfg, state=State(cfg.paths.db), log=logging.getLogger("afterglow"))


def add(state: State, name: str) -> int:
    return state.upsert_file({"path": f"/src/{name}", "rel_path": name, "kind": "raw", "ext": "cr3"})


def build_run(ctx: Context) -> int:
    st = ctx.state
    old = add(st, "old.cr3")
    st.set_step(old, "group", "done")  # finished before this run starts
    time.sleep(1.1)
    run_id = st.start_run(["index", "cull", "group"])
    a, b, evil = add(st, "a.cr3"), add(st, "b.cr3"), add(st, "<script>x.jpg")
    st.set_step(a, "index", "done")
    st.set_step(b, "index", "done")
    st.set_step(evil, "index", "failed", error="boom <img src=x onerror=alert(1)>")
    st.set_step(a, "cull", "skipped")
    for f in (a, b):
        st.set_step(f, "group", "done")
    st.create_group("event", [a, b], name="苏格兰高地")
    st.create_group("event", [old, a], name="旧事件")  # contains an older member -> not new
    st.set_analysis(a, pick=1)
    st.set_analysis(b, reject_reason="blurry")
    st.add_output("export", "/out/a.jpg", file_id=a)
    st.add_output("export", "/out/b.jpg", file_id=b)
    st.add_output("crop", "/out/a_3x2.jpg", file_id=a)
    st.finish_run(run_id, ok=4, failed=1, skipped=1)
    return run_id


def test_run_summary(tmp_path: Path) -> None:
    ctx = make_ctx(tmp_path)
    run_id = build_run(ctx)
    s = ctx.state.run_summary(run_id)
    assert s["run"]["id"] == run_id and s["run"]["stages"] == ["index", "cull", "group"]
    assert (s["run"]["ok"], s["run"]["failed"], s["run"]["skipped"]) == (4, 1, 1)
    assert s["stages"]["index"] == {"done": 2, "failed": 1, "skipped": 0}
    assert s["stages"]["cull"]["skipped"] == 1
    assert s["stages"]["group"]["done"] == 2  # the 'old' file's step is outside the window
    assert s["failures"] == [
        {"path": "/src/<script>x.jpg", "stage": "index", "error": "boom <img src=x onerror=alert(1)>"}
    ]
    assert s["outputs"] == {"export": 2, "crop": 1}
    assert [e["name"] for e in s["new_events"]] == ["苏格兰高地"]
    assert s["library"]["files_total"] == 4
    assert s["rejects"] == {"blurry": 1}
    with pytest.raises(KeyError):
        ctx.state.run_summary(999)


def test_build_html_escapes_and_has_stage_table(tmp_path: Path) -> None:
    ctx = make_ctx(tmp_path)
    page = report.build_html(ctx.state.run_summary(build_run(ctx)), ctx.config)
    assert "<script>x.jpg" not in page and "&lt;script&gt;x.jpg" in page
    assert "<img src=x" not in page
    assert "<table>" in page and "<td>index</td>" in page and "阶段" in page and "苏格兰高地" in page
    assert "http://" not in page and "https://" not in page


def test_write_creates_report_and_latest(tmp_path: Path) -> None:
    ctx = make_ctx(tmp_path)
    run_id = build_run(ctx)
    path = report.write(ctx, run_id)
    assert path.parent == ctx.config.paths.reports
    assert path.name.startswith(f"run-{run_id}-") and path.suffix == ".html"
    latest = path.parent / "latest.html"
    assert latest.is_file() and not latest.is_symlink()
    assert latest.read_text(encoding="utf-8") == path.read_text(encoding="utf-8")


class FakeSMTP:
    instances: list[FakeSMTP] = []

    def __init__(self, host: str, port: int, timeout: float = 0) -> None:
        self.args = (host, port, timeout)
        self.calls: list[str] = []
        self.msg: Any = None
        FakeSMTP.instances.append(self)

    def __enter__(self) -> FakeSMTP:
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def starttls(self) -> None:
        self.calls.append("starttls")

    def login(self, user: str, password: str) -> None:
        self.calls.append(f"login:{user}:{password}")

    def send_message(self, msg: Any) -> None:
        self.calls.append("send")
        self.msg = msg


def email_ctx(tmp_path: Path) -> Context:
    return make_ctx(tmp_path, email_to="me@example.com", smtp_host="smtp.example.com", smtp_user="bot@example.com")


def test_notify_email(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    FakeSMTP.instances = []
    monkeypatch.setattr(report.smtplib, "SMTP", FakeSMTP)
    monkeypatch.setenv("AFTERGLOW_SMTP_PASSWORD", "pw")
    ctx = email_ctx(tmp_path)
    run_id = build_run(ctx)
    path = report.write(ctx, run_id)
    report.notify(ctx, path, ctx.state.run_summary(run_id))
    (smtp,) = FakeSMTP.instances
    assert smtp.args == ("smtp.example.com", 587, 30)
    assert smtp.calls == ["starttls", "login:bot@example.com:pw", "send"]
    assert smtp.msg["Subject"] == f"Afterglow 运行 #{run_id}：成功 4，失败 1"
    assert smtp.msg.get_body(("plain",)) is not None and smtp.msg.get_body(("html",)) is not None


def test_notify_email_missing_password(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    FakeSMTP.instances = []
    monkeypatch.setattr(report.smtplib, "SMTP", FakeSMTP)
    monkeypatch.delenv("AFTERGLOW_SMTP_PASSWORD", raising=False)
    ctx = email_ctx(tmp_path)
    run_id = build_run(ctx)
    path = report.write(ctx, run_id)
    with caplog.at_level(logging.WARNING, logger="afterglow.report"):
        report.notify(ctx, path, ctx.state.run_summary(run_id))
    assert FakeSMTP.instances == []
    assert any("AFTERGLOW_SMTP_PASSWORD" in r.getMessage() for r in caplog.records)


def test_notify_smtp_failure_is_swallowed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    def boom(*a: object, **k: object) -> None:
        raise OSError("down")

    monkeypatch.setattr(report.smtplib, "SMTP", boom)
    monkeypatch.setenv("AFTERGLOW_SMTP_PASSWORD", "pw")
    ctx = email_ctx(tmp_path)
    run_id = build_run(ctx)
    path = report.write(ctx, run_id)
    with caplog.at_level(logging.WARNING, logger="afterglow.report"):
        report.notify(ctx, path, ctx.state.run_summary(run_id))
    assert any("邮件通知失败" in r.getMessage() for r in caplog.records)


def test_notify_command_uses_argv_without_shell(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[Any, dict[str, Any]]] = []

    def fake_run(argv: Any, **kw: Any) -> None:
        calls.append((argv, kw))

    monkeypatch.setattr(report.subprocess, "run", fake_run)
    ctx = make_ctx(tmp_path, notify_command="notify-send --urgency=low 'Afterglow done'")
    run_id = build_run(ctx)
    path = report.write(ctx, run_id)
    report.notify(ctx, path, ctx.state.run_summary(run_id))
    (argv, kw), = calls
    assert argv == ["notify-send", "--urgency=low", "Afterglow done", str(path)]
    assert not kw.get("shell") and kw["timeout"] == 60


def test_finish_run_disabled_and_failure(tmp_path: Path) -> None:
    ctx = make_ctx(tmp_path, enabled=False)
    assert report.finish_run(ctx, build_run(ctx)) is None
    assert not ctx.config.paths.reports.exists()
    ctx2 = make_ctx(tmp_path / "b")
    assert report.finish_run(ctx2, 12345) is None  # unknown run: warning, no exception

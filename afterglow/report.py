"""Run report and notifications (ARCHITECTURE.md §11.3).

`finish_run` is the single entry point used by `afterglow run/retry/report` and `afterglow watch`; it never raises.
"""

from __future__ import annotations

import html
import logging
import os
import shlex
import shutil
import smtplib
import subprocess
from datetime import datetime
from email.message import EmailMessage
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from afterglow.config import Config
    from afterglow.context import Context

log = logging.getLogger("afterglow.report")

_CSS = """
body{font-family:-apple-system,"PingFang SC","Noto Sans CJK SC","Microsoft YaHei",sans-serif;margin:0;padding:12px;
background:#f4f4f6;color:#222;line-height:1.5}
.wrap{max-width:860px;margin:0 auto}
h1{font-size:1.35em;margin:.2em 0}h2{font-size:1.1em;margin:1.4em 0 .4em;border-left:4px solid #e08a2e;padding-left:8px}
.card{background:#fff;border-radius:8px;padding:10px 12px;box-shadow:0 1px 2px rgba(0,0,0,.08);overflow-x:auto}
table{border-collapse:collapse;width:100%;font-size:.92em}th,td{padding:5px 8px;border-bottom:1px solid #eee;
text-align:left;vertical-align:top}th{background:#fafafa}td.n,th.n{text-align:right;white-space:nowrap}
.ok{color:#1a7f37}.bad{color:#c62828;font-weight:600}.muted{color:#777;font-size:.9em}
code{word-break:break-all;font-size:.88em}
.kv span{display:inline-block;margin:2px 14px 2px 0}
"""

_REJECT_LABELS = {
    "blurry": "模糊", "eyes_closed": "闭眼", "duplicate": "重复", "burst_alt": "连拍备选", "tiny": "过小",
}
_GROUP_LABELS = {
    "burst": "连拍", "event": "事件", "hdr": "HDR", "focus": "景深堆栈", "pano": "全景", "timelapse": "延时",
    "duplicate": "重复",
}
_OUTPUT_LABELS = {
    "export": "导出", "enhance": "增强", "straighten": "水平校正", "crop": "裁切", "hdr": "HDR 合成",
    "focus": "景深合成", "pano": "全景", "proxy": "代理视频", "timelapse": "延时视频", "xmp": "XMP",
}


def _e(value: Any) -> str:
    return html.escape("" if value is None else str(value), quote=True)


def _fmt_bytes(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def _disk_rows(config: Config) -> list[tuple[str, str, str]]:
    rows: list[tuple[str, str, str]] = []
    seen: set[int] = set()
    for label, path in (("工作盘 work", config.paths.work), ("输出 output", config.paths.output),
                        ("整理库 organized", config.paths.organized)):
        try:
            if not Path(path).exists():
                continue
            usage = shutil.disk_usage(path)
        except OSError:
            continue
        try:
            dev = os.stat(path).st_dev
        except OSError:
            dev = id(path)
        note = "" if dev not in seen else "（与上同一磁盘）"
        seen.add(dev)
        pct = usage.used * 100 / usage.total if usage.total else 0
        rows.append((label + note, f"{path}", f"已用 {_fmt_bytes(usage.used)} / {_fmt_bytes(usage.total)}"
                                              f"（{pct:.0f}%），可用 {_fmt_bytes(usage.free)}"))
    return rows


def build_html(summary: dict[str, Any], config: Config) -> str:
    """Render the self-contained (inline CSS, no external resources) report page."""
    run = summary["run"]
    lib = summary.get("library", {})
    title = f"Afterglow 运行报告 #{run['id']}"
    status_cls = "bad" if run["failed"] or run.get("note", "").find("aborted") >= 0 else "ok"
    out: list[str] = [
        '<!DOCTYPE html><html lang="zh-CN"><head><meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width,initial-scale=1">',
        f"<title>{_e(title)}</title><style>{_CSS}</style></head><body><div class=\"wrap\">",
        f"<h1>{_e(title)}</h1>",
        f'<div class="card kv"><span>开始：{_e(run["started_at"])}</span>'
        f'<span>结束：{_e(run["finished_at"] or "未结束")}</span>'
        f'<span>阶段：{_e(", ".join(run["stages"]) or "-")}</span><br>'
        f'<span class="{status_cls}">成功 {int(run["ok"])}</span>'
        f'<span class="{status_cls}">失败 {int(run["failed"])}</span><span>跳过 {int(run["skipped"])}</span>',
    ]
    if run.get("note"):
        out.append(f'<br><span class="muted">备注：{_e(run["note"])}</span>')
    out.append("</div>")

    out.append('<h2>阶段</h2><div class="card"><table><tr><th>阶段</th><th class="n">完成</th>'
               '<th class="n">失败</th><th class="n">跳过</th></tr>')
    stages = summary.get("stages", {})
    for name, c in stages.items():
        failed_cls = ' class="n bad"' if c.get("failed") else ' class="n"'
        out.append(f'<tr><td>{_e(name)}</td><td class="n">{int(c.get("done", 0))}</td>'
                   f'<td{failed_cls}>{int(c.get("failed", 0))}</td><td class="n">{int(c.get("skipped", 0))}</td></tr>')
    if not stages:
        out.append('<tr><td colspan="4" class="muted">本次运行没有处理任何文件</td></tr>')
    out.append("</table></div>")

    rejects = summary.get("rejects", {})
    out.append('<h2>图库概览</h2><div class="card kv">')
    out.append(f'<span>文件总数 {int(lib.get("files_total", 0))}</span><span>精选 {int(lib.get("picks", 0))}</span>'
               f'<span>缺失 {int(lib.get("missing", 0))}</span>')
    if rejects:
        out.append("<br>淘汰：" + "".join(
            f"<span>{_e(_REJECT_LABELS.get(k, k))} {int(v)}</span>" for k, v in sorted(rejects.items())))
    groups = lib.get("groups", {})
    if groups:
        out.append("<br>分组：" + "".join(
            f"<span>{_e(_GROUP_LABELS.get(k, k))} {int(v)}</span>" for k, v in sorted(groups.items())))
    out.append("</div>")

    outputs = summary.get("outputs", {})
    out.append('<h2>本次新增产出</h2><div class="card kv">')
    if outputs:
        out.extend(f"<span>{_e(_OUTPUT_LABELS.get(k, k))} {int(v)}</span>" for k, v in sorted(outputs.items()))
    else:
        out.append('<span class="muted">无</span>')
    out.append("</div>")

    events = summary.get("new_events", [])
    out.append(f'<h2>新事件（{len(events)}）</h2><div class="card">')
    if events:
        out.append("<table><tr><th>事件</th><th>开始</th><th class=\"n\">照片数</th></tr>")
        for ev in events:
            name = ev.get("name") or f"事件 #{ev.get('id')}"
            out.append(f'<tr><td>{_e(name)}</td><td>{_e(ev.get("started_at") or "")}</td>'
                       f'<td class="n">{_e(ev.get("count", ""))}</td></tr>')
        out.append("</table>")
    else:
        out.append('<span class="muted">无</span>')
    out.append("</div>")

    failures = summary.get("failures", [])
    out.append(f'<h2>失败清单（{len(failures)}）</h2><div class="card">')
    if failures:
        out.append("<table><tr><th>文件</th><th>阶段</th><th>错误</th></tr>")
        for f in failures:
            out.append(f'<tr><td><code>{_e(f.get("path"))}</code></td><td>{_e(f.get("stage"))}</td>'
                       f'<td>{_e(f.get("error"))}</td></tr>')
        out.append("</table>")
        if len(failures) >= 200:
            out.append('<p class="muted">仅显示前 200 条。</p>')
    else:
        out.append('<span class="ok">没有失败</span>')
    out.append("</div>")

    out.append('<h2>磁盘占用</h2><div class="card">')
    disks = _disk_rows(config)
    if disks:
        out.append("<table><tr><th>位置</th><th>路径</th><th>使用情况</th></tr>")
        out.extend(f"<tr><td>{_e(a)}</td><td><code>{_e(b)}</code></td><td>{_e(c)}</td></tr>" for a, b, c in disks)
        out.append("</table>")
    else:
        out.append('<span class="muted">无可用路径</span>')
    out.append("</div>")
    out.append(f'<p class="muted">由 Afterglow 生成于 {_e(datetime.now().strftime("%Y-%m-%d %H:%M"))}</p>')
    out.append("</div></body></html>")
    return "".join(out)


def write(ctx: Context, run_id: int) -> Path:
    """Write reports/run-<id>-<YYYYMMDD-HHMM>.html and refresh latest.html (a copy, not a link)."""
    summary = ctx.state.run_summary(run_id)
    page = build_html(summary, ctx.config)
    folder = ctx.config.paths.reports
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"run-{run_id}-{datetime.now().strftime('%Y%m%d-%H%M')}.html"
    path.write_text(page, encoding="utf-8")
    shutil.copyfile(path, folder / "latest.html")
    return path


def _subject(summary: dict[str, Any]) -> str:
    run = summary["run"]
    return f"Afterglow 运行 #{run['id']}：成功 {run['ok']}，失败 {run['failed']}"


def _plain_text(summary: dict[str, Any], path: Path) -> str:
    run = summary["run"]
    lines = [_subject(summary), f"跳过 {run['skipped']}", f"报告文件：{path}"]
    for name, c in summary.get("stages", {}).items():
        lines.append(f"  {name}: 完成 {c.get('done', 0)} 失败 {c.get('failed', 0)} 跳过 {c.get('skipped', 0)}")
    return "\n".join(lines)


def _send_email(ctx: Context, path: Path, summary: dict[str, Any]) -> None:
    cfg = ctx.config.report
    password = None
    if cfg.smtp_user:
        password = os.environ.get(cfg.smtp_password_env)
        if not password:
            log.warning("未设置环境变量 %s，跳过邮件通知", cfg.smtp_password_env)
            return
    msg = EmailMessage()
    msg["Subject"] = _subject(summary)
    msg["From"] = cfg.smtp_user or cfg.email_to
    msg["To"] = cfg.email_to
    msg.set_content(_plain_text(summary, path))
    msg.add_alternative(path.read_text(encoding="utf-8"), subtype="html")
    with smtplib.SMTP(cfg.smtp_host, cfg.smtp_port, timeout=30) as smtp:
        smtp.starttls()
        if cfg.smtp_user:
            smtp.login(cfg.smtp_user, password or "")
        smtp.send_message(msg)


def notify(ctx: Context, path: Path, summary: dict[str, Any]) -> None:
    """E-mail and/or run `notify_command`; every failure is logged as a warning, never raised."""
    cfg = ctx.config.report
    if cfg.email_to and cfg.smtp_host:
        try:
            _send_email(ctx, path, summary)
        except Exception as exc:
            log.warning("邮件通知失败: %s: %s", type(exc).__name__, exc)
    if cfg.notify_command:
        try:
            argv = [*shlex.split(cfg.notify_command), str(path)]
            subprocess.run(argv, timeout=60, check=False, capture_output=True)
        except Exception as exc:
            log.warning("通知命令失败: %s: %s", type(exc).__name__, exc)


def finish_run(ctx: Context, run_id: int, do_notify: bool = True) -> Path | None:
    """Write the report (and notify) if enabled.  Never raises; returns the report path or None."""
    if not ctx.config.report.enabled:
        return None
    try:
        path = write(ctx, run_id)
    except Exception as exc:
        log.warning("生成报告失败: %s: %s", type(exc).__name__, exc)
        return None
    if do_notify:
        try:
            notify(ctx, path, ctx.state.run_summary(run_id))
        except Exception as exc:
            log.warning("通知失败: %s: %s", type(exc).__name__, exc)
    return path

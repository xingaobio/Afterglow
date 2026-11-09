"""Command line interface (ARCHITECTURE.md §4).  Exit codes: 0 ok, 1 config/args, 2 failures, 3 missing deps."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table

from afterglow import __version__
from afterglow import config as config_mod
from afterglow.config import Config, ConfigError
from afterglow.context import Context
from afterglow.stages.base import STAGE_ORDER
from afterglow.state import State
from afterglow.util import log as log_util

EXIT_OK, EXIT_CONFIG, EXIT_FAILED, EXIT_DEPS = 0, 1, 2, 3

# (label, argv probe, required)
_TOOLS: tuple[tuple[str, list[str], bool], ...] = (
    ("exiftool", ["exiftool", "-ver"], True),
    ("rsync", ["rsync", "--version"], True),
    ("ffmpeg", ["ffmpeg", "-version"], True),
    ("ffprobe", ["ffprobe", "-version"], True),
    ("nvidia-smi", ["nvidia-smi", "--version"], False),
)


def _console() -> Console:
    return Console(highlight=False)


def _build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", type=Path, default=argparse.SUPPRESS, help="配置文件路径")
    common.add_argument("-v", "--verbose", action="store_true", default=argparse.SUPPRESS, help="详细日志")

    parser = argparse.ArgumentParser(prog="afterglow", description="Afterglow: 无人值守、本地、可撤销的照片整理流水线")
    parser.add_argument("--config", type=Path, default=None, help="配置文件路径")
    parser.add_argument("-v", "--verbose", action="store_true", help="详细日志")
    parser.add_argument("--version", action="version", version=f"afterglow {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("setup", parents=[common], help="建目录、建库、检查依赖、创建 Immich 外部图库")

    p = sub.add_parser("run", parents=[common], help="运行流水线")
    p.add_argument("--stages", help=f"逗号分隔的阶段名，可选: {','.join(STAGE_ORDER)}")
    p.add_argument("--batch", type=int, help="每批文件数")
    p.add_argument("--dry-run", action="store_true", help="只模拟，不写任何文件")
    p.add_argument("--only", nargs="+", metavar="PATH", help="只处理这些路径（文件或目录）")

    sub.add_parser("watch", parents=[common], help="监听硬盘挂载并定时运行")
    sub.add_parser("status", parents=[common], help="查看各阶段进度")

    p = sub.add_parser("undo", parents=[common], help="撤销")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--run", type=int, dest="run_id", help="撤销指定 run")
    g.add_argument("--last", action="store_true", help="撤销最近一次有改动的 run（默认）")
    g.add_argument("--all", action="store_true", help="撤销全部")
    p.add_argument("--dry-run", action="store_true", help="只列出将要撤销的动作")

    p = sub.add_parser("retry", parents=[common], help="重试失败的步骤")
    p.add_argument("--stage", required=True, choices=STAGE_ORDER, help="阶段名")

    sub.add_parser("bench", parents=[common], help="模型基准测试（参数透传 scripts/bench_models.py）")
    sub.add_parser("doctor", parents=[common], help="依赖与服务健康检查")
    return parser


def _load_config(args: argparse.Namespace, require_source: bool) -> Config:
    cfg = config_mod.load(args.config)
    problems = cfg.validate(require_source=require_source)
    if problems:
        raise ConfigError("配置有误:\n  - " + "\n  - ".join(problems))
    return cfg


def _make_context(cfg: Config, args: argparse.Namespace, dry_run: bool = False) -> Context:
    logger = log_util.setup(cfg, verbose=getattr(args, "verbose", False))
    return Context(config=cfg, state=State(cfg.paths.db), log=logger, dry_run=dry_run)


# --------------------------------------------------------------------------- doctor
def doctor_checks(ctx: Context) -> list[dict[str, Any]]:
    """Run all health checks; each row has name, ok, required, detail."""
    rows: list[dict[str, Any]] = []
    cfg = ctx.config

    for label, argv, required in _TOOLS:
        found = shutil.which(argv[0])
        detail = "未找到"
        ok = False
        if found:
            try:
                proc = subprocess.run(argv, check=False, capture_output=True, text=True, timeout=20)
                ok = proc.returncode == 0
                text = (proc.stdout or proc.stderr).strip()
                detail = text.splitlines()[0][:80] if text else found
            except (OSError, subprocess.TimeoutExpired) as exc:
                detail = f"运行失败: {exc}"
        rows.append({"name": label, "ok": ok, "required": required, "detail": detail})

    lumen = cfg.lumen
    rows.append({"name": "lumen source_dir", "ok": lumen.source_dir.is_dir(), "required": False,
                 "detail": str(lumen.source_dir)})
    rows.append({"name": "lumen python", "ok": lumen.python.exists(), "required": False, "detail": str(lumen.python)})
    models = sorted(lumen.models_dir.glob("*.onnx")) if lumen.models_dir.is_dir() else []
    rows.append({"name": "lumen 模型 (*.onnx)", "ok": bool(models), "required": False,
                 "detail": f"{len(models)} 个 @ {lumen.models_dir}"})

    try:
        healthy = bool(ctx.llm.healthy())
        detail = cfg.llm.server
    except Exception as exc:
        healthy, detail = False, f"{type(exc).__name__}: {exc}"
    rows.append({"name": "llm 服务", "ok": healthy, "required": False, "detail": detail})

    if cfg.immich.enabled:
        try:
            client = ctx.immich
            alive = bool(client.ping()) if client is not None else False
            detail = cfg.immich.url
        except Exception as exc:
            alive, detail = False, f"{type(exc).__name__}: {exc}"
        rows.append({"name": "immich", "ok": alive, "required": False, "detail": detail})
    else:
        rows.append({"name": "immich", "ok": True, "required": False, "detail": "未启用 (api_key 为空)"})
    return rows


def _print_doctor(rows: list[dict[str, Any]]) -> None:
    table = Table(title="环境检查")
    for col in ("项目", "状态", "必需", "详情"):
        table.add_column(col)
    for r in rows:
        bad = "[red]缺失[/red]" if r["required"] else "[yellow]不可用[/yellow]"
        status = "[green]正常[/green]" if r["ok"] else bad
        table.add_row(r["name"], status, "是" if r["required"] else "否", r["detail"])
    _console().print(table)


def _doctor_exit(rows: list[dict[str, Any]]) -> int:
    return EXIT_DEPS if any(r["required"] and not r["ok"] for r in rows) else EXIT_OK


def cmd_doctor(cfg: Config, args: argparse.Namespace) -> int:
    ctx = _make_context(cfg, args)
    try:
        rows = doctor_checks(ctx)
    finally:
        ctx.close()
    _print_doctor(rows)
    code = _doctor_exit(rows)
    if code:
        _console().print("[red]缺少必需的外部工具，请先运行 deploy/install.sh[/red]")
    return code


# --------------------------------------------------------------------------- commands
def cmd_setup(cfg: Config, args: argparse.Namespace) -> int:
    p = cfg.paths
    for d in (p.work, p.output, p.organized, p.logs, p.previews, p.cache):
        d.mkdir(parents=True, exist_ok=True)
    ctx = _make_context(cfg, args)
    try:
        rows = doctor_checks(ctx)
        _print_doctor(rows)
        code = _doctor_exit(rows)
        if cfg.immich.enabled and not ctx.state.get("immich_library_id"):
            try:
                client = ctx.immich
                if client is None:
                    raise RuntimeError("Immich 客户端不可用")
                library_id = client.create_external_library(cfg.immich.external_library_name, [str(p.organized)])
                ctx.state.set("immich_library_id", library_id)
                _console().print(f"已创建 Immich 外部图库: {library_id}")
            except Exception as exc:
                ctx.log.exception("creating Immich library failed")
                _console().print(f"[red]创建 Immich 外部图库失败: {exc}[/red]")
                code = code or EXIT_DEPS
        _console().print("初始化完成。" if code == EXIT_OK else "初始化完成，但有问题需要处理。")
        return code
    finally:
        ctx.close()


def cmd_run(cfg: Config, args: argparse.Namespace) -> int:
    from afterglow import pipeline

    stages = [s.strip() for s in args.stages.split(",") if s.strip()] if args.stages else None
    ctx = _make_context(cfg, args, dry_run=args.dry_run)
    try:
        report = pipeline.run(ctx, stages=stages, batch_size=args.batch, only_paths=args.only)
    finally:
        ctx.close()
    _console().print(f"运行 #{report.run_id} 完成: 成功 {report.ok}，失败 {report.failed}，跳过 {report.skipped}")
    return EXIT_FAILED if report.failed else EXIT_OK


def cmd_watch(cfg: Config, args: argparse.Namespace) -> int:
    from afterglow import watch

    ctx = _make_context(cfg, args)
    try:
        return watch.watch(ctx)
    finally:
        ctx.close()


def cmd_status(cfg: Config, args: argparse.Namespace) -> int:
    state = State(cfg.paths.db)
    stats = state.stats()
    console = _console()

    steps = Table(title="各阶段进度")
    for col in ("阶段", "待处理", "完成", "失败", "跳过"):
        steps.add_column(col, justify="right" if col != "阶段" else "left")
    for stage in STAGE_ORDER:
        c = stats["steps"].get(stage, {})
        steps.add_row(stage, str(c.get("pending", 0)), str(c.get("done", 0)), str(c.get("failed", 0)),
                      str(c.get("skipped", 0)))
    console.print(steps)

    files = Table(title=f"文件 (共 {stats['files_total']}，精选 {stats['picks']}，缺失 {stats['missing']})")
    files.add_column("类型")
    files.add_column("数量", justify="right")
    for kind, n in sorted(stats["files"].items()):
        files.add_row(kind, str(n))
    console.print(files)

    groups = Table(title="分组")
    groups.add_column("类型")
    groups.add_column("数量", justify="right")
    for kind, n in sorted(stats["groups"].items()):
        groups.add_row(kind, str(n))
    console.print(groups)

    runs = Table(title="最近运行")
    for col in ("ID", "开始", "结束", "阶段", "成功", "失败", "跳过"):
        runs.add_column(col)
    for r in stats["runs"]:
        runs.add_row(str(r["id"]), r["started_at"], r["finished_at"] or "-", r["stages"], str(r["ok"]),
                     str(r["failed"]), str(r["skipped"]))
    console.print(runs)
    state.close()
    return EXIT_OK


def cmd_undo(cfg: Config, args: argparse.Namespace) -> int:
    ctx = _make_context(cfg, args, dry_run=args.dry_run)
    console = _console()
    try:
        if args.all:
            report = ctx.undo.undo(run_id=None, dry_run=args.dry_run)
        elif args.run_id is not None:
            report = ctx.undo.undo(run_id=args.run_id, dry_run=args.dry_run)
        else:
            report = ctx.undo.undo(last=True, dry_run=args.dry_run)
    finally:
        ctx.close()
    if args.dry_run:
        for line in report.planned:
            console.print(f"将撤销: {line}")
        console.print(f"共 {len(report.planned)} 项（dry-run，未做任何改动）")
    else:
        console.print(f"已撤销 {report.undone} 项，无需处理 {report.skipped} 项，出错 {len(report.errors)} 项")
    for err in report.errors:
        console.print(f"[red]{err}[/red]", markup=True)
    return EXIT_FAILED if report.errors else EXIT_OK


def cmd_retry(cfg: Config, args: argparse.Namespace) -> int:
    from afterglow import pipeline

    ctx = _make_context(cfg, args)
    try:
        n = ctx.state.reset_stage(args.stage, only_failed=True)
        _console().print(f"已重置 {n} 个失败步骤 ({args.stage})")
        report = pipeline.run(ctx, stages=[args.stage])
    finally:
        ctx.close()
    _console().print(f"运行 #{report.run_id} 完成: 成功 {report.ok}，失败 {report.failed}，跳过 {report.skipped}")
    return EXIT_FAILED if report.failed else EXIT_OK


def cmd_bench(rest: list[str]) -> int:
    script = Path(__file__).resolve().parent.parent / "scripts" / "bench_models.py"
    if not script.is_file():
        _console().print(f"[red]找不到基准脚本: {script}[/red]")
        return EXIT_DEPS
    proc = subprocess.run([sys.executable, str(script), *rest], check=False)
    return proc.returncode


_HANDLERS = {
    "setup": cmd_setup, "run": cmd_run, "watch": cmd_watch, "status": cmd_status,
    "undo": cmd_undo, "retry": cmd_retry, "doctor": cmd_doctor,
}
# commands that work without the source directory being present
_NO_SOURCE = {"setup", "status", "undo", "doctor", "retry"}


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    try:
        args, rest = parser.parse_known_args(argv)
    except SystemExit as exc:
        return EXIT_OK if exc.code in (0, None) else EXIT_CONFIG
    if rest and args.command != "bench":
        parser.print_usage(sys.stderr)
        print(f"afterglow: 无法识别的参数: {' '.join(rest)}", file=sys.stderr)
        return EXIT_CONFIG
    try:
        if args.command == "bench":
            return cmd_bench(rest)
        cfg = _load_config(args, require_source=args.command not in _NO_SOURCE)
        return _HANDLERS[args.command](cfg, args)
    except ConfigError as exc:
        print(f"配置错误: {exc}", file=sys.stderr)
        return EXIT_CONFIG
    except ValueError as exc:
        print(f"参数错误: {exc}", file=sys.stderr)
        return EXIT_CONFIG
    except KeyboardInterrupt:
        print("已中断", file=sys.stderr)
        return 130


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())

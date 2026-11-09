# 参与贡献 / Contributing

Afterglow 采用 **契约优先**：[ARCHITECTURE.md](ARCHITECTURE.md) 是唯一事实来源（目录结构、数据表、接口签名、阶段行为）。

## 流程

1. **先改 ARCHITECTURE.md，再写代码。** 想改表结构、接口签名、阶段行为或配置键，先在 ARCHITECTURE.md 里改并说明原因，再改实现；代码和文档不一致时，以文档为准，不一致本身就是 bug。
2. 保持三条铁律：原图只读；改变磁盘/Immich 状态的动作先写 `undo_log`；每个文件每个阶段的状态落在 `steps` 表，可断点续跑。
3. 所有 SQL 只写在 `afterglow/state.py`；子进程用 `subprocess.run([...], check=False, capture_output=True, text=True, timeout=...)`，不用 `shell=True`；不吞异常（`log.exception` 或转成 `StepOutcome(failed, error=...)`）。

## 质量闸门（合并前必须全绿）

```bash
python3.12 -m venv .venv && . .venv/bin/activate
pip install -e '.[dev]'
ruff check .          # line-length 120，select E,F,I,B,UP，零告警
pytest -q             # 必须在无 GPU、无 exiftool、无网络的环境通过
bash -n deploy/install.sh deploy/lumen/setup-lumen.sh deploy/models/download.sh   # 改了脚本时
```

## 文件归属

按 ARCHITECTURE.md §1 的目录划分；一次改动尽量只落在一个归属内，跨界改动要在 PR 里点明。

| 区域 | 内容 |
|---|---|
| 核心 | `afterglow/config.py`、`state.py`、`context.py`、`undo.py`、`gpu_lock.py`、`pipeline.py`、`watch.py`、`cli.py` |
| 工具 | `afterglow/util/*`（hashing、exif、images、fs、log） |
| 集成 | `afterglow/integrations/lumen.py`、`llm.py`、`immich.py` |
| 阶段 | `afterglow/stages/*`（base 与九个阶段） |
| 部署与文档 | `deploy/**`（install.sh、lumen、models、immich、systemd）、`scripts/*`、`README.md`、`CONTRIBUTING.md` |
| 测试 | `tests/*`（与被测模块一一对应：`test_stage_cull.py` 测 `stages/cull.py`，依此类推） |
| 契约 | `ARCHITECTURE.md`、`config/afterglow.example.toml`（键与默认值必须与 `config.py` 一致） |

## 测试里怎么用假的 LUMEN

测试不启动真正的 LUMEN。[`tests/fake_lumen_mcp.py`](tests/fake_lumen_mcp.py) 是一个假的 MCP 子进程：从 stdin 读“每行一个 JSON”的 JSON-RPC 请求，回固定结果。测试把 `LumenCLI._command` 替换成启动它的命令（`tests/test_lumen_cli.py` 里的 `make_cli`）：

```python
cli = LumenCLI(Lumen(python=Path(sys.executable), source_dir=..., models_dir=..., library_db=...), timeout=30)
monkeypatch.setattr(cli, "_command", lambda: [sys.executable, str(FAKE)])
monkeypatch.setenv("FAKE_LUMEN_FAIL_AFTER", "0")   # 非 0 = 第 N 次调用后假进程退出，用来测自动重启
with cli:
    cli.analyze(["/a/1.cr3"])
```

阶段测试通常直接用 `conftest.py` 里的 fixture 提供的假 `ctx.lumen`。需要新的假工具返回值时，改 `fake_lumen_mcp.py`，并在 ARCHITECTURE.md §6 里同步协议约定。

外部命令（`exiftool`、`rsync`、`ffmpeg`、`rawpy`）一律用 `monkeypatch` / `unittest.mock` 替换；HTTP（llama-server、Immich）用 mock transport。`conftest.py` 用 Pillow/NumPy 现场生成清晰图、模糊图、倾斜地平线、连拍和 RAW/JPEG 配对，不提交二进制样张。

## 部署脚本

`deploy/` 下的脚本必须幂等、`set -euo pipefail`、通过 `bash -n`。外部 URL、文件名、命令行标志必须来自你实际打开过的官方页面，并在注释里写出来源；查不到就写成 `TODO`/警告，不要猜。

---

*English:* contract first — edit `ARCHITECTURE.md` before code; keep `ruff check .` and `pytest -q` green (no GPU/network needed); use `tests/fake_lumen_mcp.py` to test the LUMEN protocol; deployment scripts must be idempotent, pass `bash -n`, and cite the official source for every external URL/flag.

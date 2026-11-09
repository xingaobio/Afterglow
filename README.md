# Afterglow

**Afterglow 是一条无人值守、本地运行、可撤销的照片整理流水线：把移动硬盘上约 1 TB 的 RAW / JPEG / 手机照片 / 视频，自动备份、建索引、选片、分组、用本地视觉大模型打标签、自动调色导出、合成 HDR / 景深 / 全景、生成视频代理，最后整理成目录树并同步到 Immich。**

核心图像算法来自 [LUMEN RAW](https://github.com/hety0211/lumen-raw)（MIT），通过它的 `lumen-cli` 以黑盒方式调用；视觉模型由 [llama.cpp](https://github.com/ggml-org/llama.cpp) 的 `llama-server` 在本机提供。

> 状态：0.1.0，开发中。本文只描述 [ARCHITECTURE.md](ARCHITECTURE.md) 已规定的行为；拿不准的地方会明说，而不是猜。

## 三条铁律

1. **原图只读。** 任何阶段都不会写入、改名、删除 `paths.source` 下的文件；所有输出写到 `paths.output` / `paths.organized`。
2. **一切可撤销。** 每个改变磁盘或 Immich 状态的动作先写入 `undo_log` 再执行，`afterglow undo` 按日志逆序还原。
3. **断点续跑。** 每个文件、每个阶段的状态记在 SQLite 的 `steps` 表；重跑只处理 `pending` / `failed`，崩溃不会影响已完成的记录。

## 架构

```
 移动硬盘 (paths.source, 只读)
        │
        ▼
 ┌─────────────────────────── afterglow run (pipeline.py, SQLite: afterglow.db) ───────────────────────────┐
 │ ① backup ─▶ ② index ─▶ ③ cull ─▶ ④ group ─▶ ⑤ understand ─▶ ⑥ process ─▶ ⑦ merge ─▶ ⑧ video ─▶ ⑨ publish │
 │   rsync      哈希/EXIF   清晰度     事件/HDR    VLM 标签/     调色导出/     HDR/景深/    代理/延时    XMP/目录树/ │
 │   +校验      /配对/预览   闭眼/连拍   景深/全景   描述/事件名    校正/裁切/    全景合成      视频        Immich     │
 │                          /精选                                增强                                       │
 └───────┬──────────────────────┬───────────────────────┬───────────────────────────────────┬────────────────┘
         │                      │ (GPU 阶段持 gpu.lock)  │                                   │
         ▼                      ▼                       ▼                                   ▼
   备份盘 (rsync)         LUMEN RAW                llama-server (OpenAI 兼容)        Immich (docker compose)
                       lumen-cli mcp (stdio)       :8080 Qwen3-VL-4B  (afterglow-llm)    外部图库 = paths.organized (只读挂载)
                       cull / 调色 / 导出 /         :8081 Qwen3.6-35B-A3B (可选,         相册 / 标签 / 描述 由 publish 写回
                       合成 / 延时                          afterglow-brain，与 :8080 互斥)  ML 任务在 GPU 阶段暂停
                       (AI 增强: 可选 LumenLib 直接 import)
```

## 硬件与系统要求

| 项目 | 目标配置（ARCHITECTURE.md §0） |
|---|---|
| 系统 | Ubuntu 24.04 LTS，Python 3.12（Windows / macOS 不是目标，但代码不刻意排斥） |
| GPU | NVIDIA RTX 2000 级别，8 GB 显存；需要可用的官方驱动（`nvidia-smi`） |
| 内存 | 32 GB（35B MoE 的专家权重放在内存里，见下文“模型选择”） |
| 磁盘 | 1 TB SSD 工作盘（`paths.work`）；另有源盘与可选的备份盘 |

## 安装

```bash
git clone <本仓库> afterglow && cd afterglow
deploy/install.sh                 # 视觉模型 4B
deploy/install.sh --with-brain    # 同时下载 35B MoE（约 23 GB）
deploy/install.sh --skip-immich   # 不装 Docker / NVIDIA Container Toolkit / Immich
```

用普通用户运行（脚本在需要时调用 `sudo`），可重复执行。没有 `nvidia-smi` 时脚本以退出码 **3** 结束并提示 `sudo ubuntu-drivers install`。步骤：

| # | 做什么 | 脚本 |
|---|---|---|
| 0 | 前置检查（非 root、apt、空间） | `install.sh` |
| 1 | apt：exiftool、rsync、ffmpeg、git、build-essential、cmake、python3.12(-venv)、curl、jq 等 | `install.sh` |
| 2 | 检测 NVIDIA 驱动 | `install.sh` |
| 3 | Docker Engine + compose 插件（[官方 apt 源](https://docs.docker.com/engine/install/ubuntu/)，不是 Docker Desktop）+ [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) | `install.sh` |
| 4 | LUMEN RAW：clone 到 `/opt/lumen-raw` 并固定到 `models.lock` 里的提交，建 `.venv-linux`，装 `requirements-linux.txt` + `requirements-linux-gpu.txt`，下载运行时素材和 AI 模型 | `deploy/lumen/setup-lumen.sh` |
| 5 | 编译 llama.cpp（`-DGGML_CUDA=ON`）到 `/opt/llama.cpp`，下载 GGUF 并校验 sha256 | `deploy/models/download.sh` |
| 6 | Afterglow 本体：`/opt/afterglow/.venv` + `pip install -e .`，创建 `~/.config/afterglow/afterglow.toml` | `install.sh` |
| 7 | Immich：复制 `deploy/immich/*` 到 `/opt/immich`，生成 `.env`（随机数据库密码），`docker compose up -d` | `install.sh` |
| 8 | systemd 单元（llm、可选 brain、watch、nightly 定时器、`afterglow-run@`）和 udev 规则 | `deploy/systemd/` |
| 9 | 打印下一步 | `install.sh` |

**预计时间与磁盘**（粗略估计，未在目标机器上实测，受网速影响很大）：

- 时间：下载约 4 GB（`--with-brain` 约 28 GB）+ LUMEN 素材（其 README 称约 830 MB）+ Python / CUDA 轮子 + Docker 镜像；CUDA 编译 llama.cpp 通常是十几到几十分钟。总体按“半小时到数小时”预期。
- 磁盘：GGUF 约 3.3 GB（4B + mmproj）/ 约 23 GB（35B + mmproj）；LUMEN 素材与 venv、llama.cpp 构建、Docker 镜像各数 GB。**真正大头是流水线输出**（预览、导出、增强、合成、代理视频），请给工作盘留足空间。

### 模型文件（来源见 `deploy/models/models.lock`）

| 角色 | 文件 | 仓库 |
|---|---|---|
| 视觉 LLM | `Qwen3VL-4B-Instruct-Q4_K_M.gguf`（2.5 GB） | [Qwen/Qwen3-VL-4B-Instruct-GGUF](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct-GGUF) |
| 视觉 mmproj | `mmproj-Qwen3VL-4B-Instruct-F16.gguf`（836 MB） | 同上 |
| brain LLM（可选） | `Qwen3.6-35B-A3B-UD-Q4_K_M.gguf`（22.1 GB） | [unsloth/Qwen3.6-35B-A3B-GGUF](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF) |
| brain mmproj（可选） | `mmproj-F16.gguf`（899 MB，本地改名为 `mmproj-Qwen3.6-35B-A3B-F16.gguf`） | 同上 |

`models.lock` 里的 `sha256` 初始为 `TODO-verify-on-first-download`（写文件时没能取得官方发布的校验值）：`download.sh` 首次下载后会计算并打印 sha256、写回 `models.lock`，之后每次运行都会校验。请把写回的值和 Hugging Face 文件页上的 LFS SHA256 对一下。

## 配置

复制 [`config/afterglow.example.toml`](config/afterglow.example.toml) 到 `~/.config/afterglow/afterglow.toml`（`install.sh` 已替你做了；也可用 `--config PATH`，或放在 `./afterglow.toml`）。每个键在示例文件里都有注释，最需要先改的：

| 键 | 含义 |
|---|---|
| `paths.source` | 移动硬盘挂载点，**只读** |
| `paths.work` / `output` / `organized` | SSD 上的工作目录（数据库、预览、日志）/ 导出物 / 整理后的目录树（Immich 外部图库指向 `organized`） |
| `paths.backup` | 备份盘挂载点；空字符串 = 不备份 |
| `organize.mode` | `copy_picks`（默认：精选复制到 SSD，其余建链接）\| `symlink` \| `hardlink` \| `copy` \| `move`；`move` 要求先通过备份校验（`backup.require_before_move`） |
| `organize.layout` | 目录树模板，默认 `{year}/{year}-{month:02d} {event}/{basename}` |
| `cull.*` | 清晰度阈值、连拍间隔、是否淘汰闭眼等 |
| `llm.server` / `vision_model` / `brain_server` / `brain_model` | llama-server 地址与模型名；`vision_model` / `brain_model` 须与 systemd 单元里的 `--alias` 一致；`brain_server` 留空 = 事件命名也用视觉模型 |
| `lumen.*` | LUMEN 安装位置、模型目录、`compute`、导出质量、AI 增强策略 |
| `process.*` | 水平校正、裁切比例、场景→预设映射 |
| `immich.url` / `api_key` | `api_key` 为空 = `publish` 跳过 Immich 部分 |
| `run.batch_size` / `workers` | 每批文件数 / CPU 并行数 |

## 运行

```bash
afterglow doctor            # 依赖、GPU、模型、服务健康检查
afterglow setup             # 建目录、建库、检查连通性、创建 Immich 外部图库
afterglow run --dry-run     # 只模拟：打印将要做的事，不写任何文件
afterglow run               # 正式运行（可加 --stages index,cull  --batch N  --only PATH...）
afterglow watch             # 挂载出现 / 定时 → 增量运行
afterglow status            # 各阶段 pending / done / failed 计数、最近 runs、磁盘占用
afterglow undo --last       # 撤销最近一次（也可 --run ID / --all；先加 --dry-run 看清单）
afterglow retry --stage understand   # 把该阶段 failed 的文件重置后重跑
afterglow bench --sample 200 ...     # 模型基准，参数透传 scripts/bench_models.py
```

退出码：0 成功；1 配置/参数错误；2 运行中有 failed 文件；3 外部依赖不可用。

自动触发（由 `install.sh` 装好，细节见 [`deploy/systemd/README.md`](deploy/systemd/README.md)）：`afterglow-nightly.timer` 每天 03:17 增量运行；udev 规则在 USB 盘插入时启动 `afterglow-run@<卷标>.service`；`afterglow-watch.service` 默认不启用（`install.sh --enable-watch`）。

## 预计耗时

**1 TB 素材大约需要连续运行 3–5 天。** 这是设计目标的量级估计，不是在目标硬件上的实测数；实际取决于 RAW 与视频的比例、`tag_scope`、`enhance` 等设置。各阶段的主要瓶颈：

| 阶段 | 瓶颈 |
|---|---|
| ① backup | 磁盘 / USB 带宽（rsync） |
| ② index | 全量读盘做 sha256 + 预览解码，CPU 与磁盘 |
| ③ cull | LUMEN 分析（GPU 可选） |
| ④ group | 很快（只查数据库） |
| ⑤ understand | **GPU，视觉模型逐张推理**；`tag_scope=picks` 只处理精选，可大幅减量 |
| ⑥ process | GPU，LUMEN 调色导出 + AI 增强；`enhance=auto` 只增强高 ISO / 低分辨率的精选 |
| ⑦ merge / ⑧ video | 合成与 NVENC 转码，视组数 / 视频量而定 |
| ⑨ publish | 通常很快；Immich 扫描与缩略图生成另算（最多等 30 分钟） |

想知道自己机器上的数字，用 `afterglow bench` 先测视觉模型的秒/张，再乘以精选张数估算 ⑤。流水线可随时中断，下次接着跑。

## 模型选择

- **Qwen3-VL-4B（主力）**：约 2.5 GB 权重，8 GB 显存可全部放进 GPU，速度够用，负责逐张打标签和描述。
- **Qwen3.6-35B-A3B（MoE，可选）**：只用来给**事件命名**（调用量小）。权重约 22 GB，靠 `--n-cpu-moe 99` 把专家权重留在内存，速度慢但 8 GB 显存也能跑；与 4B 服务互斥。
- **Gemma 4 12B**：在 8 GB 显存上偏紧（权重加 mmproj 与 KV 缓存很难都放进 GPU）；这是设计阶段的判断，本仓库没有实测，也没有提供下载项。

想自己比较：`python scripts/bench_models.py --source DIR --sample 200 --models 名称=http://主机:端口/v1,...`，输出秒/张（中位数、P90）、JSON 解析成功率、模型两两之间的标签 Jaccard 一致度，并写入 `bench_results.json`。8 GB 显存放不下两个模型，请逐个启动对应服务后分别测（脚本按给出的端点依次调用）。

## 与 LUMEN RAW 的关系

- **黑盒调用**：`LumenCLI` 启动 `python main.py --cli mcp --headless` 子进程，用 stdio 上“每行一个 JSON”的 JSON-RPC 2.0 通信，调用 `analyze_photos`、`open_photo`、`auto_tone`、`run_command`（`edit.preset`、`photo.export`、`video.timelapse` …）、`export_photos` 等。每处理 `lumen.restart_every` 张重启一次子进程，防内存增长。
- **可选直接 import**：`LumenLib` 把 `source_dir` 加入 `sys.path` 后直接 import LUMEN 的 culling / straighten / restoration / merge 模块；失败则 `available=False`，相关功能降级或 `skipped`。
- **许可证**：LUMEN RAW 以 MIT 发布；它的第三方模型各有各的许可证（见其 `THIRD_PARTY.md`、`MODEL.md`）。Afterglow 不修改、不再分发 LUMEN，`setup-lumen.sh` 只是把它 clone 到 `/opt/lumen-raw` 并固定到 `models.lock` 记录的提交 `f752bb40127d…`。
- **LUMEN 没有暴露、所以 Afterglow 自己实现的**：
  - **几何校正与裁切**：LUMEN 的 `edit.apply` 不支持 straighten / crop / rotation，水平校正（LSD 线段投票估角度、旋转并裁掉黑边）和按比例裁切（显著性中心）由 Afterglow 在导出图上完成。
  - **AI 增强（降噪 / 超分）只能走 `LumenLib`**：LUMEN 没有对应的 CLI 命令；`LumenLib` 不可用时该步 `skipped` 并告警。
- 安装前提：LUMEN 的 Linux 版从源码运行；其 README 说明这个 Linux 移植由社区贡献，作者本人未在 Linux 硬件上跑过——遇到问题请先看 LUMEN 的 issue。

## 与 Immich 的关系

- **外部图库、只读**：Immich 容器以 `:ro` 挂载 `paths.organized`（容器内 `/mnt/afterglow`）和 `paths.source`（`/mnt/source`，另以宿主机同一路径再挂一份，让 `symlink` 模式的链接在容器里不悬空）。Immich 永远不会写这些目录。`afterglow setup` 创建外部图库，import path 指向整理后的目录树。
- **写回**：`publish` 把每个事件建成相册（`YYYY-MM 事件名`），给精选照片写 description（caption）和层级标签 `Afterglow/<tag>`。每个 Immich 写操作先写 `undo_log`，`afterglow undo` 能删相册、还原描述、取消标签。
- **ML 任务暂停**：8 GB 显存要让给视觉模型，所以 GPU 阶段持 `gpu.lock_file` 文件锁，并（`immich.pause_ml_during_gpu_stages = true`）暂停 Immich 的 `smartSearch` 与 `faceDetection` 任务，结束后恢复。Immich 的 ML 容器用 `-cuda` 镜像并预留 GPU，见 [`deploy/immich/docker-compose.yml`](deploy/immich/docker-compose.yml)。
- **注意**：`deploy/immich/` 里的 compose 是在**没能下载到当前官方 compose 文件**的情况下写的——`install.sh` 会把官方 `docker-compose.yml` / `example.env` / `hwaccel.ml.yml` 另存到 `/opt/immich/official/` 并从中提取数据库与 valkey 的镜像标签；升级 Immich 前请先 diff。

## 目录树与撤销

```
paths.source/            原图，只读，永不被修改
paths.work/              afterglow.db  previews/  cache/  logs/  models/
paths.output/            exports/  crops/  enhanced/  merged/  proxies/  xmp/
paths.organized/         year/year-MM 事件名/…   精选（symlink / hardlink / copy / move，按 organize.mode）
                         _culled/                非最佳帧、模糊、闭眼等
                         _duplicates/            完全重复（sha256 相同）
```

XMP 附属文件写在 `output/xmp/` 下，**不会**写到原图旁边。`organize.mode = "move"` 只会在备份校验通过（`kv.last_backup_ok` 在本次运行前 24 小时内）后执行，否则降级为 `symlink` 并告警。

撤销：`afterglow undo --last --dry-run` 先看清单，再去掉 `--dry-run`。按 `undo_log` 倒序：`move` 移回；`symlink` / `hardlink` / `copy` / `write_file` 删除目标；`mkdir` 仅在目录为空时删除；`immich_*` 调用对应的删除 / 还原。每条成功后标记 `undone`。原图不会被改动，所以最坏情况下也只需删掉输出目录重来。

## v0.2 新增

- **精选复制到 SSD（默认）**：`organize.mode = "copy_picks"`。精选照片（连同调色导出图）在 SSD 上有实体副本，拔掉移动硬盘也能看；淘汰和重复的照片只建符号链接。原图始终不动。
- **人物相册**（阶段 `people`）：读取 Immich 的人脸识别结果，给每个人建一个相册，放入其质量分最高的若干张精选（`[people]`）。先在 Immich 网页里给人物起名字；没起名的默认跳过。人脸识别要在图库扫描后跑一段时间，第一次运行可能还没有人物，下次运行会补上。
- **自然语言调色**（`[grade]`）：在 LUMEN 自动调色之后，把 LUMEN 渲染出的效果图交给视觉模型，按你写的 `instruction`（例如“自然、通透，肤色准确”）给出小幅修正，再叠加到当前滑块值上。`strength` 控制应用多少；`mode = "auto"` 关闭这一步。每张图多一次模型调用，`process` 阶段会明显变慢。
- **运行报告与通知**（`[report]`）：每次运行结束在 `work/reports/` 生成一页 HTML（`latest.html` 始终是最新一份）：各阶段成功/失败数、精选与淘汰统计、新事件、失败清单、磁盘占用。可选发邮件（SMTP 密码从环境变量读）或调用自定义命令。`afterglow report` 可重新生成。
- **淘汰照片不进 Immich**：外部图库创建时排除 `_culled/` 和 `_duplicates/`，Immich 里只看到精选；这些文件仍在目录树里，随时可以翻回来。
- **SSD 空间提醒**：`copy_picks` 会把精选原图复制一份。如果精选占 1 TB 素材的三成，就是约 300 GB，再加上导出图、预览缓存和 Immich 缩略图，1 TB SSD 会比较紧。空间不够时改回 `symlink`。
- **Immich 只在本机可见**：端口绑定 `127.0.0.1:2283`。想让家里其他设备访问，改 `deploy/immich/docker-compose.yml` 里标注 `AFTERGLOW [6]` 的那一行。

## 尚未在真机验证的内容（首次部署请留意）

本仓库的代码在沙箱里只能对着假的 LUMEN / LLM / Immich 测试（186 个单元测试通过，合成照片的端到端冒烟测试通过，`undo` 验证过原图字节不变）。以下内容**没有**在真实机器上跑过，第一次部署时请对照检查：

1. **LUMEN RAW 从未被实际执行**：`LumenCLI` 的协议按 LUMEN 源码（`lumen/mcp.py`、`lumen/commands.py`）实现，`analyze_photos` 返回字段与 `lumen/commands.py` 的 `_catalog_photo` 一致，但请先用 `afterglow run --stages cull --only 某张照片` 验证。
2. **Immich API 端点**：客户端按 v1.1xx–v2 的 `/api/jobs`、`/api/search/metadata`、`/api/albums`、`/api/tags` 写成，并对 v3 可能的 `/api/queues` 做了回退；`deploy/immich/docker-compose.yml` 没有和官方最新文件逐行比对。`install.sh` 会把官方 compose 下载到 `/opt/immich/official/` 供对照。
3. **模型校验和**：`deploy/models/models.lock` 里四个 GGUF 的 sha256 是 `TODO-verify-on-first-download`，`download.sh` 首次下载后会写回，之后每次校验。
4. **systemd / udev**：单元文件和 udev 规则只做了语法检查（`systemd-analyze verify`），没有在真机触发过。
5. **Immich 是否跟随符号链接**：默认的 `copy_picks` 模式下精选是实体文件，不受影响；淘汰/重复照片和 `symlink` 模式用的是符号链接，compose 已把原图目录、整理目录、导出目录按宿主机相同路径只读挂进容器，链接在容器内不会悬空；但 Immich 外部图库对符号链接的扫描行为没有实测。若这些链接在 Immich 里看不到，把 `organize.mode` 改成 `copy`（多占一份空间）。
6. **35B“大脑”模型不会自动切换**：`afterglow-brain.service` 与 `afterglow-llm.service` 互斥，目前需要手动 `systemctl start`；`llm.brain_server` 留空时事件命名直接用 4B 模型。
7. **导出图不在撤销范围内**：`undo` 还原目录树、XMP 和 Immich 的改动；`output/` 下的导出、裁切、增强文件是可再生的产物，需要时直接删除该目录。
8. **人物相册与自然语言调色**：Immich 的人物接口（`/api/people`、按 `personIds` 搜索）和 LUMEN 的 `get_edits` / `render_preview` / `apply_edits` 调用都只对着假服务测过。建议先用 `afterglow run --stages process --only 一张照片` 看 `output/exports/` 的效果，再决定 `grade.strength`。
9. **RTX 2000 的实际速度**：所有耗时都是估算，请用 `afterglow bench` 实测。

## 常见问题

**`install.sh` 以退出码 3 结束。** 没有可用的 NVIDIA 驱动：`sudo ubuntu-drivers install && sudo reboot`，再重跑。

**`docker` 提示权限不足。** 脚本把你加入了 `docker` 组，重新登录后生效。

**Immich 扫描到 0 个文件。** 多半是 `organize.mode = "symlink"` 时链接指向宿主机路径，而容器里没有这个路径。默认 compose 已按同一路径只读再挂了一份 `PHOTO_SOURCE`；如果源盘路径变了，改 `/opt/immich/.env` 里的 `PHOTO_SOURCE` 并 `docker compose up -d`。也可以改用 `hardlink`（同分区）或 `copy`。

**`afterglow-llm.service` 起不来。** `journalctl -u afterglow-llm -e`。常见原因：新版 llama.cpp 改了二进制名或参数（`download.sh` 编译后检查 `build/bin/llama-server`，不存在会警告）；模型文件名与单元里不一致；显存被别的进程占用（brain 与 llm 互斥）。

**`models.lock` 里 sha256 是 TODO。** 首次下载后会自动写入；请与 Hugging Face 页面上的 LFS SHA256 对照确认。

**想换更多/更少的处理。** `llm.tag_scope = "all"` 让所有照片都打标签（慢）；`process.crops = []` 不裁切；`lumen.enhance = "off"` 关闭 AI 增强。

**Windows / macOS 能用吗？** 不是目标平台，也没有测试过。

## 许可证

Afterglow：MIT，见 [LICENSE](LICENSE)。第三方组件各有许可证：LUMEN RAW（MIT，其内置模型另有许可证）、Immich（AGPL-3.0）、llama.cpp（MIT）、Qwen3-VL 与 Qwen3.6 模型（见各自 Hugging Face 模型卡）。使用前请自行核对。

## 致谢

- [LUMEN RAW](https://github.com/hety0211/lumen-raw)：RAW 解码、调色、分析、合成与延时的核心算法（含其集成的 Real-ESRGAN 等模型）
- [Immich](https://immich.app)：自托管照片库
- [llama.cpp](https://github.com/ggml-org/llama.cpp)：本地推理
- [Qwen](https://huggingface.co/Qwen) 团队的 Qwen3-VL / Qwen3.6 模型，以及 [unsloth](https://huggingface.co/unsloth) 提供的 GGUF 量化
- [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN)、[ExifTool](https://exiftool.org)、[FFmpeg](https://ffmpeg.org)、[rawpy / LibRaw](https://github.com/letmaik/rawpy)、OpenCV、Pillow、rich、httpx、filelock

---

## English summary

**Afterglow** is an unattended, local, undoable photo-library pipeline for Ubuntu 24.04 (Python 3.12, NVIDIA GPU with 8 GB VRAM). Nine stages — backup, index, cull, group, understand (local vision LLM tagging), process (auto grade, straighten, crop, optional AI enhance), merge (HDR / focus stack / panorama), video (proxies, time-lapse), publish (XMP, folder tree, Immich albums/tags) — turn roughly 1 TB of RAW/JPEG/phone/video files into an organized, searchable library. Image algorithms come from [LUMEN RAW](https://github.com/hety0211/lumen-raw) (MIT) driven as a black box through `lumen-cli mcp`; vision models run in `llama-server` (Qwen3-VL-4B as the workhorse; optional Qwen3.6-35B-A3B MoE for event naming only).

Three rules: **(1) originals are read-only**; **(2) everything is undoable** (`undo_log` written before every change; `afterglow undo`); **(3) resumable** (per-file, per-stage state in SQLite).

Quick start: `deploy/install.sh [--with-brain] [--skip-immich]` → edit `~/.config/afterglow/afterglow.toml` → create an Immich API key → `afterglow doctor` → `afterglow setup` → `afterglow run --dry-run` → `afterglow run`. Expect roughly 3–5 days of continuous running for 1 TB (a design-level estimate, not measured). Geometry (straighten/crop) is implemented in Afterglow because LUMEN's CLI does not expose it; AI denoise/super-resolution needs LUMEN's Python library (`LumenLib`). Immich mounts the organized tree and the source read-only; Immich ML jobs are paused while GPU stages run.

Caveats: the Immich compose in `deploy/immich/` could not be diffed against the current official file when written (the installer saves the official files to `/opt/immich/official/` for comparison), and model checksums in `models.lock` are recorded on first download. Licensed under MIT; see Acknowledgements above.

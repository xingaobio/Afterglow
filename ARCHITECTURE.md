# Afterglow — 架构与接口契约

> 本文件是全项目的**唯一事实来源**。所有模块必须遵守这里定义的目录结构、数据表、接口签名和行为规则。
> 设计/审核：Claude Fable 5.1。实现：Claude Sonnet 5.5 子代理（按模块分工）。

Afterglow 是一条**无人值守、本地运行、可撤销**的照片整理与处理流水线：
把移动硬盘上约 1 TB 的 RAW / JPEG / 手机照片 / 视频，自动备份、建索引、选片、分组、
用本地视觉大模型打标签、自动调色导出、HDR/景深/全景合成、生成视频代理，
最后整理成目录树并同步到 Immich。核心图像算法通过 **LUMEN RAW**
（https://github.com/hety0211/lumen-raw，MIT）的 `lumen-cli` 以黑盒方式调用。

## 0. 三条铁律（所有代码必须遵守）

1. **原图只读。** 任何阶段都不得写入、改名、删除 `paths.source` 下的任何文件。所有输出写到 `paths.output` / `paths.organized`。
2. **一切可撤销。** 每个改变磁盘或 Immich 状态的动作先写 `undo_log`，再执行。`afterglow undo` 能按日志逆序还原。
3. **断点续跑。** 每个文件每个阶段的状态记在 `steps` 表；重跑只处理 `pending`/`failed`，`done` 的不重复做。任何阶段崩溃不得影响已完成的记录。

目标平台：**Ubuntu 24.04 LTS，Python 3.12**，NVIDIA RTX 2000（8 GB VRAM），32 GB RAM，1 TB SSD 工作盘。
Windows/macOS 不是目标，但代码不应有意排斥。

## 1. 目录结构

```
Afterglow/
├── ARCHITECTURE.md              # 本文件
├── README.md                    # 中文为主，附英文摘要；安装、配置、运行、撤销
├── LICENSE                      # MIT
├── pyproject.toml               # 包名 afterglow，console_script: afterglow = afterglow.cli:main
├── config/
│   └── afterglow.example.toml   # 完整示例配置，每个键有注释
├── afterglow/
│   ├── __init__.py              # __version__
│   ├── cli.py                   # 子命令：setup run watch status undo bench doctor
│   ├── config.py                # TOML → dataclasses；校验；路径展开
│   ├── state.py                 # SQLite 状态库（见 §3）；所有 SQL 只在此文件
│   ├── context.py               # Context：config + state + 集成客户端 + logger + dry_run
│   ├── undo.py                  # UndoLog：记录与回放
│   ├── gpu_lock.py              # 进程间 GPU 互斥锁 + 暂停/恢复 Immich ML 任务
│   ├── pipeline.py              # 阶段注册表、依赖解析、按批调度、runs 记录
│   ├── watch.py                 # 监听挂载/定时触发增量运行
│   ├── util/
│   │   ├── hashing.py           # sha256（分块）、dHash/pHash（感知哈希）
│   │   ├── exif.py              # exiftool -j 批量读取 → dict；日期解析；GPS
│   │   ├── images.py            # 读 RAW 内嵌预览（rawpy）/JPEG，缩放，EXIF 方向校正，保存 JPEG
│   │   ├── fs.py                # 安全的 symlink/hardlink/copy/move（经 undo）、原子写、只读检查
│   │   └── log.py               # 日志到 work/logs/afterglow-YYYYMMDD.log + 控制台（rich）
│   ├── integrations/
│   │   ├── lumen.py             # LumenCLI（MCP stdio 黑盒）+ LumenLib（可选直接 import）
│   │   ├── llm.py               # llama-server（OpenAI 兼容）客户端，图片+JSON 结构化输出
│   │   └── immich.py            # Immich REST 客户端
│   └── stages/
│       ├── base.py              # Stage 抽象类、FileRecord、StepOutcome（见 §4）
│       ├── backup.py            # ① rsync 备份 + 校验
│       ├── index.py             # ② 遍历、哈希、EXIF、RAW/JPEG 配对、预览图
│       ├── cull.py              # ③ 清晰度/闭眼/连拍/最佳帧（lumen catalog.analyze）+ 质量分 + pick
│       ├── group.py             # ④ 事件分组；HDR/景深/全景/延时序列检测
│       ├── understand.py        # ⑤ 视觉大模型打标签、描述、场景、风格；事件命名
│       ├── process.py           # ⑥ 自动调色导出、水平校正、裁切、AI 增强
│       ├── merge.py             # ⑦ HDR/景深/全景合成（lumen）
│       ├── video.py             # ⑧ ffprobe、关键帧、NVENC 代理、延时视频
│       └── publish.py           # ⑨ XMP 附属文件、目录树、Immich 相册/标签/描述
├── deploy/
│   ├── install.sh               # Ubuntu 一键安装（见 §8）
│   ├── immich/
│   │   ├── docker-compose.yml   # 官方 compose 基础上：外部图库只读挂载、CUDA ML 镜像
│   │   └── example.env
│   ├── models/
│   │   ├── download.sh          # llama.cpp 编译/下载、GGUF 模型、LUMEN 模型
│   │   └── models.lock          # 模型文件名 + 来源 URL + sha256
│   ├── systemd/
│   │   ├── afterglow-llm.service        # llama-server（视觉模型）
│   │   ├── afterglow-brain.service      # llama-server（可选 35B MoE）
│   │   ├── afterglow-watch.service      # afterglow watch
│   │   ├── afterglow-nightly.timer/.service
│   │   └── 99-afterglow-usb.rules       # udev：插盘触发
│   └── lumen/
│       └── setup-lumen.sh       # clone lumen-raw、建 .venv-linux、装 GPU 依赖、下载模型
├── scripts/
│   ├── bench_models.py          # 200 张样片对比 4B / 12B / 35B：秒/张、标签一致性
│   └── count_library.sh         # 按扩展名统计移动硬盘
└── tests/                       # pytest；不依赖外部服务（mock subprocess / HTTP）
    ├── conftest.py              # 合成图片 fixture（清晰/模糊/倾斜/连拍）
    └── test_*.py
```

## 2. 配置（`config.py`）

`afterglow` 读取 `--config PATH`，默认 `~/.config/afterglow/afterglow.toml`，其次 `./afterglow.toml`。
TOML 用标准库 `tomllib`。每个 section 对应一个 `@dataclass(frozen=True)`，缺省值在 dataclass 中，
`config.load(path) -> Config`，`Config.validate()` 检查路径存在性与互斥项（如 `organize.mode="move"` 必须 `backup.require_before_move=true`）。

```toml
[paths]
source    = "/media/xin/PHOTOS"            # 移动硬盘挂载点，只读
work      = "/data/afterglow"              # SSD：afterglow.db、previews/、cache/、logs/
output    = "/data/afterglow/output"       # 导出、增强、合成、代理视频
organized = "/data/afterglow/library"      # 整理后的目录树（Immich 外部图库指向这里）
backup    = "/media/xin/BACKUP"            # 备份盘挂载点；空字符串 = 不备份
include_ext = ["cr2","cr3","nef","arw","dng","raf","orf","rw2","jpg","jpeg","heic","png","mp4","mov","avi","mts"]

[organize]
mode       = "symlink"      # symlink | hardlink | copy | move。move 仅在 backup 校验通过后执行
layout     = "{year}/{year}-{month:02d} {event}/{basename}"
culled_dir = "_culled"      # 非最佳帧/模糊/重复 → organized/_culled/<同样 layout>
duplicates_dir = "_duplicates"

[cull]
sharpness_min      = 35.0   # LUMEN sharpness 0–100 低于此视为模糊
burst_gap_seconds  = 2.0    # 相邻拍摄间隔 ≤ 此值且同机身 → 同一连拍
keep_per_burst     = 1
reject_eyes_closed = true
min_long_edge      = 800    # 小于此的图（截图、缩略图）直接标 reject

[events]
gap_hours = 4.0             # 拍摄间隔超过此值 → 新事件
gps_km    = 5.0             # 或 GPS 跳跃超过此值 → 新事件
min_photos = 3              # 少于此的零散照片归入 "YYYY-MM 零散"

[llm]
server          = "http://127.0.0.1:8080/v1"
vision_model    = "qwen3-vl-4b"
brain_server    = ""                        # 空 = 不用第二模型，事件命名也走 vision_model
brain_model     = "qwen3.6-35b-a3b"
tag_scope       = "picks"                   # picks | all
max_image_edge  = 1024
concurrency     = 2
timeout_seconds = 120
language        = "zh"                      # 描述/事件名语言

[lumen]
source_dir     = "/opt/lumen-raw"
python         = "/opt/lumen-raw/.venv-linux/bin/python"
models_dir     = "/data/afterglow/models/lumen"     # LUMEN_MODELS
library_db     = "/data/afterglow/lumen-library.db" # LUMEN_LIBRARY（LUMEN 自己的目录库，与 afterglow.db 分开）
compute        = "auto"                              # LUMEN_COMPUTE: auto|cuda|cpu
export_format  = "jpg"
export_quality = 92
export_long_edge = 0            # 0 = 原始分辨率
enhance        = "auto"         # off | auto | all；auto = 仅对 ISO 高或分辨率低的精选
enhance_scale  = 2
restart_every  = 200            # 每处理 N 张重启 lumen MCP 子进程，防内存增长

[process]
straighten                = true
straighten_min_confidence = 0.6
straighten_max_degrees    = 5.0
crops                     = ["3:2", "4:5", "16:9"]   # 空列表 = 不裁切
scope                     = "picks"                   # picks | all
# 场景 → LUMEN 预设名（edit.preset）；空字符串 = 只做 edit.auto
[process.style_presets]
landscape = ""
portrait  = ""
night     = ""
default   = ""

[video]
proxy_height = 1080
encoder      = "h264_nvenc"     # 不可用时回退 libx264
keyframes    = 5
timelapse_min_frames = 60       # 连续等间隔 ≥ N 张 → 生成延时视频

[immich]
url     = "http://127.0.0.1:2283"
api_key = ""                    # 空 = 跳过 publish 的 Immich 部分
external_library_name = "Afterglow"
pause_ml_during_gpu_stages = true

[backup]
tool   = "rsync"
verify = "size"                 # size | sha256
require_before_move = true

[gpu]
lock_file = "/data/afterglow/gpu.lock"

[run]
batch_size = 200
workers    = 4                  # CPU 并行（索引、哈希、EXIF）
```

## 3. 状态库（`state.py`，SQLite，`work/afterglow.db`，WAL 模式）

**所有 SQL 只允许出现在 `state.py`。** 其他模块通过 `State` 的方法访问。时间一律 ISO-8601 UTC 字符串。

```sql
CREATE TABLE files (
  id INTEGER PRIMARY KEY,
  path TEXT UNIQUE NOT NULL,        -- 绝对路径（source 下）
  rel_path TEXT NOT NULL,           -- 相对 source
  size INTEGER, mtime REAL,
  sha256 TEXT,                      -- 完整文件哈希
  dhash TEXT,                       -- 64-bit 感知哈希（十六进制），图片才有
  kind TEXT NOT NULL,               -- raw | jpeg | heic | png | video | other
  ext TEXT NOT NULL,
  captured_at TEXT,                 -- EXIF DateTimeOriginal（含亚秒），无则文件名/mtime 推断，来源记在 captured_src
  captured_src TEXT,                -- exif | filename | mtime
  camera TEXT, lens TEXT, iso INTEGER, aperture REAL, shutter REAL, focal REAL,
  exposure_bias REAL, focus_distance REAL,
  gps_lat REAL, gps_lon REAL,
  width INTEGER, height INTEGER, orientation INTEGER,
  duration REAL,                    -- 视频秒数
  pair_id INTEGER REFERENCES files(id),  -- JPEG 指向同名 RAW（RAW 为主，JPEG 为伴）
  preview_path TEXT,                -- work/previews/<sha256[:2]>/<sha256>.jpg（长边 ≤ 2560）
  missing INTEGER DEFAULT 0
);
CREATE INDEX ix_files_captured ON files(captured_at);
CREATE INDEX ix_files_sha ON files(sha256);

CREATE TABLE steps (
  file_id INTEGER NOT NULL REFERENCES files(id),
  stage TEXT NOT NULL,              -- 见 §4 阶段名
  status TEXT NOT NULL,             -- pending | done | failed | skipped
  attempts INTEGER DEFAULT 0,
  error TEXT,
  updated_at TEXT NOT NULL,
  PRIMARY KEY (file_id, stage)
);

CREATE TABLE analysis (
  file_id INTEGER PRIMARY KEY REFERENCES files(id),
  sharpness REAL, faces INTEGER, eyes_closed INTEGER, blurry INTEGER,
  exposure_score REAL,              -- 0–1，直方图裁切惩罚
  quality REAL,                     -- 0–100 综合分（见 §5 cull）
  burst_id INTEGER REFERENCES groups(id),
  burst_best INTEGER DEFAULT 0,
  pick INTEGER DEFAULT 0,           -- 1 = 精选
  reject_reason TEXT                -- blurry | eyes_closed | duplicate | burst_alt | tiny | null
);

CREATE TABLE groups (
  id INTEGER PRIMARY KEY,
  kind TEXT NOT NULL,               -- burst | event | hdr | focus | pano | timelapse | duplicate
  name TEXT,                        -- 事件名（understand 阶段填）
  started_at TEXT, ended_at TEXT,
  meta TEXT                         -- JSON：gps 中心、机身、曝光序列等
);
CREATE TABLE group_members (
  group_id INTEGER NOT NULL REFERENCES groups(id),
  file_id INTEGER NOT NULL REFERENCES files(id),
  position INTEGER NOT NULL,
  PRIMARY KEY (group_id, file_id)
);

CREATE TABLE tags (
  file_id INTEGER NOT NULL REFERENCES files(id),
  tag TEXT NOT NULL,
  source TEXT NOT NULL,             -- vlm | exif | rule
  confidence REAL,
  PRIMARY KEY (file_id, tag, source)
);

CREATE TABLE descriptions (
  file_id INTEGER PRIMARY KEY REFERENCES files(id),
  caption TEXT, scene TEXT,         -- scene ∈ landscape|portrait|street|night|food|document|screenshot|pet|event|other
  style_preset TEXT,
  model TEXT, created_at TEXT
);

CREATE TABLE outputs (
  id INTEGER PRIMARY KEY,
  file_id INTEGER REFERENCES files(id),
  group_id INTEGER REFERENCES groups(id),
  kind TEXT NOT NULL,               -- export | enhance | straighten | crop | hdr | focus | pano | proxy | timelapse | xmp
  path TEXT NOT NULL,
  params TEXT,                      -- JSON
  created_at TEXT NOT NULL
);

CREATE TABLE undo_log (
  id INTEGER PRIMARY KEY,
  run_id INTEGER REFERENCES runs(id),
  ts TEXT NOT NULL,
  action TEXT NOT NULL,             -- mkdir | symlink | hardlink | copy | move | write_file | immich_album | immich_add_assets | immich_tag | immich_description
  src TEXT, dst TEXT,
  meta TEXT,                        -- JSON（如 Immich id、原 description）
  undone INTEGER DEFAULT 0
);

CREATE TABLE runs (
  id INTEGER PRIMARY KEY,
  started_at TEXT NOT NULL, finished_at TEXT,
  stages TEXT NOT NULL,             -- 逗号分隔
  ok INTEGER DEFAULT 0, failed INTEGER DEFAULT 0, skipped INTEGER DEFAULT 0,
  note TEXT
);

CREATE TABLE kv (key TEXT PRIMARY KEY, value TEXT);   -- immich_library_id、schema_version、last_backup_ok 等
```

`State` 必须提供（签名精确到参数名；返回 `FileRecord`/dict）：

```python
class State:
    def __init__(self, db_path: Path): ...                        # 建表、迁移（kv.schema_version）
    # files
    def upsert_file(self, rec: dict) -> int
    def get_file(self, file_id: int) -> FileRecord | None
    def file_by_path(self, path: str) -> FileRecord | None
    def files_where(self, **filters) -> list[FileRecord]          # kind=, pick=, missing= 等
    def mark_missing(self, present_paths: set[str]) -> int
    # steps
    def pending(self, stage: str, limit: int, file_ids: list[int] | None = None) -> list[FileRecord]
    def set_step(self, file_id: int, stage: str, status: str, error: str | None = None) -> None
    def set_steps(self, outcomes: list[StepOutcome], stage: str) -> None   # 一个事务
    def reset_stage(self, stage: str, only_failed: bool = True) -> int
    # analysis / groups / tags / descriptions / outputs
    def set_analysis(self, file_id: int, **fields) -> None
    def analysis(self, file_id: int) -> dict | None
    def create_group(self, kind: str, member_ids: list[int], name=None, meta: dict | None = None) -> int
    def groups(self, kind: str, unnamed_only: bool = False) -> list[dict]   # 含 members(file ids 有序)
    def set_group_name(self, group_id: int, name: str) -> None
    def add_tags(self, file_id: int, tags: list[tuple[str, str, float | None]]) -> None
    def set_description(self, file_id: int, **fields) -> None
    def add_output(self, kind: str, path: str, file_id=None, group_id=None, params: dict | None = None) -> int
    # undo / runs / kv
    def log_undo(self, run_id: int, action: str, src=None, dst=None, meta: dict | None = None) -> int
    def undo_entries(self, run_id: int | None = None, include_undone=False) -> list[dict]   # 按 id 倒序
    def mark_undone(self, ids: list[int]) -> None
    def start_run(self, stages: list[str]) -> int
    def finish_run(self, run_id: int, ok: int, failed: int, skipped: int, note: str = '') -> None
    def get(self, key: str, default=None) -> str | None
    def set(self, key: str, value: str) -> None
    def stats(self) -> dict                                          # status 子命令用
```

## 4. 阶段框架（`stages/base.py`、`pipeline.py`）

```python
@dataclass
class FileRecord:            # files 表一行，字段同名；另含 analysis: dict|None（按需 join）
    id: int; path: str; rel_path: str; kind: str; ext: str; ...

@dataclass
class StepOutcome:
    file_id: int
    status: Literal['done', 'failed', 'skipped']
    error: str | None = None

class Stage(ABC):
    name: ClassVar[str]                 # 'backup' | 'index' | 'cull' | 'group' | 'understand' | 'process' | 'merge' | 'video' | 'publish'
    requires: ClassVar[tuple[str, ...]] # 前置阶段
    uses_gpu: ClassVar[bool] = False    # True → pipeline 在 gpu_lock 内运行它
    per_file: ClassVar[bool] = True     # False → 全局阶段（backup、group、merge、video 的分组部分），run 的 batch 为空

    def select(self, ctx: Context, limit: int) -> list[FileRecord]:
        """默认：state.pending(self.name, limit)，且前置阶段全部 done。可覆盖以加过滤（如 kind）。"""
    @abstractmethod
    def run(self, ctx: Context, batch: list[FileRecord]) -> list[StepOutcome]: ...
```

- 阶段顺序与依赖：`backup` → `index` → `cull` → `group` → `understand` → `process` → `merge` → `video` → `publish`。
  `index` 不依赖 `backup`（备份盘可空）；`process`/`merge`/`video`/`publish` 都依赖 `understand`，但 `understand` 失败的文件允许进入 `process`（标签为空即可）。
- `pipeline.run(ctx, stages: list[str] | None, dry_run: bool)`：
  按依赖排序 → `start_run` → 每阶段循环 `select(batch_size)` 直到为空 → `set_steps` → 汇总 → `finish_run`。
  任一文件异常只标该文件 `failed`（attempts+1），**不中断阶段**；连续 20 个异常才终止该阶段并记日志。
- `uses_gpu` 阶段：`with GpuLock(ctx):`（文件锁 + 可选暂停 Immich ML 任务）。
- `Context`（`context.py`）字段：`config, state, log, run_id, dry_run, lumen: LumenCLI, lumen_lib: LumenLib|None, llm: LLMClient, immich: ImmichClient|None, undo: UndoLog, fs: SafeFS`。集成客户端惰性创建。

### CLI（`cli.py`）

```
afterglow setup   [--config]      # 建目录、建库、检查 exiftool/rsync/ffmpeg/lumen/llm/immich 连通性；创建 Immich 外部图库
afterglow run     [--stages s1,s2] [--batch N] [--dry-run] [--only PATH...]
afterglow watch                   # 挂载出现/定时 → 增量 run
afterglow status                  # 各阶段 pending/done/failed 计数、最近 runs、磁盘占用
afterglow undo    [--run ID | --last | --all] [--dry-run]
afterglow retry   --stage NAME    # reset_stage(failed) 后 run 该阶段
afterglow bench   --sample N      # 调 scripts/bench_models.py
afterglow doctor                  # 依赖、GPU、模型、服务健康检查，输出表格
```
退出码：0 成功，1 配置/参数错误，2 运行中有 failed，3 外部依赖不可用。

## 5. 各阶段行为规范

### ① backup（全局，非 GPU）
- `paths.backup` 为空 → 全部 `skipped`。
- `rsync -a --info=progress2 --exclude lost+found source/ backup/<basename(source)>/`；**绝不**加 `--delete`。
- 校验：`verify=size` 比较文件数与总字节；`sha256` 抽样 1% 文件（至少 50）比对。通过 → `kv.last_backup_ok = <ts>`。
- `organize.mode=move` 且 `require_before_move` 时，`publish` 检查该 kv 在本次 run 之前 24 h 内，否则 move 降级为 symlink 并警告。

### ② index（按文件，非 GPU，CPU 并行 `run.workers`）
- 遍历 `source`，扩展名过滤（不区分大小写），跳过 `.`、`@eaDir`、`.thumbnails`、`System Volume Information`。
- sha256 分块；dhash 用预览图算（图片）。
- EXIF：`exiftool -j -n -fast2 -api largefilesupport=1 <files...>`，每批 ≤ 200 文件；字段映射见 `util/exif.py`。
  `captured_at` 优先级：DateTimeOriginal+SubSecTimeOriginal > CreateDate > 文件名中的日期（`IMG_20240512_130501`、`PXL_20240512_130501123`、`2024-05-12 13.05.01`）> mtime。
- RAW/JPEG 配对：同目录、同主文件名、拍摄时间差 ≤ 1 s → JPEG.pair_id = RAW.id。配对的 JPEG 在 cull/understand/process 阶段 `skipped`（以 RAW 为准），publish 时跟随 RAW。
- 预览：RAW 用 `rawpy.RawPy.extract_thumb()` 取内嵌 JPEG，失败则 `postprocess(half_size=True)`；JPEG/HEIC/PNG 用 Pillow（`ImageOps.exif_transpose`）；视频用 ffmpeg 抽第 1 秒一帧。长边缩到 2560，质量 90，写 `work/previews/<sha[:2]>/<sha>.jpg`。
- 完全重复（sha256 相同）：后出现者 `reject_reason=duplicate`，加入 `duplicate` 组。
- 新发现的文件自动为后续所有阶段插入 `steps.pending`。

### ③ cull（按文件 ≤ 300/批，GPU 可选：LUMEN 的人脸模型用 ONNX）
- 只处理 `kind ∈ {raw,jpeg,heic,png}` 且未被 pair 的文件；`min(width,height) < min_long_edge` → `reject_reason=tiny`。
- 调 `ctx.lumen.analyze(paths)`（`catalog.analyze`）得到每张 `sharpness, faces, eyes_closed, blurry` 和 `bursts[{group, photos, best}]`。
  LUMEN 不可用时回退 `util/images.py` 的拉普拉斯方差（归一到 0–100）并记 warning。
- `exposure_score`：预览图亮度直方图，`1 - (clip_low + clip_high)`，clip 为 <2 或 >253 像素占比。
- `quality = 0.6*sharpness + 25*exposure_score + 15*(faces>0 且 eyes_closed==0 ? 1 : faces==0 ? 1 : 0)`，截到 0–100。
- 连拍：以 LUMEN 的 bursts 为准，存成 `groups(kind=burst)`；LUMEN 未分到的按 `burst_gap_seconds` + 同 camera 自己补分。
  每组 `keep_per_burst` 张最高 quality（LUMEN 给了 best 的优先）→ `burst_best=1`，其余 `reject_reason=burst_alt`。
- `pick = 1` 当且仅当：无 reject_reason 且 `sharpness ≥ sharpness_min` 且 `not (reject_eyes_closed and eyes_closed>0 and faces>0)`。

### ④ group（全局，非 GPU）
- 事件：所有图片+视频按 `captured_at` 排序，相邻间隔 > `gap_hours` 或 GPS 距离 > `gps_km` → 切分；少于 `min_photos` 的合并到当月 "零散"。`groups(kind=event, meta={gps_center, cameras, count})`。
- HDR 候选：同机身、间隔 ≤ 3 s、连续 3–9 张、`exposure_bias` 单调且跨度 ≥ 2 EV（或 shutter 成倍变化、aperture/ISO 不变）。
- 景深堆栈候选：间隔 ≤ 5 s、≥ 4 张、`focus_distance` 单调变化（无该字段则 dhash 汉明距离 ≤ 6 且 shutter/aperture 相同）。
- 全景候选：间隔 ≤ 10 s、3–20 张、相邻 dhash 距离 12–30（有重叠但不同）、曝光锁定。
- 延时候选：≥ `timelapse_min_frames` 张、间隔标准差/均值 < 0.2。
- 候选组的成员仍保留 pick 状态；组信息供 merge/video 用。

### ⑤ understand（按文件，GPU）
- `tag_scope=picks` 时只处理 `pick=1`（其余 `skipped`）。
- 对每张：预览图缩到 `max_image_edge`，base64 → `ctx.llm.describe(image_bytes)`，要求模型返回 JSON：
  `{"caption": str(≤40字), "scene": enum, "tags": [str ≤8], "people_count": int, "quality_notes": str, "style": enum(landscape|portrait|night|default)}`。
  用 `response_format={"type":"json_object"}`；解析失败重试 1 次，再失败 → `failed`。
- 写 `tags(source=vlm)`、`descriptions`。`concurrency` 路并行（`ThreadPoolExecutor`），因为 llama-server `-np`。
- 事件命名（全局子步骤，`groups(kind=event)` 中 `name IS NULL`）：取该事件前 6 张 pick 的 caption + 日期 + 地名（GPS 反查用离线 `reverse_geocoder` 包，无则空）→ 让 brain_model（无则 vision_model）给 ≤ 12 字中文名，形如 "苏格兰高地徒步"。写 `set_group_name`。目录名禁止 `/ \ : * ? " < > |`。

### ⑥ process（按文件，GPU，`scope` 过滤）
顺序，每步产出记 `outputs`：
1. **导出**：`ctx.lumen.open(path)` → `edit.auto` → 若 `descriptions.style_preset` 对应 `style_presets` 非空则 `edit.preset(name)` → `photo.export(output=output/exports/<rel_dir>/<stem>.jpg, quality, long_edge)`。RAW 必做；JPEG 仅当 `scope=all`。
2. **水平校正**：对导出图，`ctx.lumen_lib.straighten(rgb)`（可用时）或 `util/images.straighten_estimate`（LSD 线段投票，同算法）；`confidence ≥ min` 且 `|angle| ≤ max` → 旋转 + 内接矩形裁切，覆盖导出图（导出图是我们自己的产物，可覆盖），记 `outputs(kind=straighten, params={angle, confidence})`。
3. **裁切**：按 `crops` 比例，以显著性中心（OpenCV `saliency.StaticSaliencySpectralResidual`，失败则用人脸框中心，再失败则几何中心）裁最大矩形 → `output/crops/<stem>_<ratio>.jpg`。
4. **AI 增强**：`enhance=all`，或 `auto` 且（ISO ≥ 3200 或 长边 < 3000）→ `ctx.lumen_lib.denoise/super_resolution`（需 LumenLib；不可用则 `skipped` 并 warning）→ `output/enhanced/<stem>_x2.jpg`。
- `edit.auto` 等 LUMEN 命令失败 → 该文件 `failed`，不影响其他。

### ⑦ merge（全局，GPU）
- 对 `groups(kind ∈ hdr|focus|pano)`：成员先经 LUMEN 导出（复用 process 的导出图，没有则现场导出到 cache），
  用 `ctx.lumen_lib.merge_images(images, kind)`；LumenLib 不可用时用 OpenCV 回退：HDR = `cv2.createMergeMertens`，focus = 拉普拉斯金字塔逐像素最清晰，pano = `cv2.Stitcher_create`。
- 输出 `output/merged/<event>/<kind>_<first_stem>.jpg`，记 `outputs(group_id)`。成员文件 steps 标 done。

### ⑧ video（按文件 + 全局，GPU 编码）
- 视频文件：`ffprobe -v quiet -print_format json -show_format -show_streams` 写 duration/width/height/captured_at；
  抽 `keyframes` 帧（均匀）→ 第一帧作为 preview，各帧可选送 understand（复用 `llm.describe`，取并集标签）。
- 代理：`ffmpeg -i in -vf scale=-2:{proxy_height} -c:v {encoder} -preset p4 -cq 28 -c:a aac -b:a 128k output/proxies/<rel>/<stem>_proxy.mp4`；`h264_nvenc` 不可用（`ffmpeg -encoders` 探测）回退 `libx264 -crf 23`。
- 延时组：`ctx.lumen.timelapse(paths, output, format='h264')`。

### ⑨ publish（全局，非 GPU）
1. **XMP**：对每个有 analysis/tags/description 的图片，写 `output/xmp/<rel>/<stem>.xmp`（**不**写到 source 旁），含 `xmp:Rating`（pick → 3，burst_best → 4，quality ≥ 85 → 5）、`dc:subject`（tags）、`dc:description`（caption）、`xmp:Label`（reject_reason → "Rejected"）。用模板字符串生成，命名空间正确。
2. **目录树**：按 `layout` 在 `organized/` 下为每个 pick 文件建链接/副本（RAW 带其配对 JPEG 和 process 导出图一起放）；非 pick 放 `culled_dir`，重复放 `duplicates_dir`。每个动作经 `ctx.fs`（自动写 undo_log）。同名冲突加 `_<sha[:6]>`。
3. **Immich**（`api_key` 非空）：
   - 外部图库：`kv.immich_library_id` 为空则创建（import path = `organized`），然后触发扫描，轮询直到任务队列空闲（最多 30 min）。
   - 每个 event 组 → 相册（名 `YYYY-MM 事件名`），用 `search_by_path` 找到 asset id 加入；每张 pick 写 description（caption）和 tags（Immich tags API，层级 `Afterglow/<tag>`）。
   - 每个 Immich 写操作先 `log_undo`（记录 album id、原 description）。

## 6. 集成层接口（`integrations/`）

### `lumen.py`
```python
class LumenCLI:
    """lumen-cli mcp --headless 子进程，JSON-RPC 2.0，**每行一个 JSON**（非 Content-Length 帧）。
    启动：[python, source_dir/main.py, '--cli', 'mcp', '--headless']，cwd=source_dir，
    env 加 LUMEN_MODELS、LUMEN_LIBRARY、LUMEN_COMPUTE、ORT_DISABLE_TELEMETRY=1；stdout 读协议，stderr 转日志。
    握手：initialize(protocolVersion='2025-06-18') → notifications/initialized → tools/list（缓存工具名）。
    调用：tools/call {name, arguments}；isError 为真 → 抛 LumenError。结果取 structuredContent，无则解析 content[0].text 的 JSON。
    每 restart_every 次调用或子进程退出 → 自动重启。线程安全（一把锁）。"""
    def analyze(self, paths: list[str]) -> dict            # tool analyze_photos → {count, photos[], bursts[]}
    def open(self, path: str) -> dict                      # open_photo
    def auto_tone(self, path: str) -> dict                 # auto_tone
    def apply_preset(self, path: str, name: str, amount: int = 100) -> dict   # run_command edit.preset
    def apply_edits(self, path: str, changes: dict) -> dict # apply_edits
    def export(self, path: str, output: str, quality: int, long_edge: int | None, overwrite=True) -> dict  # run_command photo.export
    def export_batch(self, paths, folder, fmt, quality, long_edge) -> dict    # export_photos
    def timelapse(self, paths: list[str], output: str, fmt='h264', fps=24) -> dict   # run_command video.timelapse
    def write_xmp(self, paths: list[str]) -> dict          # run_command library.write_xmp（仅在 output 树上使用）
    def info(self, path: str) -> dict                      # photo.info
    def close(self) -> None

class LumenLib:
    """可选：sys.path.insert(0, source_dir) 后直接 import lumen.culling / straighten / restoration / merge。
    import 失败 → available=False。只封装四个函数；输入输出都是 RGB uint8 ndarray。"""
    available: bool
    def analyse(self, rgb) -> dict                          # culling.analyse
    def straighten(self, rgb) -> tuple[float | None, float] # straighten.estimate → (angle, confidence)
    def denoise(self, rgb, amount=70) -> ndarray            # restoration.denoise
    def super_resolution(self, rgb, scale=2) -> ndarray     # restoration.super_resolution
    def merge_images(self, images: list[ndarray], kind: str) -> ndarray   # merge.merge_images kind∈hdr|focus|pano
```
注意：LUMEN 的 `edit.apply` **不支持**几何（straighten/crop/rotation）；几何由 Afterglow 自己做（§5 ⑥）。
AI 超分/降噪也**没有** CLI 命令，只能走 `LumenLib`。

### `llm.py`
```python
class LLMClient:
    def __init__(self, server: str, model: str, timeout: float): ...
    def healthy(self) -> bool                                  # GET /models
    def chat_json(self, system: str, user: str, image_jpeg: bytes | None = None, schema_hint: str = '', max_tokens=400) -> dict
        # POST /chat/completions，image 用 data:image/jpeg;base64；response_format json_object；
        # 禁用思考：extra_body {"chat_template_kwargs": {"enable_thinking": false}}（Qwen），并在 system 末尾加 "/no_think"。
        # 剥离 ```json 围栏后 json.loads；失败重试 1 次（temperature 0）。
    def describe(self, image_jpeg: bytes, language='zh') -> dict      # §5 ⑤ 的 schema
    def name_event(self, captions: list[str], date: str, place: str | None, language='zh') -> str
```

### `immich.py`
```python
class ImmichClient:                       # httpx，header x-api-key；所有方法失败抛 ImmichError
    def ping(self) -> bool                                            # GET /api/server/ping
    def libraries(self) -> list[dict]; def create_external_library(self, name, import_paths: list[str], owner_id=None) -> str
    def scan_library(self, library_id: str) -> None
    def jobs(self) -> dict; def pause_job(self, name: str) -> None; def resume_job(self, name: str) -> None   # PUT /api/jobs/{name} {"command": "pause"|"resume", "force": false}
    def wait_idle(self, names=('library','metadataExtraction','thumbnailGeneration'), timeout=1800) -> bool
    def asset_by_path(self, original_path: str) -> dict | None       # POST /api/search/metadata {"originalPath": ...}
    def create_album(self, name: str, asset_ids: list[str]) -> str; def add_to_album(self, album_id, asset_ids) -> None; def delete_album(self, album_id) -> None
    def update_asset(self, asset_id: str, description: str | None = None, rating: int | None = None) -> dict
    def upsert_tag(self, name: str) -> str; def tag_assets(self, tag_id: str, asset_ids: list[str]) -> None; def untag_assets(self, tag_id, asset_ids) -> None
```
实现者须用 WebFetch 核对 https://immich.app/docs/api 当前端点（v3.x），端点变化以文档为准，但方法签名不变。

### `gpu_lock.py`
`class GpuLock: __enter__` 取 `filelock.FileLock(config.gpu.lock_file)`，若 `immich.pause_ml_during_gpu_stages` 且 immich 可用 → `pause_job('smartSearch')`、`pause_job('faceDetection')`；`__exit__` 恢复。暂停失败只告警。

## 7. 撤销（`undo.py`、`util/fs.py`）

`SafeFS`（`ctx.fs`）封装 `mkdir / symlink / hardlink / copy / move / write_text`：先 `state.log_undo(run_id, action, src, dst, meta)`，再执行；`dry_run` 只记日志打印不执行。
`move` 前断言 `src` 不在 `paths.source` 下**除非** `organize.mode == 'move'` 且备份校验有效。
`UndoLog.undo(run_id | None, dry_run)`：按 id 倒序：`move` → 移回；`symlink/hardlink/copy/write_file` → 删除 dst；`mkdir` → 目录为空才删；`immich_*` → 调对应删除/恢复。每条成功后 `mark_undone`。

## 8. 部署（`deploy/`）

`install.sh`（幂等，可重复执行，`set -euo pipefail`，每步打印中文说明）：
1. apt：`exiftool rsync ffmpeg git build-essential cmake python3.12-venv libimage-exiftool-perl curl jq`
2. NVIDIA 驱动检测（`nvidia-smi`），缺失则提示 `ubuntu-drivers install` 并退出 3
3. Docker Engine + compose 插件（官方 apt 源，**不是** Docker Desktop）+ NVIDIA Container Toolkit
4. `deploy/lumen/setup-lumen.sh`：clone lumen-raw 到 `/opt/lumen-raw`（固定到 `models.lock` 记录的 commit），`python3.12 -m venv .venv-linux`，装 `requirements-linux.txt` + `requirements-linux-gpu.txt`，`main.py --cli models --download`（LUMEN_MODELS 指向 models_dir）
5. `deploy/models/download.sh`：编译 llama.cpp（`-DGGML_CUDA=ON`）到 `/opt/llama.cpp`；下载 GGUF：Qwen3-VL-4B-Instruct Q4_K_M + mmproj（必选），Qwen3.6-35B-A3B Q4_K_M + mmproj（`--with-brain` 可选）；校验 sha256（`models.lock`）；实现者用 WebFetch 核对 Hugging Face 上 unsloth/ 或 Qwen/ 官方 GGUF 仓库的**实际文件名**，不得臆造
6. Afterglow 本体：`python3.12 -m venv /opt/afterglow/.venv && pip install -e .`
7. Immich：复制 `deploy/immich/*` 到 `/opt/immich`，生成 `.env`（随机 DB 密码），`docker compose up -d`；compose 文件：`immich-machine-learning` 用 `-cuda` 镜像并声明 `deploy.resources.reservations.devices` GPU；`volumes` 加 `${AFTERGLOW_ORGANIZED}:/mnt/afterglow:ro` 与 `${PHOTO_SOURCE}:/mnt/source:ro`
8. systemd：安装 `afterglow-llm.service`（`llama-server -m <gguf> --mmproj <mmproj> -ngl 99 -c 8192 -np 2 --host 127.0.0.1 --port 8080 --jinja`，`-fa on`），可选 `afterglow-brain.service`（`--n-cpu-moe 99 -c 16384 --port 8081`，**与 llm 服务互斥**：`Conflicts=`，由 afterglow 在需要时 `systemctl start`），`afterglow-watch.service`，`afterglow-nightly.timer`（03:17），udev 规则调用 `systemctl start afterglow-run@<mount>.service`
9. 最后打印下一步：编辑 `~/.config/afterglow/afterglow.toml`，在 Immich 网页创建 API key 填入，`afterglow doctor`，`afterglow setup`，`afterglow run --dry-run`

## 9. 测试要求

- `pytest -q` 在无 GPU、无 exiftool、无网络的环境必须通过。外部命令用 `monkeypatch`/`unittest.mock` 替换；`LumenCLI` 用一个假的 MCP 子进程脚本（`tests/fake_lumen_mcp.py`，读 stdin 行 JSON 回固定结果）测试协议。
- `conftest.py` 用 Pillow/NumPy 生成：清晰图、高斯模糊图、倾斜 3° 的地平线图、5 张连拍（时间差 0.5 s 的 EXIF）、一对同名 RAW(.dng 可用 tifffile 写最小 DNG，或直接 .jpg 改名 .arw 并 mock rawpy)/JPEG。
- 每个 stage 至少一个端到端测试：临时 source/work/output 目录 → `pipeline.run` → 断言 `steps` 与 `outputs`。
- `undo` 测试：publish 后 undo，organized 目录恢复为空，source 字节不变（比较 sha256）。
- `ruff check .`（`pyproject` 配置 line-length 120，select E,F,I,B,UP）零告警。

## 10. 代码风格

- Python 3.12，类型标注，`pathlib.Path`，`dataclasses`，无全局可变状态。
- 日志用 `logging.getLogger(__name__)`，中文信息面向用户，英文用于调试细节皆可，但**同一条信息不混用**。
- 不吞异常：捕获后必须 `log.exception` 或转成 `StepOutcome(failed, error=...)`。
- 子进程一律 `subprocess.run([...], check=False, capture_output=True, text=True, timeout=...)`，禁止 `shell=True`。
- 依赖（`pyproject`）：`numpy>=2`, `opencv-python-headless>=4.10`, `Pillow>=11`, `rawpy>=0.25`, `httpx>=0.27`, `filelock>=3`, `rich>=13`, `reverse_geocoder>=1.5`（可选 extra `geo`）。dev：`pytest`, `ruff`.

## 11. 第二版新增（v0.2）

配置键见 `afterglow/config.py` 的 `People`、`Grade`、`Report` 和 `Organize.mode = "copy_picks"`（精选复制到 SSD，其余建链接；已实现）。

### 11.1 人物相册（阶段 `people`，全局，非 GPU，排在 `publish` 之后）
- `immich` 未启用或 `people.enabled = false` → 直接返回。不写每文件 steps。
- `ImmichClient` 新增：`people(with_hidden=False) -> list[dict]`（id、name、可能的资产数）、`person_assets(person_id) -> list[dict]`（至少含 id、originalPath）。端点以 Immich 当前 API 为准，方法签名不变。
- 对每个人物：资产按 originalPath 映射回 `files`（宿主机路径与容器路径相同，见 compose [3c]；映射不到的资产忽略）；映射到的文件数 < `min_photos` 跳过；无名字且 `include_unnamed = false` 跳过，否则名字为 `未命名 N`。
- 相册名 `album_prefix + name`；成员 = 该人物的 pick 文件按 `analysis.quality` 降序取 `best_per_person` 张（0 = 全部）。
- 幂等：`kv["immich_person_album:<person_id>"]` 存相册 id；已存在则只 `add_to_album` 新成员（成员签名存 `kv["immich_person_sig:<person_id>"]`，相同则跳过）。
- 同时在本地 `groups(kind="person", name=人物名, meta={immich_person_id})` 记录成员（已存在则 `add_group_members`）。
- 每个 Immich 写操作先 `ctx.undo.record("immich_album" | "immich_add_assets", meta=...)`，meta 形状与 publish 相同。
- Immich 的人脸识别要在外部图库扫描后跑一段时间；首次运行可能 0 人，属正常，日志提示下次运行再建。

### 11.2 自然语言调色（`process` 阶段导出链内）
- `grade.mode = "auto"` → 行为不变。`"llm"` 且该文件 `descriptions.scene ∈ grade.scenes` 且 LLM 可用 → 在 `auto_tone` 之后、`export` 之前：
  1. `ctx.llm.grade(image_jpeg, scene, caption, instruction, language) -> dict`：让视觉模型看预览图并返回 JSON
     `{"adjustments": {exposure, contrast, highlights, shadows, whites, blacks, temperature, tint, vibrance, saturation, dehaze, clarity}, "reason": str}`；只保留这些键。
  2. 客户端钳制：`exposure` ∈ [-1.0, 1.0]（EV），其余 ∈ [-40, 40]，非数字丢弃；全为 0 或空 → 不调用 LUMEN。
  3. `ctx.lumen.apply_edits(path, {"adjustments": adj}, amount=grade.strength)`（`LumenCLI.apply_edits` 增加 `amount` 参数，映射到 LUMEN `edit.apply` 的 `amount`）。
- LLM 或 `apply_edits` 失败只告警，继续导出自动调色的结果（不让文件 failed）。
- `outputs(kind="export").params` 增加 `grade: {"adjustments": …, "reason": …, "strength": …}` 或 `grade: null`。
- 整条链仍在 `lumen.sequence()` 内。LLM 调用放在进入 sequence **之前**（先拿到参数），避免长时间占着 LUMEN 锁。

### 11.3 运行报告与通知（`afterglow/report.py`）
- `State.run_summary(run_id) -> dict`：该次 run 的起止时间、各阶段 ok/failed/skipped（从 `runs` 与 `steps.updated_at` 落在本次 run 时间窗内统计）、失败文件清单（path、stage、error，最多 200 条）、库总览（`stats()`）、本次新增 outputs 按 kind 计数、本次新建的 event 组名。
- `report.build_html(summary, config) -> str`：单文件 HTML（内联 CSS，无外部资源，中文），含：标题与时间、阶段表、精选/淘汰/重复计数、新事件列表、失败清单、磁盘占用（work/output/organized 的 `shutil.disk_usage`）。所有来自文件名/错误信息的文本必须 `html.escape`。
- `report.write(ctx, run_id) -> Path`：写 `paths.reports/run-<id>-<YYYYMMDD-HHMM>.html`，并更新 `latest.html`（复制，不用链接）。
- `report.notify(ctx, path, summary)`：`email_to` 与 `smtp_host` 非空 → `smtplib.SMTP` + STARTTLS，密码取 `os.environ[smtp_password_env]`（缺失则告警不发），主题 `Afterglow 运行 #<id>：成功 N，失败 M`，HTML 正文即报告；`notify_command` 非空 → `subprocess.run([cmd, str(path)], timeout=60)`（`shlex.split` 解析，不用 shell）。任何通知失败只告警。
- CLI：`run` / `retry` 结束后若 `report.enabled` 且非 dry-run → `write` + `notify`，并打印报告路径；新子命令 `afterglow report [--run ID]` 重新生成。`watch` 的每次运行同样生成。

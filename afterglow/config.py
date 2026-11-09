"""Configuration: TOML -> frozen dataclasses.  See ARCHITECTURE.md §2 for the schema.

Every section is a dataclass with the defaults documented in config/afterglow.example.toml.
`load()` reads the file, `Config.validate()` checks paths and mutually exclusive options.
"""

from __future__ import annotations

import dataclasses
import os
import tomllib
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

DEFAULT_LOCATIONS = (
    Path("~/.config/afterglow/afterglow.toml"),
    Path("./afterglow.toml"),
)

IMAGE_KINDS = ("raw", "jpeg", "heic", "png")
RAW_EXTENSIONS = {
    "cr2", "cr3", "nef", "nrw", "arw", "srf", "sr2", "dng", "raf", "orf", "rw2", "pef", "x3f", "3fr", "iiq",
}
JPEG_EXTENSIONS = {"jpg", "jpeg", "jpe"}
HEIC_EXTENSIONS = {"heic", "heif", "hif"}
PNG_EXTENSIONS = {"png", "tif", "tiff", "webp"}
VIDEO_EXTENSIONS = {"mp4", "mov", "m4v", "avi", "mts", "m2ts", "mkv", "3gp", "webm"}


def kind_of(ext: str) -> str:
    """Map a lower-case extension (no dot) to a files.kind value."""
    ext = ext.lower().lstrip(".")
    if ext in RAW_EXTENSIONS:
        return "raw"
    if ext in JPEG_EXTENSIONS:
        return "jpeg"
    if ext in HEIC_EXTENSIONS:
        return "heic"
    if ext in PNG_EXTENSIONS:
        return "png"
    if ext in VIDEO_EXTENSIONS:
        return "video"
    return "other"


class ConfigError(ValueError):
    """A configuration file is missing, malformed or inconsistent."""


@dataclass(frozen=True)
class Paths:
    source: Path = Path("/media/photos")
    work: Path = Path("/data/afterglow")
    output: Path = Path("/data/afterglow/output")
    organized: Path = Path("/data/afterglow/library")
    backup: Path | None = None
    include_ext: tuple[str, ...] = (
        "cr2", "cr3", "nef", "arw", "dng", "raf", "orf", "rw2",
        "jpg", "jpeg", "heic", "png", "mp4", "mov", "avi", "mts",
    )

    @property
    def db(self) -> Path:
        return self.work / "afterglow.db"

    @property
    def previews(self) -> Path:
        return self.work / "previews"

    @property
    def cache(self) -> Path:
        return self.work / "cache"

    @property
    def logs(self) -> Path:
        return self.work / "logs"


@dataclass(frozen=True)
class Organize:
    mode: str = "symlink"  # symlink | hardlink | copy | move
    layout: str = "{year}/{year}-{month:02d} {event}/{basename}"
    culled_dir: str = "_culled"
    duplicates_dir: str = "_duplicates"


@dataclass(frozen=True)
class Cull:
    sharpness_min: float = 35.0
    burst_gap_seconds: float = 2.0
    keep_per_burst: int = 1
    reject_eyes_closed: bool = True
    min_long_edge: int = 800


@dataclass(frozen=True)
class Events:
    gap_hours: float = 4.0
    gps_km: float = 5.0
    min_photos: int = 3


@dataclass(frozen=True)
class LLM:
    server: str = "http://127.0.0.1:8080/v1"
    vision_model: str = "qwen3-vl-4b"
    brain_server: str = ""
    brain_model: str = "qwen3.6-35b-a3b"
    tag_scope: str = "picks"  # picks | all
    max_image_edge: int = 1024
    concurrency: int = 2
    timeout_seconds: float = 120.0
    language: str = "zh"


@dataclass(frozen=True)
class Lumen:
    source_dir: Path = Path("/opt/lumen-raw")
    python: Path = Path("/opt/lumen-raw/.venv-linux/bin/python")
    models_dir: Path = Path("/data/afterglow/models/lumen")
    library_db: Path = Path("/data/afterglow/lumen-library.db")
    compute: str = "auto"
    export_format: str = "jpg"
    export_quality: int = 92
    export_long_edge: int = 0
    enhance: str = "auto"  # off | auto | all
    enhance_scale: int = 2
    restart_every: int = 200


@dataclass(frozen=True)
class Process:
    straighten: bool = True
    straighten_min_confidence: float = 0.6
    straighten_max_degrees: float = 5.0
    crops: tuple[str, ...] = ("3:2", "4:5", "16:9")
    scope: str = "picks"  # picks | all
    style_presets: dict[str, str] = field(
        default_factory=lambda: {"landscape": "", "portrait": "", "night": "", "default": ""}
    )


@dataclass(frozen=True)
class Video:
    proxy_height: int = 1080
    encoder: str = "h264_nvenc"
    keyframes: int = 5
    timelapse_min_frames: int = 60


@dataclass(frozen=True)
class Immich:
    url: str = "http://127.0.0.1:2283"
    api_key: str = ""
    external_library_name: str = "Afterglow"
    pause_ml_during_gpu_stages: bool = True

    @property
    def enabled(self) -> bool:
        return bool(self.api_key)


@dataclass(frozen=True)
class Backup:
    tool: str = "rsync"
    verify: str = "size"  # size | sha256
    require_before_move: bool = True


@dataclass(frozen=True)
class Gpu:
    lock_file: Path = Path("/data/afterglow/gpu.lock")


@dataclass(frozen=True)
class Run:
    batch_size: int = 200
    workers: int = 4


@dataclass(frozen=True)
class Config:
    paths: Paths = field(default_factory=Paths)
    organize: Organize = field(default_factory=Organize)
    cull: Cull = field(default_factory=Cull)
    events: Events = field(default_factory=Events)
    llm: LLM = field(default_factory=LLM)
    lumen: Lumen = field(default_factory=Lumen)
    process: Process = field(default_factory=Process)
    video: Video = field(default_factory=Video)
    immich: Immich = field(default_factory=Immich)
    backup: Backup = field(default_factory=Backup)
    gpu: Gpu = field(default_factory=Gpu)
    run: Run = field(default_factory=Run)
    path: Path | None = None  # where this config was loaded from

    def validate(self, require_source: bool = True) -> list[str]:
        """Return a list of problems (empty when valid).  Raise nothing; the CLI decides."""
        problems: list[str] = []
        if self.organize.mode not in ("symlink", "hardlink", "copy", "move"):
            problems.append(f"organize.mode 无效: {self.organize.mode}")
        if self.organize.mode == "move" and not self.backup.require_before_move:
            problems.append("organize.mode = move 时必须 backup.require_before_move = true")
        if self.organize.mode == "move" and self.paths.backup is None:
            problems.append("organize.mode = move 时必须设置 paths.backup")
        if self.llm.tag_scope not in ("picks", "all"):
            problems.append(f"llm.tag_scope 无效: {self.llm.tag_scope}")
        if self.process.scope not in ("picks", "all"):
            problems.append(f"process.scope 无效: {self.process.scope}")
        if self.lumen.enhance not in ("off", "auto", "all"):
            problems.append(f"lumen.enhance 无效: {self.lumen.enhance}")
        if self.backup.verify not in ("size", "sha256"):
            problems.append(f"backup.verify 无效: {self.backup.verify}")
        for ratio in self.process.crops:
            if ":" not in ratio or not all(p.strip().isdigit() for p in ratio.split(":", 1)):
                problems.append(f"process.crops 比例格式应为 W:H，得到 {ratio!r}")
        if require_source and not self.paths.source.is_dir():
            problems.append(f"paths.source 不存在或不是目录: {self.paths.source}")
        return problems


_PATH_FIELDS = {"source", "work", "output", "organized", "backup", "source_dir", "python", "models_dir", "library_db",
                "lock_file"}


def _coerce(cls: type, data: dict[str, Any], section: str) -> Any:
    known = {f.name: f for f in fields(cls)}
    kwargs: dict[str, Any] = {}
    for key, value in data.items():
        if key not in known:
            raise ConfigError(f"[{section}] 未知的配置项: {key}")
        f = known[key]
        if key in _PATH_FIELDS:
            if value in ("", None):
                value = None if key == "backup" else known[key].default
            else:
                value = Path(os.path.expandvars(str(value))).expanduser()
        elif isinstance(f.default, tuple) or (f.default is dataclasses.MISSING and isinstance(value, list)):
            value = tuple(value) if isinstance(value, list) else value
        kwargs[key] = value
    try:
        return cls(**kwargs)
    except TypeError as exc:
        raise ConfigError(f"[{section}] {exc}") from exc


def from_dict(data: dict[str, Any], path: Path | None = None) -> Config:
    """Build a Config from a parsed TOML mapping (unknown keys are errors)."""
    sections = {
        "paths": Paths, "organize": Organize, "cull": Cull, "events": Events, "llm": LLM, "lumen": Lumen,
        "process": Process, "video": Video, "immich": Immich, "backup": Backup, "gpu": Gpu, "run": Run,
    }
    kwargs: dict[str, Any] = {}
    for name, value in data.items():
        if name not in sections:
            raise ConfigError(f"未知的配置段: [{name}]")
        if not isinstance(value, dict):
            raise ConfigError(f"[{name}] 必须是表")
        if name == "process" and "style_presets" in value and isinstance(value["style_presets"], dict):
            value = dict(value)
            value["style_presets"] = {str(k): str(v) for k, v in value["style_presets"].items()}
        kwargs[name] = _coerce(sections[name], value, name)
    return Config(**kwargs, path=path)


def find_config(explicit: Path | None = None) -> Path | None:
    if explicit is not None:
        return explicit
    for candidate in DEFAULT_LOCATIONS:
        p = candidate.expanduser()
        if p.is_file():
            return p
    return None


def load(path: Path | None = None) -> Config:
    """Load the configuration at `path` (or the first default location).  Missing file -> ConfigError."""
    found = find_config(path)
    if found is None:
        raise ConfigError("找不到配置文件；请复制 config/afterglow.example.toml 到 ~/.config/afterglow/afterglow.toml")
    try:
        with open(found, "rb") as fh:
            data = tomllib.load(fh)
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise ConfigError(f"无法读取配置 {found}: {exc}") from exc
    return from_dict(data, path=found)

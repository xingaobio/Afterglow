"""Batch EXIF reading through exiftool and date parsing (ARCHITECTURE.md §5 ②)."""

from __future__ import annotations

import json
import logging
import re
import subprocess
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

CHUNK = 200
EXIFTOOL_TIMEOUT = 600

_DT_RE = re.compile(
    r"^\s*(\d{4})[:\-](\d{2})[:\-](\d{2})[ T](\d{2}):(\d{2}):(\d{2})(?:[.,](\d+))?\s*(Z|[+-]\d{2}:?\d{2})?\s*$"
)
_OFFSET_RE = re.compile(r"^\s*(Z|[+-])(\d{2}):?(\d{2})?\s*$")
_FILENAME_RES = (
    re.compile(
        r"(?<!\d)((?:19|20)\d{2})(0[1-9]|1[0-2])(0[1-9]|[12]\d|3[01])[_\- ]?"
        r"([01]\d|2[0-3])([0-5]\d)([0-5]\d)(\d{3})?(?!\d)"
    ),
    re.compile(
        r"(?<!\d)((?:19|20)\d{2})[-_.](0[1-9]|1[0-2])[-_.](0[1-9]|[12]\d|3[01])[ _T\-]"
        r"([01]\d|2[0-3])[.\-:_]([0-5]\d)[.\-:_]([0-5]\d)()(?!\d)"
    ),
)


def _tz_from_offset(offset: str | None) -> timezone | None:
    if not offset:
        return None
    m = _OFFSET_RE.match(str(offset))
    if not m:
        return None
    if m.group(1) == "Z":
        return UTC
    delta = timedelta(hours=int(m.group(2)), minutes=int(m.group(3) or 0))
    return timezone(delta if m.group(1) == "+" else -delta)


def parse_exif_datetime(value: Any, offset: str | None = None) -> datetime | None:
    """Parse an EXIF/ISO date string into an aware UTC datetime.

    A trailing zone in `value` or the separate `offset` ("+01:00") is honoured; without either the
    wall-clock time is taken as UTC.  Sub-seconds in `value` are kept.  Returns None when unparseable.
    """
    if value is None:
        return None
    m = _DT_RE.match(str(value))
    if not m:
        return None
    year, month, day, hour, minute, second = (int(m.group(i)) for i in range(1, 7))
    if year < 1900:
        return None
    micro = int((m.group(7) or "0")[:6].ljust(6, "0"))
    tz = _tz_from_offset(m.group(8)) or _tz_from_offset(offset) or UTC
    try:
        return datetime(year, month, day, hour, minute, second, micro, tzinfo=tz).astimezone(UTC)
    except ValueError:
        return None


def fmt_iso(dt: datetime) -> str:
    """Sortable ISO-8601 UTC string with millisecond precision, e.g. 2024-05-12T13:05:01.250Z."""
    dt = dt.astimezone(UTC)
    return dt.strftime("%Y-%m-%dT%H:%M:%S.") + f"{dt.microsecond // 1000:03d}Z"


def parse_iso(value: str | None) -> datetime | None:
    """Parse a string produced by `fmt_iso` (or any ISO-8601) into an aware UTC datetime."""
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    return dt.astimezone(UTC) if dt.tzinfo else dt.replace(tzinfo=UTC)


def captured_from_filename(name: str) -> str | None:
    """Date encoded in a file name (IMG_20240512_130501, PXL_20240512_130501123, 2024-05-12 13.05.01...)."""
    base = Path(name).name
    for rx in _FILENAME_RES:
        m = rx.search(base)
        if not m:
            continue
        year, month, day, hour, minute, second = (int(m.group(i)) for i in range(1, 7))
        micro = int(m.group(7)) * 1000 if m.group(7) else 0
        try:
            return fmt_iso(datetime(year, month, day, hour, minute, second, micro, tzinfo=UTC))
        except ValueError:
            continue
    return None


def mtime_iso(mtime: float) -> str:
    return fmt_iso(datetime.fromtimestamp(mtime, tz=UTC))


# --------------------------------------------------------------------------- exiftool JSON mapping
def _lookup(tags: dict[str, Any], *names: str) -> Any:
    """First non-empty value among `names` (tag names without group); earlier names win."""
    by_name: dict[str, list[tuple[str, Any]]] = {}
    for key, value in tags.items():
        group, _, name = key.rpartition(":")
        by_name.setdefault(name, []).append((group, value))
    for name in names:
        group_pref = None
        if ":" in name:
            group_pref, name = name.split(":", 1)
        for group, value in by_name.get(name, []):
            if group_pref and group != group_pref:
                continue
            if value not in (None, "", "undef"):
                return value
    return None


def _num(value: Any) -> float | None:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return f if f == f and abs(f) != float("inf") else None


def _int(value: Any) -> int | None:
    f = _num(value)
    return None if f is None else int(round(f))


def _camera(tags: dict[str, Any]) -> str | None:
    make = str(_lookup(tags, "Make") or "").strip()
    model = str(_lookup(tags, "Model") or "").strip()
    if not make and not model:
        return None
    if make and model.lower().startswith(make.lower().split()[0]):
        return model
    return f"{make} {model}".strip()


def _captured(tags: dict[str, Any]) -> str | None:
    original = _lookup(tags, "DateTimeOriginal")
    if original is not None:
        dt = parse_exif_datetime(original, _lookup(tags, "OffsetTimeOriginal", "OffsetTime"))
        if dt is not None:
            sub = _lookup(tags, "SubSecTimeOriginal", "SubSecTime")
            if sub is not None and dt.microsecond == 0:
                digits = re.sub(r"\D", "", str(sub))[:6]
                if digits:
                    dt = dt.replace(microsecond=int(digits.ljust(6, "0")))
            return fmt_iso(dt)
    for name in ("CreateDate", "MediaCreateDate", "CreationDate", "TrackCreateDate"):
        dt = parse_exif_datetime(_lookup(tags, name), _lookup(tags, "OffsetTimeDigitized", "OffsetTime"))
        if dt is not None and dt.year > 1904:
            return fmt_iso(dt)
    return None


def _focus_distance(tags: dict[str, Any]) -> float | None:
    for name in ("FocusDistance", "SubjectDistance", "FocusDistanceUpper"):
        v = _num(_lookup(tags, name))
        if v is not None and 0 < v < 4e9:
            return v
    return None


def map_tags(tags: dict[str, Any]) -> dict[str, Any]:
    """Map one exiftool `-G0 -n` record to the `files` columns of ARCHITECTURE.md §3 (missing -> absent)."""
    out: dict[str, Any] = {}
    captured = _captured(tags)
    if captured:
        out["captured_at"] = captured
        out["captured_src"] = "exif"
    camera = _camera(tags)
    if camera:
        out["camera"] = camera
    lens = _lookup(tags, "LensModel", "LensID", "Lens")
    if lens:
        out["lens"] = str(lens).strip()
    simple = (
        ("iso", _int(_lookup(tags, "ISO"))),
        ("aperture", _num(_lookup(tags, "FNumber", "Aperture"))),
        ("shutter", _num(_lookup(tags, "ExposureTime", "ShutterSpeed"))),
        ("focal", _num(_lookup(tags, "FocalLength"))),
        ("exposure_bias", _num(_lookup(tags, "ExposureCompensation"))),
        ("focus_distance", _focus_distance(tags)),
        ("gps_lat", _num(_lookup(tags, "GPSLatitude"))),
        ("gps_lon", _num(_lookup(tags, "GPSLongitude"))),
        ("width", _int(_lookup(tags, "ExifImageWidth", "File:ImageWidth", "ImageWidth", "SourceImageWidth"))),
        ("height", _int(_lookup(tags, "ExifImageHeight", "File:ImageHeight", "ImageHeight", "SourceImageHeight"))),
        ("orientation", _int(_lookup(tags, "Orientation"))),
        ("duration", _num(_lookup(tags, "Duration", "MediaDuration"))),
    )
    for key, value in simple:
        if value is not None:
            out[key] = value
    if out.get("duration") is not None and out["duration"] <= 0:
        del out["duration"]
    return out


def read_batch(paths: list[Path], exiftool: str = "exiftool") -> dict[str, dict]:
    """Run exiftool over `paths` (chunks of 200) and return {str(path): mapped columns}.

    exiftool missing or failing for a chunk is logged and leaves those files without EXIF data;
    the caller falls back to filename / mtime.
    """
    result: dict[str, dict] = {}
    for start in range(0, len(paths), CHUNK):
        chunk = [str(p) for p in paths[start : start + CHUNK]]
        cmd = [exiftool, "-j", "-n", "-fast2", "-api", "largefilesupport=1", "-G0", *chunk]
        try:
            proc = subprocess.run(cmd, check=False, capture_output=True, text=True, timeout=EXIFTOOL_TIMEOUT)
        except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
            log.warning("exiftool 不可用或超时，跳过这批 EXIF: %s", exc)
            continue
        if not proc.stdout.strip():
            log.warning("exiftool 没有输出 (返回码 %s): %s", proc.returncode, (proc.stderr or "").strip()[:300])
            continue
        try:
            records = json.loads(proc.stdout)
        except json.JSONDecodeError:
            log.exception("exiftool 输出不是有效 JSON")
            continue
        wanted = set(chunk)
        for rec in records:
            source = str(rec.get("SourceFile", ""))
            if source not in wanted:
                continue
            result[source] = map_tags(rec)
    return result

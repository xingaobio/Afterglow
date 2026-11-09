import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from afterglow.util import exif

CANNED = [
    {
        "SourceFile": "/src/a/IMG_0001.CR3",
        "EXIF:Make": "Canon", "EXIF:Model": "Canon EOS R5", "EXIF:LensModel": "RF24-70mm F2.8 L IS USM",
        "EXIF:ISO": 400, "EXIF:FNumber": 2.8, "EXIF:ExposureTime": 0.004, "EXIF:FocalLength": 35.0,
        "EXIF:ExposureCompensation": -0.67,
        "EXIF:DateTimeOriginal": "2024:05:12 13:05:01", "EXIF:SubSecTimeOriginal": 25,
        "EXIF:OffsetTimeOriginal": "+01:00",
        "EXIF:ExifImageWidth": 8192, "EXIF:ExifImageHeight": 5464, "EXIF:Orientation": 6,
        "Composite:GPSLatitude": 55.9533, "Composite:GPSLongitude": -3.1883,
        "MakerNotes:FocusDistance": 1.5,
    },
    {
        "SourceFile": "/src/a/clip.mov",
        "QuickTime:CreateDate": "2024:05:12 12:00:00", "QuickTime:Duration": 12.5,
        "QuickTime:ImageWidth": 1920, "QuickTime:ImageHeight": 1080, "QuickTime:Make": "Apple",
        "QuickTime:Model": "iPhone 15",
    },
    {"SourceFile": "/src/a/plain.png", "File:ImageWidth": 100, "File:ImageHeight": 50},
]


class Completed:
    def __init__(self, stdout: str, returncode: int = 0):
        self.stdout, self.stderr, self.returncode = stdout, "", returncode


def test_read_batch_maps_columns(monkeypatch):
    calls = []

    def fake_run(cmd, **kw):
        calls.append((cmd, kw))
        return Completed(json.dumps(CANNED))

    monkeypatch.setattr(exif.subprocess, "run", fake_run)
    result = exif.read_batch([Path(r["SourceFile"]) for r in CANNED])
    cmd, kw = calls[0]
    assert cmd[:7] == ["exiftool", "-j", "-n", "-fast2", "-api", "largefilesupport=1", "-G0"]
    assert not kw.get("shell") and kw["check"] is False

    raw = result["/src/a/IMG_0001.CR3"]
    assert raw["captured_at"] == "2024-05-12T12:05:01.250Z"  # +01:00 offset converted to UTC
    assert raw["captured_src"] == "exif"
    assert raw["camera"] == "Canon EOS R5"  # make not duplicated
    assert raw["lens"] == "RF24-70mm F2.8 L IS USM"
    assert (raw["iso"], raw["aperture"], raw["shutter"], raw["focal"]) == (400, 2.8, 0.004, 35.0)
    assert raw["exposure_bias"] == -0.67 and raw["focus_distance"] == 1.5
    assert (raw["width"], raw["height"], raw["orientation"]) == (8192, 5464, 6)
    assert raw["gps_lat"] == 55.9533 and raw["gps_lon"] == -3.1883

    video = result["/src/a/clip.mov"]
    assert video["captured_at"] == "2024-05-12T12:00:00.000Z"
    assert video["duration"] == 12.5 and video["camera"] == "Apple iPhone 15"

    plain = result["/src/a/plain.png"]
    assert "captured_at" not in plain and plain["width"] == 100


def test_read_batch_chunks_of_200(monkeypatch):
    sizes = []

    def fake_run(cmd, **kw):
        files = cmd[7:]
        sizes.append(len(files))
        return Completed(json.dumps([{"SourceFile": f, "EXIF:ISO": 100} for f in files]))

    monkeypatch.setattr(exif.subprocess, "run", fake_run)
    paths = [Path(f"/x/{i}.jpg") for i in range(450)]
    result = exif.read_batch(paths)
    assert sizes == [200, 200, 50] and len(result) == 450


def test_read_batch_without_exiftool_returns_empty(monkeypatch):
    def boom(cmd, **kw):
        raise FileNotFoundError("exiftool")

    monkeypatch.setattr(exif.subprocess, "run", boom)
    assert exif.read_batch([Path("/x/a.jpg")]) == {}


def test_read_batch_bad_json(monkeypatch):
    monkeypatch.setattr(exif.subprocess, "run", lambda cmd, **kw: Completed("not json"))
    assert exif.read_batch([Path("/x/a.jpg")]) == {}


def test_create_date_fallback_and_subsec_padding():
    mapped = exif.map_tags({"EXIF:CreateDate": "2023:01:02 03:04:05"})
    assert mapped["captured_at"] == "2023-01-02T03:04:05.000Z"
    mapped = exif.map_tags({"EXIF:DateTimeOriginal": "2023:01:02 03:04:05", "EXIF:SubSecTimeOriginal": "05"})
    assert mapped["captured_at"] == "2023-01-02T03:04:05.050Z"


def test_zero_date_is_ignored():
    assert "captured_at" not in exif.map_tags({"EXIF:DateTimeOriginal": "0000:00:00 00:00:00"})


def test_parse_exif_datetime_variants():
    assert exif.parse_exif_datetime("2024:05:12 13:05:01") == datetime(2024, 5, 12, 13, 5, 1, tzinfo=UTC)
    assert exif.parse_exif_datetime("2024:05:12 13:05:01", "+02:00") == datetime(2024, 5, 12, 11, 5, 1, tzinfo=UTC)
    assert exif.parse_exif_datetime("2024:05:12 13:05:01.5Z") == datetime(2024, 5, 12, 13, 5, 1, 500000, tzinfo=UTC)
    assert exif.parse_exif_datetime("2024-05-12T13:05:01-05:00") == datetime(2024, 5, 12, 18, 5, 1, tzinfo=UTC)
    assert exif.parse_exif_datetime("garbage") is None
    assert exif.parse_exif_datetime(None) is None


def test_iso_roundtrip_is_sortable():
    a = exif.fmt_iso(datetime(2024, 5, 12, 13, 5, 1, 500000, tzinfo=UTC))
    b = exif.fmt_iso(datetime(2024, 5, 12, 13, 5, 2, tzinfo=UTC))
    assert a < b and exif.parse_iso(a) == datetime(2024, 5, 12, 13, 5, 1, 500000, tzinfo=UTC)


def test_captured_from_filename_patterns():
    f = exif.captured_from_filename
    assert f("IMG_20240512_130501.jpg") == "2024-05-12T13:05:01.000Z"
    assert f("PXL_20240512_130501123.jpg") == "2024-05-12T13:05:01.123Z"
    assert f("2024-05-12 13.05.01.jpg") == "2024-05-12T13:05:01.000Z"
    assert f("20240512_130501.jpg") == "2024-05-12T13:05:01.000Z"
    assert f("VID_20240512_130501.mp4") == "2024-05-12T13:05:01.000Z"
    assert f("/some/dir/IMG_20240512_130501.HEIC") == "2024-05-12T13:05:01.000Z"
    assert f("IMG_1234.jpg") is None
    assert f("IMG_20241332_130501.jpg") is None  # month 13


def test_subprocess_module_is_real():
    assert exif.subprocess is subprocess

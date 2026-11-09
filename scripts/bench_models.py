#!/usr/bin/env python3
"""Compare vision-language models served by llama-server on the same sample of photos.

Example::

    python scripts/bench_models.py --source /media/xin/PHOTOS --sample 200 \
        --models qwen3-vl-4b=http://127.0.0.1:8080/v1,qwen3.6-35b-a3b=http://127.0.0.1:8081/v1

Only one 8 GB GPU is assumed, so models are benchmarked one after another: start the llama-server (or the
matching systemd unit, see deploy/systemd/README.md) for the model you want to test, then run this script
with only that ``name=url`` (``--models`` may list several; the script just calls each endpoint in turn, so
all of them must be reachable at the time).  Run it several times with ``--merge`` to combine results.

For every model it reports seconds per image (median, p90), the share of replies that parsed as the JSON
schema of ARCHITECTURE.md section 5 (5), and the tag agreement (mean Jaccard) between each pair of models on
the same images.  Results go to ``bench_results.json``.

No GPU is needed to import or run this file; only the endpoints need one.
"""

from __future__ import annotations

import argparse
import base64
import io
import itertools
import json
import random
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".heic", ".webp", ".tif", ".tiff"}
RAW_EXTS = {".cr2", ".cr3", ".nef", ".arw", ".dng", ".raf", ".orf", ".rw2"}
SCENES = "landscape|portrait|street|night|food|document|screenshot|pet|event|other"

# Fallback prompt, used only when afterglow.integrations.llm cannot be imported.  It mirrors the schema of
# ARCHITECTURE.md section 5 (5): {"caption","scene","tags","people_count","quality_notes","style"}.
FALLBACK_SYSTEM = (
    "你是照片整理助手。只输出一个 JSON 对象，不要任何其它文字。"
    f"字段: caption(中文，不超过40字), scene({SCENES}之一), tags(最多8个中文短标签的数组), "
    "people_count(整数), quality_notes(字符串), style(landscape|portrait|night|default 之一)。 /no_think"
)
FALLBACK_USER = "请描述这张照片。"
REQUIRED_KEYS = ("caption", "scene", "tags")


# --------------------------------------------------------------------------------------- sampling / previews
def find_images(source: Path) -> list[Path]:
    exts = IMAGE_EXTS | RAW_EXTS
    skip = {"@eaDir", ".thumbnails", "System Volume Information"}
    found: list[Path] = []
    for p in source.rglob("*"):
        if any(part.startswith(".") or part in skip for part in p.relative_to(source).parts[:-1]):
            continue
        if p.is_file() and p.suffix.lower() in exts and not p.name.startswith("."):
            found.append(p)
    return sorted(found)


def make_preview(path: Path, max_edge: int, quality: int = 85) -> bytes:
    """JPEG bytes whose long edge is <= max_edge.  RAW: embedded thumbnail via rawpy when available."""
    from PIL import Image, ImageOps

    if path.suffix.lower() in RAW_EXTS:
        import rawpy  # optional; raises ImportError if missing -> caller skips the file

        with rawpy.imread(str(path)) as raw:
            thumb = raw.extract_thumb()
        if thumb.format == rawpy.ThumbFormat.JPEG:
            img = Image.open(io.BytesIO(thumb.data))
        else:
            img = Image.fromarray(thumb.data)
    else:
        img = Image.open(path)
    img = ImageOps.exif_transpose(img).convert("RGB")
    img.thumbnail((max_edge, max_edge))
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=quality)
    return buf.getvalue()


def sample_previews(source: Path, n: int, seed: int, max_edge: int) -> list[tuple[str, bytes]]:
    files = find_images(source)
    if not files:
        raise SystemExit(f"{source}: 没有找到图片")
    rng = random.Random(seed)
    rng.shuffle(files)
    out: list[tuple[str, bytes]] = []
    for p in files:
        if len(out) >= n:
            break
        try:
            out.append((str(p), make_preview(p, max_edge)))
        except Exception as exc:  # unreadable file: skip, do not abort the benchmark
            print(f"  跳过 {p.name}: {exc}", file=sys.stderr)
    return out


# --------------------------------------------------------------------------------------- model clients
class FallbackClient:
    """Minimal OpenAI-compatible client (httpx) used when afterglow's LLMClient is not importable."""

    def __init__(self, server: str, model: str, timeout: float) -> None:
        import httpx

        self.server = server.rstrip("/")
        self.model = model
        self.http = httpx.Client(timeout=timeout)

    def describe(self, image_jpeg: bytes, language: str = "zh") -> dict[str, Any]:
        b64 = base64.b64encode(image_jpeg).decode()
        body = {
            "model": self.model,
            "temperature": 0.2,
            "max_tokens": 400,
            "response_format": {"type": "json_object"},
            "chat_template_kwargs": {"enable_thinking": False},
            "messages": [
                {"role": "system", "content": FALLBACK_SYSTEM},
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                        {"type": "text", "text": FALLBACK_USER},
                    ],
                },
            ],
        }
        last: Exception | None = None
        for _ in range(2):  # one retry on bad JSON, like LLMClient
            r = self.http.post(f"{self.server}/chat/completions", json=body)
            r.raise_for_status()
            text = r.json()["choices"][0]["message"]["content"].strip()
            if text.startswith("```"):
                text = text.strip("`")
                text = text[4:] if text.lower().startswith("json") else text
            try:
                data = json.loads(text)
                if isinstance(data, dict):
                    return data
                raise ValueError("not an object")
            except ValueError as exc:
                last = exc
                body["temperature"] = 0.0
        raise ValueError(f"unparseable reply: {last}")


def make_client(server: str, model: str, timeout: float) -> tuple[Any, str]:
    try:
        sys.path.append(str(Path(__file__).resolve().parents[1]))  # running from a checkout without `pip install -e .`
        from afterglow.integrations.llm import LLMClient

        return LLMClient(server, model, timeout), "afterglow.integrations.llm.LLMClient"
    except Exception as exc:  # ImportError, or a signature mismatch while the module is being written
        print(f"  (LLMClient 不可用: {type(exc).__name__}: {exc}; 使用内置 httpx 客户端)", file=sys.stderr)
        return FallbackClient(server, model, timeout), "inline httpx fallback"


# --------------------------------------------------------------------------------------- measurement
@dataclass
class ModelRun:
    name: str
    url: str
    client_kind: str = ""
    seconds: list[float] = field(default_factory=list)  # successful and failed calls both counted
    ok: list[bool] = field(default_factory=list)
    tags: dict[str, set[str]] = field(default_factory=dict)  # image path -> normalised tag set
    scenes: dict[str, str] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)


def norm_tags(raw: Any) -> set[str]:
    if not isinstance(raw, list):
        return set()
    return {str(t).strip().lower() for t in raw if str(t).strip()}


def valid_reply(d: Any) -> bool:
    return isinstance(d, dict) and all(k in d for k in REQUIRED_KEYS) and isinstance(d.get("tags"), list)


def run_model(
    name: str, url: str, images: list[tuple[str, bytes]], language: str, timeout: float, warmup: int
) -> ModelRun:
    run = ModelRun(name=name, url=url)
    client, run.client_kind = make_client(url, name, timeout)
    for _, data in images[:warmup]:  # first requests load the model / warm caches; not timed
        try:
            client.describe(data, language)
        except Exception as exc:
            print(f"  [{name}] 预热失败: {exc}", file=sys.stderr)
    for i, (path, data) in enumerate(images, 1):
        t0 = time.perf_counter()
        try:
            reply = client.describe(data, language)
            good = valid_reply(reply)
            if not good:
                run.errors.append(f"{Path(path).name}: schema 不符")
        except Exception as exc:
            reply, good = {}, False
            run.errors.append(f"{Path(path).name}: {type(exc).__name__}: {exc}")
        run.seconds.append(time.perf_counter() - t0)
        run.ok.append(good)
        if good:
            run.tags[path] = norm_tags(reply.get("tags"))
            run.scenes[path] = str(reply.get("scene", ""))
        if i % 10 == 0 or i == len(images):
            print(f"  [{name}] {i}/{len(images)}", file=sys.stderr)
    return run


def percentile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    s = sorted(values)
    k = (len(s) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def agreement(a: ModelRun, b: ModelRun) -> dict[str, Any]:
    common = sorted(set(a.tags) & set(b.tags))
    if not common:
        return {"images": 0, "tag_jaccard_mean": None, "scene_agreement": None}
    return {
        "images": len(common),
        "tag_jaccard_mean": statistics.fmean(jaccard(a.tags[p], b.tags[p]) for p in common),
        "scene_agreement": sum(a.scenes[p] == b.scenes[p] for p in common) / len(common),
    }


def summarize(run: ModelRun) -> dict[str, Any]:
    n = len(run.seconds)
    return {
        "name": run.name,
        "url": run.url,
        "client": run.client_kind,
        "images": n,
        "seconds_median": statistics.median(run.seconds) if n else None,
        "seconds_p90": percentile(run.seconds, 0.9) if n else None,
        "seconds_mean": statistics.fmean(run.seconds) if n else None,
        "json_ok_rate": sum(run.ok) / n if n else None,
        "errors": run.errors[:20],
    }


# --------------------------------------------------------------------------------------- output
def print_tables(summaries: list[dict[str, Any]], pairs: list[dict[str, Any]]) -> None:
    try:
        from rich.console import Console
        from rich.table import Table
    except ImportError:  # rich is an Afterglow dependency, but keep the script usable standalone
        for s in summaries:
            print(s)
        for p in pairs:
            print(p)
        return
    console = Console()
    t = Table(title="Afterglow 模型对比 — 速度与 JSON 成功率")
    for col in ("模型", "图片数", "中位数 s/张", "P90 s/张", "JSON 成功率"):
        t.add_column(col, justify="left" if col == "模型" else "right")
    for s in summaries:
        t.add_row(
            s["name"],
            str(s["images"]),
            f"{s['seconds_median']:.2f}",
            f"{s['seconds_p90']:.2f}",
            f"{s['json_ok_rate'] * 100:.1f}%",
        )
    console.print(t)
    if pairs:
        t2 = Table(title="两两一致性(同一批图片)")
        for col in ("A", "B", "共同成功图片", "标签 Jaccard 均值", "场景一致率"):
            t2.add_column(col)
        for p in pairs:
            j, sc = p["tag_jaccard_mean"], p["scene_agreement"]
            t2.add_row(
                p["a"],
                p["b"],
                str(p["images"]),
                "-" if j is None else f"{j:.3f}",
                "-" if sc is None else f"{sc * 100:.1f}%",
            )
        console.print(t2)


def parse_models(spec: str) -> list[tuple[str, str]]:
    out = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise SystemExit(f"--models 格式应为 name=url,...；收到 {part!r}")
        name, url = part.split("=", 1)
        out.append((name.strip(), url.strip()))
    if not out:
        raise SystemExit("--models 为空")
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", type=Path, required=True, help="照片目录(递归；只读)")
    ap.add_argument("--sample", type=int, default=200, help="抽样张数，默认 200")
    ap.add_argument("--models", required=True, help="name=url,... 例如 qwen3-vl-4b=http://127.0.0.1:8080/v1")
    ap.add_argument("--seed", type=int, default=42, help="抽样随机种子")
    ap.add_argument("--max-edge", type=int, default=1024, help="预览图长边，应与 [llm].max_image_edge 一致")
    ap.add_argument("--language", default="zh")
    ap.add_argument("--timeout", type=float, default=120.0)
    ap.add_argument("--warmup", type=int, default=1, help="每个模型不计时的预热请求数")
    ap.add_argument("--out", type=Path, default=Path("bench_results.json"))
    args = ap.parse_args(argv)

    if not args.source.is_dir():
        raise SystemExit(f"{args.source}: 不是目录")
    models = parse_models(args.models)

    print(f"抽样 {args.sample} 张 (seed={args.seed}) 自 {args.source} ...", file=sys.stderr)
    images = sample_previews(args.source, args.sample, args.seed, args.max_edge)
    print(f"得到 {len(images)} 张预览", file=sys.stderr)

    runs: list[ModelRun] = []
    for name, url in models:
        print(f"== {name} @ {url}", file=sys.stderr)
        runs.append(run_model(name, url, images, args.language, args.timeout, args.warmup))

    summaries = [summarize(r) for r in runs]
    pairs = []
    for a, b in itertools.combinations(runs, 2):
        pairs.append({"a": a.name, "b": b.name, **agreement(a, b)})

    print_tables(summaries, pairs)
    result = {
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "source": str(args.source),
        "seed": args.seed,
        "sample": len(images),
        "max_edge": args.max_edge,
        "models": summaries,
        "agreement": pairs,
        "images": [p for p, _ in images],
    }
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"已写入 {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

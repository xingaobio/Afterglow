"""Standalone fake of LUMEN RAW's MCP stdio server (newline-delimited JSON-RPC) for tests.

Env: FAKE_LUMEN_FAIL_AFTER=N  (or fail_after=N) -> exit abruptly after answering N tools/call requests.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

TOOLS = [
    "analyze_photos", "open_photo", "auto_tone", "apply_edits", "export_photos", "photo_info", "run_command",
    "get_edits", "render_preview",
]
COMMANDS = {"photo.export", "edit.preset", "video.timelapse", "library.write_xmp"}


def send(message: dict) -> None:
    sys.stdout.write(json.dumps(message, ensure_ascii=False, separators=(",", ":")) + "\n")
    sys.stdout.flush()


def ok(result: dict) -> dict:
    return {
        "content": [{"type": "text", "text": json.dumps(result)}],
        "structuredContent": result,
        "isError": False,
    }


def err(text: str) -> dict:
    return {"content": [{"type": "text", "text": text}], "isError": True}


def sharpness(path: str) -> float:
    return int(hashlib.sha256(path.encode()).hexdigest()[:4], 16) % 100


def write_jpeg(output: str) -> None:
    from PIL import Image

    Image.new("RGB", (32, 24), (120, 90, 60)).save(output, "JPEG")


def run_command(command: str, args: dict) -> dict:
    if command == "photo.export":
        output = args["output"]
        if not Path(output).parent.is_dir():
            return err(f"folder not found: {Path(output).parent}")
        write_jpeg(output)
        return ok({"path": args.get("path"), "output": output, "bytes": Path(output).stat().st_size})
    if command == "edit.preset":
        return ok({"preset": args["name"], "applied": [args.get("path")]})
    if command == "video.timelapse":
        return ok({"output": args["output"], "frames": len(args.get("paths", []))})
    if command == "library.write_xmp":
        return ok({"written": [p + ".xmp" for p in args.get("paths", [])], "errors": []})
    return err(f"unknown command {command!r}")


def call_tool(name: str, args: dict) -> dict:
    if name == "analyze_photos":
        paths = args.get("paths", [])
        photos = [{"path": p, "sharpness": sharpness(p), "faces": 0, "eyes_closed": 0} for p in paths]
        bursts = [{"group": 1, "photos": paths[:3], "best": paths[0]}] if len(paths) >= 3 else []
        return ok({"count": len(photos), "photos": photos, "bursts": bursts})
    if name == "open_photo":
        return ok({"name": Path(args["path"]).name, "width": 6000, "height": 4000})
    if name == "apply_edits":
        return ok({"path": args.get("path"), "tool": name, "amount": args.get("amount"),
                   "changes": args.get("changes"), "label": args.get("label")})
    if name == "get_edits":
        return ok({"path": args.get("path"), "state": {"adjustments": {"exposure": 0.5, "contrast": 10}}})
    if name == "render_preview":
        import base64
        import io

        from PIL import Image

        buf = io.BytesIO()
        Image.new("RGB", (64, 48), (90, 120, 150)).save(buf, "JPEG")
        result = ok({"path": args.get("path"), "width": 64, "height": 48})
        result["content"].insert(0, {"type": "image", "data": base64.b64encode(buf.getvalue()).decode(),
                                     "mimeType": "image/jpeg"})
        return result
    if name in ("auto_tone", "photo_info"):
        return ok({"path": args.get("path"), "tool": name})
    if name == "export_photos":
        folder = args["folder"]
        outs = []
        for p in args.get("paths", []):
            out = str(Path(folder) / (Path(p).stem + "-Lumen." + args.get("format", "jpg")))
            write_jpeg(out)
            outs.append({"path": p, "output": out})
        return ok({"exported": outs, "failed": []})
    if name == "run_command":
        return run_command(args.get("command", ""), args.get("args") or {})
    return err(f"unknown tool {name}")


def main() -> int:
    fail_after = int(os.environ.get("FAKE_LUMEN_FAIL_AFTER") or os.environ.get("fail_after") or 0)
    served = 0
    for raw in sys.stdin:
        raw = raw.strip()
        if not raw:
            continue
        msg = json.loads(raw)
        method, ident, params = msg.get("method"), msg.get("id"), msg.get("params") or {}
        if ident is None:
            continue  # notification
        if method == "initialize":
            send({"jsonrpc": "2.0", "id": ident, "result": {
                "protocolVersion": params.get("protocolVersion"),
                "capabilities": {"tools": {"listChanged": False}},
                "serverInfo": {"name": "fake-lumen", "version": "0"},
            }})
        elif method == "tools/list":
            send({"jsonrpc": "2.0", "id": ident, "result": {"tools": [{"name": t} for t in TOOLS]}})
        elif method == "tools/call":
            result = call_tool(params.get("name"), params.get("arguments") or {})
            send({"jsonrpc": "2.0", "id": ident, "result": result})
            served += 1
            if fail_after and served >= fail_after:
                os._exit(3)
        else:
            send({"jsonrpc": "2.0", "id": ident, "error": {"code": -32601, "message": f"Method not found: {method}"}})
    return 0


if __name__ == "__main__":
    sys.exit(main())

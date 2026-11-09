from __future__ import annotations

import json

import httpx
import pytest

from afterglow.integrations.llm import LLMClient, LLMError


def make_client(replies: list, seen: list | None = None) -> LLMClient:
    """replies: list of content strings, or ints for HTTP error status codes."""
    queue = list(replies)

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/models"):
            return httpx.Response(200, json={"data": []})
        if seen is not None:
            seen.append(json.loads(request.content))
        item = queue.pop(0)
        if isinstance(item, int):
            return httpx.Response(item, text="boom")
        return httpx.Response(200, json={"choices": [{"message": {"content": item}}]})

    return LLMClient("http://llm/v1", "m", 5, transport=httpx.MockTransport(handler))


def test_healthy():
    assert make_client([]).healthy() is True


def test_request_shape():
    seen: list = []
    client = make_client(['{"a": 1}'], seen)
    assert client.chat_json("sys", "usr", image_jpeg=b"\xff\xd8abc") == {"a": 1}
    body = seen[0]
    assert body["response_format"] == {"type": "json_object"}
    assert body["chat_template_kwargs"] == {"enable_thinking": False}
    assert body["messages"][0]["content"].endswith("/no_think")
    parts = body["messages"][1]["content"]
    assert parts[0]["image_url"]["url"].startswith("data:image/jpeg;base64,")


def test_fence_and_think_stripping():
    assert make_client(['```json\n{"a": 1}\n```']).chat_json("s", "u") == {"a": 1}
    assert make_client(['<think>\nhmm {"x": 2}\n</think>\n{"a": 3}']).chat_json("s", "u") == {"a": 3}
    assert make_client(['<think>x</think>```\n{"a": 4}\n```']).chat_json("s", "u") == {"a": 4}


def test_retry_on_bad_json():
    seen: list = []
    client = make_client(["not json at all", '{"ok": true}'], seen)
    assert client.chat_json("s", "u") == {"ok": True}
    assert [b["temperature"] for b in seen][-1] == 0.0


def test_double_failure_and_http_error():
    with pytest.raises(LLMError):
        make_client(["nope", "still nope"]).chat_json("s", "u")
    with pytest.raises(LLMError):
        make_client([500]).chat_json("s", "u")


def test_describe_normalises():
    raw = {
        "caption": " 海边日落 ",
        "scene": "Sunset",
        "tags": ["海", "海", " 日落 ", "", "a", "b", "c", "d", "e", "f", "g"],
        "people_count": -2,
        "style": "weird",
    }
    out = make_client([json.dumps(raw)]).describe(b"img", "zh")
    assert out["caption"] == "海边日落"
    assert out["scene"] == "other" and out["style"] == "default"
    assert out["tags"][:2] == ["海", "日落"] and len(out["tags"]) == 8 and len(set(out["tags"])) == 8
    assert out["people_count"] == 0

    good = {"caption": "c", "scene": "night", "tags": [], "people_count": "3", "style": "night"}
    out = make_client([json.dumps(good)]).describe(b"img", "en")
    assert out["scene"] == "night" and out["style"] == "night" and out["people_count"] == 3


def test_describe_prompt_language():
    seen: list = []
    make_client(['{"caption": "x"}'], seen).describe(b"i", "zh")
    make_client(['{"caption": "x"}'], seen).describe(b"i", "en")
    assert "照片" in seen[0]["messages"][0]["content"]
    assert "photo" in seen[1]["messages"][0]["content"].lower()


def test_name_event_sanitises():
    reply = '{"name": "苏格兰/高地:徒步?*\\"<>|\\n之旅与更多更多"}'
    name = make_client([reply]).name_event(["a"], "2024-05-01", "Skye")
    assert not set('/\\:*?"<>|\n') & set(name)
    assert 0 < len(name) <= 12


def test_name_event_fallback():
    assert make_client(['{"name": "  "}']).name_event(["a"], "2024-05-01", "Skye") == "Skye"
    assert make_client(['{"name": "///"}']).name_event(["a"], "2024-05-01", None) == "照片"
    assert make_client([500]).name_event(["a"], "2024-05-01", None) == "照片"

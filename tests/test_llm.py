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


def test_grade_normalises_and_clamps():
    from afterglow.integrations.llm import GRADE_KEYS

    reply = json.dumps({
        "adjustments": {
            "exposure": 3.456, "contrast": 99, "highlights": -55.6, "shadows": "12.4", "whites": 0,
            "blacks": "abc", "temperature": 0.2, "tint": None, "vibrance": True, "saturation": 1e999,
            "dehaze": -40.4, "clarity": 7.5, "sharpness": 30, "hsl": {"red": 1},
        },
        "reason": " 压高光 ",
    })
    out = make_client([reply]).grade(b"img", "landscape", "海边", "自然")
    assert out["reason"] == "压高光"
    adj = out["adjustments"]
    assert set(adj) <= set(GRADE_KEYS) and len(GRADE_KEYS) == 12
    assert adj == {"exposure": 1.0, "contrast": 40, "highlights": -40, "shadows": 12, "dehaze": -40, "clarity": 8}


def test_grade_exposure_rounding_and_empty():
    reply = '{"adjustments": {"exposure": -0.456, "contrast": 0.3}, "reason": "x"}'
    out = make_client([reply]).grade(b"i", "night", "", "")
    assert out["adjustments"] == {"exposure": -0.46}
    assert make_client(['{"adjustments": "oops"}']).grade(b"i", "night", "", "") == {"adjustments": {}, "reason": ""}
    with pytest.raises(LLMError):
        make_client(["not json", "still not"]).grade(b"i", "night", "", "")


def test_grade_prompt_language_and_image():
    seen: list = []
    make_client(['{"adjustments": {}}'], seen).grade(b"JPEGDATA", "landscape", "a lake", "natural look", "zh")
    make_client(['{"adjustments": {}}'], seen).grade(b"JPEGDATA", "landscape", "a lake", "natural look", "en")
    zh, en = seen[0]["messages"][0]["content"], seen[1]["messages"][0]["content"]
    assert "调色" in zh and "偏暖" in zh and "洋红" in zh and "JSON" in zh
    assert "colourist" in en and "warmer" in en and "magenta" in en and "JSON" in en
    for body in seen:
        assert body["messages"][1]["content"][0]["type"] == "image_url"
    assert "a lake" in seen[0]["messages"][1]["content"][1]["text"]
    assert "natural look" in seen[1]["messages"][1]["content"][1]["text"]

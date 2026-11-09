"""llama-server (OpenAI-compatible) client with image input and JSON-structured output."""

from __future__ import annotations

import base64
import json
import logging
import re
from typing import Any

import httpx

log = logging.getLogger(__name__)

SCENES = ("landscape", "portrait", "street", "night", "food", "document", "screenshot", "pet", "event", "other")
STYLES = ("landscape", "portrait", "night", "default")
_FORBIDDEN = re.compile(r'[/\\:*?"<>|\r\n\t]')
_THINK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_FENCE = re.compile(r"^```[a-zA-Z0-9]*\s*\n?(.*?)\n?```\s*$", re.DOTALL)


class LLMError(RuntimeError):
    """The model server failed or returned unusable output."""


def _clean_json_text(text: str) -> str:
    text = _THINK.sub("", text)
    # an unterminated think block (truncated output): drop everything before the closing tag if any
    if "</think>" in text.lower():
        text = re.split(r"</think>", text, flags=re.IGNORECASE)[-1]
    text = text.strip()
    match = _FENCE.match(text)
    if match:
        text = match.group(1).strip()
    if not text.startswith(("{", "[")):
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end > start:
            text = text[start : end + 1]
    return text


class LLMClient:
    def __init__(
        self,
        server: str,
        model: str,
        timeout: float = 120.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.server = server.rstrip("/")
        self.model = model
        self._client = httpx.Client(timeout=timeout, transport=transport)

    def close(self) -> None:
        self._client.close()

    def healthy(self) -> bool:
        try:
            return self._client.get(f"{self.server}/models").status_code == 200
        except httpx.HTTPError:
            return False

    def _post(
        self, system: str, user: str, image_jpeg: bytes | None, schema_hint: str, max_tokens: int, temperature: float
    ) -> str:
        system_text = system.rstrip()
        if schema_hint:
            system_text += f"\n{schema_hint}"
        system_text += "\n/no_think"
        content: Any
        if image_jpeg is not None:
            b64 = base64.b64encode(image_jpeg).decode("ascii")
            content = [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                {"type": "text", "text": user},
            ]
        else:
            content = user
        body = {
            "model": self.model,
            "messages": [{"role": "system", "content": system_text}, {"role": "user", "content": content}],
            "response_format": {"type": "json_object"},
            "chat_template_kwargs": {"enable_thinking": False},
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        try:
            resp = self._client.post(f"{self.server}/chat/completions", json=body)
            resp.raise_for_status()
            data = resp.json()
            return str(data["choices"][0]["message"]["content"] or "")
        except httpx.HTTPError as exc:
            raise LLMError(f"LLM request failed: {exc}") from exc
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise LLMError(f"LLM returned an unexpected response: {exc}") from exc

    def chat_json(
        self,
        system: str,
        user: str,
        image_jpeg: bytes | None = None,
        schema_hint: str = "",
        max_tokens: int = 400,
    ) -> dict:
        last = ""
        for temperature in (0.2, 0.0):
            raw = self._post(system, user, image_jpeg, schema_hint, max_tokens, temperature)
            last = raw
            try:
                parsed = json.loads(_clean_json_text(raw))
            except ValueError:
                log.debug("LLM JSON parse failed, raw=%r", raw[:200])
                continue
            if isinstance(parsed, dict):
                return parsed
        raise LLMError(f"LLM did not return valid JSON: {last[:200]!r}")

    # -- task helpers --------------------------------------------------------------------------

    def describe(self, image_jpeg: bytes, language: str = "zh") -> dict:
        if language == "zh":
            system = (
                "你是照片整理助手。只输出一个 JSON 对象，不要任何其他文字。"
                "字段：caption（不超过40字的中文描述）、"
                f"scene（{'|'.join(SCENES)} 之一）、tags（最多8个简短中文标签的数组）、"
                "people_count（人数，整数）、quality_notes（一句话画质备注）、"
                f"style（{'|'.join(STYLES)} 之一）。"
            )
            user = "请描述这张照片。"
        else:
            system = (
                "You are a photo-library assistant. Output a single JSON object and nothing else. "
                "Fields: caption (English, at most 40 words), "
                f"scene (one of {', '.join(SCENES)}), tags (array of at most 8 short tags), "
                "people_count (integer), quality_notes (one short sentence about image quality), "
                f"style (one of {', '.join(STYLES)})."
            )
            user = "Describe this photo."
        raw = self.chat_json(system, user, image_jpeg, max_tokens=400)
        return self._normalise(raw)

    @staticmethod
    def _normalise(raw: dict) -> dict:
        scene = str(raw.get("scene", "")).strip().lower()
        style = str(raw.get("style", "")).strip().lower()
        tags_in = raw.get("tags")
        if isinstance(tags_in, str):
            tags_in = re.split(r"[,，、;；]", tags_in)
        tags: list[str] = []
        for tag in tags_in if isinstance(tags_in, list) else []:
            text = str(tag).strip().strip("#").strip()
            if text and len(text) <= 30 and text not in tags:
                tags.append(text)
        try:
            people = max(0, int(float(raw.get("people_count", 0))))
        except (TypeError, ValueError):
            people = 0
        return {
            "caption": str(raw.get("caption", "") or "").strip(),
            "scene": scene if scene in SCENES else "other",
            "tags": tags[:8],
            "people_count": people,
            "quality_notes": str(raw.get("quality_notes", "") or "").strip(),
            "style": style if style in STYLES else "default",
        }

    def name_event(self, captions: list[str], date: str, place: str | None, language: str = "zh") -> str:
        lines = "\n".join(f"- {c}" for c in captions if c)
        if language == "zh":
            system = (
                "你为一组照片起一个事件名称，不超过12个字，例如“苏格兰高地徒步”。"
                '只输出 JSON：{"name": "..."}。名称不要包含日期和特殊符号。'
            )
            user = f"日期：{date}\n地点：{place or '未知'}\n照片描述：\n{lines}"
        else:
            system = (
                "Give this photo event a short name of at most 12 characters, e.g. 'Highland Hike'. "
                'Output JSON only: {"name": "..."}. No dates or special symbols.'
            )
            user = f"Date: {date}\nPlace: {place or 'unknown'}\nPhoto captions:\n{lines}"
        fallback = place or "照片"
        try:
            raw = self.chat_json(system, user, max_tokens=100)
        except LLMError:
            log.warning("Event naming failed; using fallback %r", fallback)
            return self._sanitize_name(fallback) or "照片"
        name = self._sanitize_name(str(raw.get("name", "") or ""))
        return name or self._sanitize_name(fallback) or "照片"

    @staticmethod
    def _sanitize_name(name: str) -> str:
        name = _FORBIDDEN.sub("", name).strip().strip(".").strip()
        return name[:12].strip()

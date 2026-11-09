from __future__ import annotations

import json

import httpx
import pytest

from afterglow.integrations.immich import ImmichClient, ImmichError


def make(handler) -> ImmichClient:
    return ImmichClient("http://immich:2283/", "KEY", transport=httpx.MockTransport(handler))


def test_ping_and_header():
    seen = []

    def handler(req: httpx.Request) -> httpx.Response:
        seen.append(req)
        return httpx.Response(200, json={"res": "pong"})

    assert make(handler).ping() is True
    assert seen[0].url.path == "/api/server/ping" and seen[0].headers["x-api-key"] == "KEY"
    assert make(lambda r: httpx.Response(503)).ping() is False


def test_asset_by_path():
    def handler(req: httpx.Request) -> httpx.Response:
        assert req.url.path == "/api/search/metadata"
        body = json.loads(req.content)
        items = [{"id": "A1"}] if body["originalPath"].endswith("a.jpg") else []
        return httpx.Response(200, json={"assets": {"items": items}})

    client = make(handler)
    assert client.asset_by_path("/mnt/afterglow/a.jpg") == {"id": "A1"}
    assert client.asset_by_path("/mnt/afterglow/b.jpg") is None


def test_albums():
    calls = []

    def handler(req: httpx.Request) -> httpx.Response:
        calls.append((req.method, req.url.path, json.loads(req.content) if req.content else None))
        if req.method == "POST":
            return httpx.Response(201, json={"id": "ALB"})
        if req.method == "PUT":
            return httpx.Response(200, json=[{"id": "A1", "success": True}])
        return httpx.Response(204)

    client = make(handler)
    assert client.create_album("2024-05 Skye", ["A1"]) == "ALB"
    client.add_to_album("ALB", ["A2"])
    client.delete_album("ALB")
    assert calls[0] == ("POST", "/api/albums", {"albumName": "2024-05 Skye", "assetIds": ["A1"]})
    assert calls[1] == ("PUT", "/api/albums/ALB/assets", {"ids": ["A2"]})
    assert calls[2][:2] == ("DELETE", "/api/albums/ALB")


def test_pause_resume():
    calls = []

    def handler(req: httpx.Request) -> httpx.Response:
        calls.append((req.method, req.url.path, json.loads(req.content)))
        return httpx.Response(200, json={})

    client = make(handler)
    client.pause_job("smartSearch")
    client.resume_job("smartSearch")
    assert calls[0] == ("PUT", "/api/jobs/smartSearch", {"command": "pause", "force": False})
    assert calls[1][2]["command"] == "resume"


def test_pause_falls_back_to_queues():
    calls = []

    def handler(req: httpx.Request) -> httpx.Response:
        calls.append((req.method, req.url.path))
        return httpx.Response(404) if "/jobs/" in req.url.path else httpx.Response(200, json={})

    make(handler).pause_job("faceDetection")
    assert calls == [("PUT", "/api/jobs/faceDetection"), ("PUT", "/api/queues/faceDetection")]


def test_wait_idle():
    states = iter([3, 1, 0])

    def handler(req: httpx.Request) -> httpx.Response:
        n = next(states, 0)
        return httpx.Response(200, json={"library": {"jobCounts": {"active": n, "waiting": 0}}})

    assert make(handler).wait_idle(names=("library",), timeout=5, poll=0.01) is True


def test_wait_idle_timeout():
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"library": {"jobCounts": {"active": 0, "waiting": 5}}})

    assert make(handler).wait_idle(names=("library",), timeout=0.05, poll=0.01) is False


def test_errors_raise():
    with pytest.raises(ImmichError):
        make(lambda r: httpx.Response(401, text="no")).libraries()

    def boom(req: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("down")

    with pytest.raises(ImmichError):
        make(boom).scan_library("L")


def test_library_and_tags():
    def handler(req: httpx.Request) -> httpx.Response:
        p = req.url.path
        if p == "/api/users/me":
            return httpx.Response(200, json={"id": "U1"})
        if p == "/api/libraries":
            assert json.loads(req.content)["ownerId"] == "U1"
            return httpx.Response(201, json={"id": "L1"})
        if p == "/api/tags":
            return httpx.Response(200, json=[{"id": "T1", "value": "Afterglow/sea"}])
        return httpx.Response(200, json={"id": "A1"})

    client = make(handler)
    assert client.create_external_library("Afterglow", ["/mnt/afterglow"]) == "L1"
    assert client.upsert_tag("Afterglow/sea") == "T1"
    client.tag_assets("T1", ["A1"])
    client.untag_assets("T1", ["A1"])
    assert client.update_asset("A1", description="d", rating=3)["id"] == "A1"

"""Immich REST client.

Endpoint notes: each method names the endpoint and the Immich API docs page it belongs to
(https://api.immich.app/endpoints/<tag>/<operationId>). NOTE: these were written from the Immich v1.1xx/v2 API
as known when this module was authored; live verification against https://immich.app/docs/api could not be
done in the build sandbox, so re-check on first real run (`afterglow doctor`).
"""

from __future__ import annotations

import logging
import time
from typing import Any

import httpx

log = logging.getLogger(__name__)


class ImmichError(RuntimeError):
    """An Immich request failed."""


class ImmichClient:
    def __init__(
        self,
        url: str,
        api_key: str,
        timeout: float = 60.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.base = url.rstrip("/") + "/api"
        self._client = httpx.Client(
            headers={"x-api-key": api_key, "Accept": "application/json"}, timeout=timeout, transport=transport
        )

    def close(self) -> None:
        self._client.close()

    def _req(self, method: str, path: str, json_body: Any = None, ok_404: bool = False) -> Any:
        try:
            resp = self._client.request(method, self.base + path, json=json_body)
        except httpx.HTTPError as exc:
            raise ImmichError(f"Immich {method} {path} failed: {exc}") from exc
        if ok_404 and resp.status_code == 404:
            return None
        if resp.status_code >= 400:
            raise ImmichError(f"Immich {method} {path} -> HTTP {resp.status_code}: {resp.text[:200]}")
        if not resp.content:
            return None
        try:
            return resp.json()
        except ValueError:
            return resp.text

    # -- server --------------------------------------------------------------------------------

    # GET /api/server/ping  (docs: Server -> pingServer) -> {"res": "pong"}
    def ping(self) -> bool:
        try:
            data = self._req("GET", "/server/ping")
        except ImmichError:
            return False
        return isinstance(data, dict) and data.get("res") == "pong"

    # GET /api/users/me  (docs: Users -> getMyUser); needed because POST /libraries wants an ownerId
    def _my_user_id(self) -> str:
        data = self._req("GET", "/users/me")
        try:
            return str(data["id"])
        except (TypeError, KeyError) as exc:
            raise ImmichError("Immich /users/me returned no id") from exc

    # -- libraries -----------------------------------------------------------------------------

    # GET /api/libraries  (docs: Libraries -> getAllLibraries)
    def libraries(self) -> list[dict]:
        data = self._req("GET", "/libraries")
        return data if isinstance(data, list) else []

    # POST /api/libraries {name, ownerId, importPaths}  (docs: Libraries -> createLibrary) -> {id, ...}
    def create_external_library(
        self, name: str, import_paths: list[str], owner_id: str | None = None,
        exclusion_patterns: list[str] | None = None,
    ) -> str:
        body: dict[str, Any] = {
            "name": name, "ownerId": owner_id or self._my_user_id(), "importPaths": list(import_paths),
        }
        if exclusion_patterns:
            body["exclusionPatterns"] = list(exclusion_patterns)  # glob patterns, e.g. "**/_culled/**"
        data = self._req("POST", "/libraries", body)
        try:
            return str(data["id"])
        except (TypeError, KeyError) as exc:
            raise ImmichError("Immich createLibrary returned no id") from exc

    # POST /api/libraries/{id}/scan  (docs: Libraries -> scanLibrary) -> 204
    def scan_library(self, library_id: str) -> None:
        self._req("POST", f"/libraries/{library_id}/scan")

    # -- jobs / queues -------------------------------------------------------------------------

    # GET /api/jobs  (docs: Jobs -> getAllJobsStatus) -> {name: {jobCounts: {active, waiting, ...}, queueStatus}}
    # Newer servers expose the same data at GET /api/queues; used as fallback when /jobs is 404.
    def jobs(self) -> dict:
        data = self._req("GET", "/jobs", ok_404=True)
        if data is None:
            data = self._req("GET", "/queues")
        if isinstance(data, list):  # /queues may return a list of {name, ...}
            data = {str(q.get("name")): q for q in data if isinstance(q, dict)}
        return data if isinstance(data, dict) else {}

    # PUT /api/jobs/{name} {"command": "pause"|"resume", "force": false}  (docs: Jobs -> sendJobCommand)
    # Fallback for newer servers: PUT /api/queues/{name} {"isPaused": bool}  (docs: Queues -> updateQueue)
    def _job_command(self, name: str, command: str) -> None:
        body = {"command": command, "force": False}
        try:
            resp = self._client.request("PUT", f"{self.base}/jobs/{name}", json=body)
        except httpx.HTTPError as exc:
            raise ImmichError(f"Immich PUT /jobs/{name} failed: {exc}") from exc
        if resp.status_code == 404:
            self._req("PUT", f"/queues/{name}", {"isPaused": command == "pause"})
        elif resp.status_code >= 400:
            raise ImmichError(f"Immich PUT /jobs/{name} -> HTTP {resp.status_code}: {resp.text[:200]}")

    def pause_job(self, name: str) -> None:
        self._job_command(name, "pause")

    def resume_job(self, name: str) -> None:
        self._job_command(name, "resume")

    @staticmethod
    def _busy(entry: Any) -> bool:
        if not isinstance(entry, dict):
            return False
        counts = entry.get("jobCounts") or entry.get("statistics") or {}
        status = entry.get("queueStatus") or {}
        if status.get("isPaused"):
            return False  # a paused queue never drains; do not wait on it
        return bool(counts.get("active", 0)) or bool(counts.get("waiting", 0))

    def wait_idle(
        self,
        names: tuple[str, ...] = ("library", "metadataExtraction", "thumbnailGeneration"),
        timeout: float = 1800,
        poll: float = 3.0,
    ) -> bool:
        deadline = time.monotonic() + timeout
        while True:
            jobs = self.jobs()
            if not any(self._busy(jobs.get(n)) for n in names):
                return True
            if time.monotonic() >= deadline:
                log.warning("Immich queues still busy after %.0fs", timeout)
                return False
            time.sleep(min(poll, max(0.0, deadline - time.monotonic())))

    # -- assets --------------------------------------------------------------------------------

    # POST /api/search/metadata {"originalPath": ...}  (docs: Search -> searchAssets) -> {assets: {items: [...]}}
    def asset_by_path(self, original_path: str) -> dict | None:
        data = self._req("POST", "/search/metadata", {"originalPath": original_path, "size": 1})
        items = ((data or {}).get("assets") or {}).get("items") or []
        return items[0] if items else None

    # PUT /api/assets/{id} {description?, rating?}  (docs: Assets -> updateAsset) -> asset
    def update_asset(self, asset_id: str, description: str | None = None, rating: int | None = None) -> dict:
        body: dict[str, Any] = {}
        if description is not None:
            body["description"] = description
        if rating is not None:
            body["rating"] = rating
        data = self._req("PUT", f"/assets/{asset_id}", body)
        return data if isinstance(data, dict) else {}

    # -- people --------------------------------------------------------------------------------

    # NOTE: the two endpoints below are UNVERIFIED against the current Immich docs (the sandbox had no access);
    # they follow the API as remembered. Re-check on first real run.
    #
    # GET /api/people?withHidden=<bool>&page=<n>&size=<n>  (docs: People -> getAllPeople)
    #   -> {"people": [{id, name, isHidden, ...}], "total", "hidden", "hasNextPage"}; older servers return a bare list.
    def people(self, with_hidden: bool = False) -> list[dict]:
        out: list[dict] = []
        page = 1
        while page <= 1000:  # hard stop against a server that never clears hasNextPage
            path = f"/people?withHidden={'true' if with_hidden else 'false'}&page={page}&size=500"
            data = self._req("GET", path)
            if isinstance(data, list):
                out.extend(p for p in data if isinstance(p, dict))
                break
            if not isinstance(data, dict):
                break
            out.extend(p for p in (data.get("people") or []) if isinstance(p, dict))
            if not data.get("hasNextPage"):
                break
            page += 1
        return out

    # POST /api/search/metadata {"personIds": [id], "page": n, "size": 250}  (docs: Search -> searchAssets)
    #   -> {"assets": {"items": [{id, originalPath, ...}], "nextPage": "2" | null}}
    def person_assets(self, person_id: str) -> list[dict]:
        out: list[dict] = []
        page: int | None = 1
        seen_pages: set[int] = set()
        while page is not None and page not in seen_pages:
            seen_pages.add(page)
            data = self._req("POST", "/search/metadata", {"personIds": [person_id], "page": page, "size": 250})
            block = (data or {}).get("assets") if isinstance(data, dict) else None
            block = block if isinstance(block, dict) else {}
            out.extend(a for a in (block.get("items") or []) if isinstance(a, dict))
            nxt = block.get("nextPage")
            try:
                page = int(nxt) if nxt not in (None, "") else None
            except (TypeError, ValueError):
                page = None
        return out

    # -- albums --------------------------------------------------------------------------------

    # POST /api/albums {albumName, assetIds}  (docs: Albums -> createAlbum) -> {id, ...}
    def create_album(self, name: str, asset_ids: list[str]) -> str:
        data = self._req("POST", "/albums", {"albumName": name, "assetIds": list(asset_ids)})
        try:
            return str(data["id"])
        except (TypeError, KeyError) as exc:
            raise ImmichError("Immich createAlbum returned no id") from exc

    # PUT /api/albums/{id}/assets {ids}  (docs: Albums -> addAssetsToAlbum) -> per-asset results
    def add_to_album(self, album_id: str, asset_ids: list[str]) -> None:
        if asset_ids:
            self._req("PUT", f"/albums/{album_id}/assets", {"ids": list(asset_ids)})

    # DELETE /api/albums/{id}  (docs: Albums -> deleteAlbum) -> 204
    def delete_album(self, album_id: str) -> None:
        self._req("DELETE", f"/albums/{album_id}")

    # -- tags ----------------------------------------------------------------------------------

    # PUT /api/tags {"tags": [name]}  (docs: Tags -> upsertTags); "a/b" names create hierarchy -> [{id, value, ...}]
    def upsert_tag(self, name: str) -> str:
        data = self._req("PUT", "/tags", {"tags": [name]})
        try:
            return str(data[0]["id"])
        except (TypeError, KeyError, IndexError) as exc:
            raise ImmichError(f"Immich upsertTags returned no id for {name!r}") from exc

    # PUT /api/tags/{id}/assets {ids}  (docs: Tags -> tagAssets)
    def tag_assets(self, tag_id: str, asset_ids: list[str]) -> None:
        if asset_ids:
            self._req("PUT", f"/tags/{tag_id}/assets", {"ids": list(asset_ids)})

    # DELETE /api/tags/{id}/assets {ids}  (docs: Tags -> untagAssets)
    def untag_assets(self, tag_id: str, asset_ids: list[str]) -> None:
        if asset_ids:
            self._req("DELETE", f"/tags/{tag_id}/assets", {"ids": list(asset_ids)})

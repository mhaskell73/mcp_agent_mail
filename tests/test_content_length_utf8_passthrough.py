"""Regression: _base_passthrough must not forward a stale Content-Length.

Bug (agent-mail #C / bead content-length-fix-6164229-mcw6): the base-path
passthrough copied every upstream response header verbatim -- including
`content-length` -- and then re-rendered the body via ``JSONResponse``.
Starlette's ``init_headers`` keeps a caller-supplied ``content-length`` and
never recomputes it (``populate_content_length = b"content-length" not in
keys``), so a re-rendered multibyte body shipped under the upstream's stale
byte count -> uvicorn "Response content longer than Content-Length" -> crash /
intermittent fetch_inbox read failures.

Two distinct defects are pinned here, each via the REAL ``_base_passthrough``
running inside a real ``build_http_app`` FastAPI app (the passthrough closure
re-dispatches to the mounted ``stateless_app``, which we stand in for with a
controllable ASGI app that emits exactly the bytes/headers we choose):

1. ``test_stale_content_length_recomputed_on_multibyte_body`` -- upstream
   declares a wrong Content-Length (character count, not byte count) for a
   multibyte JSON body. The passthrough must ship the body verbatim with a
   Content-Length equal to its actual byte length.
2. ``test_multichunk_body_accumulated_verbatim`` -- upstream streams the body
   across two ASGI chunks (split mid-multibyte-char). The passthrough must
   concatenate the raw chunks verbatim, not overwrite/re-parse per chunk.

Both assertions fail against the pre-fix code (stale CL retained; per-chunk
overwrite collapses the body to ``{}``) and pass after the fix.

Reference: bead mi-child-support-calculator-content-length-fix-6164229-mcw6.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pytest
from httpx import ASGITransport, AsyncClient

from mcp_agent_mail import config as _config
from mcp_agent_mail.http import build_http_app

# The passthrough POST route is always registered on the trailing-slash-free
# "/api" alias, regardless of the configured base path (which owns the mount).
# Posting here forces the request through _base_passthrough rather than the
# raw mount.
_PASSTHROUGH_PATH = "/api"


class _CraftedUpstreamApp:
    """Minimal ASGI stand-in for the mounted MCP app.

    Emits a caller-chosen (status, headers, body-chunks) response so we can
    exercise _base_passthrough's response reconstruction against a body whose
    real byte length disagrees with the declared Content-Length.
    """

    def __init__(self, raw_headers: list[tuple[bytes, bytes]], chunks: list[bytes]) -> None:
        self._raw_headers = raw_headers
        self._chunks = chunks
        # _HeaderFixupMCPApp._ensure_lifespan short-circuits when the wrapped
        # app already exposes a running session manager, so we never need a
        # real lifespan in-test.
        self.state = SimpleNamespace(session_manager=SimpleNamespace(_task_group=object()))

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        assert scope["type"] == "http"
        await send(
            {
                "type": "http.response.start",
                "status": 200,
                "headers": self._raw_headers,
            }
        )
        last = len(self._chunks) - 1
        for i, chunk in enumerate(self._chunks):
            await send(
                {
                    "type": "http.response.body",
                    "body": chunk,
                    "more_body": i != last,
                }
            )


class _StubServer:
    """Stand-in for the FastMCP server: only .http_app() is consumed by build_http_app."""

    def __init__(self, upstream: _CraftedUpstreamApp) -> None:
        self._upstream = upstream

    def http_app(self, *args: Any, **kwargs: Any) -> _CraftedUpstreamApp:
        return self._upstream


def _build_app_with_upstream(raw_headers: list[tuple[bytes, bytes]], chunks: list[bytes]):
    settings = _config.get_settings()
    upstream = _CraftedUpstreamApp(raw_headers, chunks)
    return build_http_app(settings, _StubServer(upstream))


@pytest.mark.asyncio
async def test_stale_content_length_recomputed_on_multibyte_body(isolated_env):
    """A wrong upstream Content-Length must not survive: CL must equal actual bytes."""
    # Multibyte payload; byte length > character length (em-dash, arrow, star,
    # emoji, accented chars -- the exact class that crashed the old code).
    body = json.dumps(
        {"note": "café — naïve → π ★ 🚀 employer LRAP"},
        ensure_ascii=False,
    ).encode("utf-8")
    char_count = len(body.decode("utf-8"))
    assert char_count < len(body), "fixture must be genuinely multibyte"

    raw_headers = [
        (b"content-type", b"application/json"),
        # Deliberately WRONG: character count, not byte count (what a naive
        # upstream serializer that miscounts multibyte would declare).
        (b"content-length", str(char_count).encode("latin-1")),
    ]
    app = _build_app_with_upstream(raw_headers, [body])

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(_PASSTHROUGH_PATH, content=body,
                                 headers={"Content-Type": "application/json"})

    assert resp.status_code == 200
    # The body reaches the client verbatim...
    assert resp.content == body
    # ...and the declared Content-Length matches the actual byte length (the fix:
    # the stale character-count CL must have been dropped and recomputed).
    assert int(resp.headers["content-length"]) == len(resp.content) == len(body)


@pytest.mark.asyncio
async def test_multichunk_body_accumulated_verbatim(isolated_env):
    """A body streamed across ASGI chunks must be concatenated verbatim, not per-chunk re-parsed."""
    full = json.dumps(
        {"items": ["café", "→", "🚀"], "count": 3, "sym": "★π"},
        ensure_ascii=False,
    ).encode("utf-8")
    # Split at a byte offset that lands inside a multibyte character, so neither
    # half is independently valid UTF-8/JSON. The old per-chunk json.loads
    # overwrite collapses this to {}; the fix concatenates the raw bytes.
    mid = len(full) // 2
    chunk1, chunk2 = full[:mid], full[mid:]
    assert chunk1 and chunk2

    raw_headers = [
        (b"content-type", b"application/json"),
        (b"content-length", str(len(full)).encode("latin-1")),
    ]
    app = _build_app_with_upstream(raw_headers, [chunk1, chunk2])

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(_PASSTHROUGH_PATH, content=full,
                                 headers={"Content-Type": "application/json"})

    assert resp.status_code == 200
    assert resp.content == full
    assert int(resp.headers["content-length"]) == len(resp.content) == len(full)

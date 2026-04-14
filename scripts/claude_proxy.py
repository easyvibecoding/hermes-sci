#!/usr/bin/env python3
"""Minimal Anthropic-compatible proxy that delegates to `claude -p`.

Why: let Anthropic-SDK-based code (AI Scientist, upstream libs) run against
the user's Claude subscription via `claude -p` subprocess instead of requiring
an ANTHROPIC_API_KEY + per-token billing.

Scope: POST /v1/messages, non-streaming, text-only content blocks.
Streaming requests are accepted but buffered server-side then replayed as a
single fake SSE stream (good enough for clients that only check token totals).

Start:
    python3 claude_proxy.py --host 127.0.0.1 --port 9099

Env:
    CLAUDE_PROXY_BIN        override `claude` binary path (default: claude)
    CLAUDE_PROXY_MAX_TURNS  claude -p --max-turns (default: 1)
    CLAUDE_PROXY_LOG        log file path (default: ~/.hermes/logs/claude_proxy.log)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import pathlib
import subprocess
import sys
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

CLAUDE_BIN = os.environ.get("CLAUDE_PROXY_BIN", "claude")
MAX_TURNS = os.environ.get("CLAUDE_PROXY_MAX_TURNS", "1")
LOG_PATH = pathlib.Path(
    os.environ.get("CLAUDE_PROXY_LOG", str(pathlib.Path.home() / ".hermes" / "logs" / "claude_proxy.log"))
)
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    filename=str(LOG_PATH),
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("claude_proxy")


def _flatten_content(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
            elif isinstance(block, dict) and block.get("type") == "tool_result":
                parts.append(f"[tool_result] {block.get('content', '')}")
            elif isinstance(block, str):
                parts.append(block)
        return "\n".join(parts)
    return str(content)


def build_prompt(system, messages) -> str:
    lines = []
    if system:
        sys_text = _flatten_content(system)
        if sys_text:
            lines.append(f"# System Instructions\n\n{sys_text}\n")
    for m in messages or []:
        role = m.get("role", "user").capitalize()
        text = _flatten_content(m.get("content", ""))
        lines.append(f"## {role}\n\n{text}\n")
    return "\n".join(lines).strip()


def call_claude(prompt: str) -> dict:
    cmd = [
        CLAUDE_BIN, "-p",
        "--output-format", "json",
        "--max-turns", str(MAX_TURNS),
        "--permission-mode", "bypassPermissions",
    ]
    log.info("spawn: %s (prompt=%d chars)", " ".join(cmd), len(prompt))
    t0 = time.time()
    proc = subprocess.run(
        cmd, input=prompt, capture_output=True, text=True, timeout=600,
    )
    elapsed = time.time() - t0
    if proc.returncode != 0:
        log.error("claude exit=%d stderr=%s", proc.returncode, proc.stderr[:500])
        raise RuntimeError(f"claude -p failed: {proc.stderr[:500]}")
    try:
        data = json.loads(proc.stdout)
    except json.JSONDecodeError as e:
        log.error("non-JSON stdout[:500]=%s", proc.stdout[:500])
        raise RuntimeError(f"claude -p did not emit JSON: {e}")
    log.info("ok in %.2fs, tokens=%s/%s",
             elapsed,
             data.get("usage", {}).get("input_tokens"),
             data.get("usage", {}).get("output_tokens"))
    return data


def to_anthropic_response(claude_json: dict, req_model: str) -> dict:
    text = claude_json.get("result", "")
    usage = claude_json.get("usage", {}) or {}
    return {
        "id": f"msg_{uuid.uuid4().hex[:24]}",
        "type": "message",
        "role": "assistant",
        "model": req_model or "claude-proxy",
        "content": [{"type": "text", "text": text}],
        "stop_reason": claude_json.get("stop_reason", "end_turn"),
        "stop_sequence": None,
        "usage": {
            "input_tokens": int(usage.get("input_tokens") or 0),
            "output_tokens": int(usage.get("output_tokens") or 0),
            "cache_creation_input_tokens": int(usage.get("cache_creation_input_tokens") or 0),
            "cache_read_input_tokens": int(usage.get("cache_read_input_tokens") or 0),
        },
    }


def fake_stream_events(response: dict):
    msg_id = response["id"]
    # message_start
    yield {
        "type": "message_start",
        "message": {**response, "content": [], "usage": {"input_tokens": response["usage"]["input_tokens"], "output_tokens": 0}},
    }
    yield {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}}
    text = response["content"][0]["text"]
    # chunk the text in one go (buffered)
    yield {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": text}}
    yield {"type": "content_block_stop", "index": 0}
    yield {"type": "message_delta", "delta": {"stop_reason": response["stop_reason"], "stop_sequence": None}, "usage": {"output_tokens": response["usage"]["output_tokens"]}}
    yield {"type": "message_stop"}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        log.info("%s - %s", self.address_string(), fmt % args)

    def _send_json(self, code: int, obj: dict):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_sse_stream(self, events):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        for evt in events:
            data = json.dumps(evt)
            self.wfile.write(f"event: {evt['type']}\ndata: {data}\n\n".encode("utf-8"))
            self.wfile.flush()

    def do_GET(self):
        if self.path.startswith("/health"):
            self._send_json(200, {"ok": True, "backend": "claude -p", "bin": CLAUDE_BIN})
            return
        self._send_json(404, {"error": "not found"})

    def do_POST(self):
        if not self.path.startswith("/v1/messages"):
            self._send_json(404, {"type": "error", "error": {"type": "not_found", "message": self.path}})
            return

        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length) if length else b""
        try:
            payload = json.loads(raw.decode("utf-8")) if raw else {}
        except json.JSONDecodeError as e:
            self._send_json(400, {"type": "error", "error": {"type": "invalid_request_error", "message": str(e)}})
            return

        stream = bool(payload.get("stream"))
        req_model = payload.get("model", "claude-proxy")
        prompt = build_prompt(payload.get("system"), payload.get("messages", []))
        if not prompt:
            self._send_json(400, {"type": "error", "error": {"type": "invalid_request_error", "message": "empty prompt"}})
            return

        try:
            claude_json = call_claude(prompt)
        except Exception as e:
            log.exception("proxy error")
            self._send_json(500, {"type": "error", "error": {"type": "api_error", "message": str(e)}})
            return

        resp = to_anthropic_response(claude_json, req_model)
        if stream:
            self._send_sse_stream(fake_stream_events(resp))
        else:
            self._send_json(200, resp)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=9099)
    args = p.parse_args()

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"claude_proxy listening on http://{args.host}:{args.port}  (log: {LOG_PATH})")
    log.info("server start %s:%d bin=%s", args.host, args.port, CLAUDE_BIN)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        log.info("server stop (KeyboardInterrupt)")
        server.server_close()


if __name__ == "__main__":
    sys.exit(main() or 0)

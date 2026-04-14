#!/usr/bin/env bash
# claude_proxy_ctl.sh — start / stop / status for the Anthropic-compatible shim.
#
# Usage:
#   bash claude_proxy_ctl.sh start [--host 127.0.0.1] [--port 9099]
#   bash claude_proxy_ctl.sh stop
#   bash claude_proxy_ctl.sh status
#   bash claude_proxy_ctl.sh ensure        # start if not running, no-op if up

set -u

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROXY_PY="$SCRIPT_DIR/claude_proxy.py"
PID_FILE="$HOME/.hermes/logs/claude_proxy.pid"
LOG_FILE="$HOME/.hermes/logs/claude_proxy.out"
mkdir -p "$(dirname "$PID_FILE")"

HOST="${CLAUDE_PROXY_HOST:-127.0.0.1}"
PORT="${CLAUDE_PROXY_PORT:-9099}"

is_running() {
    [[ -f "$PID_FILE" ]] || return 1
    local pid
    pid=$(cat "$PID_FILE" 2>/dev/null || true)
    [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null
}

health_check() {
    curl -fsS -m 3 "http://${HOST}:${PORT}/health" >/dev/null 2>&1
}

cmd_start() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --host) HOST="$2"; shift 2 ;;
            --port) PORT="$2"; shift 2 ;;
            *) echo "❌ unknown arg: $1" >&2; exit 2 ;;
        esac
    done

    if is_running; then
        echo "already running (pid=$(cat "$PID_FILE"))"
        return 0
    fi

    nohup python3 "$PROXY_PY" --host "$HOST" --port "$PORT" >>"$LOG_FILE" 2>&1 &
    echo $! > "$PID_FILE"
    sleep 0.7
    if ! is_running; then
        echo "❌ failed to start (check $LOG_FILE)" >&2
        return 1
    fi
    for _ in 1 2 3 4 5; do
        health_check && { echo "✅ claude_proxy up on http://${HOST}:${PORT} (pid=$(cat "$PID_FILE"))"; return 0; }
        sleep 0.5
    done
    echo "⚠️  started but /health did not respond yet; tail $LOG_FILE"
    return 0
}

cmd_stop() {
    if ! is_running; then
        echo "not running"
        rm -f "$PID_FILE"
        return 0
    fi
    local pid
    pid=$(cat "$PID_FILE")
    kill "$pid" 2>/dev/null || true
    sleep 0.3
    kill -0 "$pid" 2>/dev/null && kill -9 "$pid" 2>/dev/null || true
    rm -f "$PID_FILE"
    echo "stopped (pid=$pid)"
}

cmd_status() {
    if is_running; then
        echo "running  pid=$(cat "$PID_FILE")  endpoint=http://${HOST}:${PORT}"
        health_check && echo "health=ok" || echo "health=unreachable"
    else
        echo "stopped"
    fi
}

cmd_ensure() {
    if is_running && health_check; then
        return 0
    fi
    cmd_stop >/dev/null 2>&1 || true
    cmd_start
}

case "${1:-}" in
    start)   shift; cmd_start "$@" ;;
    stop)    cmd_stop ;;
    status)  cmd_status ;;
    ensure)  cmd_ensure ;;
    ""|--help|-h)
        echo "Usage: $0 {start|stop|status|ensure} [--host H] [--port P]"
        exit 0
        ;;
    *) echo "❌ unknown cmd: $1" >&2; exit 2 ;;
esac

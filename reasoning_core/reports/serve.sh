#!/usr/bin/env bash
# Serve the built reports over Tailscale so other agents / machines can open them.
#
#   reasoning_core/reports/serve.sh [port] [dir]
#
# Binds to the Tailscale address only, never 0.0.0.0: the magnet host is on a shared network and a
# wildcard bind would expose unpublished results to it. If tailscale is not up, this refuses rather
# than silently falling back to a public bind.
set -euo pipefail
PORT="${1:-8778}"
DIR="${2:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/build}"

IP="$(tailscale ip -4 2>/dev/null | head -1 || true)"
if [ -z "$IP" ]; then
  echo "tailscale has no IPv4 address here (is 'tailscale up' done?)." >&2
  echo "Refusing to bind 0.0.0.0 -- pass an explicit address if you really want that." >&2
  exit 1
fi
[ -d "$DIR" ] || { echo "no such directory: $DIR" >&2; exit 1; }

echo "serving $DIR"
echo "  http://$IP:$PORT/   (tailnet only)"
NAME="$(tailscale status --json 2>/dev/null | python3 -c 'import json,sys; print(json.load(sys.stdin)["Self"]["DNSName"].rstrip("."))' 2>/dev/null || true)"
[ -n "$NAME" ] && echo "  http://$NAME:$PORT/"
exec python3 -m http.server "$PORT" --bind "$IP" --directory "$DIR"

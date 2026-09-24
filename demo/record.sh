#!/usr/bin/env bash
# Record a narrated demo and regenerate its checked-in asciicast and GIF.
#
#   ./demo/record.sh                  # greeks-on-gpu, on Vulkan
#   ./demo/record.sh greeks-on-gpu    # explicit equivalent
#
# Set NABLATENSOR_DEMO_ENGINE to cuda, rocm, or cpu-jit only when deliberately
# recording for that backend. The default must stay Vulkan: README embeds the
# Greeks recording as the Vulkan GPU demonstration.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCENARIO="${1:-greeks-on-gpu}"
ENGINE="${NABLATENSOR_DEMO_ENGINE:-vulkan}"
ASSETS="$ROOT/docs/assets"
DEMO="$ROOT/demo/$SCENARIO.sh"
CAST="$ASSETS/$SCENARIO.cast"
GIF="$ASSETS/$SCENARIO.gif"

usage() {
  echo "usage: $0 [scenario]" >&2
  exit 2
}

[[ $# -le 1 ]] || usage
[[ "$SCENARIO" =~ ^[a-z0-9][a-z0-9-]*$ ]] || usage
[[ -x "$DEMO" ]] || { echo "demo not found: $DEMO" >&2; exit 2; }
case "$ENGINE" in
  cuda|vulkan|rocm|cpu-jit) ;;
  *) echo "unsupported NABLATENSOR_DEMO_ENGINE: $ENGINE" >&2; exit 2 ;;
esac
command -v asciinema >/dev/null || { echo "asciinema is required" >&2; exit 1; }
command -v docker >/dev/null || { echo "docker is required to render the GIF" >&2; exit 1; }

mkdir -p "$ASSETS"
TMP_CAST="$(mktemp "$ASSETS/$SCENARIO.XXXXXX.cast")"
TMP_GIF="$(mktemp "$ASSETS/$SCENARIO.XXXXXX.gif")"
cleanup() { rm -f "$TMP_CAST" "$TMP_GIF"; }
trap cleanup EXIT

title="${SCENARIO//-/ }"
asciinema rec --overwrite --quiet --title "$title" --cols 95 --rows 38 \
  --idle-time-limit 2 \
  -c "NABLATENSOR_DEMO_ENGINE=$ENGINE $DEMO" \
  "$TMP_CAST"

docker run --rm --user "$(id -u):$(id -g)" \
  -v "$ASSETS:/assets" \
  ghcr.io/asciinema/agg:latest \
  --theme github-dark --font-size 14 --idle-time-limit 2 --last-frame-duration 3 \
  "/assets/$(basename "$TMP_CAST")" "/assets/$(basename "$TMP_GIF")"

mv "$TMP_CAST" "$CAST"
mv "$TMP_GIF" "$GIF"
trap - EXIT

printf 'recorded %s on %s:\n  %s\n  %s\n' "$SCENARIO" "$ENGINE" "$CAST" "$GIF"

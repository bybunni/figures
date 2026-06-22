#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  scripts/join_partition_mpg.sh [2d.mpg] [3d.mpg] [joined.mpg]

Defaults:
  2d.mpg      output/partition_iterations.mpg
  3d.mpg      output/partition_iterations_3d.mpg
  joined.mpg  output/partition_iterations_joined.mpg

Environment:
  JOIN_MPG_HEIGHT              output height for each side, default: 720
  JOIN_MPG_BITRATE             MPEG-1 video bitrate, default: 8000k
  JOIN_MPG_DURATION_TOLERANCE  allowed duration delta in seconds, default: 0.05
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

input_2d="${1:-output/partition_iterations.mpg}"
input_3d="${2:-output/partition_iterations_3d.mpg}"
output="${3:-output/partition_iterations_joined.mpg}"
height="${JOIN_MPG_HEIGHT:-720}"
bitrate="${JOIN_MPG_BITRATE:-8000k}"
tolerance="${JOIN_MPG_DURATION_TOLERANCE:-0.05}"

for cmd in ffmpeg ffprobe awk mkdir dirname; do
  if ! command -v "$cmd" >/dev/null 2>&1; then
    echo "error: required command not found: $cmd" >&2
    exit 1
  fi
done

for input in "$input_2d" "$input_3d"; do
  if [[ ! -f "$input" ]]; then
    echo "error: input file does not exist: $input" >&2
    exit 1
  fi
done

duration_seconds() {
  ffprobe \
    -v error \
    -show_entries format=duration \
    -of default=noprint_wrappers=1:nokey=1 \
    "$1"
}

duration_2d="$(duration_seconds "$input_2d")"
duration_3d="$(duration_seconds "$input_3d")"
duration_delta="$(
  awk -v a="$duration_2d" -v b="$duration_3d" 'BEGIN {
    d = a - b
    if (d < 0) d = -d
    printf "%.6f", d
  }'
)"

if ! awk -v d="$duration_delta" -v t="$tolerance" 'BEGIN { exit(d <= t ? 0 : 1) }'; then
  cat >&2 <<EOF
error: input durations differ by ${duration_delta}s, above tolerance ${tolerance}s
  2D: ${input_2d} (${duration_2d}s)
  3D: ${input_3d} (${duration_3d}s)

Regenerate matching 2D/3D animations from the same run, or set
JOIN_MPG_DURATION_TOLERANCE if this mismatch is expected.
EOF
  exit 1
fi

mkdir -p "$(dirname "$output")"

ffmpeg \
  -hide_banner \
  -loglevel error \
  -y \
  -i "$input_2d" \
  -i "$input_3d" \
  -filter_complex "[0:v]setpts=PTS-STARTPTS,scale=-2:${height}[left];[1:v]setpts=PTS-STARTPTS,scale=-2:${height}[right];[left][right]hstack=inputs=2[v]" \
  -map "[v]" \
  -an \
  -r 24 \
  -c:v mpeg1video \
  -b:v "$bitrate" \
  -pix_fmt yuv420p \
  "$output"

echo "wrote $output"

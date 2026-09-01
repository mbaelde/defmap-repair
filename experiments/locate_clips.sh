#!/bin/sh
# Where does each musdb18-7s test clip sit inside its full track? Run from the
# host, like daneel.sh. Decodes the 50 full tracks and their clips to raw mono
# 8 kHz with the host ffmpeg (the uv container has none), then localises each
# clip. Writes scratch/locate_offsets.tsv, which is what TEST_ANCHORS reads:
#
#   sh experiments/locate_clips.sh
#   MUSDB_TEST=$HOME/data/gasm-demos/musdb18 TEST_ANCHORS=/scratch/locate_offsets.tsv \
#   TEST_OFFSETS=-30,0,30 SLICES="0:13 13:25 25:38 38:50" \
#   LOG=/scratch/logs/lot5_anchored sh experiments/daneel.sh defmap
#
# Decoding is skipped for a pair already present, so a re-run costs the FFTs only.
set -e
G=${ROOT:-$HOME/data/gasm-demos}
RAW=$G/scratch/locate_raw
mkdir -p "$RAW"

n=0
for f in "$G/musdb18/test"/*.stem.mp4; do
    slug=$(basename "$f" .stem.mp4)
    clip="$G/musdb18-7s/test/$slug.stem.mp4"
    [ -f "$clip" ] || { echo "PAS DE CLIP: $slug" >&2; continue; }
    [ -s "$RAW/$slug.full.raw" ] || ffmpeg -v error -i "$f" \
        -map 0:a:0 -ac 1 -ar 8000 -f f32le "$RAW/$slug.full.raw"
    [ -s "$RAW/$slug.clip.raw" ] || ffmpeg -v error -i "$clip" \
        -map 0:a:0 -ac 1 -ar 8000 -f f32le "$RAW/$slug.clip.raw"
    n=$((n + 1))
done
echo "decodage fait: $n pistes" >&2

HERE=$(cd "$(dirname "$0")" && pwd)
docker run --rm -v "$G:/data" -v "$HERE:/w:ro" -w /data \
    -e UV_CACHE_DIR=/data/scratch/uv_cache -e HOME=/data/scratch \
    ghcr.io/astral-sh/uv:python3.13-bookworm \
    uv run --quiet --with numpy python /w/locate_clips.py \
    /data/scratch/locate_raw | tee "$G/scratch/locate_offsets.tsv"

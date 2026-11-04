#!/usr/bin/env bash
# 100 outer x 100 inner NestBoot run on the ENCODE K562 table behind REPORT.md,
# with the preprocessing chosen from the 10 x 10 runs.
# Run from the pyGS root: bash nestboot_report/run_100x100.sh
set -euo pipefail

REAL=.gs_cache/downloads/134c84bebbd65203.csv
OUT=nestboot_report/runs_100x100
mkdir -p "$OUT"

uv run pygs nestboot "$REAL" -m lsco -i 100 -o 100 --zeta=-4:0:41 --seed 42 \
    --dropout-below -10 --max-knockdown-z 0 \
    --output "$OUT/real.npy" > "$OUT/real.log" 2>&1
echo "$(date +%H:%M:%S) real done"

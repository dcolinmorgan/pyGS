#!/usr/bin/env bash
# 10 outer x 10 inner NestBoot runs on the ENCODE K562 table behind REPORT.md:
# each preprocessing option, then two more seeds of the chosen configuration.
# Run from the pyGS root: bash nestboot_report/run_10x10.sh [real|seeds|all]
set -euo pipefail

PART=${1:-all}
REAL=.gs_cache/downloads/134c84bebbd65203.csv
OUT=nestboot_report/runs_10x10
ZETA=-4:0:41
mkdir -p "$OUT"

run() {
    local name=$1 file=$2 seed=$3
    shift 3
    uv run pygs nestboot "$file" -m lsco -i 10 -o 10 --zeta="$ZETA" --seed "$seed" \
        --output "$OUT/$name.npy" "$@" > "$OUT/$name.log" 2>&1
    echo "$(date +%H:%M:%S) $name done"
}

if [[ $PART == real || $PART == all ]]; then
    run real_none "$REAL" 42
    run real_dropout "$REAL" 42 --dropout-below -10
    run real_dropout_kd "$REAL" 42 --dropout-below -10 --max-knockdown-z 0
    run real_dropout_kd_scale "$REAL" 42 --dropout-below -10 --max-knockdown-z 0 --scale-genes
fi

if [[ $PART == seeds || $PART == all ]]; then
    for seed in 1 2; do
        run "real_dropout_kd_seed$seed" "$REAL" "$seed" --dropout-below -10 --max-knockdown-z 0
    done
fi

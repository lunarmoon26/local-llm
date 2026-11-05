#!/usr/bin/env bash
# HTTP client only: never starts a model or changes serving configuration.
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
MODE="${1:-off}"
if [[ "$MODE" != off && "$MODE" != on ]]; then
  echo 'Usage: run_all.sh off|on' >&2
  exit 2
fi
: "${BASE:?set BASE to the separately validated 2.9 endpoint, including /v1}"
: "${DATA:?set DATA to the pinned sample directory}"
: "${OUT:?set OUT to a fresh output directory}"
: "${STRATA_EVAL_API_KEY:?set STRATA_EVAL_API_KEY privately}"
PYTHON="${EVAL_PYTHON:-python3}"
MODEL="${MODEL:-deepseek-v4.1-flash-1017}"
if [[ -e "$OUT" ]]; then
  echo 'OUT already exists; use a fresh directory and preserve previous results.' >&2
  exit 2
fi
mkdir -m 700 -p -- "$OUT"
E=("$PYTHON" -u "$HERE/eval1017.py" --base "$BASE" --model "$MODEL" --out "$OUT" --data "$DATA")
if [[ "$MODE" == off ]]; then
  "${E[@]}" tools --think off --conc 1
fi
for task in gsm8k math500 humaneval mmlupro ceval; do
  "${E[@]}" "$task" --think "$MODE" --conc 1
done
"$PYTHON" "$HERE/compare.py" --out "$OUT" > "$OUT/comparison.json"

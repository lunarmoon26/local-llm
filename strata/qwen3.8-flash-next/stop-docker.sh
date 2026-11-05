#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

docker --context default compose -f "$ROOT/compose.yaml" down "$@"
echo 'Strata container stopped and removed; image and model files preserved.'

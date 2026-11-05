#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

if ! mountpoint -q /mnt/deepseek-nvme1; then
  echo 'SN8100 data filesystem is not mounted.' >&2
  exit 1
fi

echo 'Starting Strata Docker service (native Linux daemon)...'
docker --context default compose -f "$ROOT/compose.yaml" up \
  --no-build -d --wait --wait-timeout 300 "$@"
echo 'Strata is ready: http://127.0.0.1:8080/v1'
echo 'Logs: docker --context default compose logs -f'

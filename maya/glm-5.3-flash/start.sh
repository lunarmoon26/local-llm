#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
exec bash "$ROOT/../../start.sh" maya/glm-5.3-flash "$@"

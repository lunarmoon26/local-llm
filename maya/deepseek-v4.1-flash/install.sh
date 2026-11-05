#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
exec python3 "$ROOT/../../deployment/prepare_maya.py" maya/deepseek-v4.1-flash "$@"

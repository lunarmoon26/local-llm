#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
exec bash "$ROOT/../../smoke-test.sh" maya/deepseek-v4.1-flash "$@"

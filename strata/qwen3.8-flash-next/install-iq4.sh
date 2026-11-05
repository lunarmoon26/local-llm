#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DATA=/mnt/deepseek-nvme1/strata-data
if ! mountpoint -q /mnt/deepseek-nvme1; then
  echo 'SN8100 data filesystem is not mounted; refusing to use the system disk.' >&2
  exit 1
fi
mkdir -p "$ROOT/logs" "$DATA"
# Preparation and serving share ownership of the draft assets.
exec 9>"$ROOT/logs/assets.lock"
flock -n -E 73 9
if [[ -L "$DATA/mtp" ]]; then
  echo 'MTP storage must be a real directory on the data disk, not a symlink.' >&2
  exit 1
fi
cd "$ROOT/Strata"
export STRATA_UV=1
bash ./setup.sh --setup --yes --family unsloth --model UD-IQ4_XS \
  --backend cuda --gpu 1 --context 262144 --vision no \
  --resident-budget-gib 12 --kv-streaming off \
  --host 127.0.0.1 --port 8080 --data-dir "$DATA" --no-start \
  2>&1 | tee "$ROOT/logs/setup-iq4.log"
# Setup's Unsloth split path drops the RAM budget. Configure the verified
# engine split directly after setup has prepared the bounded-RAM config.
.venv/bin/python - <<'PY'
import json
from pathlib import Path

path = Path("strata-unsloth-ud-iq4_xs.json")
config = json.loads(path.read_text())
config["gpu"] = [0, 1]
config["layer_split"] = "auto"
# Apply the extended window after setup's single-card memory-fit check.
args = config["args"]
args[args.index("--max-context") + 1] = "524288"
for flag, value in (("--rope-scaling", "yarn"), ("--rope-scale", "2")):
    if flag in args:
        args[args.index(flag) + 1] = value
    else:
        args.extend([flag, value])
path.write_text(json.dumps(config, indent=1) + "\n")
PY

# Local LLM

Run one local model at **127.0.0.1:8080**. Switch models explicitly with scripts.

This repository contains controller and recipe source. Engine checkouts, model
weights, Python environments and installed server configs are local prerequisites,
not bundled downloads. On a new host, prepare the pinned upstream engines and
models, then supply the installed config/interpreter paths selected by each
`recipe.json`. The Maya preparation adapter requires that installed config;
it is not a fresh-machine bootstrapper.

```bash
./list.sh
./check.sh maya/glm-5.3-flash
./start.sh maya/glm-5.3-flash
./status.sh
./smoke-test.sh
./stop.sh

# Start another model after stopping:
./start.sh maya/deepseek-v4.1-flash
```

## Recipes

| Recipe | API model | Context capacity |
| --- | --- | --- |
| [maya/glm-5.3-flash](maya/glm-5.3-flash/) | `glm-5.3-flash` | 524288 |
| [maya/deepseek-v4.1-flash](maya/deepseek-v4.1-flash/) | `deepseek-v4.1-flash-1017` | 524288 |
| [strata/qwen3.8-flash-next](strata/qwen3.8-flash-next/) | `qwen3.8-flash-next-unsloth-ud-iq4_xs` | 524288 |

Use `http://127.0.0.1:8080/v1` in OpenAI-compatible clients. Select the model
matching `status.sh`; the endpoint does not route requests between models.
Context capacity is not proof of full-window input quality or memory usage.

## OpenCode

Use one provider ID, `local-llm`, for all three models. Merge the
[provider fragment](deployment/opencode-provider.json) under `providers.local-llm`
in your global OpenCode config. With GLM running, select
`local-llm/glm-5.3-flash#fast` or `local-llm/glm-5.3-flash#thinking`.
The root default uses `"model": "local-llm/glm-5.3-flash"` (no variant suffix).

DeepSeek and Qwen use the same `local-llm/` prefix and their API model IDs above.
Selecting a client model does not switch the server: stop/start the matching
recipe first. OpenCode reloads config changes automatically; `/reload` forces a
refresh. Keep authentication values in private environment/file references.

## Preparation

Maya recipes use the pinned upstream Maya installer, not a custom build/pack pipeline:

```bash
bash maya/glm-5.3-flash/install.sh               # preview
bash maya/glm-5.3-flash/install.sh --run         # execute; stop models first
```

Setup uses local weights, `--no-start`, text-only mode and no calibration or model
download request. It may install Python dependencies, compile and generate packs
and configs. Review its output before changing the serving config selected by
`recipe.json`; installer defaults need not match the selected profile.
Qwen preparation is documented in its recipe.

## Layout

```text
deployment/          shared lifecycle, preparation adapter and read-only asset checks
maya/<model>/        Maya recipes
strata/<model>/      Strata recipes
tests/               CPU-only tests
state/               private process receipts and per-run configurations
logs/                private run logs
```

Each recipe owns its engine revision, interpreter, installed config and integrity
fingerprint. Model weights, packs and draft assets live on the NVMe. Check/start
do not download, build or modify them. Secrets belong in private environments,
not tracked files. Optional `LOCAL_LLM_API_KEY` enables API authentication.

Model folders under `/mnt/deepseek-nvme1/strata-data/models/`:

```text
DeepSeek-V4.1-Flash-1017/          DeepSeek EXL3 files (Maya default)
Maya-L/                          GLM GGUF files and adjacent pack/ (Maya default)
DeepSeek-V4.1-Flash-1017-Engram/   shared real Engram shards
unsloth-UD-IQ4_XS/                Qwen GGUF files (Strata default)
```

Model revisions identify the source in manifests, not directory names.
Qwen's MTP draft assets live directly in `/mnt/deepseek-nvme1/strata-data/mtp/`;
native and Docker configs use its `rt/` directory, without a Workspace data-folder dependency.

`start.sh` runs the server in the background and waits for readiness. A different
model, busy port, locked assets or insufficient VRAM causes refusal, not process
termination. Closed TCP connections do not delay a Linux restart; an active
listener still causes refusal. `stop.sh` sends SIGTERM to the verified wrapper
and native engine children, including engines left behind after the wrapper exits.
It reports success only when all have exited. `stop.sh --force` explicitly sends
SIGKILL to those verified processes; normal stops never escalate automatically.
If a process remains stuck even after SIGKILL, status stays `stopping` and the
command reports its PID instead of claiming VRAM was released. Unrelated GPU
processes are never stopped. `--timeout SECONDS` controls either command's wait;
startup timeout leaves a loading server running. Maya model management and boot
services are off.

### Sleep protection

Every managed start acquires a logind `sleep` inhibitor in `block` mode before
launching the server. Normal suspend/hibernate requests are blocked while the
model is loading, ready or stopping. A separate guardian retains the inhibitor
and asset locks if the wrapper exits with a native engine still running; it
releases them only after both have exited. `status.sh` reports
`sleep_inhibitor.held`; successful stop waits for the guardian to release it.
Startup refuses to load a model if the inhibitor cannot be acquired or sleep is
already in progress. A healthy server without its guardian reports `unprotected`
and must be stopped before starting again.

The controller requires systemd-logind and `python3-dbus` for its system Python.
No boot service or global sleep setting is installed. Shutdown/reboot and display
blanking are not inhibited. Stop the model before sleeping; do not override the
inhibitor with privileged/forced sleep commands. This protects managed starts,
not engines launched outside the controller, and is not a fix for NVIDIA driver
suspend/resume faults.

## Tests

```bash
python3 -m unittest discover -s tests -v
```

Tests use tiny fixtures and a fake HTTP server, not model allocations. Run GPU
checks sequentially, only for the selected recipe. Requires Linux, Python 3.10+,
NVIDIA drivers and the configured recipe environments.

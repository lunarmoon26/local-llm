# Local Qwen3.8-Flash-Next with Strata

Native serving uses the [common controller](../../README.md), at
**127.0.0.1:8080**. Stop with `bash ../../stop.sh` before switching models.
Engine: [Strata](https://github.com/Niko1221/Strata), pinned to
`fb58e0dbc8399662c0e47c76578c6e878b14f6cf`.

## Configuration

- Qwen3.8-Flash-Next Unsloth UD-IQ4_XS (roughly 94 GB download).
- CUDA, GPUs 0 and 1: two RTX PRO 6000 Blackwell 96 GB cards, automatic layer split.
- 524,288-token context window using experimental YaRN factor 2 (native: 262,144); images disabled.
- Local-only server: http://127.0.0.1:8080.
- Weights and prepared packs: `/mnt/deepseek-nvme1/strata-data/` on the SN8100.
- Draft assets: `/mnt/deepseek-nvme1/strata-data/mtp/`, stored directly on the NVMe;
  native and Docker serving use its `rt/` directory. Private Python environment: `Strata/.venv/`.
- 12 GiB expert RAM budget; int8 KV stays in VRAM.

## Commands

```bash
cd ~/Workspace/local-llm
./check.sh strata/qwen3.8-flash-next
./start.sh strata/qwen3.8-flash-next
./smoke-test.sh
./stop.sh
```

Open http://127.0.0.1:8080 for chat. OpenAI-compatible clients use
`http://127.0.0.1:8080/v1`, model `qwen3.8-flash-next-unsloth-ud-iq4_xs`.
Do not expose this unauthenticated server to other devices or through a tunnel.

The first launch can take several minutes and temporarily slow the desktop.
Do not run a second server while one is loading. Startup logs are under
`Strata/strata-*.log`; split-test server output is in `logs/server-iq4-split.log`.
No system driver changes or automatic boot services are installed.

## Docker (Linux, NVIDIA)

The image builds pinned Strata source for Blackwell `sm_120` using CUDA 13.0.
The host needs a compatible NVIDIA driver, Docker Compose, and NVIDIA CDI
device specifications (this PC already has `/var/run/cdi/nvidia.yaml`). It does
not need the host CUDA toolkit or Python environment. The first build downloads
CUDA base images, source, and Python dependencies, but never model weights.

```bash
cd ~/Workspace/local-llm/strata/qwen3.8-flash-next
docker --context default compose build
docker --context default compose up       # foreground; Ctrl+C stops the container
```

For a manually started background service:

```bash
bash start-docker.sh
docker --context default compose logs -f
bash stop-docker.sh
```

The wrappers work from any directory and always select the native daemon.
Start checks the NVMe mount, uses the existing image without rebuilding, and
waits up to five minutes for Docker health before reporting readiness. Build
the image first if it is missing. Stop removes the container, not the image or
model files. Neither wrapper changes the native `bash start.sh` launch path.

Use the native Linux Docker daemon (`--context default`), not `desktop-linux`:
Docker Desktop's VM cannot access this host's NVMe mount and NVIDIA CDI devices.
The commands above do not change your selected Docker context.

The container uses the same IQ4, two-GPU split, 524K YaRN ×2 context, int8 KV,
and MTP settings as the native launch. `docker/strata.json` owns its config;
after changing that file, rebuild the image. Model, pack, and draft directories
are mounted read-only from their existing locations; missing directories cause
startup to fail instead of creating empty mount sources. Keep the SN8100 mounted.
No automatic model setup or download runs inside the container.

Linux host networking preserves the listener at **127.0.0.1:8080** without
publishing it to the LAN. OpenCode's existing endpoint remains unchanged. Do not
run the native server and container simultaneously: they use the same port and
GPUs. Automatic restart is disabled; this does not install a boot service.
Container logs are available through `docker --context default compose logs`; the engine log is
also inside the container at `/opt/strata/logs/strata.log`.

## Preparation and limits

Full-window prompts, tool calling and sustained performance remain unqualified.
YaRN applies to short prompts too, not only tokens beyond 262K. Native context
uses `--max-context 262144` without the scaling flags; client limits must agree.

`start-iq4.sh` is an alias for the same model launch used by `start.sh`.
`install-iq4.sh` installs with setup's single-card RAM budget, then configures
the verified engine split and 524K startup configuration directly. This avoids setup's Unsloth multi-GPU path,
which would remove the RAM budget and load all experts into host RAM. Do not
rerun plain `setup.sh --gpus` to configure this model on the 48 GB RAM host.
The NVMe `mtp/` is a real directory, not a symlink into the project. Preparation
and serving share an asset lock; the installer refuses symlinked MTP storage.

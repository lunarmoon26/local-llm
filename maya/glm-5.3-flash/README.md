# GLM-5.3-Flash · Maya

- Engine: Maya v1.1.55, `c4effee67cb95fb4303720919b075fddf04b4e69`.
- Weights: `peasantsmith/GLM-5.3-Flash-Maya-GGUF`, Maya-L IQ3_S, revision
  `a1c2a554ea72a7e8e66cc28b3eaf5036c9313860`. File hashes: `deployment/model.json`.
- Two RTX PRO 6000 Blackwell 96-GB GPUs, automatic layer split, 4096-MiB reserve.
- 524288 context capacity, FP16 KV, KV streaming off, automatic prefill, one conversation.
- Built-in MTP; no vision, external drafter, suffix drafting or approximate skipping.
- Periodic prompt checkpoints every 65536 tokens (`MAYA_GLM_CKPT_EVERY`);
  system-prompt and turn-boundary checkpoints remain enabled.
- API model: `glm-5.3-flash` at `http://127.0.0.1:8080/v1`.

## Commands

```bash
bash install.sh        # preview upstream Maya preparation
bash check.sh          # read-only asset/profile checks
bash start.sh          # managed background service
bash smoke-test.sh
bash ../../stop.sh
```

`install.sh --run` executes pinned upstream preparation after all models are
stopped. It passes local weights, `--no-start`, MTP, text-only mode and no speed
calibration. It may build or regenerate the adjacent pack; no model download is
requested. Review generated settings before selecting them in `recipe.json`.
See the [controller guide](../../README.md#preparation).

The installed config is `logs/maya-glm.json`. `recipe.json` records its reviewed
profile fingerprint. Shared checks validate the model/pack receipts and engine
hash without rereading all weight bytes. Weights stay under the configured NVMe
data root; the native `pack/` stays beside the GGUF shards.
The model folder is Maya's default `models/Maya-L/`.

Start requires at least 84000 MiB free on each selected GPU. Keep API keys in
the private environment. Full 512K-input quality and memory usage are unqualified.

## Measured performance

On this machine with Maya v1.1.55, a fresh 51681-token system prompt and 64-token
greedy output takes 30.93 seconds (two runs), versus 39.36 seconds with 16384-token
periodic checkpoints (four bracketing runs): 21.4% lower latency. Two cached
128-token follow-ups total 2.78 seconds, versus 2.76 seconds in the control.
Arithmetic, JSON, tool arguments and English/Chinese retrieval of records near
the beginning, middle and end of the input pass. This is not a full quality suite.

Wider checkpoint spacing reduces prefill pipeline drains, but editing older long
input can require rereading more tokens. Prefill boundaries can change greedy
wording; bit-identical output is not guaranteed. Decode settings and model weights
are unchanged. The local receipts are under
`state/experiments/checkpoint-confirm-20261106/` at the repository root (untracked).

CPU tests: `(cd ../.. && python3 -m unittest discover -s tests -v)`.

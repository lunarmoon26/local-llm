# DeepSeek-V4.1-Flash · Maya

- Engine: Maya v1.1.55, `c4effee67cb95fb4303720919b075fddf04b4e69`.
- Weights: `LibertAIDAI/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw`, revision
  `c178a01284fc5c0e859d8668cffaac81ca8f4042`.
- Two RTX PRO 6000 Blackwell GPUs; layer split (`--tp 1`), DSpark, mmap experts.
- 524288 context capacity, prefill chunk 4096, text-only, suffix drafting off.
- API model: `deepseek-v4.1-flash-1017` at `http://127.0.0.1:8080/v1`.

## Commands

```bash
bash install.sh        # preview upstream Maya preparation
bash check.sh          # read-only asset/profile checks
bash start.sh          # managed background service
bash smoke-test.sh
bash ../../stop.sh
```

`install.sh --run` executes pinned upstream preparation after all models are
stopped. It uses the local EXL3 model, `--no-start`, text-only mode and no speed
calibration. It may install dependencies, compile and generate a pack inside the
model directory; no model download is requested. Review generated output before
selecting it in `recipe.json`, particularly TP1, chunk4096 and the pack path.
See the [controller guide](../../README.md#preparation).

The installed config is `logs/maya-deepseek-1017.json`. `recipe.json` records its
reviewed profile fingerprint. Shared checks validate the source pin, pack index,
model identity and Engram links. The two canonical Engram shards are real files
under `models/DeepSeek-V4.1-Flash-1017-Engram/`; the EXL3 model links to them.
The EXL3 files live directly in Maya's default `models/DeepSeek-V4.1-Flash-1017/`.

Start requires at least 84000 MiB free on each selected GPU. API keys stay in the
private environment. Actual 512K inputs and broad quality remain unqualified.

## Evaluation

[evaluation/](evaluation/README.md) provides explicit dataset preparation,
CPU tests and contained code evaluation. See
[prefill analysis](docs/maya-prefill-analysis.md) for measured performance.

Deployment CPU tests: `(cd ../.. && python3 -m unittest discover -s tests -v)`.

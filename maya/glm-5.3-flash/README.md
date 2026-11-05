# GLM-5.3-Flash · Maya

- Engine: Maya v1.1.55, `c4effee67cb95fb4303720919b075fddf04b4e69`.
- Weights: `peasantsmith/GLM-5.3-Flash-Maya-GGUF`, Maya-L IQ3_S, revision
  `a1c2a554ea72a7e8e66cc28b3eaf5036c9313860`. File hashes: `deployment/model.json`.
- Two RTX PRO 6000 Blackwell 96-GB GPUs, automatic layer split, 4096-MiB reserve.
- 524288 context capacity, FP16 KV, KV streaming off, automatic prefill, one conversation.
- Built-in MTP; no vision, external drafter, suffix drafting or approximate skipping.
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

CPU tests: `(cd ../.. && python3 -m unittest discover -s tests -v)`.

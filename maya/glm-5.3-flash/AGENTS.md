# GLM Maya recipe

- Follow the root README for shared lifecycle and preparation. Do not implement
  model-local installers, builders or serving fallbacks.
- Engine/model identities live in `recipe.json` and `deployment/model.json`.
  Do not modify the pinned `Project-Maya/` engine checkout or requantize weights.
- Use Maya-L IQ3_S, two SM120 96-GB GPUs, FP16 KV, KV streaming off, MTP,
  one conversation and text-only mode. No approximate skipping or extra drafter.
- Keep the native pack beside its GGUF shards. Preserve model/build/pack receipts;
  check them read-only. Do not run setup while a model owns the shared locks.
- Keep weights, environments, checkouts, logs and `.env` untracked. Never publish
  API keys or private device identifiers.
- Commands: `bash install.sh` previews upstream setup; `--run` executes it.
  `bash check.sh`, `bash start.sh`, `bash smoke-test.sh`, `bash ../../stop.sh`.
- CPU tests: `(cd ../.. && python3 -m unittest discover -s tests -v)`.
  Configured 524288 capacity is not actual 512K-input qualification.

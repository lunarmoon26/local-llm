# DeepSeek Maya recipe

- Follow the root README for shared lifecycle and preparation. Do not implement
  model-local installers, builders, Docker paths or serving fallbacks.
- Engine/model identities live in `recipe.json`. Use the pinned LibertAIDAI
  1017 2.9-bpw EXL3 weights; never requantize them or modify engine checkouts.
- Keep the selected profile: TP1 layer split, DSpark, mmap experts, chunk4096,
  text-only and suffix drafting off. Review upstream defaults before activation.
- Preserve the real Engram shards in `models/DeepSeek-V4.1-Flash-1017-Engram/`
  and the active model's links to them. Do not move or delete model data.
- `Strata/` is an independent engine checkout; follow its instructions if edited.
- Keep `.env`, logs, weights, packs and environments untracked. Do not publish
  API keys, device IDs or private measurements.
- Commands: `bash install.sh` previews upstream setup; `--run` executes it.
  `bash check.sh`, `bash start.sh`, `bash smoke-test.sh`, `bash ../../stop.sh`.
- CPU tests: `(cd ../.. && python3 -m unittest discover -s tests -v)`.
  Configured 524288 capacity is not actual 512K-input qualification.

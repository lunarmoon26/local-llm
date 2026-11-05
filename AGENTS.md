# Local LLM controller

- Read README.md for the lifecycle contract and the selected recipe's instructions.
- `<engine>/<model>` directories are independent deployment Git repositories;
  preserve their history and all uncommitted work. Do not break absolute asset,
  interpreter or build references.
- Use common start/stop/status/check scripts. One loopback listener at port 8080,
  one model at a time. Never terminate unrelated processes, remove model data,
  requantize weights, download automatically or install a boot service.
- Maya model management stays off by user choice; use explicit script switching.
- Keep model, pack, Engram and Qwen MTP assets at their configured NVMe paths.
- Document only the current setup. Do not add migration narratives or fallback
  implementations. Maya preparation uses upstream; serving uses the controller.
- Recipe manifests own engine/model identity; ignored state/ owns runtime process
  receipts and generated configs. Keep secrets and local logs out of Git.
- No commits, pushes, broad resets or discarded changes without a separate request.
- CPU tests use unittest; live GPU tests run sequentially and only for the selected
  recipe. A configured context limit does not qualify an actual long prompt.

# DeepSeek 1017 quality evaluation

Status: **implemented; 18 CPU tests pass; real-model evaluation is separate**.
This directory adapts cliecy's small-subset evaluation harness.
The evaluation target is our existing **LibertAIDAI 1017 EXL3 2.9-bpw** checkpoint,
now selected by the Maya root launchers. Evaluation does not change the serving
configuration, start a server or download any weights.

## Reference and scope

Upstream: [cliecy/DeepSeek-V4.1-Flash-1017-2xRTXPRO6000](https://github.com/cliecy/DeepSeek-V4.1-Flash-1017-2xRTXPRO6000),
commit `c3a37665c93b49a3dbc63d0f0c92d036475b2fc6`, `results/eval/`.
The MIT license is retained in `LICENSE.cliecy`. `reference/cliecy-1017.json`
owns the published counts, exact item IDs, gold-answer hashes and pinned dataset
revisions used for comparison. No dataset questions or generated solutions are
bundled here. Datasets retain their own licenses.

| Task | Questions | Cliecy thinking off | Thinking on |
| --- | ---: | ---: | ---: |
| GSM8K test | 100 | 99/100 | 98/100 |
| MATH-500 | 50 | 44/50 | 48/50 |
| HumanEval | 50 | 49/50 | 45/50 |
| MMLU-Pro test | 200 | 150/200 | 46/64, stopped early |
| C-Eval validation | 100 | 93/100 | 92/100 |
| Tool-call smoke cases | 10 | 10/10 | Not reported |

These are small subsets, not full benchmark scores. The five thinking-on
HumanEval failures hit the output budget; serial retries recovered four, which
is a separate result. Tool scoring checks call selection and selected arguments,
not actual tool execution or multi-step agent behavior. Our engine replay tests
are not substitutes for these quality scores. No 2.9 quality score is assumed.

## Contract

- `make_samples.py` explicitly downloads **dataset files only**, pins revisions,
  rebuilds seed-1234 subsets and verifies exact reference IDs/non-code gold
  hashes. Model evaluation never downloads datasets or weights implicitly.
- `eval1017.py` sends requests to an explicitly selected OpenAI-compatible
  endpoint; it never starts or replaces a server. Verify that endpoint actually
  serves 2.9: the model alias alone does not distinguish quantizations.
- Original task prompts and grading are retained. Temperature is zero, budgets
  are 2048 tokens off and 16384 on; tool cases use 1024. The client sends
  `chat_template_kwargs.enable_thinking`, which our server understands.
- Quality concurrency defaults to **one**, unlike cliecy's eight, so quality
  can be compared per item but wall time is not a matched throughput comparison.
- HumanEval requires bubblewrap and fails closed without it. Generated code has
  no network, inherited credentials, writable host mount or visible host home;
  execution has time, CPU, address-space and output limits.
- Requests/responses, predictions, errors and truncations stay in the selected
  output directory. API keys are read from `STRATA_EVAL_API_KEY` and are not
  written to receipts. Use fresh output directories for different model builds.
- The primary MMLU-Pro on run uses all 200 questions. Comparison with their
  incomplete on result uses its exact 64 completed IDs, not the first 64 rows.
- No production activation, requantization, model download, automatic boot
  service, benchmark claim or second model-sized workload is included.

## Setup and execution

Use a separate environment so evaluation dependencies do not alter serving:

```bash
python3 -m venv "$STRATA_DATA_DIR/envs/ds41-29-eval"
"$STRATA_DATA_DIR/envs/ds41-29-eval/bin/pip" install -r evaluation/requirements.txt
export EVAL_PYTHON="$STRATA_DATA_DIR/envs/ds41-29-eval/bin/python"

"$EVAL_PYTHON" evaluation/make_samples.py \
  --out logs/quality-29/samples --cache-dir "$STRATA_DATA_DIR/evaluation-cache"
"$EVAL_PYTHON" -m unittest discover -s evaluation -p 'test_*.py' -v
"$EVAL_PYTHON" evaluation/eval1017.py --base http://127.0.0.1:1/v1 \
  --model unused --out logs/quality-29/cpu self-test
```

Start the Maya 2.9 server separately with `bash start.sh` only after its resource
and GPU-exclusion checks. It binds to `127.0.0.1`; set `STRATA_API_KEY` privately
for authenticated testing. Set `STRATA_EVAL_API_KEY` to the same value for the
evaluator. Then select the server's address, for example:

```bash
export BASE=http://127.0.0.1:8081/v1
export MODEL=deepseek-v4.1-flash-1017
export DATA=logs/quality-29/samples
export OUT=logs/quality-29/off
# Set STRATA_EVAL_API_KEY privately, without putting it in shell history.
bash evaluation/run_all.sh off
export OUT=logs/quality-29/on
bash evaluation/run_all.sh on
```

`run_all.sh` runs tools (off only) and the five quality tasks, stops on a task
process failure, and writes a reference comparison. A score below the reference
is evidence, not an infrastructure failure. Preserve low scores and failed runs.
Keep the shared service lock for the server's whole lifetime; do not load a
second model to run evaluation. Stop the temporary server when finished.

`compare.py --out RESULTS` also re-creates a comparison offline without new model
requests. Keep raw outputs when changing a grader or budget. Dependency versions
are pinned in `requirements.txt`; the reference file records source identities.

The inherited `speed` and `needle` commands remain diagnostic. Their character-
estimated context lengths, prefix-cache behavior and SSE-event timing do not
constitute a matched performance benchmark; our server does not implement their
`ignore_eos` request extension. The requested quality suite does not use it.

## Acceptance evidence

CPU acceptance covers task prompts/scoring, request-mode mapping, HTTP receipt
capture, exact sample identity, fail-closed sandboxing, code isolation and
timeouts. Static AST checks retain the original task-prompt/scorer functions;
offline comparison reproduces the ten available complete task/mode reference
counts (the incomplete MMLU-Pro on subset has an exact-ID unit test). All 500
quality question IDs and available non-code gold answers match the reference.
The local evaluator uses Python 3.12.13 and the sandbox Python 3.14.4; record
these versions when interpreting HumanEval results rather than assuming the
reference container's interpreter is identical.

GPU evaluation is a separate explicit step. Results and private server
configs belong under ignored `logs/`, never in source control. The 2.9 target
still needs broader quality and long-input evidence beyond its bounded smoke checks.

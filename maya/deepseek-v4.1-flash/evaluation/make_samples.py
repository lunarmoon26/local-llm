#!/usr/bin/env python3
"""Rebuild cliecy's fixed subsets from pinned datasets; never downloads weights.

Adapted from cliecy's results/eval/make_samples.py at c3a37665c93b49a3dbc63d0f0c92d036475b2fc6.
See LICENSE.cliecy. Dataset questions/solutions are not distributed by this repo.
"""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import random

REFERENCE = Path(__file__).parent / 'reference/cliecy-1017.json'


def verify_samples(task, rows, reference):
    expected = reference['tasks'][task]['samples']
    by_id = {str(row['id']): row for row in rows}
    if len(by_id) != len(rows) or set(by_id) != {str(row['id']) for row in expected}:
        raise ValueError(task + ': sample IDs differ from the published subset')
    for row in expected:
        if 'gold_sha256' in row:
            actual = hashlib.sha256(str(by_id[str(row['id'])]['answer']).encode()).hexdigest()
            if actual != row['gold_sha256']:
                raise ValueError(task + ': gold answer differs for ID ' + str(row['id']))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--cache-dir', type=Path, required=True, help='Explicit dataset cache, preferably under STRATA_DATA_DIR')
    parser.add_argument('--reference', type=Path, default=REFERENCE)
    parser.add_argument('--local-files-only', action='store_true', help='Use already cached dataset files; no network')
    args = parser.parse_args(argv)
    reference = json.loads(args.reference.read_text())
    revisions = reference['datasets']
    if args.out.exists() and any(args.out.iterdir()):
        parser.error('--out must be a new or empty directory; preserve earlier samples/receipts')
    # Optional heavy dependencies load only for explicit sample preparation.
    import pandas as pd
    from huggingface_hub import hf_hub_download

    def download(repo, filename):
        return hf_hub_download(repo, filename, repo_type='dataset', revision=revisions[repo],
            local_dir=args.cache_dir / repo, local_files_only=args.local_files_only)

    def pick(frame, count):
        indices = sorted(random.Random(reference['sampling_seed']).sample(range(len(frame)), count))
        return frame.iloc[indices]

    g = pd.read_parquet(download('openai/gsm8k', 'main/test-00000-of-00001.parquet'))
    m = pd.read_json(download('HuggingFaceH4/MATH-500', 'test.jsonl'), lines=True)
    h = pd.read_parquet(download('openai/openai_humaneval', 'openai_humaneval/test-00000-of-00001.parquet'))
    p = pd.read_parquet(download('TIGER-Lab/MMLU-Pro', 'data/test-00000-of-00001.parquet'))
    # Preserve the inspected subject order, not the order from a future Hub listing.
    files = reference['ceval_subject_files']
    c = pd.concat([pd.read_parquet(download('ceval/ceval-exam', f)).assign(subject=f.split('/')[0])
                   for f in files]).reset_index(drop=True)
    tasks = {
        'gsm8k': [{'id': int(i), 'question': r.question,
                   'answer': r.answer.split('####')[-1].strip().replace(',', '')}
                  for i, r in pick(g, 100).iterrows()],
        'math500': [{'id': r.unique_id, 'problem': r.problem, 'answer': r.answer,
                     'level': int(r.level), 'subject': r.subject} for _, r in pick(m, 50).iterrows()],
        'humaneval': [{'id': r.task_id, 'prompt': r.prompt, 'test': r.test, 'entry_point': r.entry_point}
                      for _, r in pick(h, 50).iterrows()],
        'mmlupro': [{'id': int(r.question_id), 'question': r.question, 'options': list(r.options),
                    'answer': r.answer, 'category': r.category} for _, r in pick(p, 200).iterrows()],
        'ceval': [{'id': f'{r.subject}-{r.id}', 'subject': r.subject, 'question': r.question,
                  'options': [r.A, r.B, r.C, r.D], 'answer': r.answer} for _, r in pick(c, 100).iterrows()],
    }
    for task, rows in tasks.items():
        verify_samples(task, rows, reference)
    args.out.mkdir(parents=True, exist_ok=True, mode=0o700)
    manifest = {'seed': reference['sampling_seed'], 'datasets': revisions, 'ceval_subject_files': files,
        'reference_sha256': hashlib.sha256(args.reference.read_bytes()).hexdigest(),
        'dependencies': {k: importlib.metadata.version(k) for k in ['pandas', 'pyarrow', 'huggingface-hub', 'math-verify']},
        'tasks': {}}
    for task, rows in tasks.items():
        path = args.out / (task + '.jsonl')
        with path.open('x') as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + '\n')
        manifest['tasks'][task] = {'n': len(rows), 'IDs': [row['id'] for row in rows],
            'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'published_IDs_and_gold_verified': True}
        print(task, len(rows), 'reference IDs/gold verified', flush=True)
    on_ids = reference['tasks']['mmlupro']['on_ids']
    by_id = {str(row['id']): row for row in tasks['mmlupro']}
    with (args.out / 'mmlupro-cliecy-on64.jsonl').open('x') as handle:
        for item_id in on_ids:
            handle.write(json.dumps(by_id[str(item_id)], ensure_ascii=False) + '\n')
    with (args.out / 'manifest.json').open('x') as handle:
        json.dump(manifest, handle, indent=2); handle.write('\n')


if __name__ == '__main__':
    main()

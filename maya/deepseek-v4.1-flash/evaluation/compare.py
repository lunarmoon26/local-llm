#!/usr/bin/env python3
"""Compare saved quality outputs with the pinned cliecy reference, offline."""
import argparse
import json
from pathlib import Path

REFERENCE = Path(__file__).parent / 'reference/cliecy-1017.json'


def compare(task, mode, rows, reference):
    if task not in reference['tasks'] or mode not in ('off', 'on'):
        raise ValueError('unknown task/mode')
    task_reference = reference['tasks'][task]
    published = task_reference.get(mode)
    if task == 'tools':
        if len(rows) != 10:
            raise ValueError('tools: expected exactly ten cases')
    else:
        ids = [str(row['id']) for row in rows]
        expected = {str(row['id']) for row in task_reference['samples']}
        if len(ids) != len(set(ids)) or set(ids) != expected:
            raise ValueError(task + ': missing, duplicate, or unexpected item IDs')
    ours = {'n': len(rows), 'correct': sum(bool(row['ok']) for row in rows),
            'errors': sum('error' in row for row in rows),
            'truncated': sum(row.get('finish') == 'length' for row in rows)}
    ours['acc'] = ours['correct'] / ours['n']
    result = {'task': task, 'mode': mode, 'ours': ours, 'cliecy': published,
              'accuracy_difference_pp': None}
    if published is not None:
        if published['n'] == ours['n']:
            result['accuracy_difference_pp'] = 100 * (ours['acc'] - published['correct']/published['n'])
        else:
            matched_ids = {str(item_id) for item_id in task_reference['on_ids']}
            matched = [row for row in rows if str(row['id']) in matched_ids]
            if len(matched) != published['n']:
                raise ValueError(task + ': incomplete matched reference subset')
            result['matched_published_subset'] = {'n': len(matched), 'correct': sum(bool(row['ok']) for row in matched)}
            result['matched_published_subset']['acc'] = result['matched_published_subset']['correct']/len(matched)
            result['accuracy_difference_pp'] = 100 * (result['matched_published_subset']['acc'] - published['correct']/published['n'])
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True, help='Directory containing saved task JSONL')
    parser.add_argument('--reference', type=Path, default=REFERENCE)
    args = parser.parse_args(argv)
    reference = json.loads(args.reference.read_text())
    results = []
    for task in reference['tasks']:
        for mode in ['off', 'on']:
            tag = 'tools' if task == 'tools' else task + '-think' + mode
            path = args.out / (tag + '.jsonl')
            if task == 'tools' and mode == 'on' or not path.is_file():
                continue
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            results.append(compare(task, mode, rows, reference))
    if not results:
        parser.error('no saved quality outputs found')
    # A derived comparison only: never replace raw outputs or historical receipts.
    print(json.dumps({'reference_commit': reference['commit'], 'results': results}, indent=2))


if __name__ == '__main__':
    main()

import hashlib
import json
from pathlib import Path
import unittest

import make_samples


class SampleTests(unittest.TestCase):
    def reference(self):
        return {'tasks': {'gsm8k': {'samples': [{'id': 7, 'gold_sha256': hashlib.sha256(b'42').hexdigest()}]}}}

    def test_matching_ids_and_gold(self):
        make_samples.verify_samples('gsm8k', [{'id': 7, 'answer': '42'}], self.reference())

    def test_wrong_gold_refused(self):
        with self.assertRaisesRegex(ValueError, 'gold answer differs'):
            make_samples.verify_samples('gsm8k', [{'id': 7, 'answer': '43'}], self.reference())

    def test_wrong_and_duplicate_ids_refused(self):
        for rows in [[{'id': 8, 'answer': '42'}], [{'id': 7, 'answer': '42'}]*2]:
            with self.assertRaisesRegex(ValueError, 'sample IDs differ'):
                make_samples.verify_samples('gsm8k', rows, self.reference())

    def test_reference_counts_and_pinned_revisions(self):
        reference = json.loads(make_samples.REFERENCE.read_text())
        for task, n in [('gsm8k', 100), ('math500', 50), ('humaneval', 50), ('mmlupro', 200), ('ceval', 100)]:
            rows = reference['tasks'][task]['samples']
            self.assertEqual(len(rows), n)
            self.assertEqual(len({str(row['id']) for row in rows}), n)
        self.assertEqual(len(reference['tasks']['mmlupro']['on_ids']), 64)
        for revision in reference['datasets'].values():
            self.assertRegex(revision, r'^[0-9a-f]{40}$')


if __name__ == '__main__':
    unittest.main()

import unittest

from compare import compare


class CompareTests(unittest.TestCase):
    def test_full_subset_delta(self):
        reference = {'tasks': {'gsm8k': {'samples': [{'id': 1}, {'id': 2}],
                      'off': {'n': 2, 'correct': 1}, 'on': None}}}
        result = compare('gsm8k', 'off', [{'id': 1, 'ok': True}, {'id': 2, 'ok': True}], reference)
        self.assertEqual(result['accuracy_difference_pp'], 50)

    def test_partial_on_reference_matches_exact_ids_not_prefix(self):
        reference = {'tasks': {'mmlupro': {'samples': [{'id': 1}, {'id': 2}, {'id': 3}],
                     'on_ids': [3, 2], 'on': {'n': 2, 'correct': 1}}}}
        result = compare('mmlupro', 'on', [{'id': 1, 'ok': False}, {'id': 2, 'ok': True}, {'id': 3, 'ok': True}], reference)
        self.assertEqual(result['ours']['n'], 3)
        self.assertEqual(result['matched_published_subset']['correct'], 2)
        self.assertEqual(result['accuracy_difference_pp'], 50)

    def test_duplicate_and_incomplete_results_refused(self):
        reference = {'tasks': {'gsm8k': {'samples': [{'id': 1}, {'id': 2}], 'off': {'n': 2, 'correct': 1}}}}
        for rows in [[{'id': 1, 'ok': True}], [{'id': 1, 'ok': True}]*2]:
            with self.assertRaisesRegex(ValueError, 'missing, duplicate, or unexpected'):
                compare('gsm8k', 'off', rows, reference)

    def test_tool_case_count(self):
        reference = {'tasks': {'tools': {'off': {'n': 10, 'correct': 10}, 'on': None}}}
        with self.assertRaisesRegex(ValueError, 'exactly ten'):
            compare('tools', 'off', [{'ok': True}], reference)
        result = compare('tools', 'off', [{'ok': True}]*10, reference)
        self.assertEqual(result['accuracy_difference_pp'], 0)


if __name__ == '__main__':
    unittest.main()

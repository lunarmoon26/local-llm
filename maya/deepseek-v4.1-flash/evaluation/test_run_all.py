import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parent


class RunAllTests(unittest.TestCase):
    def fixture(self):
        logs = ROOT.parent / 'logs'
        logs.mkdir(exist_ok=True)
        temp = tempfile.TemporaryDirectory(prefix='eval-wrapper-', dir=logs)
        self.addCleanup(temp.cleanup)
        directory = Path(temp.name).resolve()
        fake = directory / 'fake-python'
        fake.write_text(f'#!{sys.executable}\nimport json,os,sys\n'
            'with open(os.environ["RECEIPT"],"a") as f:f.write(json.dumps(sys.argv[1:])+"\\n")\n'
            'if "math500" in sys.argv and os.environ.get("FAIL_MATH"):sys.exit(9)\n'
            'print("{}")\n')
        fake.chmod(0o700)
        env = {**os.environ, 'EVAL_PYTHON': str(fake), 'BASE': 'http://127.0.0.1:1/v1',
               'DATA': str(directory / 'samples'), 'OUT': str(directory / 'out'),
               'STRATA_EVAL_API_KEY': 'not-a-real-key', 'RECEIPT': str(directory / 'calls.jsonl')}
        return directory, env

    def test_off_includes_tools_and_all_five_tasks(self):
        directory, env = self.fixture()
        subprocess.run(['bash', str(ROOT / 'run_all.sh'), 'off'], env=env, capture_output=True, check=True)
        calls = [json.loads(line) for line in (directory / 'calls.jsonl').read_text().splitlines()]
        self.assertEqual(len(calls), 7)
        for task in ['tools', 'gsm8k', 'math500', 'humaneval', 'mmlupro', 'ceval']:
            self.assertEqual(sum(task in args for args in calls), 1)
        self.assertTrue(calls[-1][0].endswith('compare.py'))
        self.assertNotIn('not-a-real-key', (directory / 'calls.jsonl').read_text())

    def test_on_and_existing_output_refusal(self):
        directory, env = self.fixture()
        subprocess.run(['bash', str(ROOT / 'run_all.sh'), 'on'], env=env, capture_output=True, check=True)
        before = (directory / 'calls.jsonl').read_bytes()
        self.assertNotIn('tools', before.decode())
        result = subprocess.run(['bash', str(ROOT / 'run_all.sh'), 'on'], env=env, capture_output=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual((directory / 'calls.jsonl').read_bytes(), before)

    def test_task_process_failure_stops_subsequent_requests(self):
        directory, env = self.fixture()
        result = subprocess.run(['bash', str(ROOT / 'run_all.sh'), 'off'], env={**env, 'FAIL_MATH': '1'}, capture_output=True)
        self.assertEqual(result.returncode, 9)
        calls = (directory / 'calls.jsonl').read_text()
        self.assertNotIn('humaneval', calls)
        self.assertNotIn('compare.py', calls)


if __name__ == '__main__':
    unittest.main()

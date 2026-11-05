"""CPU-only tests of the local client adaptation; never loads model weights."""
import importlib.util
import json
from pathlib import Path
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest import mock

ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('eval_local', ROOT / 'eval1017.py')
ev = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ev)


class AdapterTests(unittest.TestCase):
    def setUp(self):
        logs = ROOT.parent / 'logs'
        logs.mkdir(exist_ok=True)
        self.directory = tempfile.TemporaryDirectory(prefix='eval-cpu-', dir=logs)
        self.addCleanup(self.directory.cleanup)
        ev.configure(['--base', 'http://127.0.0.1:1/v1', '--model', 'unused', '--out', self.directory.name, 'gsm8k'])

    def test_thinking_mapping(self):
        self.assertEqual(ev.kw(False), {'chat_template_kwargs': {'enable_thinking': False}})
        self.assertEqual(ev.kw(True), {'chat_template_kwargs': {'enable_thinking': True}})
        self.assertEqual(ev.kw(None), {})

    def test_output_identity_rejects_another_model(self):
        with self.assertRaisesRegex(ValueError, 'output directory identity differs'):
            ev.configure(['--base', 'http://127.0.0.1:1/v1', '--model', 'another-model',
                          '--out', self.directory.name, 'gsm8k'])

    def test_gsm8k_original_prompt_and_grading(self):
        reply = {'choices': [{'message': {'content': 'Steps.\nAnswer: 1,234'}, 'finish_reason': 'stop'}],
                 'usage': {'completion_tokens': 12}}
        with mock.patch.object(ev, 'chat', return_value=(1.0, reply)) as chat:
            result = ev.task_item('gsm8k', {'id': 1, 'question': 'Question?', 'answer': '1234'}, False)
        self.assertTrue(result['ok'])
        self.assertEqual(chat.call_args.args[1:], (2048, False))
        self.assertIn("Answer: <number>", chat.call_args.args[0][0]['content'])

    def test_math_and_choice_grading(self):
        self.assertTrue(ev.math_eq(r'\frac{1}{2}', '0.5'))
        self.assertFalse(ev.math_eq('2', '3'))
        self.assertEqual(ev.choice_letter('The answer is (J)', 10), 'J')
        self.assertEqual(ev.choice_letter('答案：C', 4), 'C')

    def test_sandbox_fail_closed(self):
        with mock.patch.object(ev.shutil, 'which', return_value=None):
            with self.assertRaisesRegex(RuntimeError, 'refusing unsandboxed'):
                ev.run_code('raise AssertionError("must not run")')

    def test_grading_error_is_recorded_then_stops_task(self):
        with mock.patch.object(ev, 'load', return_value=[{'id': 1}]), \
             mock.patch.object(ev, 'task_item', side_effect=RuntimeError('grading failure')):
            with self.assertRaisesRegex(RuntimeError, 'request/grading errors'):
                ev.quality('gsm8k')
        summary = json.loads((Path(self.directory.name) / 'gsm8k-thinkoff.summary.json').read_text())
        self.assertEqual(summary['errors'], 1)
        self.assertEqual(summary['correct'], 0)

    def test_http_request_and_receipts(self):
        received = []
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_): pass
            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                received.append(body)
                response = json.dumps({'choices': [{'message': {'content': '323'}, 'finish_reason': 'stop'}],
                                       'usage': {'completion_tokens': 1}}).encode()
                self.send_response(200); self.send_header('Content-Length', str(len(response)))
                self.end_headers(); self.wfile.write(response)
        server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True); thread.start()
        try:
            ev.BASE = 'http://127.0.0.1:' + str(server.server_port) + '/v1'
            _, response = ev.chat([{'role': 'user', 'content': 'public canary'}], 2048, False)
            self.assertEqual(response['choices'][0]['message']['content'], '323')
            self.assertEqual(received[0]['chat_template_kwargs'], {'enable_thinking': False})
            self.assertEqual(received[0]['temperature'], 0)
            for name in ['requests', 'responses']:
                self.assertEqual(len((Path(self.directory.name) / (name + '.jsonl')).read_text().splitlines()), 1)
        finally:
            server.shutdown(); server.server_close(); thread.join()


if __name__ == '__main__':
    unittest.main()

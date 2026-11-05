"""Opt-in real HTTP/browser checks against a running localhost 1017 server."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import http.client
import json
import os
from pathlib import Path
import time
import urllib.error
import urllib.request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=18082)
    parser.add_argument('--context', type=int, default=32768)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--browser', action='store_true')
    a = parser.parse_args()
    a.out.mkdir(exist_ok=False, parents=True)
    results = []

    def save():
        (a.out / 'results.json').write_text(json.dumps(results, ensure_ascii=False, indent=2) + '\n')

    def call(path, body=None, stream=False):
        headers = {'Content-Type': 'application/json'}
        if os.environ.get('STRATA_API_KEY'):
            headers['Authorization'] = 'Bearer ' + os.environ['STRATA_API_KEY']
        req = urllib.request.Request(f'http://127.0.0.1:{a.port}{path}',
                                     data=None if body is None else json.dumps(body).encode(),
                                     headers=headers)
        with urllib.request.urlopen(req, timeout=180) as r:
            data = r.read().decode()
        return data if stream else json.loads(data)

    def chat(text, **extra):
        return {'model': 'deepseek-v4.1-flash-1017', 'messages': [{'role': 'user', 'content': text}],
                'temperature': 0, 'max_tokens': 128,
                'chat_template_kwargs': {'enable_thinking': False}, **extra}

    def test(name, fn):
        start = time.monotonic()
        try:
            value = fn()
            results.append({'test': name, 'status': 'pass', 'seconds': time.monotonic() - start, 'result': value})
        except Exception as e:
            failure = {'test': name, 'status': 'fail', 'error': repr(e)}
            if isinstance(e, urllib.error.HTTPError):
                failure['response'] = e.read().decode(errors='replace')
            results.append(failure)
            save(); raise
        save(); print('PASS', name, flush=True)

    def health():
        r = call('/health'); assert r['status'] == 'ok' and not r.get('unloaded') and r['max_context'] == a.context, r
        models = call('/v1/models'); assert any(x['id'] == 'deepseek-v4.1-flash-1017' for x in models['data']), models
        return r
    test('loaded-health-and-models', health)

    def arithmetic():
        r = call('/v1/chat/completions', chat('What is 17 * 19? Reply with only the integer.'))
        assert r['choices'][0]['message']['content'].strip() == '323', r
        return r
    test('openai-nonstream', arithmetic)

    def streaming():
        raw = call('/v1/chat/completions', chat('What is 2 + 2? Reply with only the integer.', stream=True), True)
        chunks = [json.loads(s[6:]) for s in raw.splitlines() if s.startswith('data: ') and s != 'data: [DONE]']
        text = ''.join(c['choices'][0]['delta'].get('content', '') for c in chunks if c.get('choices'))
        assert text.strip() == '4' and 'data: [DONE]' in raw, raw
        return {'text': text, 'chunks': len(chunks)}
    test('openai-stream', streaming)

    def copy_text():
        phrase = 'ALPHA BRAVO CHARLIE DELTA ECHO FOXTROT GOLF HOTEL INDIA JULIET'
        r = call('/v1/chat/completions', chat('Copy the following text exactly, without quotes, explanation or changes:\n' + phrase))
        assert r['choices'][0]['message']['content'].strip() == phrase, r
        return r
    test('verbatim-copy-for-suffix-drafts', copy_text)

    def responses():
        r = call('/v1/responses', {'model': 'deepseek-v4.1-flash-1017', 'input': 'What is the capital of France? Reply with only the city name.',
                                 'max_output_tokens': 128, 'temperature': 0, 'reasoning': {'effort': 'none'}})
        text = ''.join(c.get('text', '') for x in r['output'] for c in x.get('content', []))
        assert 'Paris' in text, r
        return r
    test('responses-api', responses)

    def anthropic():
        r = call('/v1/messages', {'model': 'deepseek-v4.1-flash-1017', 'messages': [{'role': 'user', 'content': 'What is 2 + 2?'}],
                                'max_tokens': 256, 'temperature': 0, 'thinking': {'type': 'disabled'}})
        text = ''.join(c.get('text', '') for c in r['content'])
        assert '4' in text, r
        return r
    test('anthropic-messages', anthropic)

    def tools():
        tool = {'type': 'function', 'function': {'name': 'lookup', 'description': 'Look up a city. Use this tool when asked.',
                'parameters': {'type': 'object', 'properties': {'city': {'type': 'string'}}, 'required': ['city']}}}
        r = call('/v1/chat/completions', chat('Call the lookup tool with city="Paris". Do not answer with plain text.', tools=[tool], max_tokens=256))
        calls = r['choices'][0]['message'].get('tool_calls', [])
        assert calls and calls[0]['function']['name'] == 'lookup', r
        assert json.loads(calls[0]['function']['arguments'])['city'] == 'Paris', r
        return r
    test('real-model-tool-call', tools)

    def concurrent():
        def one(word):
            r = call('/v1/chat/completions', chat(f'Reply with exactly the word {word} and nothing else.'))
            text = r['choices'][0]['message']['content'].strip(); assert text == word, r
            return text
        with ThreadPoolExecutor(2) as pool:
            values = list(pool.map(one, ['BRONZE', 'SILVER']))
        return values
    test('concurrent-session-isolation', concurrent)

    def cancel():
        conn = http.client.HTTPConnection('127.0.0.1', a.port, timeout=180)
        headers = {'Content-Type': 'application/json'}
        if os.environ.get('STRATA_API_KEY'):
            headers['Authorization'] = 'Bearer ' + os.environ['STRATA_API_KEY']
        conn.request('POST', '/v1/chat/completions', json.dumps(chat('Write an extremely long story, at least 2000 words.',
                                                               max_tokens=2048, stream=True)), headers)
        response = conn.getresponse(); assert response.status == 200
        while True:
            line = response.readline()
            if not line:
                raise AssertionError('stream ended before cancellation')
            if b'"content"' in line:
                break
        response.close(); conn.close()
        return arithmetic()
    test('disconnect-cancellation-and-recovery', cancel)

    def overflow():
        try:
            call('/v1/chat/completions', chat('hello', max_tokens=a.context))
        except urllib.error.HTTPError as e:
            assert e.code == 400, e.code
            return json.loads(e.read())
        raise AssertionError('context overflow was accepted')
    test('context-overflow-refused', overflow)

    if a.browser:
        def browser():
            from playwright.sync_api import sync_playwright
            candidates = list((Path.home() / '.cache/ms-playwright').glob('chromium_headless_shell-*/chrome-headless-shell-linux64/chrome-headless-shell'))
            with sync_playwright() as p:
                b = p.chromium.launch(headless=True, **({'executable_path': str(sorted(candidates)[-1])} if candidates else {}))
                page = b.new_page(); errors = []
                page.on('pageerror', lambda e: errors.append(str(e)))
                page.goto(f'http://127.0.0.1:{a.port}', wait_until='networkidle')
                page.screenshot(path=str(a.out / 'ui-loaded.png'), full_page=True)
                page.locator('#input').fill('What is 17 * 19? Reply with only the integer.')
                page.locator('#send-btn').click()
                page.get_by_text('323', exact=True).wait_for(timeout=180000)
                page.screenshot(path=str(a.out / 'ui-answer.png'), full_page=True)
                assert not errors, errors
                b.close()
            return {'answer': '323', 'page_errors': errors}
        test('browser-chat', browser)


if __name__ == '__main__':
    main()

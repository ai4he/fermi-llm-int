#!/usr/bin/env python3
"""End-to-end test against a running instance.

Drives the real HTTP API the way the browser does: create a task, generate
code, review, approve, watch progress, download the outputs. It needs a
server and executes a real FermiPy analysis, so it takes ~12 minutes.

    FERMI_LLM_TEST_URL=http://localhost:8769 python tests/test_e2e.py
    python tests/test_e2e.py --model gemini-2.5-flash-lite
"""

import argparse
import json
import os
import sys
import time

import requests

BASE_URL = os.environ.get('FERMI_LLM_TEST_URL', 'http://localhost:8765')
PROMPT = ('Perform spectral analysis of Markarian 421 between 1 GeV and '
          '1 TeV, compute the spectral energy distribution')


def check(condition, message):
    print(('  PASS: ' if condition else '  FAIL: ') + message, flush=True)
    if not condition:
        raise AssertionError(message)


def test_static_and_metadata():
    print('\n[1/6] Static assets and metadata')
    for path in ('/', '/static/core/app.js', '/static/core/plugins.js',
                 '/api/ui/skin.css', '/api/models', '/api/versions',
                 '/api/demo_samples'):
        resp = requests.get(BASE_URL + path, timeout=30)
        check(resp.status_code == 200, f'{path} -> {resp.status_code}')


def test_platform_view():
    print('\n[2/6] Platform view')
    data = requests.get(BASE_URL + '/api/platform', timeout=30).json()
    check(not data['plugin_failures'],
          f'no plugin failures: {data["plugin_failures"]}')
    for kind in ('model_provider', 'guardrail', 'validator', 'exporter',
                 'execution_backend', 'session_store', 'data_source', 'skin'):
        check(bool(data['active'].get(kind)), f'{kind} has an active component')


def test_chat(model):
    print(f'\n[3/6] Chat ({model})')
    session_id = requests.post(BASE_URL + '/api/session/create',
                               timeout=30).json()['session_id']
    print(f'  session: {session_id}')
    started = time.time()
    resp = requests.post(f'{BASE_URL}/api/session/{session_id}/chat',
                         json={'message': PROMPT, 'model': model}, timeout=900)
    check(resp.status_code == 200, f'chat -> {resp.status_code}')
    data = resp.json()
    check(len(data['yaml']) > 50, f'YAML generated ({len(data["yaml"])} chars)')
    check(len(data['python']) > 50,
          f'Python generated ({len(data["python"])} chars)')
    print(f'  generated in {time.time() - started:.0f}s')
    return session_id


def test_review(session_id):
    print('\n[4/6] Review (first click runs nothing)')
    resp = requests.post(f'{BASE_URL}/api/session/{session_id}/run_pipeline',
                         json={}, timeout=600)
    data = resp.json()
    check(data.get('status') == 'review_required',
          f'status is review_required (got {data.get("status")})')
    check(bool(data.get('approval_token')), 'an approval token was issued')
    for name, url in (data.get('downloads') or {}).items():
        check(requests.get(BASE_URL + url, timeout=60).status_code == 200,
              f'review download: {name}')
    return data['approval_token']


def test_run(session_id, token, timeout=3600):
    print('\n[5/6] Approved run')
    resp = requests.post(f'{BASE_URL}/api/session/{session_id}/run_pipeline',
                         json={'approval_token': token, 'confirm': True},
                         timeout=120)
    check(resp.json().get('status') == 'started', 'run started')
    started, last = time.time(), None
    while time.time() - started < timeout:
        status = requests.get(
            f'{BASE_URL}/api/session/{session_id}/pipeline_status',
            timeout=60).json()
        if status.get('status') != last:
            last = status.get('status')
            print(f'  [{time.time() - started:5.0f}s] {last}', flush=True)
        if status.get('result') or last not in ('running', 'idle'):
            break
        time.sleep(15)
    result = status.get('result') or {}
    print(f'  status={result.get("status")} error={result.get("error")}')
    check(result.get('success'), 'the run succeeded')
    check(result.get('expected_sed_bins') == result.get('actual_sed_bins'),
          f'SED bin contract honoured '
          f'({result.get("actual_sed_bins")}/{result.get("expected_sed_bins")})')
    return result


def test_downloads(session_id, result):
    print('\n[6/6] Downloads')
    for name, url in (result.get('downloads') or {}).items():
        check(requests.get(BASE_URL + url, timeout=120).status_code == 200,
              f'run download: {name}')
    exports = requests.get(f'{BASE_URL}/api/session/{session_id}/exports',
                           timeout=60).json()['exports']
    for export in exports:
        if not export['available']:
            continue
        resp = requests.get(BASE_URL + export['url'], timeout=300)
        check(resp.status_code == 200, f'export: {export["name"]}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='template',
                        help='model id to generate with (default: template)')
    parser.add_argument('--url', default=BASE_URL)
    args = parser.parse_args()

    global BASE_URL
    BASE_URL = args.url
    print(f'End-to-end test against {BASE_URL}')
    try:
        test_static_and_metadata()
        test_platform_view()
        session_id = test_chat(args.model)
        token = test_review(session_id)
        result = test_run(session_id, token)
        test_downloads(session_id, result)
    except AssertionError as exc:
        print(f'\nFAILED: {exc}')
        return 1
    print('\nAll end-to-end checks passed.')
    return 0


if __name__ == '__main__':
    sys.exit(main())

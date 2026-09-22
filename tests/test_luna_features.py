#!/usr/bin/env python3
"""
Tests for the features added in commit 3d4ec5e:

  1. OpenAI "ChatGPT Luna 5.6" backend            (backend='openai')
  2. "ChatGPT Luna 5.6 (Codex session)" backend   (backend='codex')
  3. Run-outcome continuity for API/local models  (_build_run_outcome_summary)

Unit tests (default) mock all network / subprocess I/O, so they are fast, free,
and deterministic while still exercising the REAL code paths in
webapp/model_manager.py and webapp/server.py.

Live smoke tests (opt-in with `--live`) make real OpenAI API + Codex CLI calls
and drive the running web app on :8765. They cost a little API/subscription
usage.

Run:
    python test_luna_features.py            # unit tests only
    python test_luna_features.py --live     # unit + live smoke tests
"""

import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import types

import conftest  # noqa: F401  (bootstraps the context)
import compat

mm = compat.model_manager()

LIVE = "--live" in sys.argv
BASE_URL = "http://localhost:8765"

# ---------------------------------------------------------------------------
# Tiny test harness: collect pass/fail without aborting on first failure.
# ---------------------------------------------------------------------------
_RESULTS = []


def run(test):
    name = test.__name__
    try:
        test()
        _RESULTS.append((name, True, ""))
        print(f"  PASS: {name}")
    except AssertionError as e:
        _RESULTS.append((name, False, str(e)))
        print(f"  FAIL: {name} :: {e}")
    except Exception as e:  # unexpected error is also a failure
        _RESULTS.append((name, False, f"{type(e).__name__}: {e}"))
        print(f"  ERROR: {name} :: {type(e).__name__}: {e}")


# ===========================================================================
# 1. Registry / model listing
# ===========================================================================

def test_registry_has_both_luna_options():
    reg = mm.MODEL_REGISTRY
    assert 'gpt-5.6-luna' in reg, "openai Luna entry missing"
    assert reg['gpt-5.6-luna']['backend'] == 'openai'
    assert reg['gpt-5.6-luna']['group'] == 'OpenAI API'
    assert reg['gpt-5.6-luna']['openai_model'] == 'gpt-5.6-luna'
    assert 'gpt-5.6-luna-codex' in reg, "codex Luna entry missing"
    assert reg['gpt-5.6-luna-codex']['backend'] == 'codex'
    assert reg['gpt-5.6-luna-codex']['group'] == 'OpenAI Codex'


def test_default_model_is_luna_api():
    assert mm.DEFAULT_MODEL_ID == 'gpt-5.6-luna'


def test_models_endpoint_advertises_default():
    import asyncio
    srv = _load_server()
    payload = asyncio.run(srv.list_models())
    assert payload['default_model'] == 'gpt-5.6-luna'
    assert any(m['id'] == payload['default_model'] for m in payload['models'])


def test_openai_entry_budget_is_generous():
    # Reasoning tokens count against the completion budget; must be large enough
    # not to truncate YAML+Python after hidden reasoning.
    assert mm.MODEL_REGISTRY['gpt-5.6-luna']['max_new_tokens'] >= 8000


def test_list_models_openai_availability_tracks_key():
    m = mm.ModelManager()
    m._openai_key = "sk-test"
    entry = next(x for x in m.list_models() if x['id'] == 'gpt-5.6-luna')
    assert entry['available'] is True
    m._openai_key = None
    entry = next(x for x in m.list_models() if x['id'] == 'gpt-5.6-luna')
    assert entry['available'] is False
    assert entry['status'] == 'no_api_key'


def test_list_models_codex_availability_tracks_binary(monkeypatch_attr):
    m = mm.ModelManager()
    # Force codex "not found"
    with monkeypatch_attr(mm, 'CODEX_BIN', '/definitely/not/here'):
        entry = next(x for x in m.list_models() if x['id'] == 'gpt-5.6-luna-codex')
        assert entry['available'] is False
        assert entry['status'] == 'not_running'
    # Real environment: should be ready (binary + auth present on this host)
    ok, detail = m._codex_ready()
    assert ok is True, f"codex not ready on host: {detail}"


# ===========================================================================
# 2. OpenAI backend — payload shape (mock urllib)
# ===========================================================================

def _fake_openai_response(content="```yaml\na: 1\n```\n```python\nx=1\n```",
                          finish_reason="stop"):
    body = json.dumps({
        "choices": [{"message": {"content": content},
                     "finish_reason": finish_reason}],
        "usage": {"completion_tokens": 10},
    }).encode()
    return io.BytesIO(body)


def test_openai_payload_obeys_gpt5_contract(monkeypatch_attr):
    """max_completion_tokens present, NO temperature, reasoning_effort set."""
    import urllib.request
    captured = {}

    def fake_urlopen(req, timeout=None):
        captured['url'] = req.full_url
        captured['auth'] = req.headers.get('Authorization')
        captured['payload'] = json.loads(req.data.decode())
        return _fake_openai_response()

    m = mm.ModelManager()
    m._openai_key = "sk-unit-test"
    with monkeypatch_attr(urllib.request, 'urlopen', fake_urlopen):
        text = m._generate_openai('gpt-5.6-luna', "make a fermipy config")

    p = captured['payload']
    assert p['model'] == 'gpt-5.6-luna'
    assert 'max_completion_tokens' in p, "must use max_completion_tokens"
    assert 'max_tokens' not in p, "legacy max_tokens must NOT be sent"
    assert 'temperature' not in p, "temperature must not be sent (only default 1 allowed)"
    assert p['reasoning_effort'] == 'high'
    assert p['response_format']['type'] == 'json_schema'
    assert p['response_format']['json_schema']['strict'] is True
    schema = p['response_format']['json_schema']['schema']
    assert set(schema['required']) == {'intent', 'yaml', 'python'}
    assert captured['auth'] == 'Bearer sk-unit-test'
    assert captured['url'].endswith('/chat/completions')
    assert '```yaml' in text


def test_openai_truncation_raises_helpful_error(monkeypatch_attr):
    """Empty content + finish_reason='length' => clear 'increase budget' error."""
    import urllib.request

    def fake_urlopen(req, timeout=None):
        return _fake_openai_response(content="", finish_reason="length")

    m = mm.ModelManager()
    m._openai_key = "sk-unit-test"
    raised = None
    with monkeypatch_attr(urllib.request, 'urlopen', fake_urlopen):
        try:
            m._generate_openai('gpt-5.6-luna', "prompt")
        except RuntimeError as e:
            raised = str(e)
    assert raised is not None, "should raise on truncated-before-content"
    assert 'truncat' in raised.lower() or 'budget' in raised.lower()


def test_openai_missing_key_raises():
    m = mm.ModelManager()
    m._openai_key = None
    try:
        m._generate_openai('gpt-5.6-luna', "prompt")
        assert False, "should raise without a key"
    except RuntimeError as e:
        assert 'key' in str(e).lower()


# ===========================================================================
# 3. Codex backend — command construction, resume, fallback (mock subprocess)
# ===========================================================================

class _FakeProc:
    def __init__(self, returncode, stdout="", stderr=""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def _install_fake_subprocess(monkeypatch_attr, calls, script):
    """Patch subprocess.run to record argv and drive behaviour via `script`,
    a list of dicts: {returncode, thread_id, last_message}. Writes last_message
    to the -o file so _generate_codex can read it."""
    import subprocess
    state = {'i': 0}

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        step = script[state['i']]
        state['i'] += 1
        # honor -o <file>: write the fake final message there
        if '-o' in cmd:
            outpath = cmd[cmd.index('-o') + 1]
            if step.get('last_message') is not None:
                with open(outpath, 'w') as f:
                    f.write(step['last_message'])
        stdout = ""
        if step.get('thread_id'):
            stdout = json.dumps({"type": "thread.started",
                                 "thread_id": step['thread_id']}) + "\n"
        return _FakeProc(step['returncode'], stdout=stdout,
                         stderr=step.get('stderr', ''))

    return monkeypatch_attr(subprocess, 'run', fake_run)


def test_codex_fresh_session_builds_exec_and_captures_thread(monkeypatch_attr):
    calls = []
    script = [{'returncode': 0, 'thread_id': 'TID-NEW',
               'last_message': '```yaml\na: 1\n```\n```python\nx=1\n```'}]
    m = mm.ModelManager()
    with _install_fake_subprocess(monkeypatch_attr, calls, script):
        text, tid = m._generate_codex('gpt-5.6-luna-codex', "prompt", thread_id=None)
    argv = calls[0]
    assert argv[1] == 'exec' and 'resume' not in argv, "fresh call must be plain exec"
    assert '--json' in argv and '-m' in argv
    assert argv[argv.index('-m') + 1] == mm.CODEX_MODEL
    assert '-s' in argv and argv[argv.index('-s') + 1] == mm.CODEX_SANDBOX
    assert tid == 'TID-NEW', "thread_id must be captured from thread.started"
    assert '```yaml' in text


def test_codex_resume_uses_resume_subcommand_and_config_sandbox(monkeypatch_attr):
    calls = []
    # resume succeeds; resume does NOT re-emit thread.started, so tid stays.
    script = [{'returncode': 0, 'thread_id': None,
               'last_message': 'ok resumed'}]
    m = mm.ModelManager()
    with _install_fake_subprocess(monkeypatch_attr, calls, script):
        text, tid = m._generate_codex('gpt-5.6-luna-codex', "edit", thread_id='TID-OLD')
    argv = calls[0]
    assert argv[1] == 'exec' and argv[2] == 'resume' and argv[3] == 'TID-OLD'
    assert '-s' not in argv, "resume must NOT pass -s (unsupported)"
    joined = ' '.join(argv)
    assert 'sandbox_mode=' in joined, "resume must set sandbox via -c"
    assert tid == 'TID-OLD', "resume keeps the same thread id"
    assert text == 'ok resumed'


def test_codex_resume_failure_falls_back_to_fresh(monkeypatch_attr):
    calls = []
    script = [
        {'returncode': 1, 'thread_id': None, 'stderr': 'session not found'},  # resume fails
        {'returncode': 0, 'thread_id': 'TID-FRESH', 'last_message': 'recovered'},  # fresh ok
    ]
    m = mm.ModelManager()
    with _install_fake_subprocess(monkeypatch_attr, calls, script):
        text, tid = m._generate_codex('gpt-5.6-luna-codex', "edit", thread_id='DEAD')
    assert len(calls) == 2, "should retry once as a fresh session"
    assert calls[0][2] == 'resume', "first attempt is resume"
    assert 'resume' not in calls[1], "second attempt is a fresh exec"
    assert tid == 'TID-FRESH' and text == 'recovered'


def test_codex_empty_output_raises(monkeypatch_attr):
    calls = []
    script = [{'returncode': 0, 'thread_id': 'T', 'last_message': '   '}]
    m = mm.ModelManager()
    raised = False
    with _install_fake_subprocess(monkeypatch_attr, calls, script):
        try:
            m._generate_codex('gpt-5.6-luna-codex', "p", thread_id=None)
        except RuntimeError:
            raised = True
    assert raised, "empty codex message must raise"


def test_codex_hard_failure_raises(monkeypatch_attr):
    calls = []
    script = [{'returncode': 2, 'thread_id': None, 'stderr': 'boom'}]
    m = mm.ModelManager()
    raised = False
    with _install_fake_subprocess(monkeypatch_attr, calls, script):
        try:
            m._generate_codex('gpt-5.6-luna-codex', "p", thread_id=None)
        except RuntimeError as e:
            raised = 'boom' in str(e) or 'failed' in str(e).lower()
    assert raised, "fresh exec non-zero must raise with stderr"


# ===========================================================================
# 4. Run-outcome continuity (server._build_run_outcome_summary + build_prompt)
# ===========================================================================

def _static_dir():
    """The frontend now ships inside the package."""
    import fermi_llm
    return os.path.join(os.path.dirname(fermi_llm.__file__), 'web', 'static')


def _load_server():
    """The former ``server`` module, now split (see tests/compat.py)."""
    return compat.server()


def test_summary_empty_when_no_run():
    srv = _load_server()
    s = types.SimpleNamespace(pipeline_result=None)
    assert srv._build_run_outcome_summary(s) == ''
    s = types.SimpleNamespace(pipeline_result={})
    assert srv._build_run_outcome_summary(s) == ''
    s = types.SimpleNamespace(pipeline_result="not-a-dict")
    assert srv._build_run_outcome_summary(s) == ''


def test_summary_success_has_fit_and_repairs():
    srv = _load_server()
    res = {
        'status': 'complete', 'success': True, 'run_mode': 'full',
        'level4': {'fit_ok': True, 'fit_quality': 3, 'target_ts': 1234.5,
                   'target_flux': 7.3e-14, 'spectral_index': 1.63,
                   'convergence_ok': True, 'ts_check': True, 'flux_check': False},
        'repair_history': [{'iteration': 1, 'repairs': ['set isodiff']}],
        'target_unmatched': False,
    }
    s = types.SimpleNamespace(pipeline_result=res)
    out = srv._build_run_outcome_summary(s)
    assert 'Status: complete' in out and 'run_mode: full' in out
    assert 'TS=' in out and 'fit_ok=True' in out
    assert 'ts_check=True' in out and 'flux_check=False' in out
    assert 'Auto-repairs applied' in out
    assert 'NOT matched' not in out  # target matched => no warning


def test_summary_failure_has_error_and_traceback():
    srv = _load_server()
    res = {
        'status': 'error', 'success': False, 'run_mode': 'full',
        'error': 'gta.fit() failed: RuntimeError: NaN in likelihood',
        'traceback': 'Traceback ...\n  line 1\nRuntimeError: NaN in likelihood',
        'target_unmatched': True,
    }
    s = types.SimpleNamespace(pipeline_result=res)
    out = srv._build_run_outcome_summary(s)
    assert 'Error:' in out and 'NaN in likelihood' in out
    assert 'Traceback (last line):' in out
    assert 'NOT matched' in out  # unmatched target => warning present


def test_summary_survives_weird_shapes():
    srv = _load_server()
    # level4 None, repair_history non-list, missing keys — must not crash
    for res in [
        {'success': True, 'level4': None, 'repair_history': None},
        {'success': False},
        {'success': True, 'level4': {'target_ts': None}, 'repair_history': ['x']},
    ]:
        s = types.SimpleNamespace(pipeline_result=res)
        out = srv._build_run_outcome_summary(s)
        assert isinstance(out, str)


def test_abort_endpoint_records_aborted_result(monkeypatch_attr):
    """Abort state is persisted without starting any FermiPy work."""
    import asyncio
    srv = _load_server()
    sid = f"abort-unit-{os.getpid()}"

    async def scenario():
        session = srv.AnalysisSession(sid)
        session.current_yaml = 'selection:\n  target: Crab\n'
        session.pipeline_status = 'running'
        session.save()
        srv.sessions[sid] = session
        future = asyncio.get_running_loop().create_future()
        entry = {
            'run_id': 'unit-run', 'future': future,
            'executor': types.SimpleNamespace(), 'aborted': False,
        }
        srv.active_pipeline_runs[sid] = entry

        async def fake_terminate(session_dir, active_entry):
            assert session_dir == session.session_dir
            assert active_entry is entry
            return True

        with monkeypatch_attr(srv, '_terminate_pipeline_worker', fake_terminate):
            payload = await srv.abort_pipeline(sid)

        assert payload['status'] == 'aborted'
        assert payload['result']['aborted'] is True
        assert payload['result']['worker_terminated'] is True
        assert session.pipeline_status == 'aborted'
        assert sid not in srv.active_pipeline_runs
        with open(os.path.join(session.session_dir, 'pipeline_result.json')) as f:
            saved = json.load(f)
        assert saved['run_id'] == 'unit-run'
        assert saved['status'] == 'aborted'

    try:
        asyncio.run(scenario())
    finally:
        srv.sessions.pop(sid, None)
        srv.active_pipeline_runs.pop(sid, None)
        shutil.rmtree(os.path.join(srv.SESSIONS_DIR, sid), ignore_errors=True)


def test_sidebar_brand_and_abort_controls_present():
    static_dir = _static_dir()
    with open(os.path.join(static_dir, 'index.html')) as f:
        html = f.read()
    with open(os.path.join(static_dir, 'core', 'app.js')) as f:
        js = f.read()
    assert '<h1>Fermi LLM</h1>' in html
    assert 'Fermi-LAT Gamma-Ray Analysis' not in html
    assert '&#9776;' in html
    assert 'title="Open sidebar"' in html
    assert 'id="abort-btn" onclick="abortPipeline()" class="btn-danger" disabled' in html
    assert '/abort_pipeline' in js
    assert 'id = \'abort-confirm-overlay\'' in js
    assert 'Yes, Abort Run' in js
    assert 'Abort Run selected — waiting for confirmation.' in js
    assert 'Abort completed — the pipeline process was stopped.' in js
    assert 'executeAbortPipeline(abortSessionId)' in js
    assert "abortBtn.disabled = abortDisabled" in js
    assert "abortBtn.classList.toggle('hidden'" not in js
    # The stylesheet is served by the active skin component now.
    assert '/api/ui/skin.css' in html
    assert '/static/core/app.js?v=20260914-1' in html
    assert "const label = open ? 'Close sidebar' : 'Open sidebar'" in js


def test_vertical_layout_preserves_pipeline_progress_height():
    """The content workspace must not squeeze the fixed top/bottom chrome."""
    static_dir = _static_dir()
    with open(os.path.join(static_dir, 'skins', 'classic.css')) as f:
        css = f.read()

    chrome_rule = css[css.index('.header,'):css.index('/* Header */')]
    assert '.pipeline-progress,' in chrome_rule
    assert '.action-bar {' in chrome_rule
    assert 'flex: 0 0 auto;' in chrome_rule

    progress_rule = css[css.index('.pipeline-progress {'):
                        css.index('.pipeline-progress.hidden')]
    assert 'overflow-y: hidden;' in progress_rule
    assert '.workspace-row { flex: 1 1 0;' in css
    assert '.download-card {' in css
    assert '.download-file-grid {' in css
    assert '.download-utility-links {' in css


def test_pipeline_controls_are_scoped_to_the_active_run():
    """UI regressions: chat generation is not a run, and HTTP errors recover."""
    static_dir = _static_dir()
    with open(os.path.join(static_dir, 'core', 'app.js')) as f:
        js = f.read()
    chat_block = js[js.index('async function sendMessage()'):
                    js.index('function startChatSSE()')]
    assert 'showProgressBar();' not in chat_block
    assert 'hideProgressBar();' in chat_block
    assert 'if (!resp.ok)' in js
    assert "pipelineStatus === 'running'" in js
    assert 'connectedSessionId !== sessionId' in js
    assert "setEditorValue('python', result.final_python)" in js
    assert "data.status === 'review_required'" in js
    assert 'approval_token: pendingRunApproval' in js
    assert 'Nothing has run yet.' in js
    assert 'download-card' in js and 'download-bundle-btn' in js
    assert 'download-file-btn' in js and 'download-utility-links' in js
    assert 'Final config' in js and 'Analysis script' in js
    assert 'YAML + Python' in js


def test_abort_terminates_isolated_worker_process():
    """Exercise the real signal path with only a lightweight sleeping child."""
    import asyncio
    srv = _load_server()
    temp_dir = tempfile.mkdtemp(prefix='fermi-abort-test-')
    proc = subprocess.Popen(
        [sys.executable, '-c', 'import time; time.sleep(30)'],
        start_new_session=True,
    )

    class FakeFuture:
        def __init__(self):
            self.cancelled = False

        def done(self):
            return False

        def cancel(self):
            self.cancelled = True

    class FakeExecutor:
        _processes = {}

        def __init__(self):
            self.shutdown_called = False

        def shutdown(self, **kwargs):
            self.shutdown_called = True

    future = FakeFuture()
    executor = FakeExecutor()
    entry = {
        'run_id': 'signal-test', 'future': future,
        'executor': executor, 'aborted': True,
    }
    with open(os.path.join(temp_dir, 'pipeline_worker.json'), 'w') as f:
        json.dump({
            'run_id': 'signal-test', 'pid': proc.pid,
            'process_group_isolated': True,
        }, f)

    try:
        terminated = asyncio.run(srv._terminate_pipeline_worker(temp_dir, entry))
        proc.wait(timeout=5)
        assert terminated is True
        assert future.cancelled is True
        assert executor.shutdown_called is True
        assert proc.returncode != 0
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=5)
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_build_prompt_renders_run_outcome_in_edit_mode():
    ctx = {'prompt': 'Analyze Mrk 421',
           'yaml': 'selection:\n  target: 4FGL J1104.4+3812\n',
           'python': 'gta = 1',
           'run_outcome': 'Status: error\nError: NaN in likelihood'}
    p = mm.build_prompt("fix it", [], n_shots=0, context=ctx)
    assert 'last EXECUTED with this outcome' in p
    assert 'NaN in likelihood' in p
    assert 'INCREMENTAL EDIT' in p


def test_build_prompt_no_outcome_section_when_absent():
    ctx = {'prompt': 'Analyze Mrk 421',
           'yaml': 'selection:\n  target: X\n', 'python': 'y=1'}
    p = mm.build_prompt("add SED", [], n_shots=0, context=ctx)
    assert 'last EXECUTED with this outcome' not in p
    assert 'INCREMENTAL EDIT' in p  # still an edit


def test_build_prompt_fresh_has_examples_no_edit_block():
    train = [{'prompt': 'Example task', 'response': {'yaml': 'a: 1', 'script': 'x=1'}}]
    p = mm.build_prompt("new analysis", train, n_shots=1, context=None)
    assert 'Examples' in p
    assert 'INCREMENTAL EDIT' not in p


def _intent(**overrides):
    value = {
        'mode': 'edit',
        'summary': 'Apply the requested incremental edit.',
        'preserve_unspecified': True,
        'target': {
            'action': 'keep', 'query': '', 'evidence': '',
            'ra_deg': None, 'dec_deg': None, 'position_evidence': '',
            'spatial_model': 'unspecified',
        },
        'products': [],
        'selection_changes': [],
        'ambiguities': [],
    }
    value.update(overrides)
    return value


def test_luna_prompt_requests_semantic_intent_for_codex():
    p = mm.build_prompt(
        'Remove the spectrum and add a weekly light curve.', [], n_shots=0,
        context={'yaml': 'selection: {}', 'python': 'x = 1'},
        include_intent=True, structured_envelope=False)
    assert '### Semantic Intent:' in p
    assert 'understand paraphrases, negation' in p
    assert 'preserve_unspecified=true' in p
    assert '### YAML Configuration:' in p and '### Python Script:' in p


def test_luna_structured_prompt_uses_raw_file_fields():
    p = mm.build_prompt(
        'Analyze M87.', [], n_shots=0, include_intent=True,
        structured_envelope=True)
    assert 'provided JSON schema' in p
    assert 'raw YAML and raw Python' in p
    assert '### Semantic Intent:' not in p


def test_extract_semantic_intent_rejects_bad_operations():
    raw = _intent(
        products=[
            {'name': 'sed', 'action': 'remove', 'evidence': 'remove spectrum'},
            {'name': 'not-real', 'action': 'explode', 'evidence': 'x'},
        ],
        selection_changes=[
            {'path': 'selection.emin', 'action': 'set',
             'value_json': '1000', 'evidence': 'lower bound to 1 GeV'},
            {'path': '', 'action': 'set', 'value_json': '1', 'evidence': ''},
        ])
    text = '### Semantic Intent:\n```json\n' + json.dumps(raw) + '\n```'
    parsed = mm.extract_semantic_intent(text)
    assert parsed is not None
    assert parsed['products'] == [
        {'name': 'sed', 'action': 'remove', 'evidence': 'remove spectrum'}]
    assert len(parsed['selection_changes']) == 1


def test_generate_openai_extracts_structured_envelope(monkeypatch_attr):
    m = mm.ModelManager()
    intent = _intent(
        mode='create',
        target={'action': 'set', 'query': 'M87', 'evidence': 'Analyze M87'})
    envelope = json.dumps({
        'intent': intent,
        'yaml': 'selection:\n  target: 4FGL J1230.8+1223\n',
        'python': 'from fermipy.gtanalysis import GTAnalysis\n',
    })
    with monkeypatch_attr(m, '_generate_openai',
                          lambda model_id, prompt: envelope):
        yaml_text, python_text, meta = m.generate(
            'gpt-5.6-luna', 'Analyze M87.', n_shots=0)
    assert '4FGL J1230.8+1223' in yaml_text
    assert 'GTAnalysis' in python_text
    assert meta['semantic_intent']['target']['query'] == 'M87'
    assert meta['has_yaml'] is True and meta['has_python'] is True


def test_semantic_merge_honors_paraphrased_remove():
    srv = _load_server()
    previous = (
        'selection:\n  target: 4FGL J1104.4+3812\n  emin: 100\n'
        'sed:\n  bins_per_decade: 8\n'
        'tsmap:\n  model: point\n')
    generated = (
        'selection:\n  target: 4FGL J1104.4+3812\n  emin: 100\n'
        'lightcurve:\n  binsz: 604800\n')
    intent = _intent(products=[
        {'name': 'sed', 'action': 'remove', 'evidence': 'remove the spectrum'},
        {'name': 'lightcurve', 'action': 'add',
         'evidence': 'add a weekly light curve'},
    ])
    merged, carried = srv._merge_edit_yaml(
        generated, previous,
        'Remove the spectrum and add a weekly light curve.',
        semantic_intent=intent)
    cfg = srv.yaml.safe_load(merged)
    assert 'sed' not in cfg, 'explicit semantic removal must not be restored'
    assert 'lightcurve' in cfg
    assert 'tsmap' in cfg, 'unmentioned product must be preserved'
    assert 'tsmap' in carried


def test_semantic_merge_can_remove_selection_key():
    srv = _load_server()
    previous = 'selection:\n  target: Old\n  tmin: 1\n  tmax: 2\n'
    generated = 'selection:\n  target: Old\n  tmin: 1\n'
    intent = _intent(selection_changes=[
        {'path': 'selection.tmax', 'action': 'reset', 'value_json': '',
         'evidence': 'use the default end time'},
    ])
    merged, carried = srv._merge_edit_yaml(
        generated, previous, 'Use the default end time.',
        semantic_intent=intent)
    assert 'tmax' not in srv.yaml.safe_load(merged)['selection']
    assert 'selection.tmax' not in carried


def test_semantic_target_change_is_not_reverted():
    srv = _load_server()
    previous = 'selection:\n  target: 4FGL J1104.4+3812\n'
    generated = 'selection:\n  target: 4FGL J1230.8+1223\n'
    intent = _intent(target={
        'action': 'set', 'query': 'M87', 'evidence': 'Use M87 now'})
    out, reverted, _ = srv._preserve_target_on_edit(
        generated, previous, 'Use M87 now.', semantic_intent=intent)
    assert reverted is None
    assert '4FGL J1230.8+1223' in out


def test_non_catalog_target_with_verified_coordinates_is_injected():
    srv = _load_server()
    phrase = 'RA=150.123, Dec=-20.456'
    intent = _intent(
        mode='create',
        target={
            'action': 'set',
            'query': 'Candidate Alpha',
            'evidence': 'Candidate Alpha',
            'ra_deg': 10.0,  # deliberately wrong: evidence must win
            'dec_deg': 20.0,
            'position_evidence': phrase,
            'spatial_model': 'point',
        })
    analysis = srv._analysis_from_semantic_intent(
        intent, user_message=f'Analyze Candidate Alpha at {phrase}')
    assert analysis['target'] == 'Candidate Alpha'
    assert analysis['custom_target']['ra'] == 150.123
    out = srv._ensure_custom_target(
        'selection:\n  target: Wrong\nmodel:\n  catalogs: [4FGL-DR3]\n',
        analysis['custom_target'])
    cfg = srv.yaml.safe_load(out)
    assert cfg['selection']['target'] == 'Candidate Alpha'
    assert cfg['selection']['ra'] == 150.123
    sources = cfg['model']['sources']
    assert len(sources) == 1
    assert sources[0]['name'] == 'Candidate Alpha'
    assert sources[0]['SpatialModel'] == 'PointSource'
    assert sources[0]['fermillm_custom_target'] is True


def test_non_catalog_target_rejects_unverified_coordinates():
    srv = _load_server()
    intent = _intent(
        mode='create',
        target={
            'action': 'set',
            'query': 'Candidate Beta',
            'evidence': 'Candidate Beta',
            'ra_deg': 10.0,
            'dec_deg': 20.0,
            'position_evidence': 'RA=10 Dec=20',
            'spatial_model': 'point',
        })
    analysis = srv._analysis_from_semantic_intent(
        intent, user_message='Analyze Candidate Beta')
    assert analysis.get('target') is None
    assert not analysis.get('custom_target')
    assert any('not verified' in x for x in analysis['ambiguities'])


def test_semantic_mixed_product_edit_still_reconciles_addition():
    srv = _load_server()
    intent = _intent(products=[
        {'name': 'tsmap', 'action': 'remove', 'evidence': 'Remove the TS map'},
        {'name': 'lightcurve', 'action': 'add',
         'evidence': 'add a weekly light curve'},
    ])
    out, injected, _ = srv._reconcile_planned_products(
        'selection:\n  target: X\n', ['Light Curve'],
        'Remove the TS map and add a weekly light curve.',
        semantic_intent=intent)
    cfg = srv.yaml.safe_load(out)
    assert injected == ['lightcurve']
    assert 'lightcurve' in cfg and 'tsmap' not in cfg


# ===========================================================================
# 5. Jupyter notebook export
# ===========================================================================

def _fake_session_for_nb(**over):
    base = dict(session_id='nbtest', session_dir='.', current_yaml='',
                current_python='', current_prompt='', pipeline_result=None)
    base.update(over)
    return types.SimpleNamespace(**base)


def test_notebook_structure_and_serialisable():
    srv = _load_server()
    s = _fake_session_for_nb(
        current_yaml="selection:\n  target: 4FGL J1104.4+3812\n  emin: 100.0\n",
        current_python="from fermipy.gtanalysis import GTAnalysis\n",
        current_prompt="Analyze Mrk 421",
        pipeline_result={'status': 'complete', 'success': True,
                         'level4': {'fit_ok': True, 'target_ts': 1234.5}})
    nb = srv._build_notebook(s)
    assert nb['nbformat'] == 4 and isinstance(nb['cells'], list)
    for c in nb['cells']:
        assert c['cell_type'] in ('markdown', 'code')
        assert isinstance(c['source'], list)
    json.dumps(nb)  # must be JSON-serialisable (endpoint returns it)
    srcs = ["".join(c['source']) for c in nb['cells']]
    assert any('open("config.yaml"' in x for x in srcs), "no config writer cell"
    assert any('GTAnalysis' in x for x in srcs), "no analysis cell"
    assert any('Last run results' in x for x in srcs), "no results cell"


def test_notebook_filename_is_safe():
    srv = _load_server()
    s = _fake_session_for_nb(current_yaml="selection:\n  target: 4FGL J1104.4+3812\n")
    fn = srv._notebook_filename(s)
    assert fn.endswith('.ipynb')
    assert ' ' not in fn and '/' not in fn, "filename must be filesystem/URL safe"
    assert 'nbtest' in fn


def test_notebook_empty_session_still_valid():
    srv = _load_server()
    nb = srv._build_notebook(_fake_session_for_nb())
    assert nb['nbformat'] == 4 and len(nb['cells']) >= 2
    json.dumps(nb)


def test_analysis_script_isolates_failing_products_and_prepares_psmap():
    srv = _load_server()
    script = srv._build_analysis_python(
        "selection:\n  target: 4FGL J1104.4+3812\n  emin: 1000\n"
        "sed:\n  make_plots: true\npsmap:\n  nbinloge: 20\n")
    compile(script, 'analysis.py', 'exec')
    assert 'gta.write_model_map(model_name="psmap_model")' in script
    assert 'run_product("psmap", lambda: gta.psmap(**psmap_options()))' in script

    calls = []

    class FakeGTA:
        workdir = '/work'
        components = [types.SimpleNamespace(
            files={'ccube': '/work/ccube_00.fits'},
            config={'file_suffix': '_00'})]

        def __init__(self, *_):
            pass

        def __getattr__(self, name):
            return lambda *a, **k: calls.append((name, a, k)) or {}

        def sed(self, *a, **k):
            raise RuntimeError('sed exploded')

        def psmap(self, **k):
            calls.append(('psmap', (), k))
            return {'ok': True}

    body = script.replace('from fermipy.gtanalysis import GTAnalysis', '')
    with tempfile.TemporaryDirectory() as run_dir:
        cfg = os.path.join(run_dir, 'final_config.yaml')
        with open(cfg, 'w') as f:
            f.write("selection:\n  target: X\n  emin: 1000\n"
                    "sed: {}\npsmap:\n  nbinloge: 20\n")
        ns = {'__file__': os.path.join(run_dir, 'analysis.py'),
              '__name__': '__main__', 'GTAnalysis': FakeGTA}
        cwd = os.getcwd()
        old_env = os.environ.get('FERMI_LLM_CONFIG_PATH')
        os.environ['FERMI_LLM_CONFIG_PATH'] = cfg
        try:
            exec(compile(body, 'analysis.py', 'exec'), ns)
        finally:
            os.chdir(cwd)
            if old_env is None:
                os.environ.pop('FERMI_LLM_CONFIG_PATH', None)
            else:
                os.environ['FERMI_LLM_CONFIG_PATH'] = old_env
    state = ns['analysis_state']
    assert state['product_errors'] == {'sed': 'RuntimeError: sed exploded'}
    assert state['fit_result'] == {}
    psmap_kwargs = [k for name, _, k in calls if name == 'psmap'][0]
    assert psmap_kwargs['cmap'] == '/work/ccube_00.fits'
    assert psmap_kwargs['mmap'] == '/work/mcube_psmap_model_00.fits'
    assert psmap_kwargs['emin'] == 1000.0 and psmap_kwargs['nbinloge'] == 20


def test_run_exports_include_final_yaml_and_runnable_python():
    srv = _load_server()
    yaml_text = (
        "selection:\n"
        "  target: 4FGL J1104.4+3812\n"
        "sed:\n"
        "  make_plots: true\n"
        "lightcurve:\n"
        "  binsz: 604800\n"
    )
    result = {
        'status': 'complete_l4', 'success': True,
        'run_mode': 'full', 'final_yaml': yaml_text,
    }
    with tempfile.TemporaryDirectory() as session_dir:
        srv._write_run_exports(result, session_dir, 'run123')
        run_dir = os.path.join(session_dir, 'runs', 'run123')
        with open(os.path.join(run_dir, 'final_config.yaml')) as f:
            assert f.read() == yaml_text
        with open(os.path.join(run_dir, 'analysis.py')) as f:
            script = f.read()
        compile(script, 'analysis.py', 'exec')
        assert "GTAnalysis(str(config_path))" in script
        assert 'run_product("sed", gta.sed, target, **(config["sed"] or {}))' in script
        assert 'run_product("lightcurve", gta.lightcurve, target,' in script
        assert '**(config["lightcurve"] or {})' in script
        assert "FERMI_LLM_CONFIG_PATH" in script
        assert "analysis_state = {" in script
        # Product failures are recorded visibly, never allowed to discard the fit.
        assert '"product_errors": product_errors,' in script
        assert result['final_python'] == script
        assert result['downloads']['yaml'].endswith('/run123/final_config.yaml')
        assert result['downloads']['python'].endswith('/run123/analysis.py')
        with srv.zipfile.ZipFile(os.path.join(run_dir, 'analysis_files.zip')) as zf:
            assert set(zf.namelist()) == {
                'final_config.yaml', 'analysis.py', 'manifest.json'}


def test_six_bins_per_decade_produces_an_18_bin_contract():
    srv = _load_server()
    yaml_text = (
        "binning:\n"
        "  roiwidth: 10\n"
        "  binsz: 0.1\n"
        "  binsperdec: 6\n"
        "selection:\n"
        "  target: 4FGL J2253.9+1609\n"
        "  emin: 100\n"
        "  emax: 100000\n"
        "sed:\n"
        "  make_plots: true\n"
    )
    assert srv._expected_sed_bin_count(yaml_text) == 18
    script = srv._build_analysis_python(yaml_text)
    assert 'run_product("sed", gta.sed, target, **(config["sed"] or {}))' in script
    assert "gta.optimize()" in script and "fit_result = gta.fit()" in script
    compile(script, 'analysis.py', 'exec')


def test_validator_moves_sed_edges_out_of_yaml_and_keeps_the_count():
    """sed.loge_bins is a gta.sed() argument, so it cannot stay in the YAML.
    18 uniform edges over 3 decades is 6 bins per decade exactly, so the count
    survives as binning.binsperdec and needs no edges in the script."""
    from fermi_llm.fermipy.run_execution_validated import repair_config
    from fermi_llm.fermipy.fermipy_schema import fermipy_option_keys

    assert 'loge_bins' not in fermipy_option_keys('sed')
    assert 'num_bins' not in fermipy_option_keys('sed')
    edges = ', '.join(str(2.0 + index / 6.0) for index in range(19))
    yaml_text = (
        "binning:\n"
        "  roiwidth: 10\n"
        "  binsz: 0.1\n"
        "  binsperdec: 8\n"
        "selection:\n"
        "  target: 4FGL J1104.4+3812\n"
        "  emin: 100\n"
        "  emax: 100000\n"
        f"sed:\n  loge_bins: [{edges}]\n  make_plots: true\n"
    )
    repaired, repairs = repair_config(yaml_text, 0)
    import yaml
    config = yaml.safe_load(repaired)
    assert 'loge_bins' not in config['sed']
    assert config['binning']['binsperdec'] == 6
    assert any('removed sed.loge_bins' in item for item in repairs)
    assert any('binsperdec=6' in item for item in repairs)


def test_sed_bin_request_reads_a_total_and_ignores_lightcurve_bins():
    from fermi_llm.fermipy.fermipy_schema import sed_bin_request

    assert sed_bin_request('compute an SED with 7 energy bins') == 7
    assert sed_bin_request('produce a spectrum with 12 bins') == 12
    assert sed_bin_request('give me 10 spectral bins') == 10
    # The light-curve parser owns these; an SED count must not shadow them.
    assert sed_bin_request('make a lightcurve with 30 bins') is None
    assert sed_bin_request('a lightcurve with 30 time bins') is None
    # Both products in one sentence: each parser takes its own number.
    assert sed_bin_request('SED with 6 bins and a lightcurve with 30 bins') == 6
    assert sed_bin_request('analyze Mrk 421') is None


def test_sed_bin_plan_uses_binsperdec_only_for_a_whole_number():
    """binsperdec is bins per decade: it can carry "N bins" only when
    N / log10(emax/emin) is whole. Otherwise the count needs loge_bins."""
    from fermi_llm.fermipy.fermipy_schema import sed_bin_plan

    # 6 bins over 3 decades is exactly 2 per decade: the analysis grid IS the
    # requested grid, and gta.sed() uses it by default.
    plan = sed_bin_plan(100.0, 100000.0, 6)
    assert plan == {'mode': 'binsperdec', 'nbins': 6, 'binsperdec': 2}
    plan = sed_bin_plan(100.0, 100000.0, 24)
    assert plan['mode'] == 'binsperdec' and plan['binsperdec'] == 8
    # 7 over 3 decades is 2.33 per decade, which binsperdec cannot express.
    plan = sed_bin_plan(100.0, 100000.0, 7)
    assert plan['mode'] == 'loge_bins'
    edges = plan['loge_bins']
    assert len(edges) == 8
    assert edges[0] == 2.0 and edges[-1] == 5.0
    # Edges are rounded to 6 dp so the generated literal stays readable, so
    # the steps agree with 3/7 dex to within that rounding, not exactly.
    steps = [b - a for a, b in zip(edges, edges[1:])]
    assert all(abs(step - 3.0 / 7) < 2e-6 for step in steps), \
        'edges must be logarithmically spaced'
    # A range that survives float log10 noise still counts as whole.
    assert sed_bin_plan(50.0, 50000.0, 6)['binsperdec'] == 2
    assert sed_bin_plan(100.0, 100.0, 6) is None
    assert sed_bin_plan(100.0, 100000.0, 1) is None


def test_review_proposes_loge_bins_only_when_binsperdec_cannot_express_it():
    from fermi_llm.fermipy import script_validator

    yaml_text = (
        "binning:\n  roiwidth: 10\n  binsz: 0.1\n  binsperdec: 8\n"
        "selection:\n  target: 4FGL J1104.4+3812\n  emin: 100\n  emax: 100000\n"
        "sed:\n  make_plots: true\n"
    )
    script = (
        "import os\n"
        "from fermipy.gtanalysis import GTAnalysis\n"
        "gta = GTAnalysis(os.environ['FERMI_LLM_CONFIG_PATH'])\n"
        "gta.setup()\n"
        "gta.optimize()\n"
        "gta.fit()\n"
        "gta.sed('4FGL J1104.4+3812', make_plots=True)\n"
    )
    # 7 bins over 3 decades is not a whole number per decade -> edges.
    review = script_validator.review_script(
        script, yaml_text,
        requests=['Compute an SED of Mrk 421 with 7 energy bins'])
    proposal = review.get('proposal')
    assert proposal, 'a loge_bins proposal should be offered'
    assert 'loge_bins=[2.0, 2.428571,' in proposal['script']
    assert '7 energy-bin edges' in proposal['reason']
    assert not review['blocking'], 'the fix is advisory, never blocking'
    compile(proposal['script'], 'analysis.py', 'exec')
    # 6 bins IS a whole number per decade: repair_config puts it in the YAML
    # as binsperdec, so the script must be left alone.
    assert script_validator.apply_sed_loge_bins(script, yaml_text, 6) == script
    six = script_validator.review_script(
        script, yaml_text,
        requests=['Compute an SED of Mrk 421 with 6 energy bins'])
    assert 'loge_bins' not in ((six.get('proposal') or {}).get('script') or '')
    # A script that already sets its own edges is left alone either way.
    own = script.replace("make_plots=True",
                         "make_plots=True, loge_bins=[2.0, 3.0, 4.0, 5.0]")
    assert script_validator.script_sets_sed_bins(own)
    assert script_validator.apply_sed_loge_bins(own, yaml_text, 7) == own


def test_enumbins_survives_repair_and_sets_the_bin_contract():
    from fermi_llm.fermipy.run_execution_validated import repair_config
    srv = _load_server()

    yaml_text = (
        "binning:\n  roiwidth: 10\n  binsz: 0.1\n  binsperdec: 8\n"
        "  enumbins: 12\n"
        "selection:\n  target: 4FGL J1104.4+3812\n  emin: 100\n  emax: 100000\n"
        "sed:\n  make_plots: true\n"
    )
    repaired, _ = repair_config(yaml_text, 0)
    import yaml
    assert yaml.safe_load(repaired)['binning']['enumbins'] == 12
    # enumbins overrides binsperdec in FermiPy, so the contract follows it.
    assert srv._expected_sed_bin_count(repaired) == 12


def _fit_fixture():
    from fermi_llm.fermipy import script_validator
    yaml_text = (
        "binning:\n  roiwidth: 10\n  binsz: 0.1\n  binsperdec: 8\n"
        "selection:\n  target: 4FGL J1104.4+3812\n  emin: 100\n  emax: 100000\n"
        "model:\n  galdiff:\n    - /x/gll_iem_v07.fits\n"
        "  isodiff:\n    - /x/iso_P8R3_SOURCE_V3_v1.txt\n"
        "sed:\n  make_plots: true\n"
    )
    bare = (
        "import os\n"
        "from fermipy.gtanalysis import GTAnalysis\n"
        "gta = GTAnalysis(os.environ['FERMI_LLM_CONFIG_PATH'])\n"
        "gta.setup()\n"
        "gta.sed('4FGL J1104.4+3812', make_plots=True)\n"
    )
    return script_validator, yaml_text, bare


def test_fit_directions_are_detected_only_when_actually_given():
    sv, _, _ = _fit_fixture()
    # No fitting procedure stated: the default applies.
    assert not sv.fit_directions_requested(['Compute the SED of Mrk 421'])
    assert not sv.fit_directions_requested(
        ['Fit the spectrum of Vela and make a lightcurve'])
    assert not sv.fit_directions_requested([])
    # A stated procedure: the user's directions win instead.
    assert sv.fit_directions_requested(['free all sources within 5 degrees'])
    assert sv.fit_directions_requested(['keep the background fixed'])
    assert sv.fit_directions_requested(['free sources with TS > 25'])
    assert sv.fit_directions_requested(['run an iterative fit in three rounds'])
    assert sv.fit_directions_requested(['do not fit, just make a TS map'])
    assert sv.fit_directions_requested(['free only the normalizations'])


def test_default_fit_is_proposed_when_the_user_gives_no_directions():
    sv, yaml_text, bare = _fit_fixture()
    gaps = sv.default_fit_gaps(bare, yaml_text)
    assert gaps == ['galdiff', 'isodiff', '4FGL J1104.4+3812',
                    'optimize', 'fit']
    review = sv.review_script(bare, yaml_text,
                              requests=['Compute the SED of Mrk 421'])
    proposal = review.get('proposal')
    assert proposal, 'the default fitting procedure should be proposed'
    fixed = proposal['script']
    for call in ("gta.free_source('galdiff')", "gta.free_source('isodiff')",
                 "gta.free_source('4FGL J1104.4+3812')",
                 'gta.optimize()', 'gta.fit()'):
        assert call in fixed, call
    compile(fixed, 'analysis.py', 'exec')
    # The fit must land after setup and before the product it feeds.
    assert (fixed.index('gta.setup()') < fixed.index('gta.fit()')
            < fixed.index('gta.sed('))
    assert not review['blocking'], 'the default fit is advisory, never blocking'
    # Nothing beyond the default is added.
    assert 'free_sources' not in fixed
    # A script that already does it is left alone.
    assert sv.default_fit_gaps(fixed, yaml_text) == []
    assert sv.apply_default_fit(fixed, yaml_text) == fixed


def test_user_fitting_directions_are_not_overridden_by_the_default():
    sv, yaml_text, bare = _fit_fixture()
    # The user asked for something that contradicts the default (no freeing of
    # the background). Their direction wins: nothing is imposed on top.
    review = sv.review_script(
        bare, yaml_text,
        requests=['Make an SED of Mrk 421 but keep the background fixed'])
    proposed = (review.get('proposal') or {}).get('script') or ''
    assert "free_source('galdiff')" not in proposed
    assert "free_source('isodiff')" not in proposed


def test_default_fit_skips_an_incremental_run_and_absent_backgrounds():
    sv, yaml_text, bare = _fit_fixture()
    # load_roi reuses a saved fit, so there is nothing to free or refit.
    reuse = bare.replace("gta.setup()", "gta.load_roi('fit_model')")
    assert sv.default_fit_gaps(reuse, yaml_text) == []
    # A YAML with no diffuse backgrounds declared frees only the target.
    no_diffuse = yaml_text[:yaml_text.index('model:')] + "sed:\n  make_plots: true\n"
    assert sv.default_fit_gaps(bare, no_diffuse) == [
        '4FGL J1104.4+3812', 'optimize', 'fit']


def test_approved_worker_rejects_script_content_mismatch_without_execution():
    srv = _load_server()
    yaml_text = "selection:\n  target: 4FGL J1104.4+3812\n"
    reviewed_python = "# reviewed\n"
    digest = srv._run_content_digest(yaml_text, reviewed_python)
    with tempfile.TemporaryDirectory() as session_dir:
        script_path = os.path.join(session_dir, 'analysis.py')
        with open(script_path, 'w') as stream:
            stream.write("# changed after approval\n")
        result = srv.run_pipeline_isolated(
            session_dir, yaml_text, reviewed_python, test_idx=0,
            run_id='mismatch', approved_script_path=script_path,
            approved_digest=digest)
    assert result['status'] == 'approval_mismatch'
    assert result['success'] is False
    assert 'nothing was run' in result['error']


def test_notebook_prefers_final_run_exports():
    srv = _load_server()
    s = _fake_session_for_nb(
        current_yaml="selection:\n  target: Draft\n",
        current_python="# draft\n",
        pipeline_result={
            'final_yaml': "selection:\n  target: Final Target\n",
            'final_python': "# final runnable script\n",
        })
    nb = srv._build_notebook(s)
    srcs = ["".join(c['source']) for c in nb['cells']]
    assert any('Final Target' in source for source in srcs)
    assert any('# final runnable script' in source for source in srcs)
    assert not any('# draft' in source for source in srcs)


# ===========================================================================
# 6. Google Sign-In / auth / task ownership
# ===========================================================================

def test_auth_token_roundtrip():
    srv = _load_server()
    tok = srv._make_auth_token({'sub': 's1', 'email': 'e@x.com', 'name': 'N', 'picture': 'p'})
    v = srv._verify_auth_token(tok)
    assert v and v['sub'] == 's1' and v['email'] == 'e@x.com'


def test_auth_token_rejects_tamper_and_garbage():
    srv = _load_server()
    tok = srv._make_auth_token({'sub': 's1'})
    tampered = tok[:-2] + ('aa' if not tok.endswith('aa') else 'bb')
    assert srv._verify_auth_token(tampered) is None
    assert srv._verify_auth_token('not.a.token') is None
    assert srv._verify_auth_token('') is None


def test_auth_token_rejects_expiry():
    srv = _load_server()
    import time as _t, json as _j, base64, hmac, hashlib
    payload = {'sub': 's1', 'exp': int(_t.time()) - 5}
    pj = base64.urlsafe_b64encode(_j.dumps(payload).encode()).rstrip(b'=').decode()
    sig = base64.urlsafe_b64encode(
        hmac.new(srv._AUTH_SECRET, pj.encode(), hashlib.sha256).digest()).rstrip(b'=').decode()
    assert srv._verify_auth_token(f"{pj}.{sig}") is None


def test_google_verify_requires_configuration():
    srv = _load_server()
    if srv.GOOGLE_CLIENT_ID:
        return  # configured on this host; nothing to assert
    raised = False
    try:
        srv._verify_google_credential('anything')
    except Exception as e:
        raised = 'not configured' in str(e).lower()
    assert raised, "must refuse to verify when no client id is configured"


def test_list_user_sessions_filters_by_owner():
    srv = _load_server()
    import os, json, uuid, shutil
    owner = 'owner-' + uuid.uuid4().hex[:8]
    sid = 't' + uuid.uuid4().hex[:6]
    sdir = os.path.join(srv.SESSIONS_DIR, sid)
    os.makedirs(sdir, exist_ok=True)
    try:
        json.dump({'session_id': sid, 'owner': owner, 'title': 'Owned task',
                   'current_prompt': 'p', 'current_yaml': 'selection: {}',
                   'created_at': '2026-01-01T00:00:00', 'pipeline_status': 'idle'},
                  open(os.path.join(sdir, 'session.json'), 'w'))
        mine = srv._list_user_sessions(owner)
        assert any(s['session_id'] == sid for s in mine), "owned session missing"
        assert srv._list_user_sessions('someone-else') == [] or \
            all(s['session_id'] != sid for s in srv._list_user_sessions('someone-else'))
    finally:
        shutil.rmtree(sdir, ignore_errors=True)


# ===========================================================================
# LIVE smoke tests (opt-in): real OpenAI API, real Codex CLI, real HTTP.
# ===========================================================================

def test_live_openai_generation():
    m = mm.ModelManager()
    assert m._openai_key, "no OpenAI key configured on host"
    y, p, meta = m.generate(
        'gpt-5.6-luna',
        'Analyze blazar Mrk 421, 2 deg ROI, 100 MeV to 300 GeV, P8R3 SOURCE class.',
        progress_cb=lambda s, d: None)
    assert y.strip() and p.strip(), "empty YAML/Python from openai backend"
    import yaml as _y
    cfg = _y.safe_load(y)
    assert isinstance(cfg, dict) and 'selection' in cfg, "YAML not a valid fermipy config"


def test_live_codex_fresh_then_resume():
    m = mm.ModelManager()
    ok, detail = m._codex_ready()
    assert ok, detail
    y1, p1, meta1 = m.generate(
        'gpt-5.6-luna-codex',
        'Analyze blazar Mrk 421, 2 deg ROI, 100 MeV to 300 GeV, P8R3 SOURCE class.',
        progress_cb=lambda s, d: None)
    tid = meta1.get('codex_thread_id')
    assert tid, "no thread id captured on fresh codex turn"
    assert meta1.get('codex_resumed') is False
    # resume same thread with an edit
    ctx = {'prompt': '...', 'yaml': y1, 'python': p1}
    y2, p2, meta2 = m.generate(
        'gpt-5.6-luna-codex',
        'Change the lower energy bound to 1 GeV; keep everything else the same.',
        progress_cb=lambda s, d: None, context=ctx, codex_thread_id=tid)
    assert meta2.get('codex_thread_id') == tid, "resume did not reuse the same thread"
    assert meta2.get('codex_resumed') is True
    import re
    em = re.search(r'emin:\s*([0-9.eE+]+)', y2)
    assert em and float(em.group(1)) == 1000.0, f"edit not applied (emin={em and em.group(1)})"


def test_live_http_continuity_after_failed_run():
    """Seed a failed run on disk, then a 'fix it' follow-up must succeed via HTTP."""
    import requests, uuid, subprocess
    sid = "t" + uuid.uuid4().hex[:6]
    sdir = os.path.join(os.path.dirname(__file__), 'sessions', sid)
    os.makedirs(sdir, exist_ok=True)
    yaml_txt = ("logging:\n  verbosity: 3\nselection:\n  target: 4FGL J1104.4+3812\n"
                "  emin: 100.0\n  emax: 300000.0\n")
    session = {
        "session_id": sid, "chat_history": [], "current_yaml": yaml_txt,
        "current_python": "from fermipy.gtanalysis import GTAnalysis\n",
        "current_prompt": "Analyze Mrk 421", "validation_log": [],
        "pipeline_status": "error",
        "pipeline_result": {
            "status": "error", "success": False, "run_mode": "full",
            "error": "gta.fit() failed: RuntimeError: NaN in likelihood",
            "level4": {"fit_ok": False, "convergence_ok": False},
            "repair_history": [{"iteration": 1, "repairs": ["set isodiff"]}],
            "target_unmatched": False},
        "created_at": "2026-08-11T00:00:00", "last_executed_yaml": yaml_txt,
        "last_run_test_idx": None, "last_run_roi_ready": False, "codex_thread_id": None,
    }
    with open(os.path.join(sdir, 'session.json'), 'w') as f:
        json.dump(session, f)
    # force the server to load from disk (not a cached in-memory session)
    subprocess.run(['systemctl', '--user', 'restart', 'fermi-webapp.service'], check=False)
    time.sleep(10)
    try:
        r = requests.post(f"{BASE_URL}/api/session/{sid}/chat",
                          json={"message": "The fit failed. Fix it so it converges.",
                                "model": "gemini-3.5-flash"}, timeout=120)
        assert r.status_code == 200, f"HTTP {r.status_code}"
        d = r.json()
        assert d.get('yaml', '').strip(), "no YAML returned on continuity follow-up"
    finally:
        import shutil
        shutil.rmtree(sdir, ignore_errors=True)


def test_live_http_notebook_export():
    """Generate a config over HTTP, then the notebook endpoint returns a valid
    downloadable .ipynb; a fresh session with no analysis returns 400."""
    import requests
    # 400 before any analysis
    sid = requests.post(f"{BASE_URL}/api/session/create", timeout=15).json()['session_id']
    r0 = requests.get(f"{BASE_URL}/api/session/{sid}/notebook", timeout=15)
    assert r0.status_code == 400, f"expected 400 for empty session, got {r0.status_code}"
    # generate, then export
    g = requests.post(f"{BASE_URL}/api/session/{sid}/chat",
                      json={"message": "Analyze blazar Mrk 421, 2 deg ROI, 100 MeV to 300 GeV.",
                            "model": "gemini-3.5-flash"}, timeout=120)
    assert g.status_code == 200
    try:
        r = requests.get(f"{BASE_URL}/api/session/{sid}/notebook", timeout=20)
        assert r.status_code == 200, f"notebook HTTP {r.status_code}"
        assert 'attachment' in r.headers.get('content-disposition', '')
        fname = r.headers.get('x-notebook-filename', '')
        assert fname.endswith('.ipynb')
        nb = r.json()
        assert nb['nbformat'] == 4 and len(nb['cells']) >= 4
    finally:
        import shutil
        shutil.rmtree(os.path.join(os.path.dirname(__file__), 'sessions', sid),
                      ignore_errors=True)


def test_live_http_auth_and_ownership():
    """Full auth flow over HTTP with a forged (validly-signed) session token:
    config/me/guest-401, owned create+list+claim, and guest session GET."""
    import requests, shutil, os
    srv = _load_server()
    token = srv._make_auth_token({'sub': 'test-owner-xyz', 'email': 't@x.com',
                                  'name': 'Tester', 'picture': ''})
    H = {'Authorization': f'Bearer {token}'}

    # public config + guest/me + 401 gating
    assert requests.get(f"{BASE_URL}/api/auth/config", timeout=10).status_code == 200
    assert requests.get(f"{BASE_URL}/api/auth/me", timeout=10).json()['authenticated'] is False
    assert requests.get(f"{BASE_URL}/api/auth/me", headers=H, timeout=10).json()['authenticated'] is True
    assert requests.get(f"{BASE_URL}/api/my/sessions", timeout=10).status_code == 401

    created = []
    try:
        # owned session shows up in my/sessions
        r = requests.post(f"{BASE_URL}/api/session/create", headers=H, timeout=10).json()
        sid = r['session_id']; created.append(sid)
        assert r.get('owner') == 'test-owner-xyz'
        mine = requests.get(f"{BASE_URL}/api/my/sessions", headers=H, timeout=10).json()['sessions']
        assert any(s['session_id'] == sid for s in mine), "owned session not listed"

        # guest session: GET works (continuity) and can be claimed
        g = requests.post(f"{BASE_URL}/api/session/create", timeout=10).json()
        gsid = g['session_id']; created.append(gsid)
        assert g.get('owner') is None
        assert requests.get(f"{BASE_URL}/api/session/{gsid}", timeout=10).status_code == 200
        claim = requests.post(f"{BASE_URL}/api/session/{gsid}/claim", headers=H, timeout=10).json()
        assert claim['claimed'] is True and claim['owner'] == 'test-owner-xyz'
    finally:
        for s in created:
            shutil.rmtree(os.path.join(os.path.dirname(__file__), 'sessions', s),
                          ignore_errors=True)


# ===========================================================================
# 7. Validation Agent script review: the model's analysis.py is executed
# ===========================================================================

def _pipeline_module(name):
    """Analysis modules now live in the ``fermi_llm.fermipy`` package."""
    import importlib
    return importlib.import_module(f'fermi_llm.fermipy.{name}')


REVIEW_YAML = ("selection:\n  target: 4FGL J1104.4+3812\n"
               "fileio:\n  outdir: output\nsed:\n  make_plots: true\n")
PLAIN_SCRIPT = (
    "from fermipy.gtanalysis import GTAnalysis\n"
    "gta = GTAnalysis('config.yaml')\n"
    "gta.setup()\n"
    "gta.free_source('galdiff')\n"
    "gta.optimize()\n"
    "fit = gta.fit()\n"
    "gta.sed('4FGL J1104.4+3812', make_plots=True)\n"
    "print('done', fit is not None)\n"
)


def _errors(review, category=None):
    return [f for f in review['findings'] if f['severity'] == 'error'
            and (category is None or f['category'] == category)]


def test_script_validator_accepts_a_plain_fermipy_script():
    sv = _pipeline_module('script_validator')
    review = sv.static_review(PLAIN_SCRIPT, REVIEW_YAML)
    assert not review['blocking'], review['findings']
    assert review['runs_fit'] and review['products'] == ['sed']
    assert review['config_ref'] == 'config.yaml'
    assert sv.infer_run_mode(review) == 'full'


def test_script_validator_blocks_unsafe_operations():
    sv = _pipeline_module('script_validator')
    script = PLAIN_SCRIPT + (
        "import subprocess\nimport os\nos.system('ls')\nimport shutil\n"
        "shutil.rmtree('output')\nopen('/tmp/x.txt', 'w')\neval('1')\n"
        "().__class__.__subclasses__()\nimport requests\n"
        "open('../../configs/openai_key.txt').read()\n")
    review = sv.static_review(script, REVIEW_YAML)
    assert review['blocking']
    text = ' '.join(f['message'] for f in _errors(review, 'safety'))
    for needle in ('subprocess', 'os.system', 'shutil.rmtree', "/tmp/x.txt",
                   'eval', '__subclasses__', 'requests', 'sensitive path'):
        assert needle in text, (needle, text)
    # Relative writes inside the working directory stay allowed.
    ok = sv.static_review(PLAIN_SCRIPT + "open('notes.txt', 'w').write('x')\n"
                          "import numpy as np\nnp.save('flux.npy', [1])\n",
                          REVIEW_YAML)
    assert not ok['blocking'], ok['findings']


def test_script_validator_checks_fermipy_api_and_call_order():
    sv = _pipeline_module('script_validator')
    script = (
        "from fermipy.gtanalysis import GTAnalysis\n"
        "gta = GTAnalysis('config.yaml')\n"
        "gta.sed('4FGL J1104.4+3812')\n"
        "gta.setup()\n"
        "gta.fitt()\n"
        "gta.lightcurve('4FGL J1104.4+3812', fix_background=True)\n")
    review = sv.static_review(script, REVIEW_YAML)
    where = {(f['category'], f['line']) for f in _errors(review)}
    assert ('order', 3) in where, review['findings']
    assert ('api', 5) in where, review['findings']
    if sv._gtanalysis_class() is not None:
        # keyword options are read from the installed FermiPy's ConfigSchema
        assert ('api', 6) in where, review['findings']
        assert 'free_background' in ' '.join(
            f['message'] for f in _errors(review, 'api'))
    lc_warning = [f for f in review['findings'] if f['category'] == 'order'
                  and f['severity'] == 'warning' and f['line'] == 6]
    assert lc_warning, 'lightcurve before fit() must be flagged'


def test_script_validator_checks_yaml_consistency_and_offers_config_fix():
    sv = _pipeline_module('script_validator')
    script = (
        "import yaml\n"
        "from fermipy.gtanalysis import GTAnalysis\n"
        "cfg = yaml.safe_load(open('config.yaml'))\n"
        "gta = GTAnalysis('/data/elsewhere/config.yaml')\n"
        "gta.setup()\n"
        "gta.fit()\n"
        "print(cfg['tsmap'])\n")
    review = sv.static_review(script, REVIEW_YAML)
    text = ' '.join(f['message'] for f in _errors(review, 'consistency'))
    assert '/data/elsewhere/config.yaml' in text and "'tsmap'" in text
    fixed = sv.fix_config_reference(script)
    assert "FERMI_LLM_CONFIG_PATH" in fixed and fixed.startswith('import os\n')
    fixed_review = sv.static_review(fixed, REVIEW_YAML)
    assert '/data/elsewhere' not in ' '.join(
        f['message'] for f in fixed_review['findings'])


def _review_json(revised='', findings=None, summary='checked'):
    return json.dumps({'summary': summary, 'findings': findings or [],
                       'revised_script': revised})


def test_review_script_offers_llm_revision_as_a_proposal():
    sv = _pipeline_module('script_validator')
    revised = PLAIN_SCRIPT + "gta.lightcurve('4FGL J1104.4+3812', nbins=5)\n"
    prompts = []

    def llm(prompt):
        prompts.append(prompt)
        return "Here you go:\n```json\n" + _review_json(revised, [{
            'severity': 'error', 'category': 'fidelity', 'line': None,
            'message': 'The requested 5-bin light curve is missing.'}]) + "\n```"

    requests = ['Fit Mrk 421 and make a light curve with 5 bins']
    review = sv.review_script(PLAIN_SCRIPT, REVIEW_YAML, requests, llm=llm)
    assert prompts and requests[0] in prompts[0] and '   7| gta.sed' in prompts[0]
    assert review['llm_used'] and review['proposal']['kind'] == 'fix'
    assert "nbins=5" in review['proposal']['script']
    assert any(f['source'] == 'llm' for f in review['findings'])
    # LLM findings are advisory: they never block on their own.
    assert review['blocking'] is False


def test_review_script_rejects_unsafe_revision_and_falls_back_to_heuristics():
    sv = _pipeline_module('script_validator')
    unsafe = PLAIN_SCRIPT + "import os\nos.system('curl http://x')\n"
    review = sv.review_script(PLAIN_SCRIPT, REVIEW_YAML, ['fit it'],
                              llm=lambda prompt: _review_json(unsafe))
    assert review['proposal'] is None
    assert any('failed the deterministic checks' in f['message']
               for f in review['findings'])

    def broken(prompt):
        raise RuntimeError('backend down')
    review = sv.review_script(PLAIN_SCRIPT, REVIEW_YAML,
                              ['Fit Mrk 421 and make a light curve'],
                              llm=broken)
    assert not review['llm_used'] and 'backend down' in review['llm_error']
    assert any(f['source'] == 'heuristic' and 'lightcurve' in f['message']
               for f in review['findings'])
    # "remove the SED" after asking for one: the SED call is now unwanted.
    review = sv.review_script(PLAIN_SCRIPT, REVIEW_YAML,
                              ['make an SED', 'please remove the SED'])
    assert any('drop sed' in f['message'] for f in review['findings'])


def test_propose_load_roi_rewrites_only_simple_top_level_calls():
    sv = _pipeline_module('script_validator')
    out = sv.propose_load_roi(PLAIN_SCRIPT)
    assert "gta.load_roi('fit_model')" in out
    assert 'fit = None' in out and '# gta.optimize()' in out
    assert "gta.sed('4FGL J1104.4+3812', make_plots=True)" in out
    assert sv.infer_run_mode(sv.static_review(out, REVIEW_YAML)) == 'incremental'
    review = sv.review_script(PLAIN_SCRIPT, REVIEW_YAML, [], allow_incremental=True)
    assert review['proposal']['kind'] == 'incremental'
    nested = PLAIN_SCRIPT.replace("fit = gta.fit()", "fit = gta.fit(\n    tol=1e-3)")
    assert sv.propose_load_roi(nested) is None


FAKE_GTA_MODULE = """
class _ROI:
    sources = []

class _SEDMixin:                 # FermiPy defines products on mixins
    def sed(self, name, **kw):
        self.fit()                # internal call: must not be recorded
        raise RuntimeError('sed exploded')


class GTAnalysis(_SEDMixin):
    def __init__(self, config):
        self.config_path = config
        self.roi = _ROI()

    def setup(self):
        return None

    def optimize(self):
        return {'loglike': 1.0}

    def fit(self):
        self.optimize()           # internal call: must not be recorded
        return {'fit_quality': 3}

    def lightcurve(self, name, **kw):
        return {'flux': [1.0, 2.0]}
"""


def test_sed_bin_contract_follows_the_edges_the_script_passed():
    """A script's own loge_bins sets the expected count, not the YAML.

    The recorder stores kwargs as truncated repr text, so the edge count has
    to come from kwarg_lens; reading the repr string would leave the YAML's
    binsperdec count in place and fail a run that did exactly what was asked.
    """
    srv = _load_server()
    # 7 edges = 6 bins, while the YAML (0.1-100 GeV at 8/decade) implies 24.
    edges = [2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
    calls = [{'method': 'fit', 'kwargs': {}, 'kwarg_lens': {}},
             {'method': 'sed',
              'kwargs': {'loge_bins': repr(edges), 'make_plots': 'True'},
              'kwarg_lens': {'loge_bins': len(edges)}}]
    assert srv._sed_bins_from_script_calls(24, calls) == 6
    # a truncated repr must not be mistaken for "no edges given"
    long_edges = [2.0 + 0.05 * i for i in range(41)]
    truncated = {'method': 'sed',
                 'kwargs': {'loge_bins': repr(long_edges)[:117] + '...'},
                 'kwarg_lens': {'loge_bins': len(long_edges)}}
    assert srv._sed_bins_from_script_calls(24, [truncated]) == 40
    # no sed call, or a sed call without edges: the YAML count stands
    assert srv._sed_bins_from_script_calls(24, []) == 24
    assert srv._sed_bins_from_script_calls(
        24, [{'method': 'sed', 'kwargs': {'make_plots': 'True'},
              'kwarg_lens': {}}]) == 24
    assert srv._sed_bins_from_script_calls(None, calls) == 6
    # a hand-built trace holding literal edges still works
    assert srv._sed_bins_from_script_calls(
        24, [{'method': 'sed', 'kwargs': {'loge_bins': edges}}]) == 6
    # the review panel shows the same count, read statically from the script
    sv = _pipeline_module('script_validator')
    script = ("gta.sed('src', make_plots=True,\n"
              "        loge_bins=[2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0])\n")
    assert sv.script_sed_bin_count(script) == 6
    assert sv.script_sed_bin_count("gta.sed('src', make_plots=True)") is None
    assert sv.script_sed_bin_count("gta.sed('src', loge_bins=edges)") is None
    assert sv.script_sed_bin_count("gta.sed(") is None


def test_script_runner_records_top_level_calls_and_output():
    sr = _pipeline_module('script_runner')
    import importlib
    with tempfile.TemporaryDirectory() as tmp:
        with open(os.path.join(tmp, 'fakegta_rec.py'), 'w') as f:
            f.write(FAKE_GTA_MODULE)
        sys.path.insert(0, tmp)
        try:
            fake = importlib.import_module('fakegta_rec')
            run_dir = os.path.join(tmp, 'run')
            work_dir = os.path.join(tmp, 'work')
            os.makedirs(run_dir)
            config_path = os.path.join(run_dir, 'final_config.yaml')
            with open(config_path, 'w') as f:
                f.write(REVIEW_YAML)
            script_path = os.path.join(run_dir, 'analysis.py')
            with open(script_path, 'w') as f:
                f.write(
                    "import os\nfrom fakegta_rec import GTAnalysis\n"
                    "gta = GTAnalysis('config.yaml')\n"
                    "assert open('config.yaml').read().startswith('selection')\n"
                    "gta.setup()\nres = gta.fit()\n"
                    "try:\n    gta.sed('src')\nexcept RuntimeError:\n    pass\n"
                    "gta.lightcurve('src', nbins=2, "
                    "loge_bins=[2.0, 2.5, 3.0])\n"
                    "print('hello from the script', os.getcwd())\n")
            cwd = os.getcwd()
            state = sr.execute_script(
                script_path, config_path, work_dir,
                config_aliases=['config.yaml', '../escape.yaml'],
                stdout_path=os.path.join(run_dir, 'script_output.txt'),
                recorder=sr.GTARecorder(fake.GTAnalysis))
            assert os.getcwd() == cwd
            assert not os.path.exists(os.path.join(tmp, 'escape.yaml'))
            assert [c['method'] for c in state['calls']] == [
                'setup', 'fit', 'sed', 'lightcurve'], state['calls']
            assert state['fit_result'] == {'fit_quality': 3}
            assert state['setup_ok'] and state['script_error'] is None
            assert state['product_errors']['sed'].startswith('RuntimeError')
            assert state['product_results'] == {'lightcurve': {'flux': [1.0, 2.0]}}
            # kwargs are truncated repr text, so sequence sizes are recorded
            # separately (the SED bin-count contract reads them).
            lc_call = state['calls'][-1]
            assert lc_call['kwargs']['loge_bins'] == '[2.0, 2.5, 3.0]'
            assert lc_call['kwarg_lens'] == {'loge_bins': 3}
            assert 'hello from the script' in state['stdout_tail']
            with open(os.path.join(run_dir, 'script_output.txt')) as f:
                assert 'hello from the script' in f.read()
            # the class is restored after the run (mixin methods included)
            assert 'wrapper' not in fake.GTAnalysis.fit.__qualname__
            assert 'sed' not in fake.GTAnalysis.__dict__

            with open(script_path, 'w') as f:
                f.write("from fakegta_rec import GTAnalysis\n"
                        "gta = GTAnalysis('config.yaml')\n"
                        "raise ValueError('bad bin choice')\n")
            state = sr.execute_script(script_path, config_path, work_dir,
                                      recorder=sr.GTARecorder(fake.GTAnalysis))
            assert state['script_error'].startswith('ValueError')
            assert 'analysis.py line 3' in state['script_traceback']
            assert not state['setup_ok']
        finally:
            sys.path.remove(tmp)
            sys.modules.pop('fakegta_rec', None)


def test_script_runner_scrubs_credentials_from_the_child_environment(monkeypatch_attr):
    sr = _pipeline_module('script_runner')
    fake_env = {'PATH': '/usr/bin', 'OPENAI_API_KEY': 'sk-x',
                'FERMI_LLM_AUTH_SECRET': 's', 'FERMI_LLM_OPENAI_KEY_FILE': 'k',
                'FERMI_LLM_GOOGLE_OAUTH_FILE': 'o', 'FERMI_DIR': '/opt/st',
                'CALDB': '/caldb'}
    with monkeypatch_attr(os, 'environ', fake_env):
        env = sr.scrubbed_environment({'PYTHONPATH': '/x'})
    assert env['PATH'] == '/usr/bin' and env['FERMI_DIR'] == '/opt/st'
    assert env['CALDB'] == '/caldb' and env['PYTHONPATH'] == '/x'
    for key in ('OPENAI_API_KEY', 'FERMI_LLM_AUTH_SECRET',
                'FERMI_LLM_OPENAI_KEY_FILE', 'FERMI_LLM_GOOGLE_OAUTH_FILE'):
        assert key not in env, key


class _FakeRequest:
    def __init__(self, body):
        self._body = body

    async def json(self):
        return self._body


def _review_session(srv, session_dir, script, model_id):
    session = srv.AnalysisSession.__new__(srv.AnalysisSession)
    session.__dict__.update(dict(
        session_id='revtest', session_dir=session_dir,
        chat_history=[{'role': 'user', 'model': model_id,
                       'content': 'Fit Mrk 421 and make a light curve'}],
        current_yaml=REVIEW_YAML, current_python=script, current_prompt='',
        validation_log=[], pipeline_status='idle', pipeline_result=None,
        pending_run=None, created_at='', last_executed_yaml='',
        last_run_test_idx=None, last_run_roi_ready=False,
        codex_thread_id=None, owner=None, owner_email=None, title='',
        script_proposal=None, script_reviews={}))
    return session


def test_run_review_proposes_fix_then_runs_the_model_script(monkeypatch_attr):
    srv = _load_server()
    rev = _pipeline_module('run_execution_validated')
    MODEL_REGISTRY = mm.MODEL_REGISTRY
    model_id = next(k for k, v in MODEL_REGISTRY.items()
                    if v['backend'] == 'gemini')
    revised = PLAIN_SCRIPT + "gta.lightcurve('4FGL J1104.4+3812')\n"
    srv_sv = _pipeline_module('script_validator')
    llm_calls = []

    def fake_complete(mid, prompt, response_schema=None):
        llm_calls.append(mid)
        assert response_schema is not None
        return _review_json(revised, [{
            'severity': 'warning', 'category': 'fidelity', 'line': None,
            'message': 'light curve missing'}])

    import asyncio
    with tempfile.TemporaryDirectory() as tmp, \
            monkeypatch_attr(rev, 'repair_config', lambda y, i, iteration=0, **kw: (y, [])), \
            monkeypatch_attr(rev, 'validate_level1', lambda y: {'fermipy_load': True}), \
            monkeypatch_attr(rev, 'validate_level2', lambda y: {'gta_init': True}), \
            monkeypatch_attr(srv, '_test_idx_for_target', lambda t: 0), \
            monkeypatch_attr(srv.model_manager, 'complete', fake_complete):
        session = _review_session(srv, tmp, PLAIN_SCRIPT, model_id)
        srv.sessions['revtest'] = session
        try:
            first = srv._prepare_run_review(session)
            assert first['status'] == 'script_changes_proposed', first
            # The proposal carries the reviewer's revision with the
            # deterministic guardrails layered on: the target was never freed,
            # and the request states no fitting procedure of its own.
            proposed = first['proposed_python']
            assert "gta.lightcurve('4FGL J1104.4+3812')" in proposed
            assert "gta.free_source('4FGL J1104.4+3812')" in proposed
            assert proposed.rstrip() == srv_sv.apply_default_fit(
                revised, REVIEW_YAML).rstrip()
            assert first['python'] == PLAIN_SCRIPT and 'approval_token' not in first
            assert llm_calls == [model_id]

            # Decline: the model's own script runs, and the LLM is not re-asked.
            asyncio.run(srv.script_review_decision('revtest', _FakeRequest(
                {'proposal_id': first['proposal_id'], 'decision': 'decline'})))
            second = srv._prepare_run_review(session)
            assert second['status'] == 'review_required', second
            assert second['python'] == PLAIN_SCRIPT and llm_calls == [model_id]
            assert any(f['message'] == 'light curve missing'
                       for f in second['findings'])
            pending = session.pending_run
            assert pending['final_python'] == PLAIN_SCRIPT
            assert pending['config_aliases'] == ['config.yaml']
            with open(pending['script_path']) as f:
                assert f.read() == PLAIN_SCRIPT

            # Accept path: the proposal replaces the script, still not re-asked.
            session.script_reviews.clear()
            third = srv._prepare_run_review(session)
            asyncio.run(srv.script_review_decision('revtest', _FakeRequest(
                {'proposal_id': third['proposal_id'], 'decision': 'accept'})))
            assert session.current_python.rstrip() == srv_sv.apply_default_fit(
                revised, REVIEW_YAML).rstrip()
            fourth = srv._prepare_run_review(session)
            assert fourth['status'] == 'review_required', fourth
            assert 'gta.lightcurve' in fourth['python']
            assert len(llm_calls) == 2
        finally:
            srv.sessions.pop('revtest', None)


def test_run_review_blocks_unsafe_script_without_llm(monkeypatch_attr):
    srv = _load_server()
    rev = _pipeline_module('run_execution_validated')
    unsafe = PLAIN_SCRIPT + "import subprocess\nsubprocess.run(['ls'])\n"
    with tempfile.TemporaryDirectory() as tmp, \
            monkeypatch_attr(rev, 'repair_config', lambda y, i, iteration=0, **kw: (y, [])), \
            monkeypatch_attr(rev, 'validate_level1', lambda y: {'fermipy_load': True}), \
            monkeypatch_attr(rev, 'validate_level2', lambda y: {'gta_init': True}), \
            monkeypatch_attr(srv, '_test_idx_for_target', lambda t: 0):
        session = _review_session(srv, tmp, unsafe, 'template')
        result = srv._prepare_run_review(session)
    assert result['status'] == 'script_blocked', result
    assert 'approval_token' not in result and session.pending_run is None
    assert 'cannot review scripts' in result['llm_note']


def test_worker_refuses_a_blocking_script_even_when_approved(monkeypatch_attr):
    srv = _load_server()
    rev = _pipeline_module('run_execution_validated')
    unsafe = PLAIN_SCRIPT + "import os\nos.system('echo pwned')\n"
    digest = srv._run_content_digest(REVIEW_YAML, unsafe)
    with tempfile.TemporaryDirectory() as session_dir, \
            monkeypatch_attr(rev, 'validate_level1', lambda y: {'fermipy_load': True}), \
            monkeypatch_attr(rev, 'validate_level2', lambda y: {'gta_init': True}):
        script_path = os.path.join(session_dir, 'analysis.py')
        with open(script_path, 'w') as f:
            f.write(unsafe)
        result = srv.run_pipeline_isolated(
            session_dir, REVIEW_YAML, unsafe, test_idx=0, run_id='unsafe',
            approved_script_path=script_path, approved_digest=digest)
    assert result['status'] == 'validation_failed', result
    assert 'os.system' in result['error'] and result['level4'] is None


# ---------------------------------------------------------------------------
# monkeypatch helper (context manager) — no pytest dependency
# ---------------------------------------------------------------------------
import contextlib


@contextlib.contextmanager
def _set_attr(obj, name, value):
    sentinel = object()
    old = getattr(obj, name, sentinel)
    setattr(obj, name, value)
    try:
        yield
    finally:
        if old is sentinel:
            delattr(obj, name)
        else:
            setattr(obj, name, old)


# Inject the helper into tests that declare a `monkeypatch_attr` parameter.
def _call(test):
    import inspect, functools
    if 'monkeypatch_attr' in inspect.signature(test).parameters:
        @functools.wraps(test)
        def wrapper():
            return test(_set_attr)
        return wrapper
    return test


UNIT_TESTS = [
    test_script_validator_accepts_a_plain_fermipy_script,
    test_script_validator_blocks_unsafe_operations,
    test_script_validator_checks_fermipy_api_and_call_order,
    test_script_validator_checks_yaml_consistency_and_offers_config_fix,
    test_review_script_offers_llm_revision_as_a_proposal,
    test_review_script_rejects_unsafe_revision_and_falls_back_to_heuristics,
    test_propose_load_roi_rewrites_only_simple_top_level_calls,
    test_sed_bin_contract_follows_the_edges_the_script_passed,
    test_script_runner_records_top_level_calls_and_output,
    test_script_runner_scrubs_credentials_from_the_child_environment,
    test_run_review_proposes_fix_then_runs_the_model_script,
    test_run_review_blocks_unsafe_script_without_llm,
    test_worker_refuses_a_blocking_script_even_when_approved,
    test_registry_has_both_luna_options,
    test_default_model_is_luna_api,
    test_models_endpoint_advertises_default,
    test_openai_entry_budget_is_generous,
    test_list_models_openai_availability_tracks_key,
    test_list_models_codex_availability_tracks_binary,
    test_openai_payload_obeys_gpt5_contract,
    test_openai_truncation_raises_helpful_error,
    test_openai_missing_key_raises,
    test_codex_fresh_session_builds_exec_and_captures_thread,
    test_codex_resume_uses_resume_subcommand_and_config_sandbox,
    test_codex_resume_failure_falls_back_to_fresh,
    test_codex_empty_output_raises,
    test_codex_hard_failure_raises,
    test_summary_empty_when_no_run,
    test_summary_success_has_fit_and_repairs,
    test_summary_failure_has_error_and_traceback,
    test_summary_survives_weird_shapes,
    test_abort_endpoint_records_aborted_result,
    test_sidebar_brand_and_abort_controls_present,
    test_vertical_layout_preserves_pipeline_progress_height,
    test_pipeline_controls_are_scoped_to_the_active_run,
    test_abort_terminates_isolated_worker_process,
    test_build_prompt_renders_run_outcome_in_edit_mode,
    test_build_prompt_no_outcome_section_when_absent,
    test_build_prompt_fresh_has_examples_no_edit_block,
    test_luna_prompt_requests_semantic_intent_for_codex,
    test_luna_structured_prompt_uses_raw_file_fields,
    test_extract_semantic_intent_rejects_bad_operations,
    test_generate_openai_extracts_structured_envelope,
    test_semantic_merge_honors_paraphrased_remove,
    test_semantic_merge_can_remove_selection_key,
    test_semantic_target_change_is_not_reverted,
    test_non_catalog_target_with_verified_coordinates_is_injected,
    test_non_catalog_target_rejects_unverified_coordinates,
    test_semantic_mixed_product_edit_still_reconciles_addition,
    test_notebook_structure_and_serialisable,
    test_notebook_filename_is_safe,
    test_notebook_empty_session_still_valid,
    test_run_exports_include_final_yaml_and_runnable_python,
    test_analysis_script_isolates_failing_products_and_prepares_psmap,
    test_six_bins_per_decade_produces_an_18_bin_contract,
    test_validator_moves_sed_edges_out_of_yaml_and_keeps_the_count,
    test_sed_bin_request_reads_a_total_and_ignores_lightcurve_bins,
    test_sed_bin_plan_uses_binsperdec_only_for_a_whole_number,
    test_review_proposes_loge_bins_only_when_binsperdec_cannot_express_it,
    test_enumbins_survives_repair_and_sets_the_bin_contract,
    test_fit_directions_are_detected_only_when_actually_given,
    test_default_fit_is_proposed_when_the_user_gives_no_directions,
    test_user_fitting_directions_are_not_overridden_by_the_default,
    test_default_fit_skips_an_incremental_run_and_absent_backgrounds,
    test_approved_worker_rejects_script_content_mismatch_without_execution,
    test_notebook_prefers_final_run_exports,
    test_auth_token_roundtrip,
    test_auth_token_rejects_tamper_and_garbage,
    test_auth_token_rejects_expiry,
    test_google_verify_requires_configuration,
    test_list_user_sessions_filters_by_owner,
]

LIVE_TESTS = [
    test_live_openai_generation,
    test_live_codex_fresh_then_resume,
    test_live_http_continuity_after_failed_run,
    test_live_http_notebook_export,
    test_live_http_auth_and_ownership,
]


def main():
    print("=" * 70)
    print("UNIT TESTS (mocked I/O — free, deterministic)")
    print("=" * 70)
    for t in UNIT_TESTS:
        run(_call(t))

    if LIVE:
        print("\n" + "=" * 70)
        print("LIVE SMOKE TESTS (real OpenAI + Codex + HTTP)")
        print("=" * 70)
        for t in LIVE_TESTS:
            run(_call(t))
    else:
        print("\n(skipping live tests — pass --live to run them)")

    passed = sum(1 for _, ok, _ in _RESULTS if ok)
    total = len(_RESULTS)
    print("\n" + "=" * 70)
    print(f"RESULT: {passed}/{total} passed")
    if passed != total:
        print("FAILURES:")
        for name, ok, msg in _RESULTS:
            if not ok:
                print(f"  - {name}: {msg}")
    print("=" * 70)
    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()

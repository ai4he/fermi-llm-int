"""Execute the reviewed LLM-generated analysis.py and observe what it did.

The script is run as written. Instead of requiring it to follow a result
contract, ``GTARecorder`` wraps the public GTAnalysis methods for the duration
of the run and records every call the script itself makes (FermiPy's own
internal calls, e.g. the setup/fit of the clones inside ``lightcurve()``, are
not recorded). The pipeline reads the fit result, product outputs and
per-call errors from that trace.

Isolation (defence in depth on top of ``script_validator``):

* ``run_level4_subprocess`` runs Level 4 in a *fresh* Python interpreter, so
  the script never shares memory with the web server (API keys, sessions).
* The child's environment is scrubbed of credentials, its working directory is
  the session's fermipy work dir, and CPU/memory/file-size/core limits are
  applied before the script is imported (``FERMI_LLM_SCRIPT_MAX_MEMORY_GB``,
  ``FERMI_LLM_SCRIPT_MAX_FILE_GB``; 0 disables a limit).
* The child stays in the pipeline worker's process group, so /abort_pipeline
  terminates it together with the worker.
"""

import contextlib
import functools
import inspect
import io
import json
import os
import re
import runpy
import subprocess
import sys
import threading
import time
import traceback


# The methods the pipeline reads results from (all public methods are
# recorded; this list documents the ones that matter and is the fallback
# when the class cannot be introspected).
RECORDED_METHODS = (
    'setup', 'load_roi', 'optimize', 'fit', 'sed', 'lightcurve', 'tsmap',
    'tscube', 'residmap', 'psmap', 'localize', 'extension', 'find_sources',
    'curvature', 'free_source', 'free_sources', 'free_norm', 'free_index',
    'free_shape', 'free_parameter', 'set_parameter', 'set_norm',
    'set_source_spectrum', 'set_source_morphology', 'add_source',
    'delete_source', 'delete_sources', 'zero_source', 'write_roi',
    'write_model_map', 'write_config', 'write_xml', 'profile_norm', 'profile',
    'set_energy_range', 'simulate_roi',
)
PRODUCT_METHODS = ('sed', 'lightcurve', 'tsmap', 'tscube', 'residmap',
                   'psmap', 'localize', 'extension', 'find_sources',
                   'curvature')

SECRET_ENV_RE = re.compile(
    r'(KEY|SECRET|TOKEN|PASSWORD|PASSWD|CREDENTIAL|AUTH|COOKIE|SESSION)',
    re.I)


def _short_repr(value, limit=120):
    try:
        text = repr(value)
    except Exception:
        text = f'<{type(value).__name__}>'
    return text if len(text) <= limit else text[:limit - 3] + '...'


def _kwarg_lengths(kwargs):
    """Element counts of the sequence keyword arguments of a call.

    ``kwargs`` is recorded as ``_short_repr`` text, which is truncated and, for
    a numpy array or other computed sequence, not parseable back into values.
    Callers that only need the size of an argument (e.g. the ``loge_bins``
    edges handed to ``gta.sed()``) read it from here instead.
    """
    lengths = {}
    for key, value in kwargs.items():
        if isinstance(value, (str, bytes, dict)):
            continue
        try:
            lengths[key] = len(value)
        except Exception:
            continue
    return lengths


class GTARecorder:
    """Record the GTAnalysis calls made directly by the user script."""

    def __init__(self, cls=None):
        if cls is None:
            from fermipy.gtanalysis import GTAnalysis as cls
        self.cls = cls
        self.calls = []            # ordered call trace (JSON-safe)
        self.returns = {}          # method -> return value of last success
        self.instances = []        # GTAnalysis objects built by the script
        self._originals = {}
        self._local = threading.local()

    def _depth(self):
        return getattr(self._local, 'depth', 0)

    def _wrap(self, name, original):
        recorder = self

        @functools.wraps(original)
        def wrapper(obj, *args, **kwargs):
            if recorder._depth() > 0:
                return original(obj, *args, **kwargs)
            recorder._local.depth = 1
            entry = {
                'method': name,
                'args': [_short_repr(a) for a in args],
                'kwargs': {k: _short_repr(v) for k, v in kwargs.items()},
                'kwarg_lens': _kwarg_lengths(kwargs),
                'ok': False, 'error': None, 'seconds': None,
            }
            recorder.calls.append(entry)
            start = time.time()
            try:
                value = original(obj, *args, **kwargs)
            except BaseException as exc:
                entry['error'] = f'{type(exc).__name__}: {str(exc)[:500]}'
                raise
            finally:
                entry['seconds'] = round(time.time() - start, 2)
                recorder._local.depth = 0
            entry['ok'] = True
            recorder.returns[name] = value
            return value
        return wrapper

    def __enter__(self):
        recorder = self
        original_init = self.cls.__init__
        self._originals['__init__'] = original_init

        @functools.wraps(original_init)
        def init_wrapper(obj, *args, **kwargs):
            top_level = recorder._depth() == 0
            recorder._local.depth = recorder._depth() + 1
            try:
                original_init(obj, *args, **kwargs)
            finally:
                recorder._local.depth -= 1
            if top_level:
                recorder.instances.append(obj)
        self.cls.__init__ = init_wrapper

        # FermiPy defines most products (sed, lightcurve, tsmap, ...) on mixin
        # base classes, so look each method up along the MRO; the wrapper is
        # installed on GTAnalysis itself and removed again on exit.
        # Every public method is wrapped (not just RECORDED_METHODS): an
        # unwrapped method's internal calls would otherwise be recorded as
        # if the script had made them.
        self._own = set()
        names = {n for n in dir(self.cls) if not n.startswith('_')}
        for name in sorted(names | set(RECORDED_METHODS)):
            original = next((klass.__dict__[name] for klass in self.cls.__mro__
                             if name in klass.__dict__), None)
            if not inspect.isfunction(original):
                continue
            if name in self.cls.__dict__:
                self._own.add(name)
            self._originals[name] = original
            setattr(self.cls, name, self._wrap(name, original))
        return self

    def __exit__(self, *exc):
        for name, original in self._originals.items():
            if name == '__init__' or name in self._own:
                setattr(self.cls, name, original)
            else:
                delattr(self.cls, name)
        self._originals.clear()
        return False

    # -- summaries ---------------------------------------------------------
    def called(self, method, ok=None):
        return any(c['method'] == method and (ok is None or c['ok'] == ok)
                   for c in self.calls)

    def analysis_gta(self):
        """The GTAnalysis the results should be read from."""
        for gta in reversed(self.instances):
            if getattr(gta, '_roi', None) is not None or getattr(
                    gta, 'roi', None) is not None:
                return gta
        return self.instances[-1] if self.instances else None

    def state(self):
        product_results, product_errors = {}, {}
        for call in self.calls:
            method = call['method']
            if method not in PRODUCT_METHODS:
                continue
            if call['ok']:
                product_errors.pop(method, None)
            elif call['error']:
                product_errors[method] = call['error']
        for method in PRODUCT_METHODS:
            if method in self.returns:
                product_results[method] = self.returns[method]
        return {
            'gta': self.analysis_gta(),
            'fit_result': self.returns.get('fit'),
            'fit_called': self.called('fit'),
            'optimize_ok': self.called('optimize', ok=True),
            'load_roi_ok': self.called('load_roi', ok=True),
            'setup_ok': (self.called('setup', ok=True)
                         or self.called('load_roi', ok=True)),
            'product_results': product_results,
            'product_errors': product_errors,
            'calls': list(self.calls),
        }


class _Tee(io.TextIOBase):
    def __init__(self, *streams):
        self.streams = streams

    def write(self, text):
        for stream in self.streams:
            try:
                stream.write(text)
            except Exception:
                pass
        return len(text)

    def flush(self):
        for stream in self.streams:
            try:
                stream.flush()
            except Exception:
                pass


def _script_traceback(exc, script_path):
    """Traceback lines that point into analysis.py (plus the final error)."""
    frames = traceback.extract_tb(exc.__traceback__)
    mine = [f for f in frames if os.path.abspath(f.filename)
            == os.path.abspath(script_path)]
    lines = [f'  analysis.py line {f.lineno}: {(f.line or "").strip()}'
             for f in mine]
    lines.append(f'{type(exc).__name__}: {str(exc)[:800]}')
    return '\n'.join(lines)


def execute_script(script_path, config_path, work_dir, config_aliases=(),
                   stdout_path=None, recorder=None):
    """Run ``script_path`` as ``__main__`` inside ``work_dir``.

    ``config_aliases`` are the relative file names the script passes to
    GTAnalysis(); the reviewed YAML is copied to each of them in ``work_dir``
    so the script loads it without being edited. Returns the recorder state
    plus ``script_error``/``script_traceback`` (None on success) and the
    script's captured stdout.
    """
    recorder = recorder or GTARecorder()
    os.makedirs(work_dir, exist_ok=True)
    with open(config_path) as stream:
        reviewed_yaml = stream.read()
    for alias in config_aliases or ():
        target = os.path.normpath(os.path.join(work_dir, alias))
        if not target.startswith(os.path.normpath(work_dir) + os.sep):
            continue
        os.makedirs(os.path.dirname(target), exist_ok=True)
        with open(target, 'w') as stream:
            stream.write(reviewed_yaml)

    saved_env = {k: os.environ.get(k) for k in (
        'FERMI_LLM_CONFIG_PATH', 'FERMI_LLM_WORK_DIR', 'MPLBACKEND')}
    os.environ['FERMI_LLM_CONFIG_PATH'] = config_path
    os.environ['FERMI_LLM_WORK_DIR'] = work_dir
    os.environ['MPLBACKEND'] = 'Agg'
    previous_cwd = os.getcwd()
    captured = io.StringIO()
    log_stream = open(stdout_path, 'w') if stdout_path else None
    streams = [sys.stdout, captured] + ([log_stream] if log_stream else [])
    script_error = script_tb = None
    try:
        os.chdir(work_dir)
        with recorder, contextlib.redirect_stdout(_Tee(*streams)):
            try:
                runpy.run_path(script_path, run_name='__main__')
            except SystemExit as exc:
                if exc.code not in (None, 0):
                    script_error = f'SystemExit: exit code {exc.code}'
                    script_tb = _script_traceback(exc, script_path)
            except Exception as exc:
                if type(exc).__name__ == 'FitTimeoutError':
                    raise
                script_error = f'{type(exc).__name__}: {str(exc)[:500]}'
                script_tb = _script_traceback(exc, script_path)
    finally:
        os.chdir(previous_cwd)
        for key, value in saved_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        if log_stream:
            log_stream.close()
    state = recorder.state()
    state['script_error'] = script_error
    state['script_traceback'] = script_tb
    output = captured.getvalue()
    state['stdout_tail'] = output[-8000:]
    return state


# ---------------------------------------------------------------------------
# Fresh-interpreter execution
# ---------------------------------------------------------------------------

def scrubbed_environment(extra=None):
    """The parent environment minus anything that looks like a credential."""
    env = {k: v for k, v in os.environ.items() if not SECRET_ENV_RE.search(k)}
    for key in list(env):
        if key.endswith('_KEY_FILE') or key.endswith('_OAUTH_FILE') or (
                key.startswith(('OPENAI', 'GEMINI', 'GOOGLE_', 'ANTHROPIC'))):
            env.pop(key, None)
    env['MPLBACKEND'] = 'Agg'
    env['PYTHONUNBUFFERED'] = '1'
    env.update(extra or {})
    return env


def apply_resource_limits(cpu_seconds=None):
    try:
        import resource
    except ImportError:
        return

    def _set(which, value):
        try:
            _soft, hard = resource.getrlimit(which)
            if hard != resource.RLIM_INFINITY:
                value = min(value, hard)
            resource.setrlimit(which, (value, hard))
        except (ValueError, OSError):
            pass

    _set(resource.RLIMIT_CORE, 0)
    mem_gb = float(os.environ.get('FERMI_LLM_SCRIPT_MAX_MEMORY_GB', 64))
    if mem_gb > 0:
        _set(resource.RLIMIT_AS, int(mem_gb * 1024 ** 3))
    file_gb = float(os.environ.get('FERMI_LLM_SCRIPT_MAX_FILE_GB', 20))
    if file_gb > 0:
        _set(resource.RLIMIT_FSIZE, int(file_gb * 1024 ** 3))
    if cpu_seconds:
        _set(resource.RLIMIT_CPU, int(cpu_seconds))


def run_level4_subprocess(job, job_dir, timeout, python=None):
    """Run ``validate_level4(**job)`` in a fresh interpreter; return its dict.

    ``timeout`` is the wall-clock limit for the child (the Level 4 fit
    timeout plus a margin); on expiry the child is killed and an error
    result is returned.
    """
    os.makedirs(job_dir, exist_ok=True)
    job_path = os.path.join(job_dir, 'level4_job.json')
    result_path = os.path.join(job_dir, 'level4_result.json')
    log_path = os.path.join(job_dir, 'level4_subprocess.log')
    if os.path.exists(result_path):
        os.remove(result_path)
    with open(job_path, 'w') as stream:
        json.dump(job, stream, indent=2, default=str)

    # The child runs this module as ``python -m fermi_llm.fermipy.script_runner``
    # so its package-relative imports resolve; ``src`` therefore has to be on
    # the child's PYTHONPATH (the parent may be running from an installed
    # package or from a checkout).
    package_root = os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))))
    pythonpath = os.pathsep.join(
        p for p in (package_root, os.environ.get('PYTHONPATH', '')) if p)
    env = scrubbed_environment({'PYTHONPATH': pythonpath})
    cmd = [python or sys.executable, '-m', 'fermi_llm.fermipy.script_runner',
           job_path, result_path]
    cwd = job.get('work_dir') or job_dir
    os.makedirs(cwd, exist_ok=True)
    started = time.time()
    with open(log_path, 'w') as log:
        try:
            proc = subprocess.Popen(
                cmd, cwd=cwd, env=env, stdin=subprocess.DEVNULL,
                stdout=log, stderr=subprocess.STDOUT)
        except OSError as exc:
            return {'level': 4, 'error': (
                f'could not start the analysis subprocess: '
                f'{type(exc).__name__}: {exc}'), 'elapsed_seconds': 0}
        try:
            returncode = proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
            return {'level': 4, 'error': (
                f'analysis.py exceeded the {int(timeout)} s time limit and '
                f'was stopped'), 'script_error': 'timeout',
                'elapsed_seconds': time.time() - started}
    try:
        with open(result_path) as stream:
            return json.load(stream)
    except (OSError, ValueError):
        tail = ''
        try:
            with open(log_path, errors='replace') as stream:
                tail = stream.read()[-2000:]
        except OSError:
            pass
        return {'level': 4, 'error': (
            f'the analysis subprocess exited with code {returncode} without '
            f'a result'), 'traceback': tail,
            'elapsed_seconds': time.time() - started}


def _json_safe(value):
    try:
        json.dumps(value)
        return value
    except (TypeError, ValueError):
        pass
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    try:
        import numpy as np
        if isinstance(value, np.ndarray):
            return _json_safe(value.tolist())
        if isinstance(value, np.generic):
            return value.item()
    except ImportError:
        pass
    return str(value)


def main(argv):
    job_path, result_path = argv[1], argv[2]
    with open(job_path) as stream:
        job = json.load(stream)
    apply_resource_limits(cpu_seconds=job.pop('cpu_seconds', None))
    from .run_level4_validation import validate_level4
    try:
        result = validate_level4(**job)
    except BaseException as exc:
        result = {'level': 4, 'error': f'{type(exc).__name__}: {exc}',
                  'traceback': traceback.format_exc()[-2000:]}
    with open(result_path, 'w') as stream:
        json.dump(_json_safe(result), stream, indent=2, default=str)
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))

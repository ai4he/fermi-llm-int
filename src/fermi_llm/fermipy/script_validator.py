"""Review of the LLM-generated analysis.py before it is executed.

The pipeline executes the model's own Python script (not a canonical
replacement), so this module decides whether that script may run and what
the user should be asked to change first. Two layers:

* ``static_review`` -- deterministic AST checks, no LLM:
    - safety: import allowlist, no shell/network/dynamic code, no destructive
      file operations, no writes outside the working directory;
    - FermiPy API: methods exist on GTAnalysis, keyword options are accepted
      (read from the installed FermiPy's ConfigSchema definitions), call order
      (setup/load_roi first, products after the fit);
    - consistency with the reviewed YAML: config file reference, source
      names, YAML sections the script reads.
* ``review_script`` -- runs the static checks, optionally asks an LLM to judge
  fidelity to the user's requests (and to propose a minimal corrected script),
  and returns the findings plus at most one proposed script. Nothing is ever
  applied here: the server shows the proposal as a diff and the user decides.

The allowlist is defence in depth, not a sandbox: the runner additionally
executes the script in a fresh interpreter with a scrubbed environment and
resource limits (see ``script_runner.py``).
"""

import ast
import difflib
import functools
import inspect
import json
import os
import re
import textwrap

import yaml

from .fermipy_schema import sed_bin_plan, sed_bin_request


SEVERITIES = ('error', 'warning', 'info')
CATEGORIES = ('safety', 'api', 'order', 'consistency', 'fidelity', 'syntax')

# ---------------------------------------------------------------------------
# Safety policy
# ---------------------------------------------------------------------------

ALLOWED_IMPORT_ROOTS = frozenset({
    '__future__', 'argparse', 'astropy', 'collections', 'contextlib', 'copy',
    'csv', 'dataclasses', 'datetime', 'decimal', 'enum', 'fermipy', 'fnmatch',
    'fractions', 'functools', 'gammapy', 'glob', 'gt_apps', 'healpy',
    'iminuit', 'io', 'itertools', 'json', 'logging', 'math', 'matplotlib',
    'numbers', 'numpy', 'operator', 'os', 'pandas', 'pathlib', 'pprint',
    'pyLikelihood', 'BinnedAnalysis', 'UnbinnedAnalysis', 'SummedLikelihood',
    'random', 're', 'regions', 'scipy', 'shutil', 'statistics', 'string',
    'sys', 'tabulate', 'textwrap', 'time', 'typing', 'uncertainties',
    'warnings', 'yaml', 'cmath',
})

FORBIDDEN_IMPORT_REASONS = {
    'subprocess': 'runs arbitrary shell commands',
    'socket': 'opens network connections',
    'ssl': 'opens network connections',
    'urllib': 'accesses the network',
    'urllib3': 'accesses the network',
    'http': 'accesses the network',
    'requests': 'accesses the network',
    'httpx': 'accesses the network',
    'aiohttp': 'accesses the network',
    'ftplib': 'accesses the network',
    'smtplib': 'accesses the network',
    'telnetlib': 'accesses the network',
    'xmlrpc': 'accesses the network',
    'webbrowser': 'accesses the network',
    'paramiko': 'accesses the network',
    'astroquery': 'queries remote services over the network',
    'pyvo': 'queries remote services over the network',
    'ctypes': 'calls native code directly',
    'cffi': 'calls native code directly',
    'importlib': 'imports modules dynamically',
    'pickle': 'can execute arbitrary code when loading',
    'marshal': 'can execute arbitrary code when loading',
    'shelve': 'can execute arbitrary code when loading',
    'multiprocessing': 'spawns processes outside the run supervisor',
    'signal': 'can disable the run timeout',
    'resource': 'can change the run resource limits',
    'builtins': 'exposes the interpreter internals',
    'gc': 'exposes the interpreter internals',
    'inspect': 'exposes the interpreter internals',
    'code': 'executes arbitrary code',
    'codeop': 'executes arbitrary code',
    'pty': 'spawns a shell',
    'os.system': 'runs arbitrary shell commands',
}

FORBIDDEN_BUILTINS = {
    'eval': 'evaluates arbitrary code',
    'exec': 'executes arbitrary code',
    'compile': 'compiles arbitrary code',
    '__import__': 'imports modules dynamically',
    'breakpoint': 'would hang the run waiting for a debugger',
    'input': 'would hang the run waiting for keyboard input',
    'globals': 'exposes the interpreter internals',
    'locals': 'exposes the interpreter internals',
    'vars': 'exposes the interpreter internals',
    'memoryview': 'exposes raw memory',
}

# Fully-qualified calls that are never allowed.
FORBIDDEN_CALLS = {
    'os.system': 'runs a shell command',
    'os.popen': 'runs a shell command',
    'os.fork': 'forks the process',
    'os.forkpty': 'forks the process',
    'os.kill': 'signals other processes',
    'os.killpg': 'signals other processes',
    'os.remove': 'deletes files',
    'os.unlink': 'deletes files',
    'os.rmdir': 'deletes directories',
    'os.removedirs': 'deletes directories',
    'os.rename': 'moves files',
    'os.renames': 'moves files',
    'os.replace': 'moves files',
    'os.chmod': 'changes file permissions',
    'os.chown': 'changes file ownership',
    'os.lchown': 'changes file ownership',
    'os.symlink': 'creates symbolic links',
    'os.link': 'creates hard links',
    'os.truncate': 'truncates files',
    'os.setuid': 'changes the process identity',
    'os.setgid': 'changes the process identity',
    'os.chroot': 'changes the filesystem root',
    'os.putenv': 'changes the process environment',
    'os.unsetenv': 'changes the process environment',
    'os.posix_spawn': 'spawns processes',
    'os.posix_spawnp': 'spawns processes',
    'shutil.rmtree': 'deletes directory trees',
    'shutil.move': 'moves files',
    'shutil.chown': 'changes file ownership',
    'shutil.unpack_archive': 'extracts archives to arbitrary locations',
    'shutil.make_archive': 'writes archives to arbitrary locations',
    'astropy.utils.data.download_file': 'downloads data from the network',
    'astropy.coordinates.SkyCoord.from_name': 'queries a name resolver over the network',
    'sys.settrace': 'exposes the interpreter internals',
    'sys.setprofile': 'exposes the interpreter internals',
}
FORBIDDEN_CALL_PREFIXES = {
    'os.exec': 'replaces the process with another program',
    'os.spawn': 'spawns processes',
}
# Destructive methods, whatever object they are called on (pathlib etc.).
FORBIDDEN_METHODS = {
    'unlink': 'deletes files',
    'rmdir': 'deletes directories',
    'rmtree': 'deletes directory trees',
    'symlink_to': 'creates symbolic links',
    'hardlink_to': 'creates hard links',
    'chmod': 'changes file permissions',
    'lchmod': 'changes file permissions',
    'from_name': 'queries a name resolver over the network',
}
ALLOWED_DUNDERS = frozenset({'__name__', '__file__', '__doc__', '__version__',
                             '__init__', '__main__'})

SENSITIVE_PATH_RE = re.compile(
    r'(/etc/(passwd|shadow|sudoers)|(^|/)\.ssh(/|$)|(^|/)\.aws(/|$)|'
    r'\.netrc|/proc/|(^|/)configs/|gemini_keys|_key\.txt|auth_secret|'
    r'client_secret|google_oauth|openai_key|clemson_vllm_key)', re.I)

# Calls that write to a path: dotted name or bare method name -> (arg index,
# keyword name). Paths given as literals must stay inside the working dir.
PATH_WRITE_CALLS = {
    'os.makedirs': (0, 'name'),
    'os.mkdir': (0, 'path'),
    'os.chdir': (0, 'path'),
    'numpy.save': (0, 'file'),
    'numpy.savez': (0, 'file'),
    'numpy.savez_compressed': (0, 'file'),
    'numpy.savetxt': (0, 'fname'),
    'shutil.copy': (1, 'dst'),
    'shutil.copy2': (1, 'dst'),
    'shutil.copyfile': (1, 'dst'),
    'shutil.copytree': (1, 'dst'),
}
PATH_WRITE_METHODS = {
    'savefig': (0, 'fname'),
    'writeto': (0, 'fileobj'),
    'to_csv': (0, 'path_or_buf'),
    'write_roi': (0, 'outfile'),
    'write_config': (0, 'outfile'),
    'write_xml': (0, 'xmlfile'),
}

# ---------------------------------------------------------------------------
# FermiPy API knowledge
# ---------------------------------------------------------------------------

SETUP_METHODS = ('setup', 'load_roi')
FIT_METHODS = ('fit', 'optimize')
PRODUCT_METHODS = ('sed', 'lightcurve', 'tsmap', 'tscube', 'residmap', 'psmap',
                   'localize', 'extension', 'find_sources', 'curvature')
# Methods whose first argument is an ROI source name.
SOURCE_ARG_METHODS = frozenset({
    'free_source', 'sed', 'lightcurve', 'localize', 'extension', 'curvature',
    'get_src_model', 'free_norm', 'free_index', 'free_shape', 'set_norm',
    'set_parameter', 'set_source_spectrum', 'set_source_morphology',
    'delete_source', 'zero_source', 'unzero_source', 'profile_norm',
    'profile', 'free_parameter', 'set_parameter_bounds',
    'set_parameter_scale', 'set_parameter_error', 'lock_parameter',
    'set_norm_scale', 'set_norm_bounds', 'set_edisp_flag',
})
# Methods that are harmless before setup(): they only read the config.
PRE_SETUP_OK = frozenset({'print_roi', 'print_params', 'print_model',
                          'config', 'outdir', 'workdir'})
BACKGROUND_NAMES = frozenset({'galdiff', 'isodiff'})

# Offline fallback when FermiPy cannot be imported (e.g. unit tests on a
# machine without ScienceTools): method names only, no keyword checks.
_FALLBACK_GTA_METHODS = frozenset({
    'setup', 'load_roi', 'optimize', 'fit', 'sed', 'lightcurve', 'tsmap',
    'tscube', 'residmap', 'psmap', 'localize', 'extension', 'find_sources',
    'curvature', 'free_source', 'free_sources', 'free_norm', 'free_index',
    'free_shape', 'free_parameter', 'set_parameter', 'set_norm',
    'set_source_spectrum', 'set_source_morphology', 'delete_source',
    'delete_sources', 'add_source', 'zero_source', 'unzero_source',
    'get_src_model', 'get_sources', 'get_source_name', 'write_roi',
    'write_model_map', 'write_weight_map', 'write_config', 'write_xml',
    'print_roi', 'print_params', 'print_model', 'profile', 'profile_norm',
    'like', 'model_counts_map', 'model_counts_spectrum', 'counts_map',
    'set_energy_range', 'set_free_param_vector', 'get_free_param_vector',
    'reload_source', 'reload_sources', 'simulate_roi', 'simulate_source',
    'stage_input', 'stage_output', 'lock_parameter', 'set_edisp_flag',
    'generate_model', 'roi', 'config', 'outdir', 'workdir', 'components',
    'tmin', 'tmax', 'energies', 'npix', 'create', 'bowtie', 'constrain_norms',
    'remove_prior', 'set_parameter_bounds', 'set_parameter_scale',
    'set_parameter_error', 'set_norm_scale', 'set_norm_bounds',
    'get_params', 'get_config', 'get_norm', 'update_source', 'residmap',
})


@functools.lru_cache(maxsize=1)
def _gtanalysis_class():
    try:
        from fermipy.gtanalysis import GTAnalysis
        return GTAnalysis
    except Exception:
        return None


@functools.lru_cache(maxsize=1)
def gta_method_names():
    cls = _gtanalysis_class()
    if cls is None:
        return _FALLBACK_GTA_METHODS
    return frozenset(name for name in dir(cls) if not name.startswith('_'))


@functools.lru_cache(maxsize=None)
def gta_method_spec(name):
    """Describe what a GTAnalysis method accepts.

    Returns ``{'positional': [...], 'keywords': set|None, 'max_positional':
    int|None}``. ``keywords`` is None when the accepted options cannot be
    determined (then no keyword check is made). FermiPy's product methods
    validate ``**kwargs`` strictly through ``ConfigSchema.create_config`` --
    an unknown option raises KeyError at run time -- so their accepted keys
    are read from the method source: the schema's config section plus any
    ``add_option``/constructor extras.
    """
    cls = _gtanalysis_class()
    fn = getattr(cls, name, None) if cls is not None else None
    if fn is None or not callable(fn):
        return None
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return None
    params = [p for p in sig.parameters.values() if p.name != 'self']
    positional = [p.name for p in params if p.kind in (
        p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]
    has_varargs = any(p.kind == p.VAR_POSITIONAL for p in params)
    has_varkw = any(p.kind == p.VAR_KEYWORD for p in params)
    named = {p.name for p in params if p.kind in (
        p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)}
    spec = {
        'positional': positional,
        'max_positional': None if has_varargs else len(positional),
        'keywords': None if has_varkw else named,
    }
    if has_varkw:
        schema_keys = _config_schema_keys(fn)
        if schema_keys is not None:
            spec['keywords'] = named | schema_keys
    return spec


def _config_schema_keys(fn):
    try:
        tree = ast.parse(textwrap.dedent(inspect.getsource(fn)))
        from fermipy import defaults as fermipy_defaults
    except Exception:
        return None
    sections, extras, strict = [], set(), False
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fname = _call_name(node.func)
        if fname == 'ConfigSchema':
            for arg in node.args:
                if (isinstance(arg, ast.Subscript)
                        and isinstance(arg.slice, ast.Constant)):
                    sections.append(arg.slice.value)
            extras.update(kw.arg for kw in node.keywords if kw.arg)
        elif fname == 'add_option' and node.args and isinstance(
                node.args[0], ast.Constant):
            extras.add(node.args[0].value)
        elif fname == 'create_config':
            strict = True
    if not strict or not sections:
        return None
    keys = set(extras)
    for section in sections:
        table = getattr(fermipy_defaults, section, None)
        if not isinstance(table, dict):
            return None
        keys.update(table.keys())
    return keys


def _call_name(func):
    if isinstance(func, ast.Attribute):
        return func.attr
    if isinstance(func, ast.Name):
        return func.id
    return None


# ---------------------------------------------------------------------------
# Static review
# ---------------------------------------------------------------------------

def _finding(severity, category, message, line=None, source='static',
             fix=None):
    item = {'severity': severity, 'category': category, 'message': message,
            'line': line, 'source': source}
    if fix:
        item['fix'] = fix
    return item


def _load_yaml(yaml_text):
    try:
        data = yaml.safe_load(yaml_text or '') or {}
    except yaml.YAMLError:
        return {}
    return data if isinstance(data, dict) else {}


def _norm_source(name):
    return re.sub(r'[\s_]+', '', str(name or '')).lower().replace(
        '4fgl', '').replace('mkn', 'mrk')


def _is_escaping_path(path):
    """True for literal paths that leave the run's working directory."""
    if not isinstance(path, str) or not path:
        return False
    if path.startswith('~') or os.path.isabs(path):
        return True
    return '..' in re.split(r'[\\/]+', path)


class _Analyzer(ast.NodeVisitor):
    def __init__(self, tree, yaml_cfg):
        self.tree = tree
        self.yaml_cfg = yaml_cfg
        self.findings = []
        self.aliases = {}          # local name -> dotted module/object path
        self.gta_vars = set()
        self.yaml_vars = set()     # names bound to yaml.safe_load(...) output
        self.str_consts = {}       # module-level NAME = 'literal'
        self.gta_ctor_calls = []   # ast.Call nodes constructing GTAnalysis
        self.parents = {}
        for parent in ast.walk(tree):
            for child in ast.iter_child_nodes(parent):
                self.parents[child] = parent

    # -- helpers -----------------------------------------------------------
    def add(self, severity, category, message, node=None, fix=None):
        self.findings.append(_finding(
            severity, category, message,
            line=getattr(node, 'lineno', None), fix=fix))

    def dotted(self, node):
        """Resolve ``a.b.c`` through import aliases to a dotted path."""
        parts = []
        while isinstance(node, ast.Attribute):
            parts.append(node.attr)
            node = node.value
        if isinstance(node, ast.Call):
            # Path('x').unlink() / SkyCoord.from_name(...)
            inner = self.dotted(node.func)
            if inner:
                parts.append(inner)
                return '.'.join(reversed(parts))
            return None
        if not isinstance(node, ast.Name):
            return None
        parts.append(self.aliases.get(node.id, node.id))
        return '.'.join(reversed(parts))

    def literal(self, node):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value
        if isinstance(node, ast.Name) and node.id in self.str_consts:
            return self.str_consts[node.id]
        return None

    def call_arg(self, call, index, keyword):
        for kw in call.keywords:
            if kw.arg == keyword:
                return kw.value
        if index is not None and len(call.args) > index:
            return call.args[index]
        return None

    def enclosing_function(self, node):
        cur = self.parents.get(node)
        while cur is not None:
            if isinstance(cur, (ast.FunctionDef, ast.AsyncFunctionDef,
                                ast.Lambda)):
                return cur
            cur = self.parents.get(cur)
        return None

    # -- pass 1: bindings --------------------------------------------------
    def collect_bindings(self):
        for node in ast.walk(self.tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    local = alias.asname or alias.name.split('.')[0]
                    self.aliases[local] = (alias.name if alias.asname
                                           else alias.name.split('.')[0])
            elif isinstance(node, ast.ImportFrom) and node.module:
                for alias in node.names:
                    self.aliases[alias.asname or alias.name] = (
                        f'{node.module}.{alias.name}')
        for node in self.tree.body:
            if (isinstance(node, ast.Assign) and len(node.targets) == 1
                    and isinstance(node.targets[0], ast.Name)
                    and isinstance(node.value, ast.Constant)
                    and isinstance(node.value.value, str)):
                self.str_consts[node.targets[0].id] = node.value.value
        for node in ast.walk(self.tree):
            if not isinstance(node, (ast.Assign, ast.AnnAssign, ast.With)):
                continue
            if isinstance(node, ast.With):
                pairs = [(item.optional_vars, item.context_expr)
                         for item in node.items if item.optional_vars]
            else:
                targets = (node.targets if isinstance(node, ast.Assign)
                           else [node.target])
                pairs = [(t, node.value) for t in targets]
            for target, value in pairs:
                if not isinstance(target, ast.Name) or not isinstance(
                        value, ast.Call):
                    continue
                name = self.dotted(value.func) or ''
                if name.endswith('GTAnalysis') or name.endswith(
                        'GTAnalysis.create'):
                    self.gta_vars.add(target.id)
                    self.gta_ctor_calls.append(value)
                elif name in ('yaml.safe_load', 'yaml.load',
                              'yaml.full_load'):
                    self.yaml_vars.add(target.id)
        for node in ast.walk(self.tree):
            if isinstance(node, ast.Call):
                name = self.dotted(node.func) or ''
                if (name.endswith('GTAnalysis') or name.endswith(
                        'GTAnalysis.create')) and node not in self.gta_ctor_calls:
                    self.gta_ctor_calls.append(node)

    # -- pass 2: safety ----------------------------------------------------
    def check_safety(self):
        for node in ast.walk(self.tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                self._check_import(node)
            elif isinstance(node, ast.Attribute):
                if (node.attr.startswith('__') and node.attr.endswith('__')
                        and node.attr not in ALLOWED_DUNDERS):
                    self.add('error', 'safety',
                             f'Accesses the interpreter internal '
                             f'`{node.attr}`, which is not allowed in an '
                             f'analysis script.', node)
            elif isinstance(node, ast.Name):
                if (node.id.startswith('__') and node.id.endswith('__')
                        and node.id not in ALLOWED_DUNDERS):
                    self.add('error', 'safety',
                             f'Uses the interpreter internal `{node.id}`.',
                             node)
            elif isinstance(node, ast.Constant) and isinstance(
                    node.value, str):
                if SENSITIVE_PATH_RE.search(node.value):
                    self.add('error', 'safety',
                             f'References a sensitive path '
                             f'({node.value[:80]!r}).', node)
            elif isinstance(node, ast.Call):
                self._check_call_safety(node)

    def _check_import(self, node):
        names = ([a.name for a in node.names] if isinstance(node, ast.Import)
                 else [node.module or ''])
        if isinstance(node, ast.ImportFrom) and node.level:
            self.add('error', 'safety',
                     'Relative imports are not allowed in the analysis script.',
                     node)
            return
        for name in names:
            root = name.split('.')[0]
            full_reason = FORBIDDEN_IMPORT_REASONS.get(name)
            reason = full_reason or FORBIDDEN_IMPORT_REASONS.get(root)
            if reason:
                self.add('error', 'safety',
                         f'Imports `{name}`, which {reason}.', node)
            elif root not in ALLOWED_IMPORT_ROOTS:
                self.add('error', 'safety',
                         f'Imports `{name}`, which is not on the analysis '
                         f'library allowlist.', node)
        if isinstance(node, ast.ImportFrom) and node.module:
            for alias in node.names:
                dotted = f'{node.module}.{alias.name}'
                if dotted in FORBIDDEN_CALLS:
                    self.add('error', 'safety',
                             f'Imports `{dotted}`, which '
                             f'{FORBIDDEN_CALLS[dotted]}.', node)

    def _check_call_safety(self, call):
        name = self.dotted(call.func)
        bare = call.func.id if isinstance(call.func, ast.Name) else None
        if bare in FORBIDDEN_BUILTINS and bare not in self.aliases:
            self.add('error', 'safety',
                     f'Calls `{bare}()`, which {FORBIDDEN_BUILTINS[bare]}.',
                     call)
            return
        if name:
            if name in FORBIDDEN_CALLS:
                self.add('error', 'safety',
                         f'Calls `{name}()`, which {FORBIDDEN_CALLS[name]}.',
                         call)
                return
            for prefix, reason in FORBIDDEN_CALL_PREFIXES.items():
                if name.startswith(prefix):
                    self.add('error', 'safety',
                             f'Calls `{name}()`, which {reason}.', call)
                    return
        if isinstance(call.func, ast.Attribute) and (
                call.func.attr in FORBIDDEN_METHODS):
            self.add('error', 'safety',
                     f'Calls `.{call.func.attr}()`, which '
                     f'{FORBIDDEN_METHODS[call.func.attr]}.', call)
            return
        if bare in ('getattr', 'setattr', 'delattr') and len(call.args) >= 2:
            attr = self.literal(call.args[1])
            if attr is None:
                self.add('warning', 'safety',
                         f'`{bare}()` with a computed attribute name cannot '
                         f'be checked statically.', call)
            elif attr.startswith('__'):
                self.add('error', 'safety',
                         f'`{bare}()` reaches the interpreter internal '
                         f'`{attr}`.', call)
        self._check_write_paths(call, name)

    def _check_write_paths(self, call, name):
        path_node, what = None, None
        is_open = (name in ('open', 'io.open', 'builtins.open')
                   or (isinstance(call.func, ast.Name)
                       and call.func.id == 'open'))
        if is_open:
            mode_node = self.call_arg(call, 1, 'mode')
            mode = self.literal(mode_node) if mode_node is not None else 'r'
            if mode is None or any(c in mode for c in 'wax+'):
                path_node, what = self.call_arg(call, 0, 'file'), 'writes to'
            else:
                return
        elif name in PATH_WRITE_CALLS:
            index, keyword = PATH_WRITE_CALLS[name]
            path_node, what = self.call_arg(call, index, keyword), 'writes to'
            if name == 'os.chdir':
                what = 'changes directory to'
        elif isinstance(call.func, ast.Attribute):
            method = call.func.attr
            if method in PATH_WRITE_METHODS:
                index, keyword = PATH_WRITE_METHODS[method]
                path_node, what = self.call_arg(call, index, keyword), 'writes to'
            elif method in ('write_text', 'write_bytes', 'mkdir', 'touch'):
                receiver = call.func.value
                if isinstance(receiver, ast.Call) and receiver.args:
                    path_node, what = receiver.args[0], 'writes to'
            elif (isinstance(call.func.value, ast.Name)
                  and call.func.value.id in self.gta_vars):
                path_node, what = self.call_arg(call, None, 'outfile'), 'writes to'
        if path_node is None:
            return
        path = self.literal(path_node)
        if path is not None and _is_escaping_path(path):
            self.add('error', 'safety',
                     f'{(name or call.func.attr)}() {what} {path!r}, outside '
                     f'the run working directory. Use a relative path (it '
                     f'resolves inside the working directory).', call)

    # -- pass 3: FermiPy API, call order, consistency ------------------------
    def gta_call_sequence(self):
        """GTAnalysis method calls in execution order (one-level inlining)."""
        funcs = {n.name: n for n in self.tree.body
                 if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
        func_calls = {name: [] for name in funcs}
        top = []
        for node in ast.walk(self.tree):
            if not isinstance(node, ast.Call):
                continue
            owner = self.enclosing_function(node)
            if owner is not None and getattr(owner, 'name', None) in funcs:
                func_calls[owner.name].append(node)
            elif owner is None:
                top.append(node)
        for calls in func_calls.values():
            calls.sort(key=lambda c: (c.lineno, c.col_offset))
        top.sort(key=lambda c: (c.lineno, c.col_offset))
        sequence = []
        for call in top:
            if isinstance(call.func, ast.Name) and call.func.id in funcs:
                sequence.extend(
                    c for c in func_calls[call.func.id] if self._gta_method(c))
            elif self._gta_method(call):
                sequence.append(call)
        # Method calls in functions that are never called directly still
        # need the API checks.
        called = {c.func.id for c in top if isinstance(c.func, ast.Name)}
        orphans = [c for name, calls in func_calls.items()
                   if name not in called for c in calls if self._gta_method(c)]
        return sequence, orphans

    def _gta_method(self, call):
        func = call.func
        if (isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name)
                and func.value.id in self.gta_vars):
            return func.attr
        return None

    def check_api(self, calls):
        methods = gta_method_names()
        for call in calls:
            method = self._gta_method(call)
            if method not in methods:
                close = difflib.get_close_matches(method, sorted(methods), n=1)
                hint = f' Did you mean `{close[0]}`?' if close else ''
                self.add('error', 'api',
                         f'`GTAnalysis.{method}()` does not exist in the '
                         f'installed FermiPy.{hint}', call)
                continue
            spec = gta_method_spec(method)
            if not spec:
                continue
            n_pos = len(call.args)
            if any(isinstance(a, ast.Starred) for a in call.args):
                continue
            if spec['max_positional'] is not None and n_pos > spec['max_positional']:
                self.add('error', 'api',
                         f'`gta.{method}()` takes at most '
                         f'{spec["max_positional"]} positional argument(s) '
                         f'({", ".join(spec["positional"]) or "none"}); '
                         f'{n_pos} given.', call)
            keywords = spec['keywords']
            if keywords is None:
                continue
            for kw in call.keywords:
                if kw.arg is None or kw.arg in keywords:
                    continue
                close = difflib.get_close_matches(kw.arg, sorted(keywords), n=1)
                hint = f' Did you mean `{close[0]}`?' if close else ''
                self.add('error', 'api',
                         f'`gta.{method}()` does not accept the option '
                         f'`{kw.arg}` (FermiPy raises an error for unknown '
                         f'options).{hint}', call)

    def check_order(self, sequence):
        if not self.gta_ctor_calls:
            self.add('error', 'api',
                     'The script never creates a `GTAnalysis` object, so it '
                     'cannot run a FermiPy analysis.')
            return
        set_up = fitted = False
        model_map_written = False
        for call in sequence:
            method = self._gta_method(call)
            if method in SETUP_METHODS:
                set_up = True
                fitted = fitted or method == 'load_roi'
                continue
            if not set_up and method not in PRE_SETUP_OK:
                self.add('error', 'order',
                         f'`gta.{method}()` is called before `gta.setup()` '
                         f'(or `gta.load_roi()`); FermiPy needs the ROI to be '
                         f'set up first.', call)
                set_up = True  # report once
                continue
            if method in FIT_METHODS:
                fitted = fitted or method == 'fit'
            elif method == 'write_model_map':
                model_map_written = True
            elif method in PRODUCT_METHODS and not fitted:
                self.add('warning', 'order',
                         f'`gta.{method}()` runs before `gta.fit()`, so it '
                         f'uses the unfitted catalog model.', call)
            if method == 'psmap' and not model_map_written and (
                    self.call_arg(call, None, 'mmap') is not None):
                self.add('warning', 'order',
                         '`gta.psmap(mmap=...)` reads a model map, but the '
                         'script never calls `gta.write_model_map()` first.',
                         call)
        if not any(self._gta_method(c) in SETUP_METHODS for c in sequence):
            self.add('error', 'order',
                     'The script never calls `gta.setup()` (or '
                     '`gta.load_roi()`), so no analysis can run.')

    def check_consistency(self, sequence, all_calls):
        cfg = self.yaml_cfg
        target = str((cfg.get('selection') or {}).get('target') or '')
        yaml_sources = {str(s.get('name')) for s in (
            (cfg.get('model') or {}).get('sources') or [])
            if isinstance(s, dict) and s.get('name')}
        for ctor in self.gta_ctor_calls:
            arg = ctor.args[0] if ctor.args else self.call_arg(ctor, None,
                                                               'config')
            path = self.literal(arg) if arg is not None else None
            if arg is None:
                self.add('error', 'consistency',
                         '`GTAnalysis()` is created without a configuration, '
                         'so the reviewed YAML would not be used.', ctor)
            elif path is not None:
                if _is_escaping_path(path):
                    self.add(
                        'error', 'consistency',
                        f'`GTAnalysis({path!r})` loads a configuration '
                        f'outside the run directory instead of the reviewed '
                        f'YAML.', ctor,
                        fix='config_path')
                elif not path.lower().endswith(('.yaml', '.yml')):
                    self.add('warning', 'consistency',
                             f'`GTAnalysis({path!r})`: expected a .yaml '
                             f'configuration file name.', ctor)
            elif isinstance(arg, (ast.Dict, ast.Call)) and not (
                    'FERMI_LLM_CONFIG_PATH' in ast.unparse(arg)):
                self.add('warning', 'consistency',
                         '`GTAnalysis()` is given a configuration built in the '
                         'script, which may differ from the reviewed YAML.',
                         ctor)
            overrides = [kw.arg for kw in ctor.keywords
                         if kw.arg and kw.arg != 'config']
            if overrides:
                self.add('warning', 'consistency',
                         f'`GTAnalysis()` overrides YAML section(s) '
                         f'{", ".join(overrides)} at run time; the executed '
                         f'configuration will differ from the reviewed YAML.',
                         ctor)
        if len(self.gta_ctor_calls) > 1:
            self.add('info', 'consistency',
                     f'The script creates {len(self.gta_ctor_calls)} '
                     f'GTAnalysis objects; results are read from the last one '
                     f'that was set up.', self.gta_ctor_calls[1])

        for call in all_calls:
            method = self._gta_method(call)
            if method not in SOURCE_ARG_METHODS:
                continue
            arg = call.args[0] if call.args else self.call_arg(call, None,
                                                               'name')
            name = self.literal(arg) if arg is not None else None
            if not name:
                continue
            if name in BACKGROUND_NAMES or name in yaml_sources:
                continue
            if target and _norm_source(name) == _norm_source(target):
                continue
            status = _resolve_source_status(name, target)
            if status == 'same' or (status == 'other' and not target):
                continue
            if status == 'other':
                self.add('info', 'consistency',
                         f'`gta.{method}({name!r})` refers to a catalog source '
                         f'other than the YAML target {target!r}; it must lie '
                         f'inside the ROI.', call)
            else:
                self.add('warning', 'consistency',
                         f'`gta.{method}({name!r})`: {name!r} is not the YAML '
                         f'target ({target or "unset"}) and was not found in '
                         f'the 4FGL catalog or the YAML model; FermiPy raises '
                         f'an error if it is not in the ROI.', call)

        # YAML sections read from a dict the script loaded itself.
        for node in ast.walk(self.tree):
            if (isinstance(node, ast.Subscript)
                    and isinstance(node.value, ast.Name)
                    and node.value.id in self.yaml_vars
                    and isinstance(node.ctx, ast.Load)):
                key = self.literal(node.slice)
                if key and key not in cfg:
                    self.add('error', 'consistency',
                             f'Reads the YAML section {key!r}, which does not '
                             f'exist in the reviewed configuration (KeyError '
                             f'at run time).', node)

        # Files the script writes must not overwrite the reviewed config.
        config_names = {self.literal(c.args[0]) for c in self.gta_ctor_calls
                        if c.args and self.literal(c.args[0])}
        for node in ast.walk(self.tree):
            if isinstance(node, ast.Call) and (self.dotted(node.func) in (
                    'open', 'io.open')):
                mode_node = self.call_arg(node, 1, 'mode')
                mode = self.literal(mode_node) if mode_node is not None else 'r'
                path = self.literal(self.call_arg(node, 0, 'file'))
                if path and path in config_names and mode and any(
                        c in mode for c in 'wax+'):
                    self.add('error', 'consistency',
                             f'The script overwrites its configuration file '
                             f'{path!r}; the reviewed YAML must be used as-is.',
                             node)

        # psmap with relative map files when FermiPy writes to an outdir.
        outdir = ((cfg.get('fileio') or {}).get('outdir'))
        for call in all_calls:
            if self._gta_method(call) != 'psmap' or not outdir:
                continue
            for key in ('cmap', 'mmap'):
                path = self.literal(self.call_arg(call, None, key))
                if path and not os.path.isabs(path) and os.sep not in path:
                    self.add('warning', 'consistency',
                             f'`gta.psmap({key}={path!r})` is relative to the '
                             f'working directory, but FermiPy writes its maps '
                             f'to fileio.outdir ({outdir!r}); use '
                             f'`os.path.join(gta.outdir, {path!r})`.', call)


@functools.lru_cache(maxsize=256)
def _resolve_source_status(name, target):
    """'same' / 'other' / 'unknown' relative to the YAML target (4FGL)."""
    try:
        import source_resolver
    except Exception:
        return 'unknown'
    try:
        found = source_resolver.resolve(name)
    except Exception:
        found = None
    if not found:
        return 'unknown'
    try:
        tgt = source_resolver.resolve(target) if target else None
    except Exception:
        tgt = None
    if tgt and found.get('name') == tgt.get('name'):
        return 'same'
    return 'other'


def static_review(script, yaml_text):
    """Run every deterministic check. Returns a dict (see ``review_script``)."""
    out = {'findings': [], 'calls': [], 'config_ref': None,
           'uses_load_roi': False, 'runs_fit': False, 'products': []}
    if not (script or '').strip():
        out['findings'].append(_finding(
            'error', 'syntax', 'There is no Python script to execute.'))
        return _finalize(out)
    try:
        tree = ast.parse(script)
    except SyntaxError as exc:
        out['findings'].append(_finding(
            'error', 'syntax', f'Python syntax error: {exc.msg}',
            line=exc.lineno))
        return _finalize(out)

    analyzer = _Analyzer(tree, _load_yaml(yaml_text))
    analyzer.collect_bindings()
    analyzer.check_safety()
    sequence, orphans = analyzer.gta_call_sequence()
    all_calls = sequence + orphans
    analyzer.check_api(all_calls)
    analyzer.check_order(sequence)
    analyzer.check_consistency(sequence, all_calls)

    methods = [analyzer._gta_method(c) for c in sequence]
    out['calls'] = [{'method': m, 'line': c.lineno}
                    for m, c in zip(methods, sequence)]
    out['uses_load_roi'] = 'load_roi' in methods
    out['runs_fit'] = 'fit' in methods
    out['products'] = sorted({m for m in methods if m in PRODUCT_METHODS})
    for ctor in analyzer.gta_ctor_calls:
        if ctor.args:
            ref = analyzer.literal(ctor.args[0])
            if ref and not _is_escaping_path(ref):
                out['config_ref'] = ref
    out['findings'] = _dedupe(analyzer.findings)
    return _finalize(out)


def _dedupe(findings):
    seen, out = set(), []
    for item in findings:
        key = (item['severity'], item['category'], item['message'],
               item.get('line'))
        if key not in seen:
            seen.add(key)
            out.append(item)
    return out


def _finalize(out):
    out['findings'].sort(key=lambda f: (SEVERITIES.index(f['severity']),
                                        f.get('line') or 0))
    out['blocking'] = any(f['severity'] == 'error' for f in out['findings'])
    return out


def infer_run_mode(static):
    """'incremental' when the script reloads a previous fit without refitting."""
    return ('incremental' if static.get('uses_load_roi')
            and not static.get('runs_fit') else 'full')


# ---------------------------------------------------------------------------
# Deterministic fixes (offered as proposals, never applied silently)
# ---------------------------------------------------------------------------

CONFIG_ENV_EXPR = "os.environ.get('FERMI_LLM_CONFIG_PATH', 'config.yaml')"


def fix_config_reference(script):
    """Point ``GTAnalysis(<absolute path>)`` at the reviewed configuration."""
    try:
        tree = ast.parse(script)
    except SyntaxError:
        return script
    analyzer = _Analyzer(tree, {})
    analyzer.collect_bindings()
    lines = script.split('\n')
    edits = []
    for ctor in analyzer.gta_ctor_calls:
        if not ctor.args:
            continue
        arg = ctor.args[0]
        if (isinstance(arg, ast.Constant) and isinstance(arg.value, str)
                and _is_escaping_path(arg.value)
                and arg.lineno == arg.end_lineno):
            edits.append((arg.lineno - 1, arg.col_offset, arg.end_col_offset))
    if not edits:
        return script
    for row, start, end in sorted(edits, reverse=True):
        # col offsets are UTF-8 byte offsets
        raw = lines[row].encode()
        lines[row] = (raw[:start].decode() + CONFIG_ENV_EXPR
                      + raw[end:].decode())
    fixed = '\n'.join(lines)
    if not re.search(r'^\s*import\s+os\b|^\s*import\s+.*\bos\b', fixed,
                     re.M):
        fixed = 'import os\n' + fixed
    return fixed


def sed_bins_requested(requests):
    """Newest-first scan for an explicit SED bin count in the user's asks."""
    for text in reversed(list(requests or [])):
        count = sed_bin_request(text or '')
        if count:
            return count
    return None


def script_sets_sed_bins(script):
    """True when the script already passes its own SED binning."""
    try:
        tree = ast.parse(script)
    except SyntaxError:
        return True  # can't tell; never rewrite an unparseable script
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == 'sed'
                and any(kw.arg == 'loge_bins' for kw in node.keywords)):
            return True
    return False


def script_sed_bin_count(script):
    """Bin count of the literal ``loge_bins`` edges the script hands gta.sed().

    ``None`` when the script passes no edges, or builds them at run time (a
    computed argument is only knowable once the call is recorded).
    """
    try:
        tree = ast.parse(script)
    except SyntaxError:
        return None
    count = None
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == 'sed'):
            continue
        for kw in node.keywords:
            if kw.arg != 'loge_bins':
                continue
            try:
                edges = ast.literal_eval(kw.value)
            except (ValueError, TypeError, SyntaxError):
                continue
            if isinstance(edges, (list, tuple)) and len(edges) >= 2:
                count = len(edges) - 1
    return count


def apply_sed_loge_bins(script, yaml_text, nbins):
    """Pass explicit ``loge_bins`` to ``gta.sed()`` for an N-bin SED request.

    Only for a count that ``binning.binsperdec`` cannot express: when N bins
    divide the energy range into a whole number of bins per decade, the
    reviewed YAML already carries the request and the script is left alone.
    Otherwise the edges are the only way the count reaches FermiPy, so they
    are added to every ``gta.sed()`` call.
    """
    if not nbins or script_sets_sed_bins(script):
        return script
    try:
        cfg = yaml.safe_load(yaml_text) or {}
        selection = cfg.get('selection') or {}
        plan = sed_bin_plan(selection.get('emin'), selection.get('emax'),
                            nbins)
    except Exception:
        return script
    if not plan or plan['mode'] != 'loge_bins':
        return script
    edges = plan['loge_bins']
    literal = 'loge_bins=[' + ', '.join(repr(e) for e in edges) + ']'

    try:
        tree = ast.parse(script)
    except SyntaxError:
        return script
    edits = []
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == 'sed'):
            empty = not node.args and not node.keywords
            edits.append((node.end_lineno - 1, node.end_col_offset - 1, empty))
    if not edits:
        return script
    lines = script.split('\n')
    for row, col, empty in sorted(edits, reverse=True):
        # col offsets are UTF-8 byte offsets; col points at the closing paren.
        raw = lines[row].encode()
        insert = literal if empty else ', ' + literal
        lines[row] = raw[:col].decode() + insert + raw[col:].decode()
    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# Default fitting procedure
# ---------------------------------------------------------------------------

# Phrases that mean the user stated HOW the fit should run. A bare "fit" does
# not count: "fit the spectrum" or "compute an SED" asks for an analysis, not
# for a particular freeing strategy, so the default still applies there.
_FIT_DIRECTION_RE = re.compile(
    r"\bfree(ing|s|d)?\b|\bfix(ed|ing)?\b|\bfroze\w*\b|\bfreeze\b"
    r"|\bthaw\w*\b|\boptimi[sz]\w*\b|\bminmax_ts\b|\bts\s*[><]=?\s*\d"
    r"|\bminuit\b|\bminimi[sz]er\b|\biterat\w+\b|\brounds? of\b"
    r"|\bno fit\b|\b(do|does|did)(n'?t| not) fit\b|\bwithout fitting\b"
    r"|\bskip the fit\b|\bnuisance\b",
    re.I)


def fit_directions_requested(requests):
    """True when the user said how the fit should be run.

    The default fitting procedure is imposed only when they did not: an
    explicit direction always wins, even where it contradicts the default.
    """
    return any(_FIT_DIRECTION_RE.search(text or '')
               for text in (requests or []))


def default_fit_gaps(script, yaml_text):
    """What the default fitting procedure is missing from ``script``.

    With no fitting directions from the user, a full run should free the
    diffuse backgrounds and the target and then optimize and fit once.
    Returns the ordered list of missing pieces -- source names to free, and
    the strings ``'optimize'`` / ``'fit'``. An incremental run (load_roi)
    reuses an existing fit and is exempt, as is a script that never sets up.
    """
    try:
        tree = ast.parse(script)
    except SyntaxError:
        return []
    cfg = _load_yaml(yaml_text)
    analyzer = _Analyzer(tree, cfg)
    analyzer.collect_bindings()
    sequence, orphans = analyzer.gta_call_sequence()
    methods = [analyzer._gta_method(c) for c in sequence + orphans]
    if 'load_roi' in methods or 'setup' not in methods:
        return []

    freed = set()
    for call in sequence + orphans:
        if analyzer._gta_method(call) != 'free_source':
            continue
        name = analyzer.literal(analyzer.call_arg(call, 0, 'name'))
        if name:
            freed.add(name.strip().lower())

    model = cfg.get('model') or {}
    wanted = []
    # Only the backgrounds the ROI actually has: galdiff/isodiff are present
    # exactly when the reviewed YAML declares them.
    for diffuse in ('galdiff', 'isodiff'):
        if model.get(diffuse):
            wanted.append(diffuse)
    target = str((cfg.get('selection') or {}).get('target') or '').strip()
    if target:
        wanted.append(target)

    gaps = [name for name in wanted if name.lower() not in freed]
    for method in ('optimize', 'fit'):
        if method not in methods:
            gaps.append(method)
    return gaps


def apply_default_fit(script, yaml_text):
    """Add the missing pieces of the default fitting procedure.

    Everything is inserted directly after the top-level ``gta.setup()`` call,
    which keeps the freeing before any fit and the fit before any product.
    A setup call that is not a single top-level statement is left alone (no
    proposal) rather than guessed at.
    """
    gaps = default_fit_gaps(script, yaml_text)
    if not gaps:
        return script
    try:
        tree = ast.parse(script)
    except SyntaxError:
        return script
    analyzer = _Analyzer(tree, _load_yaml(yaml_text))
    analyzer.collect_bindings()

    setup_stmt = var = None
    for stmt in tree.body:
        call = stmt.value if isinstance(stmt, (ast.Expr, ast.Assign)) else None
        if not isinstance(call, ast.Call):
            continue
        if analyzer._gta_method(call) == 'setup':
            if stmt.lineno != stmt.end_lineno:
                return script
            setup_stmt = stmt
            var = call.func.value.id
            break
    if setup_stmt is None:
        return script

    lines = script.split('\n')
    row = setup_stmt.lineno - 1
    indent = lines[row][:len(lines[row]) - len(lines[row].lstrip())]
    freeing = [g for g in gaps if g not in ('optimize', 'fit')]
    added = []
    if freeing:
        added.append(f'{indent}# Default fit: free the diffuse backgrounds '
                     f'and the target.')
        added += [f'{indent}{var}.free_source({name!r})' for name in freeing]
    if 'optimize' in gaps:
        added.append(f'{indent}{var}.optimize()')
    if 'fit' in gaps:
        added.append(f'{indent}{var}.fit()')
    return '\n'.join(lines[:row + 1] + added + lines[row + 1:])


def propose_load_roi(script, roi_name='fit_model'):
    """Rewrite a full run to reuse the previous fit via ``gta.load_roi``.

    Only single-line, top-level ``gta.setup()`` / ``gta.optimize()`` /
    ``gta.fit()`` statements are rewritten; anything more complex returns
    None (no proposal) rather than guessing.
    """
    try:
        tree = ast.parse(script)
    except SyntaxError:
        return None
    analyzer = _Analyzer(tree, {})
    analyzer.collect_bindings()
    lines = script.split('\n')
    replacements = {}
    saw_setup = False
    for stmt in tree.body:
        call = stmt.value if isinstance(stmt, (ast.Expr, ast.Assign)) else None
        if not isinstance(call, ast.Call):
            continue
        method = analyzer._gta_method(call)
        if method not in ('setup', 'optimize', 'fit'):
            continue
        if stmt.lineno != stmt.end_lineno:
            return None
        row = stmt.lineno - 1
        indent = lines[row][:len(lines[row]) - len(lines[row].lstrip())]
        var = call.func.value.id
        if method == 'setup':
            saw_setup = True
            replacements[row] = (
                f"{indent}{var}.load_roi({roi_name!r})  "
                f"# reuse the fit saved by the previous run")
        elif isinstance(stmt, ast.Assign):
            target = ast.unparse(stmt.targets[0])
            replacements[row] = (
                f"{indent}{target} = None  # {method}() reused from the "
                f"previous run")
        else:
            replacements[row] = (
                f"{indent}# {lines[row].strip()}  # reused from the previous run")
    if not saw_setup or not any('fit()' in v or 'fit(' in v
                                for v in replacements.values()):
        return None
    return '\n'.join(replacements.get(i, line) for i, line in enumerate(lines))


# ---------------------------------------------------------------------------
# Request-fidelity heuristics (used when no LLM review is available)
# ---------------------------------------------------------------------------

PRODUCT_KEYWORDS = {
    'sed': r'\bseds?\b|spectral energy distribution',
    'lightcurve': r'light[\s-]?curves?',
    'tsmap': r'\bts[\s-]?maps?\b',
    'tscube': r'\bts[\s-]?cubes?\b',
    'residmap': r'\bresidual(s| map)?\b|\bresidmap\b',
    'psmap': r'\bps[\s-]?maps?\b',
    'localize': r'\blocali[sz](e|ation)\b',
    'extension': r'\bspatial extension\b|\bextension (test|analysis)\b',
    'find_sources': r'\bfind (new )?sources\b|\bsource ?find(ing)?\b',
    'curvature': r'\bcurvature\b',
}
_NEGATION_RE = re.compile(
    r"(\bno\b|\bnot\b|\bwithout\b|\bdon't\b|\bdo not\b|\bremove\b|\bdrop\b|"
    r"\bskip\b|\bdelete\b)[^.;\n]{0,40}$", re.I)


def keyword_fidelity(requests, static):
    """Very rough request-vs-script check: requested products that are missing."""
    text = '\n'.join(r for r in (requests or []) if r)
    findings = []
    called = set(static.get('products') or [])
    for product, pattern in PRODUCT_KEYWORDS.items():
        requested = removed = False
        for match in re.finditer(pattern, text, re.I):
            prefix = text[max(0, match.start() - 60):match.start()]
            if _NEGATION_RE.search(prefix):
                removed = True
            else:
                requested = True
                removed = False
        if requested and not removed and product not in called:
            findings.append(_finding(
                'warning', 'fidelity',
                f'The request mentions {product}, but the script never calls '
                f'`gta.{product}()`.', source='heuristic'))
        if removed and product in called:
            findings.append(_finding(
                'warning', 'fidelity',
                f'The request asks to drop {product}, but the script still '
                f'calls `gta.{product}()`.', source='heuristic'))
    return findings


# ---------------------------------------------------------------------------
# LLM review
# ---------------------------------------------------------------------------

REVIEW_SCHEMA = {
    'type': 'object',
    'additionalProperties': False,
    'required': ['summary', 'findings', 'revised_script'],
    'properties': {
        'summary': {'type': 'string'},
        'findings': {
            'type': 'array',
            'items': {
                'type': 'object',
                'additionalProperties': False,
                'required': ['severity', 'category', 'line', 'message'],
                'properties': {
                    'severity': {'type': 'string', 'enum': list(SEVERITIES)},
                    'category': {'type': 'string', 'enum': [
                        'fidelity', 'api', 'order', 'consistency', 'safety']},
                    'line': {'type': ['integer', 'null']},
                    'message': {'type': 'string'},
                },
            },
        },
        'revised_script': {'type': 'string'},
    },
}


def build_review_prompt(requests, yaml_text, script, static_findings):
    numbered = '\n'.join(f'{i:4d}| {line}'
                         for i, line in enumerate(script.split('\n'), 1))
    request_text = '\n'.join(f'{i}. {r.strip()}'
                             for i, r in enumerate(requests or [], 1)
                             if r and r.strip()) or '(no request text)'
    static_text = '\n'.join(
        f'- [{f["severity"]}/{f["category"]}] line {f.get("line") or "-"}: '
        f'{f["message"]}' for f in static_findings) or '- none'
    return '\n'.join([
        'You are the Validation Agent of a Fermi-LAT analysis assistant. You '
        'review a FermiPy Python script that another model generated. The '
        'script will be executed exactly as written, with the YAML below '
        'available at the file name the script passes to GTAnalysis() and in '
        'the FERMI_LLM_CONFIG_PATH environment variable. The working '
        'directory is a per-session scratch directory; relative paths '
        'resolve inside it.',
        '',
        'Check the script for:',
        '1. fidelity: every analysis step the user asked for (across all '
        'requests below; later requests amend earlier ones) is present with '
        'the requested parameters (sources, ordering, options, plots, '
        'post-processing), and nothing significant was added that was not '
        'requested. Options set in the YAML count as requested settings.',
        '2. api: valid GTAnalysis methods and arguments for FermiPy 1.x.',
        '3. order: gta.setup() (or gta.load_roi()) first, then '
        'optimize()/fit(), then products such as sed/lightcurve/tsmap.',
        '4. consistency with the YAML: configuration file, source names, '
        'output directories, and YAML sections the script relies on.',
        '5. safety: no destructive file operations, network access, shell '
        'commands, or writes outside the working directory.',
        '',
        'Do not report style issues. Do not ask for steps the user did not '
        'request. Findings the deterministic checker already reported are '
        'listed; include them only if you disagree or can add a concrete fix.',
        '',
        'If any change is needed, return the COMPLETE corrected script in '
        'revised_script, changing as little as possible and keeping the '
        "author's structure and comments. Otherwise return an empty string "
        'for revised_script. Return only a JSON object with keys summary '
        '(one sentence), findings (list of {severity: error|warning|info, '
        'category: fidelity|api|order|consistency|safety, line: integer or '
        'null, message}), and revised_script.',
        '',
        '## User requests (chronological)',
        request_text,
        '',
        '## Reviewed YAML configuration',
        '```yaml', (yaml_text or '').rstrip(), '```',
        '',
        '## Script under review (line numbers are for reference only)',
        '```', numbered, '```',
        '',
        '## Deterministic checker findings',
        static_text,
    ])


def parse_review_response(text):
    """Parse the reviewer JSON (raw, fenced, or embedded in prose)."""
    if not text or not text.strip():
        raise ValueError('empty review response')
    candidates = [text.strip()]
    fenced = re.search(r'```(?:json)?\s*\n(.*?)\n```', text, re.S)
    if fenced:
        candidates.append(fenced.group(1))
    start, end = text.find('{'), text.rfind('}')
    if 0 <= start < end:
        candidates.append(text[start:end + 1])
    data = None
    for candidate in candidates:
        try:
            data = json.loads(candidate)
            break
        except (TypeError, ValueError):
            continue
    if not isinstance(data, dict):
        raise ValueError('review response is not a JSON object')
    findings = []
    for item in data.get('findings') or []:
        if not isinstance(item, dict) or not str(item.get('message', '')).strip():
            continue
        severity = str(item.get('severity', 'warning')).lower()
        category = str(item.get('category', 'fidelity')).lower()
        line = item.get('line')
        findings.append(_finding(
            severity if severity in SEVERITIES else 'warning',
            category if category in CATEGORIES else 'fidelity',
            str(item['message']).strip()[:1000],
            line=line if isinstance(line, int) and line > 0 else None,
            source='llm'))
    revised = data.get('revised_script') or ''
    if not isinstance(revised, str):
        revised = ''
    fence = re.match(r'^\s*```(?:python)?\s*\n(.*)\n```\s*$', revised, re.S)
    if fence:
        revised = fence.group(1)
    return {'summary': str(data.get('summary') or '').strip()[:500],
            'findings': findings, 'revised_script': revised}


def _same_script(a, b):
    norm = lambda s: '\n'.join(line.rstrip() for line in (s or '').strip().split('\n'))
    return norm(a) == norm(b)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def _proposal_reason(summary, sed_bins, fit_gaps):
    """One sentence per deterministic change layered into the proposal."""
    if summary:
        return summary
    reasons = []
    if sed_bins:
        reasons.append(
            f'Pass the {sed_bins} energy-bin edges you asked for to '
            f'gta.sed(); {sed_bins} bins is not a whole number of bins per '
            f'decade, so binning.binsperdec cannot express it.')
    if fit_gaps:
        reasons.append(
            'Apply the default fitting procedure (free the diffuse '
            'backgrounds and the target, then optimize and fit once), which '
            'the script is missing: '
            + ', '.join(str(gap) for gap in fit_gaps) + '.')
    if not reasons:
        reasons.append('Load the reviewed configuration instead of an '
                       'absolute path.')
    return ' '.join(reasons)


def review_script(script, yaml_text, requests=None, llm=None,
                  allow_incremental=False):
    """Review ``script`` against ``yaml_text`` and the user's ``requests``.

    ``llm`` is an optional callable ``prompt -> response text``; when it is
    None or fails, fidelity falls back to keyword heuristics. Returns::

        {
          'findings': [...],           # static + LLM (or heuristic)
          'blocking': bool,            # error-severity findings on `script`
          'proposal': None | {
              'kind': 'fix' | 'incremental',
              'script': str,           # complete proposed script
              'reason': str,
              'findings': [...],       # static findings on the proposal
              'blocking': bool,
          },
          'llm_used': bool, 'llm_error': str | None, 'summary': str,
          'static': {...},             # static_review(script) output
        }
    """
    static = static_review(script, yaml_text)
    findings = list(static['findings'])
    result = {'static': static, 'llm_used': False, 'llm_error': None,
              'summary': '', 'proposal': None}

    revised = None
    if llm is not None and (script or '').strip():
        try:
            review = parse_review_response(llm(build_review_prompt(
                requests, yaml_text, script, static['findings'])))
            result['llm_used'] = True
            result['summary'] = review['summary']
            findings.extend(review['findings'])
            if review['revised_script'].strip() and not _same_script(
                    review['revised_script'], script):
                revised = review['revised_script']
        except Exception as exc:
            result['llm_error'] = f'{type(exc).__name__}: {str(exc)[:300]}'
    if not result['llm_used']:
        findings.extend(keyword_fidelity(requests, static))

    # Candidate proposal: the reviewer's revision, with the deterministic
    # config-path fix layered on top; or the deterministic fix alone.
    candidate = revised if revised is not None else script
    if any(f.get('fix') == 'config_path' for f in static['findings']) or (
            revised is not None):
        candidate = fix_config_reference(candidate)

    # An explicit "N energy bins" only reaches FermiPy through
    # gta.sed(loge_bins=...): binsperdec is bins per decade, so the YAML
    # cannot carry the total on its own.
    sed_bins = sed_bins_requested(requests)
    sed_fix = False
    if sed_bins and 'sed' in (static.get('products') or []):
        with_edges = apply_sed_loge_bins(candidate, yaml_text, sed_bins)
        if with_edges != candidate:
            candidate = with_edges
            sed_fix = True
            findings.append(_finding(
                'warning', 'fidelity',
                f'The request asks for {sed_bins} SED energy bins, but that '
                f'count is not a whole number of bins per decade over the '
                f'analysis energy range, so `binning.binsperdec` cannot '
                f'express it. Letting `gta.sed()` fall back to the analysis '
                f'binning would give a different number of bins; explicit '
                f'logarithmically spaced `loge_bins` edges honor the '
                f'request.', source='validator'))

    # With no fitting directions from the user, a full run still has to free
    # the diffuse backgrounds and the target and fit once. Directions the user
    # DID give always win, even where they contradict this default; the static
    # checks above are what catch a direction that cannot work.
    # With no request text at all there is nothing to judge "did the user
    # state a procedure?" against, so the default is not imposed either.
    fit_fix = []
    if requests and not fit_directions_requested(requests):
        fit_fix = default_fit_gaps(candidate, yaml_text)
        with_fit = apply_default_fit(candidate, yaml_text)
        if with_fit != candidate:
            candidate = with_fit
            missing = ', '.join(f'`{g}`' for g in fit_fix)
            findings.append(_finding(
                'warning', 'fidelity',
                f'The request gives no fitting directions, so the default '
                f'applies: free the diffuse backgrounds and the target, then '
                f'run `gta.optimize()` and `gta.fit()` once. The script is '
                f'missing {missing}.', source='validator'))
        else:
            fit_fix = []

    if candidate and not _same_script(candidate, script):
        cand_static = static_review(candidate, yaml_text)
        n_errors = lambda s: sum(f['severity'] == 'error' for f in s['findings'])
        if cand_static['blocking'] and n_errors(cand_static) >= max(
                1, n_errors(static)):
            findings.append(_finding(
                'info', 'fidelity',
                'A revision was drafted, but it failed the deterministic '
                'checks, so it is not offered.', source='validator'))
        else:
            result['proposal'] = {
                'kind': 'fix',
                'script': candidate if candidate.endswith('\n') else candidate + '\n',
                'reason': _proposal_reason(
                    result['summary'] if revised is not None else None,
                    sed_bins if sed_fix else None, fit_fix),
                'findings': cand_static['findings'],
                'blocking': cand_static['blocking'],
            }

    if (result['proposal'] is None and allow_incremental
            and not static['blocking'] and static['runs_fit']):
        incremental = propose_load_roi(script)
        if incremental:
            result['proposal'] = {
                'kind': 'incremental',
                'script': incremental,
                'reason': ('The data selection, binning and model are '
                           'unchanged since the last successful run, so the '
                           'fitted ROI can be reloaded with gta.load_roi() '
                           'instead of repeating setup/optimize/fit.'),
                'findings': static_review(incremental, yaml_text)['findings'],
                'blocking': False,
            }

    # Only deterministic errors block execution. LLM findings are advisory:
    # they reach the user as a proposal to accept or decline, so a reviewer
    # false positive can never lock the user out of running their script.
    result['findings'] = _finalize({'findings': _dedupe(findings)})['findings']
    result['blocking'] = static['blocking']
    return result

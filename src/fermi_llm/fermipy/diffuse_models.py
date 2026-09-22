"""Environment-aware Fermi diffuse-model resolution.

The web generator and the validation pipeline must agree on both the file
selected for an event type and the FermiPy value shape.  FermiPy accepts
``galdiff`` and ``isodiff`` as lists, even when only one component is used.
"""

import os
import sys

import yaml


DEFAULT_IRFS = 'P8R3_SOURCE_V3'
GALDIFF_FILENAME = 'gll_iem_v07.fits'

EVTYPE_SUFFIX = {
    3: '',
    1: '_FRONT',
    2: '_BACK',
    4: '_PSF0',
    8: '_PSF1',
    16: '_PSF2',
    32: '_PSF3',
    48: '_PSF23',
}


def get_diffuse_dir():
    """Return the absolute diffuse-model directory for this runtime."""
    explicit = os.environ.get('FERMI_DIFFUSE_DIR', '').strip()
    if explicit:
        return os.path.abspath(os.path.expandvars(os.path.expanduser(explicit)))

    fermi_dir = os.environ.get('FERMI_DIR', '').strip()
    if fermi_dir:
        return os.path.abspath(os.path.join(
            os.path.expandvars(os.path.expanduser(fermi_dir)),
            'refdata', 'fermi', 'galdiffuse'))

    # The web app is launched with the fermi-llm interpreter.  This fallback
    # also makes non-login test shells resolve the same installed files.
    return os.path.abspath(os.path.join(
        sys.prefix, 'share', 'fermitools', 'refdata', 'fermi', 'galdiffuse'))


def expected_galdiff_path():
    return os.path.join(get_diffuse_dir(), GALDIFF_FILENAME)


def expected_isodiff_path(evtype, irfs=DEFAULT_IRFS):
    try:
        event_type = int(evtype)
    except (TypeError, ValueError):
        event_type = 3
    suffix = EVTYPE_SUFFIX.get(event_type, '')
    return os.path.join(get_diffuse_dir(), f'iso_{irfs}{suffix}_v1.txt')


def get_galdiff():
    """Return the installed Galactic diffuse model, if present."""
    path = expected_galdiff_path()
    return path if os.path.exists(path) else None


def get_correct_isodiff(evtype, irfs=DEFAULT_IRFS):
    """Return the installed isotropic model matching ``evtype`` and IRFs."""
    expected = expected_isodiff_path(evtype, irfs)
    if os.path.exists(expected):
        return expected

    try:
        event_type = int(evtype)
    except (TypeError, ValueError):
        event_type = 3
    suffix = EVTYPE_SUFFIX.get(event_type, '')
    diffuse_dir = get_diffuse_dir()

    alternatives = [
        os.path.join(diffuse_dir, f'iso_{irfs}{suffix}.txt'),
        os.path.join(
            diffuse_dir,
            f'iso_{str(irfs).replace("V3", "V2")}{suffix}_v1.txt'),
    ]
    for path in alternatives:
        if os.path.exists(path):
            return path

    if os.path.isdir(diffuse_dir):
        required = ['iso_', 'P8R3', 'SOURCE']
        if suffix:
            required.append(suffix.strip('_'))
        for filename in sorted(os.listdir(diffuse_dir)):
            if filename.endswith('.txt') and all(
                    token in filename for token in required):
                return os.path.join(diffuse_dir, filename)

    fallback = expected_isodiff_path(3, DEFAULT_IRFS)
    return fallback if os.path.exists(fallback) else None


def normalize_diffuse_config(config):
    """Set canonical one-item diffuse lists on a parsed YAML config.

    Returns a list of keys whose values changed.  Missing runtime files are
    left untouched so the caller can surface the environment problem rather
    than writing a path that does not exist.
    """
    if not isinstance(config, dict):
        return []

    selection = config.get('selection')
    if not isinstance(selection, dict):
        selection = {}
    gtlike = config.get('gtlike')
    if not isinstance(gtlike, dict):
        gtlike = {}
    model = config.get('model')
    if not isinstance(model, dict):
        model = {}
        config['model'] = model

    evtype = selection.get('evtype', 3)
    irfs = str(gtlike.get('irfs') or DEFAULT_IRFS)
    desired = {
        'galdiff': get_galdiff(),
        'isodiff': get_correct_isodiff(evtype, irfs=irfs),
    }
    changed = []
    for key, path in desired.items():
        if path and model.get(key) != [path]:
            model[key] = [path]
            changed.append(key)
    return changed


def normalize_diffuse_yaml(yaml_str):
    """Canonicalize diffuse values in YAML and return ``(yaml, changed)``."""
    if not yaml_str or not yaml_str.strip():
        return yaml_str, []
    try:
        config = yaml.safe_load(yaml_str)
    except yaml.YAMLError:
        return yaml_str, []
    if not isinstance(config, dict):
        return yaml_str, []

    changed = normalize_diffuse_config(config)
    if not changed:
        return yaml_str, []
    return yaml.safe_dump(
        config, default_flow_style=False, sort_keys=False), changed


def diffuse_prompt_rules():
    """Render environment-specific rules for generation prompts."""
    gal = get_galdiff() or expected_galdiff_path()
    rows = [
        'Diffuse model values MUST be one-item YAML lists of absolute paths:',
        f'- galdiff: [{gal}]',
    ]
    for evtype, label in ((3, 'FRONT+BACK'), (1, 'FRONT'), (32, 'PSF3')):
        iso = (get_correct_isodiff(evtype, DEFAULT_IRFS)
               or expected_isodiff_path(evtype, DEFAULT_IRFS))
        rows.append(f'- evtype={evtype} ({label}) isodiff: [{iso}]')
    return '\n'.join(rows)

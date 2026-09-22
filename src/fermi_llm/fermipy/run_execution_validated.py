#!/usr/bin/env python3
"""
Execution-validated FermiPy code generation with iterative repair.

This script dramatically improves Level 3 (gta.setup()) success rates by:
1. Fixing environment issues (correct isotropic diffuse model matching)
2. Comprehensive programmatic config repair engine
3. Iterative validate-repair loop with error-specific fix strategies
4. Re-generation with error feedback when programmatic repair fails
5. Enhanced prompts with environment-aware examples

Run with:
    conda run -n fermi python run_execution_validated.py [model_name]

Or to re-validate existing outputs without model generation:
    conda run -n fermi python run_execution_validated.py --revalidate
"""

import json
import os
import sys
import re
import yaml
import ast
import math
import tempfile
import shutil
import traceback
import time
import gc
import copy
from pathlib import Path
from .diffuse_models import (diffuse_prompt_rules,
                             get_correct_isodiff,
                             get_diffuse_dir,
                             get_galdiff,
                             normalize_diffuse_yaml)
from .fermipy_schema import (fermipy_binning_prompt_rules,
                             fermipy_option_keys,
                             sed_bin_plan)

# ============================================================
# PATHS AND CONSTANTS
# ============================================================

# Paths configurable via environment variables.
PROJECT_DIR = os.environ.get(
    'FERMI_LLM_PROJECT_DIR',
    os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
)
DATA_DIR = os.environ.get('FERMI_LLM_DATA_DIR', os.path.join(PROJECT_DIR, 'data'))
RESULTS_DIR = os.environ.get('FERMI_LLM_RESULTS_DIR', os.path.join(PROJECT_DIR, 'results'))
LAT_DATA_DIR = os.environ.get('FERMI_LLM_LAT_DATA_DIR', os.path.join(PROJECT_DIR, 'lat_data'))
CACHE_DIR = os.environ.get(
    'FERMI_LLM_CACHE_DIR',
    os.path.join(os.path.expanduser('~'), '.cache', 'huggingface')
)
# Ensure results dir exists
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

FERMI_DIR = os.environ.get('FERMI_DIR', '')
DIFFUSE_DIR = get_diffuse_dir()

# Source name corrections: test reference names → actual 4FGL catalog names
# The Crab Nebula pulsar is J0534.5+2200, NOT J0534.5+2201 (which doesn't exist)
# J0534.5+2201s = synchrotron nebula, J0534.5+2201i = IC nebula
SOURCE_NAME_CORRECTIONS = {
    '4FGL J0534.5+2201': '4FGL J0534.5+2200',  # Crab pulsar (correct name)
}

# Test case metadata: maps test_idx to LAT data files and known-good params
TEST_DATA_MAP = {
    0: {  # Mrk 421
        'evfile': os.path.join(LAT_DATA_DIR, 'mrk421_PH.fits'),
        'scfile': os.path.join(LAT_DATA_DIR, 'mrk421_SC.fits'),
        'target': '4FGL J1104.4+3812',
        'name': 'Mrk421',
    },
    1: {  # Vela
        'evfile': os.path.join(LAT_DATA_DIR, 'vela_PH.fits'),
        'scfile': os.path.join(LAT_DATA_DIR, 'vela_SC.fits'),
        'target': '4FGL J0835.3-4510',
        'name': 'Vela',
    },
    2: {  # Crab
        'evfile': os.path.join(LAT_DATA_DIR, 'crab_PH.fits'),
        'scfile': os.path.join(LAT_DATA_DIR, 'crab_SC.fits'),
        'target': '4FGL J0534.5+2200',  # Correct catalog name (pulsar)
        'name': 'Crab',
    },
}


def get_test_meta(spec):
    """Return the data metadata dict for a run spec.

    ``spec`` is either an int index into TEST_DATA_MAP (the three bundled
    demo sources, pre-cut one-week files) or a dict built by
    data_registry.build_full_data_spec for an arbitrary catalog source
    running on the full weekly dataset. Every consumer that used to do
    ``TEST_DATA_MAP.get(test_idx, {})`` goes through here so both kinds
    of run flow through the same validation pipeline.
    """
    if isinstance(spec, dict):
        return spec
    return TEST_DATA_MAP.get(spec, {})

# Valid config keys for each section (keys not in these sets cause errors)
VALID_GTLIKE_KEYS = {
    'edisp', 'edisp_disable', 'irfs', 'edisp_bins',
    'llscan_npts', 'wmap', 'src_expscale',
}
VALID_BINNING_KEYS = {
    'roiwidth', 'binsz', 'binsperdec', 'enumbins', 'npix',
    'coordsys', 'proj', 'hpx_order', 'hpx_ordering_scheme',
}
VALID_MODEL_KEYS = {
    'src_radius', 'src_roiwidth', 'galdiff', 'isodiff',
    'catalogs', 'extdir', 'src_radius_roi', 'sources',
    'diffuse', 'merge_sources', 'assoc_xmatch_columns',
}
VALID_SELECTION_KEYS = {
    'emin', 'emax', 'zmax', 'target', 'radius',
    'tmin', 'tmax', 'evclass', 'evtype', 'filter',
    'glat', 'glon', 'ra', 'dec', 'convtype',
    'phasemin', 'phasemax',
}
VALID_DATA_KEYS = {
    'evfile', 'scfile', 'ltcube',
}
VALID_FILEIO_KEYS = {
    'outdir', 'logfile', 'savefits', 'workdir', 'usescratch',
}
VALID_LOGGING_KEYS = {
    'verbosity', 'chatter',
}
# Optional analysis sections that are pass-through (no key filtering needed)
ANALYSIS_SECTIONS = {
    'sed', 'lightcurve', 'psmap', 'tsmap', 'residmap',
    'plotting', 'extension', 'localize', 'roiopt',
    'sourcefind', 'tscube', 'curvature', 'mc',
    'optimizer', 'ltcube', 'components',
}

# Valid evclass values for Fermi-LAT
VALID_EVCLASS = {2, 4, 8, 16, 32, 64, 128, 256, 512}
# Valid evtype values
VALID_EVTYPE = {1, 2, 3, 4, 8, 16, 32, 48, 56, 60, 62, 63}

MAX_REPAIR_ITERATIONS = 5


# ============================================================
# ENVIRONMENT DIAGNOSTICS
# ============================================================

def _is_fits_file(path):
    """True when the file starts with the FITS magic ('SIMPLE')."""
    try:
        with open(path, 'rb') as f:
            return f.read(6) == b'SIMPLE'
    except OSError:
        return False


def read_evfile_list(path):
    """Read an event-file list (one FITS path per line, '#' comments)."""
    files = []
    with open(path) as f:
        for line in f:
            line = line.strip().lstrip('@')
            if line and not line.startswith('#'):
                files.append(line)
    return files


def get_fits_time_range(evfile):
    """Extract TSTART/TSTOP from a FITS event file or an event-file list.

    For a file list (as produced by data_registry.write_evfile_list) the
    range spans from the first file's TSTART to the last file's TSTOP;
    the member files are time-ordered by construction. nevents is None
    in that case (counting rows across hundreds of files is not worth
    the I/O -- it is only used in diagnostics output).
    """
    try:
        from astropy.io import fits
        if not _is_fits_file(evfile):
            members = read_evfile_list(evfile)
            if not members:
                raise ValueError(f'empty event-file list: {evfile}')
            with fits.open(members[0]) as hdul:
                tstart = float(hdul[1].header.get('TSTART', 0))
            with fits.open(members[-1]) as hdul:
                tstop = float(hdul[1].header.get('TSTOP', 0))
            return {'tstart': tstart, 'tstop': tstop, 'nevents': None}
        with fits.open(evfile) as hdul:
            hdr = hdul[1].header
            tstart = float(hdr.get('TSTART', 0))
            tstop = float(hdr.get('TSTOP', 0))
            nrows = hdul[1].data.shape[0] if hdul[1].data is not None else 0
            return {'tstart': tstart, 'tstop': tstop, 'nevents': nrows}
    except Exception as e:
        print(f"  Warning: could not read FITS header from {evfile}: {e}")
        return {'tstart': 504921605.0, 'tstop': 505526405.0, 'nevents': 0}


def diagnose_environment():
    """Print environment diagnostics."""
    print("=" * 60)
    print("ENVIRONMENT DIAGNOSTICS")
    print("=" * 60)
    print(f"  FERMI_DIR: {FERMI_DIR}")
    print(f"  DIFFUSE_DIR: {DIFFUSE_DIR}")
    print(f"  Diffuse dir exists: {os.path.isdir(DIFFUSE_DIR)}")

    gal = get_galdiff()
    print(f"  Galactic diffuse: {gal}")

    for evtype, label in [(3, 'FRONT+BACK'), (1, 'FRONT'), (32, 'PSF3')]:
        iso = get_correct_isodiff(evtype)
        print(f"  Isotropic (evtype={evtype}, {label}): {os.path.basename(iso) if iso else 'NOT FOUND'}")

    print(f"\n  LAT data files:")
    for idx, meta in TEST_DATA_MAP.items():
        ev_ok = os.path.exists(meta['evfile'])
        sc_ok = os.path.exists(meta['scfile'])
        info = get_fits_time_range(meta['evfile']) if ev_ok else {}
        print(f"    test_{idx} ({meta['name']}): evfile={ev_ok} scfile={sc_ok} "
              f"events={info.get('nevents', '?')} "
              f"time=[{info.get('tstart', '?')}, {info.get('tstop', '?')}]")

    try:
        import fermipy
        print(f"\n  fermipy version: {fermipy.__version__}")
    except Exception:
        print("\n  fermipy: NOT AVAILABLE")

    print("=" * 60)


# ============================================================
# YAML / PYTHON EXTRACTION
# ============================================================

def extract_yaml(text):
    """Extract YAML from markdown code block."""
    m = re.search(r'```yaml\s*\n(.*?)\n```', text, re.DOTALL)
    return m.group(1).strip() if m else ''


def extract_python(text):
    """Extract Python from markdown code block."""
    m = re.search(r'```python\s*\n(.*?)\n```', text, re.DOTALL)
    return m.group(1).strip() if m else ''


# ============================================================
# CORE-SECTION COMPARISON (for incremental / load_roi reuse)
# ============================================================
# Sections that determine whether a previous gta.setup() + gta.fit() must be
# redone. Everything else (sed, lightcurve, tsmap, residmap, psmap,
# plotting, ...) only affects which post-fit *products* are computed and
# never invalidates a previously fitted ROI.
CORE_YAML_SECTIONS = ('data', 'selection', 'binning', 'model', 'components', 'gtlike')


def extract_core_sections(yaml_str):
    """Return the subset of a parsed YAML config that affects gta.setup().

    Returns None if the YAML does not parse to a dict.
    """
    try:
        cfg = yaml.safe_load(yaml_str)
    except Exception:
        return None
    if not isinstance(cfg, dict):
        return None
    return {k: cfg.get(k) for k in CORE_YAML_SECTIONS if k in cfg}


def core_sections_equal(yaml_a, yaml_b):
    """True if two YAML configs are identical in every setup-affecting
    section, i.e. a run for ``yaml_a`` can safely reuse a fitted ROI
    produced from ``yaml_b`` via ``gta.load_roi()`` instead of re-running
    ``gta.setup()`` + ``gta.optimize()`` + ``gta.fit()``.
    """
    a = extract_core_sections(yaml_a)
    b = extract_core_sections(yaml_b)
    if a is None or b is None:
        return False
    return a == b


# ============================================================
# CONFIG REPAIR ENGINE
# ============================================================

def repair_config(yaml_str, test_idx, iteration=0, prev_error=None,
                  sed_bins=None):
    """Comprehensive programmatic repair of a FermiPy YAML config.

    Fixes environment mismatches, invalid parameters, and missing sections.
    ``sed_bins`` is the total number of SED energy bins the user asked for,
    if any. It becomes binning.binsperdec when it divides the energy range
    into whole bins per decade; otherwise the binning is left alone and the
    edges reach FermiPy through the gta.sed() call instead of the YAML.
    Returns (repaired_yaml_str, repair_log).
    """
    repairs = []

    if not yaml_str or not yaml_str.strip():
        return '', ['empty yaml - cannot repair']

    # Parse YAML
    try:
        config = yaml.safe_load(yaml_str)
        if not isinstance(config, dict):
            return '', [f'yaml parsed to {type(config).__name__}, not dict']
    except yaml.YAMLError as e:
        # Try to fix common YAML issues
        fixed = yaml_str
        # Remove lines with bare keys (no colon)
        lines = fixed.split('\n')
        cleaned = []
        for line in lines:
            stripped = line.strip()
            if stripped and not stripped.startswith('#') and ':' not in stripped and not stripped.startswith('-'):
                repairs.append(f'removed invalid YAML line: {stripped[:50]}')
                continue
            cleaned.append(line)
        fixed = '\n'.join(cleaned)
        try:
            config = yaml.safe_load(fixed)
            if not isinstance(config, dict):
                return '', [f'repaired yaml parsed to {type(config).__name__}']
            repairs.append('fixed YAML syntax by removing invalid lines')
        except yaml.YAMLError:
            return '', [f'unfixable YAML: {str(e)[:100]}']

    meta = get_test_meta(test_idx)
    evfile = meta.get('evfile', '')
    scfile = meta.get('scfile', '')

    # ---- 1. Ensure required sections exist ----
    for section in ['logging', 'fileio', 'data', 'binning', 'selection', 'gtlike', 'model']:
        if section not in config or not isinstance(config.get(section), dict):
            config[section] = {}
            repairs.append(f'added/reset missing section: {section}')

    # ---- 2. Fix data paths (evfile, scfile) ----
    if os.path.exists(evfile):
        config['data']['evfile'] = evfile
    if os.path.exists(scfile):
        config['data']['scfile'] = scfile

    # ---- 3. Remove invalid ltcube references ----
    ltcube = str(config['data'].get('ltcube', ''))
    if ltcube and ('<' in ltcube or '/Volumes/' in ltcube
                   or (ltcube and not os.path.exists(ltcube))):
        config['data'].pop('ltcube', None)
        repairs.append('removed invalid ltcube reference')

    # ---- 4. Fix selection parameters ----
    sel = config['selection']

    # Ensure valid evclass
    evclass = sel.get('evclass')
    if evclass not in VALID_EVCLASS:
        sel['evclass'] = 128  # SOURCE class
        repairs.append(f'fixed evclass {evclass} → 128')

    # Ensure valid evtype
    evtype = sel.get('evtype')
    if evtype is None or evtype not in VALID_EVTYPE:
        sel['evtype'] = 3  # FRONT+BACK
        repairs.append(f'fixed evtype {evtype} → 3')

    # Ensure valid emin/emax
    emin = sel.get('emin')
    emax = sel.get('emax')
    if not isinstance(emin, (int, float)) or emin <= 0:
        sel['emin'] = 100
        repairs.append(f'fixed emin → 100')
    if not isinstance(emax, (int, float)) or emax <= sel.get('emin', 100):
        sel['emax'] = 1000000
        repairs.append(f'fixed emax → 1000000')

    # Ensure valid zmax
    zmax = sel.get('zmax')
    if not isinstance(zmax, (int, float)) or zmax <= 0 or zmax > 180:
        sel['zmax'] = 105
        repairs.append(f'fixed zmax → 105')

    # Fix time range: clamp to actual FITS data range
    if os.path.exists(evfile):
        fits_info = get_fits_time_range(evfile)
        tstart = fits_info['tstart']
        tstop = fits_info['tstop']

        tmin = sel.get('tmin')
        tmax = sel.get('tmax')

        # If tmin/tmax missing or out of range, set to FITS range
        if tmin is None or not isinstance(tmin, (int, float)):
            sel['tmin'] = tstart
            repairs.append(f'set tmin from FITS header: {tstart}')
        elif tmin > tstop:
            sel['tmin'] = tstart
            repairs.append(f'clamped tmin {tmin} → {tstart} (was after data end)')

        if tmax is None or not isinstance(tmax, (int, float)):
            sel['tmax'] = tstop
            repairs.append(f'set tmax from FITS header: {tstop}')
        elif tmax < tstart:
            sel['tmax'] = tstop
            repairs.append(f'clamped tmax {tmax} → {tstop} (was before data start)')

        # Ensure tmin < tmax
        if sel.get('tmin', 0) >= sel.get('tmax', 1):
            sel['tmin'] = tstart
            sel['tmax'] = tstop
            repairs.append('reset time range to FITS bounds')

    # Fix known source name errors (e.g., Crab J0534.5+2201 → J0534.5+2200)
    target = sel.get('target', '')
    if target in SOURCE_NAME_CORRECTIONS:
        corrected = SOURCE_NAME_CORRECTIONS[target]
        sel['target'] = corrected
        repairs.append(f'corrected source name: {target} → {corrected}')

    # Ensure target is set
    if 'target' not in sel or not sel['target']:
        if meta.get('target'):
            sel['target'] = meta['target']
            repairs.append(f'set target from test metadata: {meta["target"]}')

    # Full-dataset runs: pin the ROI center to the resolved catalog
    # coordinates so the analysis centers on the requested source even if
    # the target name is missing from the configured catalog release.
    if meta.get('mode') == 'full' and meta.get('ra') is not None \
            and meta.get('dec') is not None:
        if not isinstance(sel.get('ra'), (int, float)) \
                or not isinstance(sel.get('dec'), (int, float)):
            sel['ra'] = float(meta['ra'])
            sel['dec'] = float(meta['dec'])
            repairs.append(f'set ROI center from catalog: '
                           f'ra={sel["ra"]:.3f}, dec={sel["dec"]:.3f}')

    # Ensure radius is set
    if not isinstance(sel.get('radius'), (int, float)) or sel['radius'] <= 0:
        sel['radius'] = 15
        repairs.append('set default radius: 15')

    # ---- 5. Fix IRFs and isotropic diffuse model ----
    gtlike = config['gtlike']
    irfs = gtlike.get('irfs', '')
    if not irfs or 'P8R3' not in str(irfs):
        gtlike['irfs'] = 'P8R3_SOURCE_V3'
        irfs = 'P8R3_SOURCE_V3'
        repairs.append('set irfs → P8R3_SOURCE_V3')

    # Ensure edisp is set
    if 'edisp' not in gtlike:
        gtlike['edisp'] = True
        repairs.append('set edisp → True')

    # Fix isotropic diffuse model - THE CRITICAL FIX
    actual_evtype = sel.get('evtype', 3)
    correct_iso = get_correct_isodiff(actual_evtype, irfs=str(irfs))
    if correct_iso and config['model'].get('isodiff') != [correct_iso]:
        config['model']['isodiff'] = [correct_iso]
        repairs.append(f'set isodiff for evtype={actual_evtype}: {os.path.basename(correct_iso)}')

    # Fix galactic diffuse model
    galdiff = get_galdiff()
    if galdiff and config['model'].get('galdiff') != [galdiff]:
        config['model']['galdiff'] = [galdiff]
        repairs.append(f'set galdiff: {os.path.basename(galdiff)}')

    # Ensure catalogs are set
    if 'catalogs' not in config['model'] or not config['model']['catalogs']:
        config['model']['catalogs'] = ['4FGL-DR3']
        repairs.append('set default catalog: 4FGL-DR3')

    # Full-dataset runs know which catalog release actually contains the
    # resolved source (DR4-only sources are invisible to '4FGL-DR3').
    # Upgrade the default DR3 choice proactively instead of burning a
    # gta.setup() iteration on a 'No source matching name' failure.
    if meta.get('mode') == 'full' and meta.get('catalogued', True) and \
            meta.get('catalog') == '4FGL-DR4' \
            and config['model'].get('catalogs') == ['4FGL-DR3']:
        config['model']['catalogs'] = ['4FGL-DR4']
        repairs.append(f'switched catalog to 4FGL-DR4: '
                       f'{meta.get("target", "?")} is not in DR3')

    # A custom target is deliberately absent from 4FGL. Preserve its explicit
    # source model so ROIModel.create_from_source can center on
    # selection.target without substituting a catalog source.
    if meta.get('mode') == 'full' and meta.get('catalogued') is False:
        custom = meta.get('source_model')
        if isinstance(custom, dict) and custom.get('name'):
            sources = config['model'].get('sources')
            if not isinstance(sources, list):
                sources = []
            key = re.sub(r'[^a-z0-9.+-]', '', custom['name'].lower())
            kept = [
                s for s in sources if isinstance(s, dict) and
                re.sub(r'[^a-z0-9.+-]', '',
                       str(s.get('name') or '').lower()) != key
            ]
            config['model']['sources'] = kept + [copy.deepcopy(custom)]
            if sel.get('target') != custom['name']:
                sel['target'] = custom['name']
                repairs.append(
                    f'restored custom target source: {custom["name"]}')

    # Ensure src_radius/src_roiwidth
    if not isinstance(config['model'].get('src_roiwidth'), (int, float)):
        config['model']['src_roiwidth'] = sel.get('radius', 15)
    if not isinstance(config['model'].get('src_radius'), (int, float)):
        config['model']['src_radius'] = sel.get('radius', 15)

    # Remove extdir if it's a placeholder
    extdir = str(config['model'].get('extdir', ''))
    if extdir and ('<' in extdir or '/Volumes/' in extdir
                   or (extdir and not os.path.exists(extdir))):
        config['model'].pop('extdir', None)
        repairs.append('removed invalid extdir')

    # ---- 6. Fix binning defaults ----
    binning = config['binning']
    if not isinstance(binning.get('roiwidth'), (int, float)) or binning['roiwidth'] <= 0:
        binning['roiwidth'] = 10
        repairs.append('set roiwidth → 10')
    if not isinstance(binning.get('binsz'), (int, float)) or binning['binsz'] <= 0:
        binning['binsz'] = 0.1
        repairs.append('set binsz → 0.1')
    if not isinstance(binning.get('binsperdec'), (int, float)) or binning['binsperdec'] <= 0:
        binning['binsperdec'] = 8
        repairs.append('set binsperdec → 8')

    # ---- 6b. radius/roiwidth consistency lint ----
    # selection.radius is the event-selection cone; binning.roiwidth is the
    # side length of the (square) analysis region. The cone must at least
    # circumscribe the square, otherwise events near the ROI corners are
    # silently excluded from gtselect. LLM output frequently confuses
    # "radius" with "width" or leaves the two inconsistent.
    roiwidth = binning.get('roiwidth')
    if isinstance(roiwidth, (int, float)) and roiwidth > 0:
        min_radius = roiwidth / 2.0 * 1.415  # circumscribing circle of the square ROI
        current_radius = sel.get('radius')
        if not isinstance(current_radius, (int, float)) or current_radius < min_radius:
            new_radius = math.ceil(min_radius)
            repairs.append(
                f'raised selection.radius {current_radius} → {new_radius} deg '
                f'to circumscribe binning.roiwidth={roiwidth} deg'
            )
            sel['radius'] = new_radius

    # model.src_radius must cover the selection radius, otherwise sources
    # near the edge of the selected events are missing from the model.
    src_radius = config['model'].get('src_radius')
    if (isinstance(src_radius, (int, float)) and isinstance(sel.get('radius'), (int, float))
            and src_radius < sel['radius']):
        repairs.append(
            f'raised model.src_radius {src_radius} → {sel["radius"]} deg '
            f'to cover selection.radius (sources near the ROI edge would '
            f'otherwise be missing from the model)'
        )
        config['model']['src_radius'] = sel['radius']

    # model.src_roiwidth must cover binning.roiwidth for the same reason.
    src_roiwidth = config['model'].get('src_roiwidth')
    if (isinstance(src_roiwidth, (int, float)) and isinstance(roiwidth, (int, float))
            and src_roiwidth < roiwidth):
        repairs.append(
            f'raised model.src_roiwidth {src_roiwidth} → {roiwidth} deg '
            f'to cover binning.roiwidth'
        )
        config['model']['src_roiwidth'] = roiwidth

    # ---- 7. Fix logging defaults ----
    logging_sec = config['logging']
    if not isinstance(logging_sec.get('verbosity'), int):
        logging_sec['verbosity'] = 3
    if not isinstance(logging_sec.get('chatter'), int):
        logging_sec['chatter'] = 3

    # ---- 8. Fix fileio ----
    if not config['fileio'].get('outdir'):
        config['fileio']['outdir'] = 'output'
    if not config['fileio'].get('logfile'):
        config['fileio']['logfile'] = meta.get('name', 'analysis')

    # ---- 9. Remove invalid keys from all sections ----
    section_key_map = {
        'gtlike': VALID_GTLIKE_KEYS,
        'binning': VALID_BINNING_KEYS,
        'model': VALID_MODEL_KEYS,
        'selection': VALID_SELECTION_KEYS,
        'data': VALID_DATA_KEYS,
        'fileio': VALID_FILEIO_KEYS,
        'logging': VALID_LOGGING_KEYS,
    }
    for section, valid_keys in section_key_map.items():
        if section in config and isinstance(config[section], dict):
            bad_keys = [k for k in config[section] if k not in valid_keys]
            for k in bad_keys:
                del config[section][k]
                repairs.append(f'removed invalid key {section}.{k}')

    # Remove entirely unknown top-level sections
    known_sections = {'logging', 'fileio', 'data', 'binning', 'selection',
                      'gtlike', 'model'} | ANALYSIS_SECTIONS
    bad_top = [k for k in config if k not in known_sections]
    for k in bad_top:
        del config[k]
        repairs.append(f'removed unknown top-level section: {k}')

    # A TOTAL number of SED bins is not something the YAML can carry on its
    # own: binsperdec is bins per DECADE, and sed.loge_bins is a gta.sed()
    # argument rather than a GTAnalysis key. binsperdec expresses the request
    # exactly when nbins / log10(emax/emin) is whole; otherwise the analysis
    # binning is left alone and the edges reach FermiPy through the script.
    sed_section = config.get('sed')
    yaml_edges = (sed_section or {}).get('loge_bins') \
        if isinstance(sed_section, dict) else None
    requested = sed_bins
    if requested is None and isinstance(yaml_edges, list) and len(yaml_edges) >= 2:
        # The model had no other way to state the count; keep the intent.
        requested = len(yaml_edges) - 1
    if isinstance(sed_section, dict) and 'loge_bins' in sed_section:
        del sed_section['loge_bins']
        repairs.append(
            'removed sed.loge_bins: it is a gta.sed() argument, not a '
            'GTAnalysis YAML key')
    if requested and isinstance(sed_section, dict):
        plan = sed_bin_plan(sel.get('emin'), sel.get('emax'), requested)
        if plan and plan['mode'] == 'binsperdec':
            if binning.get('binsperdec') != plan['binsperdec']:
                repairs.append(
                    f"set binning.binsperdec={plan['binsperdec']} so the "
                    f"analysis grid is exactly the {requested} SED bins "
                    f"requested")
                binning['binsperdec'] = plan['binsperdec']

    # ---- 9b. Remove invalid keys from analysis sections (sed, lightcurve, etc.) ----
    # FermiPy is strict about allowed keys in these sections.
    VALID_SED_KEYS = fermipy_option_keys('sed')
    VALID_LC_KEYS = fermipy_option_keys('lightcurve')
    analysis_key_map = {'sed': VALID_SED_KEYS, 'lightcurve': VALID_LC_KEYS}
    for section, valid_keys in analysis_key_map.items():
        if section in config and isinstance(config[section], dict):
            bad_keys = [k for k in config[section] if k not in valid_keys]
            for k in bad_keys:
                del config[section][k]
                repairs.append(f'removed invalid key {section}.{k}')

    # ---- 10. Error-specific repairs from previous iteration ----
    if prev_error:
        error_str = str(prev_error)

        # Generic "Invalid configuration key" error — remove the offending key
        if 'Invalid configuration key' in error_str:
            import re as _re
            m = _re.search(r'Invalid configuration key:\s*(\S+)\s*\(section\s*:\s*(\S+)\)', error_str)
            if m:
                bad_key, section = m.group(1), m.group(2)
                if section in config and isinstance(config[section], dict):
                    config[section].pop(bad_key, None)
                    repairs.append(f'removed invalid key {section}.{bad_key} (from error)')

        if 'NDSKEYS' in error_str:
            # IRF/evtype mismatch - try switching to combined evtype
            if sel.get('evtype') != 3:
                sel['evtype'] = 3
                correct_iso = get_correct_isodiff(3, irfs=str(gtlike.get('irfs', 'P8R3_SOURCE_V3')))
                if correct_iso:
                    config['model']['isodiff'] = [correct_iso]
                repairs.append('NDSKEYS fix: switched to evtype=3 (FRONT+BACK)')

        elif 'list index out of range' in error_str and 'ltcube' in error_str.lower():
            # LTCube issue - ensure ltcube is removed so FermiPy generates one
            config['data'].pop('ltcube', None)
            # Also ensure filter is set (needed for gtmktime to produce valid GTIs)
            if not sel.get('filter'):
                sel['filter'] = 'DATA_QUAL>0 && LAT_CONFIG==1'
                repairs.append('LTCube fix: added filter for GTI generation')
            repairs.append('LTCube fix: removed ltcube, will auto-generate')

        elif 'No source matching name' in error_str:
            # Source name not found in catalog
            import re as _re
            src_match = _re.search(r"No source matching name:\s*(.+?)(?:'|\"|\s*$)", error_str)
            if src_match:
                bad_name = src_match.group(1).strip()
                if meta.get('catalogued') is False:
                    # Absence from 4FGL is expected. The deterministic custom
                    # source restoration above is the only valid repair; do
                    # not rotate catalogs or fit a different source.
                    repairs.append(
                        f'Source fix: restored non-catalog target {meta.get("target", bad_name)}')
                # First check known corrections
                elif bad_name in SOURCE_NAME_CORRECTIONS:
                    corrected = SOURCE_NAME_CORRECTIONS[bad_name]
                    sel['target'] = corrected
                    repairs.append(f'Source fix: {bad_name} → {corrected} (known correction)')
                else:
                    known_target = meta.get('target', '')
                    if bad_name != known_target and known_target:
                        sel['target'] = known_target
                        repairs.append(f'Source fix: {bad_name} → {known_target}')
                    else:
                        # Try broader radius
                        current_radius = sel.get('radius', 15)
                        if current_radius < 20:
                            sel['radius'] = 20
                            config['model']['src_radius'] = 20
                            config['model']['src_roiwidth'] = 20
                            repairs.append(f'Source fix: expanded radius {current_radius} → 20')
                        # Also try a different catalog version
                        cats = config['model'].get('catalogs', [])
                        if '4FGL-DR4' in cats:
                            config['model']['catalogs'] = ['4FGL-DR3']
                            repairs.append('Source fix: switched catalog 4FGL-DR4 → 4FGL-DR3')
                        elif '4FGL-DR3' in cats:
                            config['model']['catalogs'] = ['4FGL-DR4']
                            repairs.append('Source fix: switched catalog 4FGL-DR3 → 4FGL-DR4')

        elif 'None None' in error_str or 'NoneType' in error_str:
            # Missing tmin/tmax
            if os.path.exists(evfile):
                fits_info = get_fits_time_range(evfile)
                sel['tmin'] = fits_info['tstart']
                sel['tmax'] = fits_info['tstop']
                repairs.append(f'NoneType fix: set tmin/tmax from FITS')

    # Serialize back to YAML
    repaired = yaml.dump(config, default_flow_style=False, sort_keys=False)
    return repaired, repairs


# ============================================================
# VALIDATION LEVELS (improved)
# ============================================================

def validate_level1(yaml_str):
    """Level 1: YAML parse + FermiPy ConfigManager.load()."""
    result = {'level': 1, 'yaml_parse': False, 'fermipy_load': False,
              'sections': [], 'issues': [], 'error': None}

    if not yaml_str.strip():
        result['error'] = 'empty yaml'
        return result

    try:
        parsed = yaml.safe_load(yaml_str)
        if not isinstance(parsed, dict):
            result['error'] = f'yaml parsed to {type(parsed).__name__}'
            return result
        result['yaml_parse'] = True
        result['sections'] = list(parsed.keys())
    except yaml.YAMLError as e:
        result['error'] = f'yaml parse error: {str(e)[:200]}'
        return result

    from fermipy.config import ConfigManager
    tmp = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(yaml_str)
            tmp = f.name
        config = ConfigManager.load(tmp)
        if isinstance(config, dict):
            result['fermipy_load'] = True
    except Exception as e:
        result['error'] = f'ConfigManager.load failed: {str(e)[:200]}'
    finally:
        if tmp and os.path.exists(tmp):
            os.unlink(tmp)

    return result


def validate_level2(yaml_str):
    """Level 2: GTAnalysis instantiation (no data needed)."""
    result = {'level': 2, 'gta_init': False, 'error': None, 'config_keys': []}

    if not yaml_str.strip():
        result['error'] = 'empty yaml'
        return result

    work_dir = tempfile.mkdtemp(prefix='fermipy_val_')
    config_path = os.path.join(work_dir, 'config.yaml')

    try:
        parsed = yaml.safe_load(yaml_str)
        if not isinstance(parsed, dict):
            result['error'] = 'not a dict'
            return result

        # Override fileio to use our work dir
        if 'fileio' not in parsed:
            parsed['fileio'] = {}
        parsed['fileio']['outdir'] = os.path.join(work_dir, 'output')
        os.makedirs(parsed['fileio']['outdir'], exist_ok=True)

        with open(config_path, 'w') as f:
            yaml.dump(parsed, f, default_flow_style=False)

        from fermipy.gtanalysis import GTAnalysis
        gta = GTAnalysis(config_path)
        result['gta_init'] = True
        result['config_keys'] = list(gta.config.keys()) if hasattr(gta, 'config') else []
        del gta

    except Exception as e:
        result['error'] = f'{type(e).__name__}: {str(e)[:300]}'
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)

    return result


def validate_level3(yaml_str, evfile, scfile, work_dir=None):
    """Level 3: gta.setup() with actual Fermi-LAT data.

    When ``work_dir`` is given, run setup there and KEEP the directory
    (wiping any previous contents first): a later Level-4 run with the
    same config can then reuse the ltcube/srcmaps fermipy cached on disk
    instead of redoing setup — which matters when the data is months of
    weekly files rather than the one-week demo cutouts. Without
    ``work_dir`` a throwaway temp dir is used and removed, as before.
    """
    result = {'level': 3, 'setup_ok': False, 'error': None,
              'sources': [], 'traceback': ''}

    if not yaml_str.strip():
        result['error'] = 'empty yaml'
        return result

    if not os.path.exists(evfile) or not os.path.exists(scfile):
        result['error'] = (f'data files not found: evfile={os.path.exists(evfile)}, '
                           f'scfile={os.path.exists(scfile)}')
        return result

    owns_work_dir = work_dir is None
    if owns_work_dir:
        work_dir = tempfile.mkdtemp(prefix='fermipy_run_')
    else:
        # A fresh start per validation attempt: stale caches from a
        # different config in the same dir must never be reused.
        shutil.rmtree(work_dir, ignore_errors=True)
        os.makedirs(work_dir, exist_ok=True)
    config_path = os.path.join(work_dir, 'config.yaml')

    try:
        parsed = yaml.safe_load(yaml_str)
        if not isinstance(parsed, dict):
            result['error'] = 'not a dict'
            return result

        # Override fileio to use work dir
        if 'fileio' not in parsed:
            parsed['fileio'] = {}
        parsed['fileio']['outdir'] = os.path.join(work_dir, 'output')
        os.makedirs(parsed['fileio']['outdir'], exist_ok=True)

        # Ensure data paths point to real files
        if 'data' not in parsed:
            parsed['data'] = {}
        parsed['data']['evfile'] = evfile
        parsed['data']['scfile'] = scfile

        with open(config_path, 'w') as f:
            yaml.dump(parsed, f, default_flow_style=False)

        from fermipy.gtanalysis import GTAnalysis
        gta = GTAnalysis(config_path)
        gta.setup()
        result['setup_ok'] = True
        result['n_sources'] = len(gta.roi.sources) if hasattr(gta, 'roi') else 0
        result['sources'] = ([s.name for s in gta.roi.sources[:10]]
                             if hasattr(gta, 'roi') else [])
        del gta

    except Exception as e:
        result['error'] = f'{type(e).__name__}: {str(e)[:500]}'
        result['traceback'] = traceback.format_exc()[-500:]
    finally:
        if owns_work_dir:
            shutil.rmtree(work_dir, ignore_errors=True)

    return result


# ============================================================
# ITERATIVE VALIDATE-REPAIR LOOP
# ============================================================

def validate_with_iterative_repair(yaml_str, test_idx, max_iter=MAX_REPAIR_ITERATIONS,
                                   work_dir=None):
    """Iteratively repair and validate a config until Level 3 passes.

    Returns (final_yaml, all_results, repair_history).
    """
    meta = get_test_meta(test_idx)
    evfile = meta.get('evfile', '')
    scfile = meta.get('scfile', '')

    all_results = []
    repair_history = []
    current_yaml = yaml_str
    prev_error = None

    for iteration in range(max_iter):
        print(f"      Iteration {iteration + 1}/{max_iter}...")

        # Apply repair
        repaired_yaml, repairs = repair_config(
            current_yaml, test_idx,
            iteration=iteration, prev_error=prev_error
        )

        if not repaired_yaml:
            repair_history.append({
                'iteration': iteration, 'repairs': repairs,
                'result': 'repair failed - empty yaml'
            })
            print(f"        Repair failed: {repairs}")
            break

        if repairs:
            repair_history.append({
                'iteration': iteration, 'repairs': repairs,
            })
            print(f"        Applied {len(repairs)} repairs: {repairs[:3]}{'...' if len(repairs) > 3 else ''}")

        current_yaml = repaired_yaml

        # Validate Level 1
        l1 = validate_level1(current_yaml)
        if not l1['fermipy_load']:
            print(f"        L1 FAIL: {l1.get('error', 'unknown')[:80]}")
            prev_error = l1.get('error', '')
            all_results.append({'iteration': iteration, 'level1': l1})
            continue

        # Validate Level 2
        l2 = validate_level2(current_yaml)
        if not l2['gta_init']:
            print(f"        L2 FAIL: {l2.get('error', 'unknown')[:80]}")
            prev_error = l2.get('error', '')
            all_results.append({'iteration': iteration, 'level1': l1, 'level2': l2})
            continue

        # Validate Level 3
        l3 = validate_level3(current_yaml, evfile, scfile, work_dir=work_dir)
        all_results.append({
            'iteration': iteration, 'level1': l1, 'level2': l2, 'level3': l3
        })

        if l3['setup_ok']:
            print(f"        L3 SUCCESS! Sources: {l3.get('sources', [])[:5]}")
            return current_yaml, all_results, repair_history

        print(f"        L3 FAIL: {l3.get('error', 'unknown')[:100]}")
        prev_error = l3.get('error', '') + '\n' + l3.get('traceback', '')

    return current_yaml, all_results, repair_history


# ============================================================
# ENHANCED PROMPTING (execution-aware)
# ============================================================

FERMIPY_SCHEMA_ENHANCED = """FermiPy YAML Configuration Schema (STRICT - follow exactly):

REQUIRED sections:
- logging: {{verbosity: int (2-3), chatter: int (2-3)}}
- fileio: {{outdir: str (always "output"), logfile: str}}
- data: {{evfile: str, scfile: str, ltcube: str (optional, omit if not given)}}
- binning: {{roiwidth: float (typically 10), binsz: float (0.08-0.1), binsperdec: positive number (default 8)}}
- selection: {{emin: float (MeV), emax: float (MeV), zmax: float (90-105), target: str (EXACT 4FGL name), radius: float (15-20), tmin: float (MET seconds), tmax: float (MET seconds), evclass: int, evtype: int, filter: str or null}}
- gtlike: {{edisp: true, edisp_disable: ['isodiff'], irfs: str}}
- model: {{src_radius: float, src_roiwidth: float, galdiff: list, isodiff: list, catalogs: list}}

CRITICAL RULES:
1. evclass MUST be one of: 2, 4, 8, 16, 32, 64, 128, 256, 512 (128 = SOURCE class)
2. evtype MUST be one of: 1 (FRONT), 2 (BACK), 3 (FRONT+BACK), 32 (PSF3)
""" + diffuse_prompt_rules() + """
5. tmin/tmax MUST be in MET seconds (Mission Elapsed Time), NOT calendar dates
6. filter: use 'DATA_QUAL>0 && LAT_CONFIG==1' unless explicitly told otherwise
7. target MUST use the exact 4FGL catalog name (e.g., '4FGL J1104.4+3812')
8. Do NOT add keys not listed above to gtlike, binning, or model sections

""" + fermipy_binning_prompt_rules() + """

OPTIONAL analysis sections (include only if task requires):
- sed: {{free_background: bool, free_radius: float, ul_ts_threshold: float, make_plots: bool, write_fits: bool}}
- tsmap: {{make_plots: bool, write_fits: bool, write_npy: bool}}
- residmap: {{make_plots: bool, write_fits: bool}}
- psmap: {{make_plots: bool}}
- lightcurve: {{nbins: int, free_background: bool, free_radius: float, make_plots: bool}}

FermiPy Python API (use in this order):
  from fermipy.gtanalysis import GTAnalysis
  gta = GTAnalysis('config.yaml')
  gta.setup()
  gta.free_source('galdiff')
  gta.free_source('isodiff')
  gta.free_source(target)
  gta.free_sources(minmax_ts=[threshold, None], distance=d)
  gta.optimize()
  gta.fit()
  # Then analysis-specific calls: gta.sed(), gta.tsmap(), gta.residmap(), etc.
"""


def create_execution_aware_prompt(test_ex, test_idx, train_data, n_shots=3):
    """Create a prompt with environment-specific details and strict formatting."""
    parts = [
        "You are a FermiPy expert. Generate EXACTLY one YAML configuration and one Python script.",
        "",
        FERMIPY_SCHEMA_ENHANCED,
        "",
        "IMPORTANT: Output ONLY the YAML configuration and Python script in markdown code blocks.",
        "Do NOT output thinking, planning, or explanations.",
        "Format your response EXACTLY as:",
        "### YAML Configuration:",
        "```yaml",
        "<your yaml here>",
        "```",
        "",
        "### Python Script:",
        "```python",
        "<your python here>",
        "```",
    ]

    if n_shots > 0:
        parts.append("\n## Working Reference Examples:\n")
        for j, ex in enumerate(train_data[:n_shots]):
            parts.append(f"--- Example {j+1} ---")
            parts.append(f"Task: {ex['prompt'][:300]}...")
            yaml_str = ex['response']['yaml']
            yaml_str, _ = normalize_diffuse_yaml(yaml_str)
            script = ex['response']['script']
            if yaml_str.strip():
                parts.append(f"### YAML Configuration:\n```yaml\n{yaml_str}\n```")
            parts.append(f"### Python Script:\n```python\n{script}\n```\n")

    parts.append("--- YOUR TASK ---")
    parts.append(f"Task: {test_ex['prompt']}")
    parts.append("\nGenerate the YAML configuration and Python script now:")

    return "\n".join(parts)


def create_error_feedback_prompt(test_ex, generated_text, error_msg, iteration):
    """Create a refinement prompt incorporating actual execution errors."""
    parts = [
        "/no_think",
        "Your previous FermiPy configuration FAILED during execution with this error:",
        "",
        f"ERROR: {error_msg[:500]}",
        "",
        "Common causes and fixes:",
        "- NDSKEYS error: wrong isodiff file for the evtype; use the matching absolute path from the schema below",
        "- IndexError in ltcube: remove ltcube from data section if you don't have a real file",
        "- 'No source matching name': use the exact 4FGL catalog name and ensure catalog version is correct",
        "- 'None None': you must specify tmin and tmax as MET seconds in the selection section",
        "- Invalid keys: gtlike only accepts {edisp, edisp_disable, irfs, edisp_bins}",
        "",
        FERMIPY_SCHEMA_ENHANCED,
        "",
        f"Original task: {test_ex['prompt'][:500]}",
        "",
        "Your previous output (BROKEN):",
        generated_text[:2000],
        "",
        "Generate a CORRECTED version. Output ONLY the YAML and Python:",
        "### YAML Configuration:",
        "```yaml",
    ]
    return "\n".join(parts)


# ============================================================
# GENERATION HELPERS
# ============================================================

def generate_text(model, tokenizer, prompt, max_new_tokens=2048,
                  max_input_tokens=12288, temperature=0.1, top_p=0.95):
    """Generate text from a prompt using the model.

    Default max_new_tokens reduced from 4096 to 2048 — successful FermiPy
    generations stay well under 2k tokens; the old ceiling let verbose
    models (e.g. Llama-3.1-8B-Instruct) ramble past the answer and burn
    hours of GPU time on tokens that get truncated anyway.
    """
    import torch

    # For instruction-tuned models with a chat template, wrap the prompt
    if hasattr(tokenizer, 'chat_template') and tokenizer.chat_template:
        model_name = getattr(model, 'name_or_path', '') or getattr(model.config, '_name_or_path', '')
        is_chat_model = any(k in model_name.lower() for k in
                            ('gemma-4', 'gemma4', 'gemma-3', '-it', '-instruct'))
        if is_chat_model:
            msgs = [{'role': 'user', 'content': prompt}]
            formatted = tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True)
            input_ids = tokenizer.encode(formatted, return_tensors='pt')
        else:
            input_ids = tokenizer.encode(prompt, return_tensors='pt')
    else:
        input_ids = tokenizer.encode(prompt, return_tensors='pt')

    if input_ids.shape[1] > max_input_tokens:
        input_ids = input_ids[:, :max_input_tokens]
    input_ids = input_ids.to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
        )

    new_tokens = output_ids[0][input_ids.shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True), len(new_tokens)


def validate_output_basic(generated_text):
    """Basic validation of generated text (YAML parseable + Python syntax)."""
    issues = []

    yaml_str = extract_yaml(generated_text)
    if yaml_str:
        try:
            parsed = yaml.safe_load(yaml_str)
            if isinstance(parsed, dict):
                for key in ['selection', 'model']:
                    if key not in parsed:
                        issues.append(f"YAML missing required section: {key}")
            else:
                issues.append("YAML did not parse as dict")
        except yaml.YAMLError as e:
            issues.append(f"YAML parse error: {str(e)[:100]}")
    else:
        issues.append("No YAML block found")

    script = extract_python(generated_text)
    if script:
        try:
            ast.parse(script)
        except SyntaxError as e:
            issues.append(f"Python syntax error: {str(e)[:100]}")
        if 'GTAnalysis' not in script:
            issues.append("Missing GTAnalysis")
        if 'gta.setup()' not in script:
            issues.append("Missing gta.setup()")
    else:
        issues.append("No Python block found")

    return issues


# ============================================================
# FULL PIPELINE: Generate → Repair → Validate → Iterate
# ============================================================

def run_full_pipeline(model, tokenizer, test_ex, test_idx, train_data,
                      model_config, max_regen=3):
    """Full generate-repair-validate pipeline with re-generation on failure.

    Steps:
    1. Generate initial output with enhanced prompt
    2. Extract YAML and apply programmatic repairs
    3. Run iterative validate-repair loop (up to MAX_REPAIR_ITERATIONS)
    4. If still failing, regenerate with error feedback and repeat
    5. Also try best-of-N sampling with different temperatures

    Returns dict with final results.
    """
    max_input = model_config['max_input_tokens']
    meta = get_test_meta(test_idx)

    all_attempts = []
    best_yaml = ''
    best_result = None
    success = False

    for regen in range(max_regen):
        print(f"    Generation attempt {regen + 1}/{max_regen}...")

        if regen == 0:
            # First attempt: use enhanced prompt
            prompt = create_execution_aware_prompt(test_ex, test_idx, train_data, n_shots=3)
            text, n_tok = generate_text(model, tokenizer, prompt,
                                        max_input_tokens=max_input)
        else:
            # Subsequent attempts: use error feedback prompt
            prev_error = ''
            if all_attempts:
                last = all_attempts[-1]
                if last.get('validation_results'):
                    last_result = last['validation_results'][-1]
                    l3 = last_result.get('level3', {})
                    prev_error = l3.get('error', '') + '\n' + l3.get('traceback', '')
                    if not prev_error.strip():
                        l2 = last_result.get('level2', {})
                        prev_error = l2.get('error', '')

            prompt = create_error_feedback_prompt(
                test_ex, text, prev_error, regen)
            # Slightly higher temperature for diversity
            text, n_tok = generate_text(model, tokenizer, prompt,
                                        temperature=0.1 + regen * 0.1,
                                        max_input_tokens=max_input)

        # Extract YAML
        gen_yaml = extract_yaml(text)
        gen_py = extract_python(text)

        if not gen_yaml:
            print(f"      No YAML extracted from generation")
            all_attempts.append({
                'regen': regen, 'has_yaml': False, 'has_python': bool(gen_py),
                'output_tokens': n_tok, 'validation_results': [],
            })
            continue

        # Run iterative repair + validation
        final_yaml, val_results, repair_hist = validate_with_iterative_repair(
            gen_yaml, test_idx
        )

        attempt = {
            'regen': regen,
            'has_yaml': bool(gen_yaml),
            'has_python': bool(gen_py),
            'output_tokens': n_tok,
            'validation_results': val_results,
            'repair_history': repair_hist,
            'n_repair_iterations': len(val_results),
        }

        # Check if Level 3 passed
        l3_passed = any(
            r.get('level3', {}).get('setup_ok', False)
            for r in val_results
        )

        if l3_passed:
            attempt['success'] = True
            best_yaml = final_yaml
            best_result = attempt
            success = True
            all_attempts.append(attempt)
            print(f"      SUCCESS on attempt {regen + 1}!")
            break
        else:
            attempt['success'] = False
            all_attempts.append(attempt)

        # Store the best so far (furthest validation level reached)
        if not best_result or (val_results and
                               any(r.get('level2', {}).get('gta_init') for r in val_results)):
            best_yaml = final_yaml
            best_result = attempt

    # If still no success, try best-of-3 with different temperatures
    if not success:
        print(f"    Trying Best-of-3 sampling...")
        prompt = create_execution_aware_prompt(test_ex, test_idx, train_data, n_shots=5)
        candidates = []
        for temp_idx, temp in enumerate([0.1, 0.3, 0.5]):
            text, n_tok = generate_text(model, tokenizer, prompt,
                                        temperature=temp,
                                        max_input_tokens=max_input)
            gen_yaml = extract_yaml(text)
            if gen_yaml:
                final_yaml, val_results, repair_hist = validate_with_iterative_repair(
                    gen_yaml, test_idx
                )
                l3_passed = any(
                    r.get('level3', {}).get('setup_ok', False)
                    for r in val_results
                )
                candidates.append({
                    'temp': temp, 'yaml': final_yaml,
                    'val_results': val_results, 'success': l3_passed,
                })
                if l3_passed:
                    print(f"      Best-of-3 SUCCESS at temp={temp}!")
                    success = True
                    best_yaml = final_yaml
                    best_result = {
                        'regen': 'best_of_3', 'success': True,
                        'temperature': temp,
                        'validation_results': val_results,
                        'repair_history': repair_hist,
                    }
                    break

        if not success and candidates:
            # Pick candidate that got furthest
            best_cand = max(candidates, key=lambda c: (
                c['success'],
                any(r.get('level2', {}).get('gta_init') for r in c['val_results']),
                any(r.get('level1', {}).get('fermipy_load') for r in c['val_results']),
            ))
            best_yaml = best_cand['yaml']

        all_attempts.append({
            'regen': 'best_of_3',
            'candidates': [
                {'temp': c['temp'], 'success': c['success']}
                for c in candidates
            ] if candidates else [],
        })

    return {
        'test_idx': test_idx,
        'target': meta.get('target', ''),
        'success': success,
        'final_yaml': best_yaml,
        'total_attempts': len(all_attempts),
        'attempts': all_attempts,
        'best_result': best_result,
    }


# ============================================================
# RE-VALIDATE EXISTING OUTPUTS (no model needed)
# ============================================================

def revalidate_existing_outputs():
    """Re-validate all existing generated outputs through the repair engine.

    This fixes environment issues without needing to regenerate.
    """
    print("\n" + "=" * 60)
    print("RE-VALIDATING EXISTING OUTPUTS WITH REPAIR ENGINE")
    print("=" * 60)

    all_revalidation = {}

    for model_name in ['Qwen3.5-0.8B', 'Qwen3.5-27B', 'Qwen3.5-35B-A3B']:
        fname = os.path.join(RESULTS_DIR, f'improved_{model_name}.json')
        if not os.path.exists(fname):
            print(f"\n  {model_name}: no improved results file, skipping")
            continue

        with open(fname) as f:
            data = json.load(f)

        model_results = {}
        print(f"\n{'=' * 50}")
        print(f"Model: {model_name}")
        print(f"{'=' * 50}")

        for approach, outputs in data['results'].items():
            if not isinstance(outputs, list):
                continue

            approach_results = []
            for out in outputs:
                text = out.get('generated_text', '')
                test_idx = out.get('test_idx', -1)

                gen_yaml = extract_yaml(text)
                gen_py = extract_python(text)

                if not gen_yaml:
                    approach_results.append({
                        'test_idx': test_idx,
                        'has_yaml': False,
                        'has_python': bool(gen_py),
                        'success': False,
                        'error': 'no yaml extracted',
                    })
                    print(f"  {approach} test_{test_idx}: NO YAML")
                    continue

                # Run through iterative repair + validation
                print(f"  {approach} test_{test_idx}:")
                final_yaml, val_results, repair_hist = validate_with_iterative_repair(
                    gen_yaml, test_idx
                )

                l3_passed = any(
                    r.get('level3', {}).get('setup_ok', False)
                    for r in val_results
                )

                result = {
                    'test_idx': test_idx,
                    'has_yaml': True,
                    'has_python': bool(gen_py),
                    'success': l3_passed,
                    'validation_results': val_results,
                    'repair_history': repair_hist,
                    'n_iterations': len(val_results),
                }

                status = 'SUCCESS' if l3_passed else 'FAIL'
                last_error = ''
                if val_results and not l3_passed:
                    last = val_results[-1]
                    l3 = last.get('level3', {})
                    last_error = l3.get('error', '')[:80] if l3 else ''
                    if not last_error:
                        l2 = last.get('level2', {})
                        last_error = l2.get('error', '')[:80] if l2 else ''
                print(f"    → {status} (iterations={len(val_results)})"
                      f"{f' | {last_error}' if last_error else ''}")

                approach_results.append(result)

            model_results[approach] = approach_results
        all_revalidation[model_name] = model_results

    # Save results
    out_path = os.path.join(RESULTS_DIR, 'execution_validated.json')
    with open(out_path, 'w') as f:
        json.dump(all_revalidation, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")

    # Print summary
    print_summary(all_revalidation)
    return all_revalidation


# ============================================================
# NEW GENERATION WITH MODELS
# ============================================================

# Model configurations
MODELS = {
    'Qwen3.5-0.8B': {
        'path': f'{CACHE_DIR}/hub/models--Qwen--Qwen3.5-0.8B/snapshots/2fc06364715b967f1860aea9cf38778875588b17',
        'quantize': False,
        'device_map': 'auto',
        'max_input_tokens': 12288,
    },
    'Qwen3.5-27B': {
        'path': f'{CACHE_DIR}/models--Qwen--Qwen3.5-27B/snapshots/b7ca741b86de18df552fd2cc952861e04621a4bd',
        'quantize': False,
        'device_map': 'auto',
        'max_input_tokens': 10240,
    },
    'Qwen3.5-35B-A3B': {
        'path': f'{CACHE_DIR}/models--Qwen--Qwen3.5-35B-A3B/snapshots/ec2d4ece1ffb563322cbee9a48fe0e3fcbce0307',
        'quantize': False,
        'device_map': 'auto',
        'max_input_tokens': 10240,
    },
    'Gemma-4-31B-it': {
        'path': f'{CACHE_DIR}/hub/models--google--gemma-4-31B-it/snapshots/419b2efe421994fdfd3394e621983d4cc511cd4f',
        'quantize': False,
        'device_map': 'auto',
        'max_input_tokens': 8192,
    },
}


def run_model_experiments(model_name):
    """Run the full pipeline for a single model."""
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

    model_config = MODELS[model_name]

    print(f'\n{"=" * 60}')
    print(f'Loading model: {model_name}')
    print(f'{"=" * 60}')

    tokenizer = AutoTokenizer.from_pretrained(
        model_config['path'], trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = {
        'trust_remote_code': True,
        'device_map': model_config['device_map'],
    }
    if model_config.get('quantize'):
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type='nf4',
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model_kwargs['quantization_config'] = bnb_config
    else:
        model_kwargs['torch_dtype'] = torch.bfloat16

    start = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        model_config['path'], **model_kwargs)
    model.eval()
    load_time = time.time() - start
    print(f'Loaded in {load_time:.1f}s')

    # Load training data
    with open(os.path.join(DATA_DIR, 'train.json')) as f:
        train_data = json.load(f)
    with open(os.path.join(DATA_DIR, 'test.json')) as f:
        test_data = json.load(f)

    results = []
    for i, test_ex in enumerate(test_data):
        print(f'\n  Test {i} ({TEST_DATA_MAP[i]["name"]}):')
        result = run_full_pipeline(
            model, tokenizer, test_ex, i, train_data, model_config
        )
        results.append(result)

    # Cleanup
    del model
    gc.collect()
    torch.cuda.empty_cache()

    return {
        'model': model_name,
        'load_time': load_time,
        'results': results,
    }


# ============================================================
# SUMMARY AND REPORTING
# ============================================================

def print_summary(all_results):
    """Print a summary table of results."""
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    total_tests = 0
    total_success = 0

    for model_name, model_results in all_results.items():
        print(f"\n{model_name}:")
        model_total = 0
        model_success = 0

        for approach, results in model_results.items():
            if not isinstance(results, list):
                continue
            n = len(results)
            n_ok = sum(1 for r in results if r.get('success', False))
            model_total += n
            model_success += n_ok
            print(f"  {approach:30s}: {n_ok}/{n} Level 3 passed")

        total_tests += model_total
        total_success += model_success
        if model_total > 0:
            pct = 100 * model_success / model_total
            print(f"  {'TOTAL':30s}: {model_success}/{model_total} ({pct:.1f}%)")

    if total_tests > 0:
        pct = 100 * total_success / total_tests
        print(f"\nOVERALL: {total_success}/{total_tests} ({pct:.1f}%)")


def print_generation_summary(all_results):
    """Print summary for generation experiments."""
    print("\n" + "=" * 60)
    print("GENERATION EXPERIMENT SUMMARY")
    print("=" * 60)

    for model_result in all_results:
        model_name = model_result['model']
        results = model_result['results']
        n_ok = sum(1 for r in results if r.get('success', False))
        n = len(results)
        print(f"\n{model_name}: {n_ok}/{n} Level 3 passed")
        for r in results:
            status = 'SUCCESS' if r['success'] else 'FAIL'
            target = r.get('target', f"test_{r['test_idx']}")
            n_attempts = r.get('total_attempts', 0)
            print(f"  {target}: {status} ({n_attempts} attempt(s))")


# ============================================================
# MAIN
# ============================================================

def main():
    diagnose_environment()

    mode = 'revalidate'  # default
    selected_models = []

    for arg in sys.argv[1:]:
        if arg == '--revalidate':
            mode = 'revalidate'
        elif arg == '--generate':
            mode = 'generate'
        elif arg == '--both':
            mode = 'both'
        elif arg in MODELS:
            selected_models.append(arg)
            mode = 'generate'  # if model specified, assume generate

    if mode in ('revalidate', 'both'):
        revalidation_results = revalidate_existing_outputs()

    if mode in ('generate', 'both'):
        if not selected_models:
            selected_models = list(MODELS.keys())

        all_gen_results = []
        for model_name in selected_models:
            if model_name not in MODELS:
                print(f"Unknown model: {model_name}")
                continue
            try:
                result = run_model_experiments(model_name)
                all_gen_results.append(result)

                # Save per-model results
                out_path = os.path.join(
                    RESULTS_DIR, f'exec_validated_{model_name}.json')
                with open(out_path, 'w') as f:
                    json.dump(result, f, indent=2, default=str)
                print(f'\nResults saved to {out_path}')

            except Exception as e:
                print(f'ERROR with {model_name}: {e}')
                traceback.print_exc()
                all_gen_results.append({
                    'model': model_name, 'error': str(e)
                })
                gc.collect()
                if 'torch' in sys.modules:
                    import torch
                    torch.cuda.empty_cache()

        # Save combined results
        out_path = os.path.join(RESULTS_DIR, 'exec_validated_all.json')
        with open(out_path, 'w') as f:
            json.dump(all_gen_results, f, indent=2, default=str)

        print_generation_summary(all_gen_results)

    print('\n' + '=' * 60)
    print('Execution-validated experiments complete!')
    print('=' * 60)


if __name__ == '__main__':
    main()

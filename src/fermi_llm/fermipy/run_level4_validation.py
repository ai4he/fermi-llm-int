#!/usr/bin/env python3
"""
Level 4 Validation: Full Likelihood Fitting of LLM-Generated FermiPy Configs.

Extends the Level 1-3 validation infrastructure from run_execution_validated.py
with Level 4: gta.optimize() + gta.fit() execution, convergence checks, and
science-quality metrics (TS, flux, spectral index).

Validation Levels:
  Level 1: YAML parse + FermiPy ConfigManager.load()
  Level 2: GTAnalysis instantiation
  Level 3: gta.setup() execution
  Level 4: gta.optimize() + gta.fit() + science quality checks  <-- NEW

Run with the fermi conda env:
    conda run -n fermi python run_level4_validation.py [--revalidate] [--timeout SECONDS]

Typical runtime: 5-30 minutes per source (3 sources x N configs).
"""

import json
import hashlib
import os
import runpy
import sys
import re
import yaml
import tempfile
import shutil
import traceback
import time
import math
import signal
import gc
import logging
from pathlib import Path
from datetime import datetime

# ============================================================
# IMPORT Level 1-3 infrastructure from run_execution_validated.py
# ============================================================

sys.path.insert(0, '/scratch/ctoxtli/fermi')
from .run_execution_validated import (
    PROJECT_DIR,
    RESULTS_DIR,
    LAT_DATA_DIR,
    DIFFUSE_DIR,
    TEST_DATA_MAP,
    get_test_meta,
    SOURCE_NAME_CORRECTIONS,
    extract_yaml,
    extract_python,
    repair_config,
    validate_level1,
    validate_level2,
    validate_level3,
    validate_with_iterative_repair,
    get_fits_time_range,
    get_correct_isodiff,
    get_galdiff,
    diagnose_environment,
)

os.makedirs(RESULTS_DIR, exist_ok=True)

# ============================================================
# 4FGL CATALOG REFERENCE VALUES
# ============================================================
# Approximate values from the 4FGL-DR3 catalog for our three test sources.
# Used to sanity-check fit results (within order of magnitude).
#
# Flux is the integral photon flux in ph/cm2/s (100 MeV - 1 TeV from 4FGL).
# TS is the catalog test statistic.
# Spectral index is the photon index (PowerLaw) or the low-energy index.

CATALOG_REFERENCE = {
    '4FGL J1104.4+3812': {  # Mrk 421
        'name': 'Mrk 421',
        'flux_100mev_1tev': 1.1e-8,      # ~1.1e-8 ph/cm2/s
        'flux_tolerance_log10': 1.5,      # accept if within 1.5 orders of magnitude
        'expected_ts_min': 100,           # bright blazar, TS >> 25
        'expected_ts_max': 1e7,
        'expected_index_min': 1.5,        # photon index typically 1.7-1.9
        'expected_index_max': 3.0,
        'catalog_index': 1.78,
        'spectral_type': 'LogParabola',
    },
    '4FGL J0835.3-4510': {  # Vela Pulsar
        'name': 'Vela',
        'flux_100mev_1tev': 1.1e-5,      # ~1.1e-5 ph/cm2/s (very bright)
        'flux_tolerance_log10': 1.5,
        'expected_ts_min': 1000,          # extremely bright, TS >> 25
        'expected_ts_max': 1e8,
        'expected_index_min': 1.0,        # pulsar index typically ~1.5-2.5
        'expected_index_max': 3.5,
        'catalog_index': 1.37,            # hard spectrum with exponential cutoff
        'spectral_type': 'PLSuperExpCutoff4',
    },
    '4FGL J0534.5+2200': {  # Crab Pulsar
        'name': 'Crab',
        'flux_100mev_1tev': 9.1e-7,      # ~9.1e-7 ph/cm2/s
        'flux_tolerance_log10': 1.5,
        'expected_ts_min': 100,           # bright source, TS >> 25
        'expected_ts_max': 1e8,
        'expected_index_min': 1.0,        # Crab has complex spectrum
        'expected_index_max': 3.5,
        'catalog_index': 1.69,
        'spectral_type': 'PLSuperExpCutoff4',
    },
}

# Default timeout for fit operations (seconds)
DEFAULT_FIT_TIMEOUT = 1800  # 30 minutes per source


# ============================================================
# TIMEOUT HANDLER
# ============================================================

class FitTimeoutError(Exception):
    """Raised when a fit operation exceeds the allowed time."""
    pass


def _timeout_handler(signum, frame):
    raise FitTimeoutError("Fit operation timed out")


class TimeoutContext:
    """Context manager for timeout protection using SIGALRM.

    Falls back to no-op on platforms that do not support SIGALRM (Windows).
    """

    def __init__(self, seconds, label="operation"):
        self.seconds = seconds
        self.label = label
        self._supported = hasattr(signal, 'SIGALRM')
        self._old_handler = None

    def __enter__(self):
        if self._supported and self.seconds > 0:
            self._old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
            signal.alarm(self.seconds)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._supported and self.seconds > 0:
            signal.alarm(0)  # cancel the alarm
            if self._old_handler is not None:
                signal.signal(signal.SIGALRM, self._old_handler)
        # Do not suppress exceptions
        return False


# ============================================================
# OUTPUT ARTIFACT GENERATION (plots + data for the web UI)
# ============================================================

# Per-product time budgets (seconds). These are deliberately modest so that
# producing visualisations never dominates total runtime; each is guarded by
# its own TimeoutContext and try/except and can never fail the pipeline.
SED_TIMEOUT = 600
TSMAP_TIMEOUT = 600
RESIDMAP_TIMEOUT = 400
LIGHTCURVE_TIMEOUT = 900

# How a generated PNG basename maps to a human-readable caption/kind for the UI.
_ARTIFACT_CAPTIONS = [
    ('_sedlnl', 'sed_lnl', 'SED likelihood profiles (per energy bin)'),
    ('_sed.png', 'sed', 'Spectral Energy Distribution (SED)'),
    ('_tsmap', 'tsmap', 'Test-Statistic (TS) map'),
    ('_residmap_data', 'counts', 'Counts map (observed, smoothed)'),
    ('_residmap_model', 'model', 'Model map'),
    ('_residmap_excess', 'residmap', 'Excess counts map'),
    ('_residmap', 'residmap', 'Residual significance map'),
    ('_lightcurve', 'lightcurve', 'Light curve'),
    ('_pointsource_powerlaw', 'tsmap', 'TS map (point-source, power-law)'),
    ('counts', 'counts', 'Counts map'),
    ('model', 'model', 'Model map'),
]


def _plot_lightcurve(lc, target_name, outdir):
    """Render the light-curve PNG ourselves.

    fermipy accepts make_plots=True on gta.lightcurve() but ships no
    light-curve plotter, so without this nothing reaches the artifact
    collector and the UI shows an empty result for an LC request.
    Returns the written path, or None when the LC dict is unusable.
    """
    import numpy as np
    import matplotlib
    matplotlib.use('Agg', force=False)
    import matplotlib.pyplot as plt

    if not isinstance(lc, dict):
        return None
    tmin = np.atleast_1d(np.asarray(lc.get('tmin_mjd', []), dtype=float))
    tmax = np.atleast_1d(np.asarray(lc.get('tmax_mjd', []), dtype=float))
    flux = np.atleast_1d(np.asarray(lc.get('flux', []), dtype=float))
    if flux.size == 0 or tmin.size != flux.size:
        return None
    flux_err = np.atleast_1d(np.asarray(
        lc.get('flux_err', np.zeros_like(flux)), dtype=float))
    ts = np.atleast_1d(np.asarray(
        lc.get('ts', np.zeros_like(flux)), dtype=float))
    ul = np.atleast_1d(np.asarray(
        lc.get('flux_ul95', np.full_like(flux, np.nan)), dtype=float))

    tc = 0.5 * (tmin + tmax)
    terr = 0.5 * (tmax - tmin)
    # Low-significance bins are shown as 95% upper limits.
    is_ul = (ts < 9) & np.isfinite(ul)
    det = ~is_ul & np.isfinite(flux) & (flux > 0)

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    if det.any():
        ax.errorbar(tc[det], flux[det], xerr=terr[det], yerr=flux_err[det],
                    fmt='o', ms=4, capsize=2, color='#4f46e5',
                    label='detected (TS ≥ 9)')
    if is_ul.any():
        ax.errorbar(tc[is_ul], ul[is_ul], xerr=terr[is_ul],
                    yerr=0.25 * ul[is_ul], uplims=True, fmt='none',
                    color='#9ca3af', alpha=0.9,
                    label='95% upper limit (TS < 9)')
    ax.set_xlabel('Time (MJD)')
    ax.set_ylabel(r'Photon flux (ph cm$^{-2}$ s$^{-1}$)')
    ax.set_title(f'{target_name} light curve')
    ax.set_yscale('log')
    ax.grid(alpha=0.3)
    if det.any() or is_ul.any():
        ax.legend(loc='best', fontsize=8)
    fig.tight_layout()
    out = os.path.join(
        outdir,
        target_name.lower().replace(' ', '_').replace('/', '_')
        + '_lightcurve.png')
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out


def _classify_artifact(fname):
    """Return (kind, caption) for a generated PNG filename."""
    low = fname.lower()
    for needle, kind, caption in _ARTIFACT_CAPTIONS:
        if needle in low:
            return kind, caption
    return 'plot', fname


def _to_float_list(arr):
    """Best-effort convert a numpy array / list to a JSON-safe list of floats."""
    out = []
    try:
        for v in list(arr):
            try:
                fv = float(v)
                if fv != fv or fv in (float('inf'), float('-inf')):
                    out.append(None)
                else:
                    out.append(fv)
            except (TypeError, ValueError):
                out.append(None)
    except TypeError:
        return []
    return out


def _extract_sed_data(sed):
    """Pull JSON-serialisable SED points out of fermipy's gta.sed() return dict.

    Returns a dict with energy centres (MeV), e^2 dN/dE flux points and errors,
    per-bin TS, and upper-limit values/flags, or None if nothing usable.
    """
    if not isinstance(sed, dict):
        return None
    try:
        e_ctr = sed.get('e_ctr')
        if e_ctr is None and 'e_min' in sed and 'e_max' in sed:
            import numpy as _np
            e_ctr = _np.sqrt(_np.asarray(sed['e_min']) * _np.asarray(sed['e_max']))
        data = {
            'e_ctr': _to_float_list(e_ctr),
            'e2dnde': _to_float_list(sed.get('e2dnde')),
            'e2dnde_err': _to_float_list(sed.get('e2dnde_err')),
            'e2dnde_ul95': _to_float_list(sed.get('e2dnde_ul95')),
            'ts': _to_float_list(sed.get('ts')),
        }
        # Upper-limit flag per bin: low-TS bins are plotted as ULs.
        ts = data['ts']
        data['is_ul'] = [(t is None or t < 4.0) for t in ts] if ts else []
        if not data['e_ctr'] or not any(v is not None for v in data['e2dnde']):
            return None
        return data
    except Exception:
        return None


def _build_roi_source_list(gta):
    """Build a JSON-serialisable list of every source in the fitted ROI.

    Each entry has: name, TS, npred, integral flux (ph/cm2/s), spectral
    model type, and angular offset from the ROI center (degrees) where
    available. Sorted by descending TS.
    """
    sources = []
    try:
        center = gta.roi.skydir
    except Exception:
        center = None

    for src in gta.roi.sources:
        entry = {
            'name': src.name, 'ts': None, 'npred': None,
            'flux': None, 'flux_unit': 'ph/cm2/s',
            'spectrum_type': None, 'offset_deg': None,
        }
        try:
            sm = gta.get_src_model(src.name)
        except Exception:
            sm = {}

        ts_val = sm.get('ts')
        try:
            entry['ts'] = float(ts_val) if ts_val is not None else None
        except (TypeError, ValueError):
            entry['ts'] = None

        npred_val = sm.get('npred')
        try:
            entry['npred'] = float(npred_val) if npred_val is not None else None
        except (TypeError, ValueError):
            entry['npred'] = None

        flux_val = sm.get('flux')
        if isinstance(flux_val, (list, tuple)):
            flux_val = flux_val[0] if flux_val else None
        try:
            entry['flux'] = float(flux_val) if flux_val is not None else None
        except (TypeError, ValueError):
            entry['flux'] = None

        entry['spectrum_type'] = sm.get('SpectrumType') or sm.get('spectrum_type')

        # Prefer fermipy's own precomputed offset; fall back to a manual
        # angular separation from the ROI center.
        offset = sm.get('offset')
        if offset is None and center is not None:
            try:
                offset = center.separation(src.skydir).deg
            except Exception:
                offset = None
        try:
            entry['offset_deg'] = float(offset) if offset is not None else None
        except (TypeError, ValueError):
            entry['offset_deg'] = None

        sources.append(entry)

    sources.sort(key=lambda s: (s['ts'] if s['ts'] is not None else -1), reverse=True)
    return sources


def _generate_output_artifacts(gta, target_name, parsed_config, outdir,
                               artifact_dir, result, product_results=None,
                               collect_root=None, since=None):
    """Generate analysis-product plots and copy them into ``artifact_dir``.

    Runs after a successful fit. Produces an SED by default (the headline
    product for spectral analyses) and TS/residual/light-curve maps when the
    generated config requested them. Every step is independently guarded so a
    failure here can never affect the validation verdict.

    Populates ``result['artifacts']`` (list of {filename, kind, caption}) and
    ``result['sed_data']`` (JSON SED points for an interactive plot).
    """
    artifacts = []
    result.setdefault('artifacts', [])
    result.setdefault('sed_data', None)

    try:
        os.environ.setdefault('MPLBACKEND', 'Agg')
        import matplotlib
        try:
            matplotlib.use('Agg', force=True)
        except Exception:
            pass
        os.makedirs(artifact_dir, exist_ok=True)
    except Exception as e:
        result['artifact_error'] = f'setup failed: {type(e).__name__}: {e}'
        return

    cfg = parsed_config if isinstance(parsed_config, dict) else {}
    scripted_products = product_results is not None
    product_results = product_results or {}

    # Record which products the config actually requested, so the UI can show
    # exactly what was asked for and never display an unrequested plot.
    if scripted_products:
        # A reviewed script decides its own products: report what it ran.
        ran = set(product_results) | set(result.get('product_errors') or {})
        requested = [k for k in ('sed', 'tsmap', 'residmap', 'lightcurve',
                                 'psmap', 'tscube', 'localize', 'extension',
                                 'find_sources', 'curvature') if k in ran]
    else:
        requested = [k for k in ('sed', 'tsmap', 'residmap', 'lightcurve',
                                 'psmap') if k in cfg]
    result['requested_products'] = requested

    # ---- ROI source list (name, TS, npred, flux, spectral model, offset) ----
    # Full list goes to a JSON artifact next to the plots; the run result
    # JSON only carries the 25 highest-TS sources to keep it small.
    try:
        roi_sources = _build_roi_source_list(gta)
        result['roi_sources'] = roi_sources[:25]
        with open(os.path.join(artifact_dir, 'roi_sources.json'), 'w') as f:
            json.dump(roi_sources, f, indent=2, default=str)
        print(f'        [L4] wrote roi_sources.json ({len(roi_sources)} sources)')
    except Exception as e:
        result['roi_sources_error'] = f'{type(e).__name__}: {str(e)[:200]}'

    # ---- SED (only when the config requests it) ----
    # Previously this ran unconditionally, which made an SED plot appear even
    # for analyses that never asked for one. Now it is gated like every other
    # product on the presence of a 'sed' section in the (repaired) YAML.
    if ('sed' in requested) if scripted_products else ('sed' in cfg):
        try:
            if scripted_products:
                sed = product_results.get('sed')
            else:
                with TimeoutContext(SED_TIMEOUT, label='sed'):
                    sed = gta.sed(target_name, make_plots=True)
                print('        [L4] gta.sed() completed')
            if sed is not None:
                result['sed_data'] = _extract_sed_data(sed)
        except Exception as e:
            print(f'        [L4] sed generation skipped: {type(e).__name__}: {e}')

    # ---- TS map (only if requested in the config) ----
    if 'tsmap' in cfg and not scripted_products:
        try:
            with TimeoutContext(TSMAP_TIMEOUT, label='tsmap'):
                gta.tsmap('fit_ts', make_plots=True)
            print('        [L4] gta.tsmap() completed')
        except Exception as e:
            print(f'        [L4] tsmap skipped: {type(e).__name__}: {e}')

    # ---- Residual map (only if requested) ----
    if 'residmap' in cfg and not scripted_products:
        try:
            with TimeoutContext(RESIDMAP_TIMEOUT, label='residmap'):
                gta.residmap('fit_resid', make_plots=True)
            print('        [L4] gta.residmap() completed')
        except Exception as e:
            print(f'        [L4] residmap skipped: {type(e).__name__}: {e}')

    # ---- Light curve (only if requested; can be slow) ----
    if ('lightcurve' in requested) if scripted_products else ('lightcurve' in cfg):
        try:
            if scripted_products:
                lc = product_results.get('lightcurve')
                if lc is not None:
                    _plot_lightcurve(lc, target_name, outdir)
                raise StopIteration
            lc_cfg = cfg.get('lightcurve') or {}
            if not isinstance(lc_cfg, dict):
                lc_cfg = {}
            # An explicit bin width (binsz, seconds) wins over a bin count;
            # only one of the two is passed so fermipy never has to arbitrate.
            if lc_cfg.get('binsz'):
                lc_kw = {'binsz': float(lc_cfg['binsz'])}
            else:
                lc_kw = {'nbins': int(lc_cfg.get('nbins', 6))}
            print(f'        [L4] light-curve binning: {lc_kw}')
            with TimeoutContext(LIGHTCURVE_TIMEOUT, label='lightcurve'):
                lc = gta.lightcurve(target_name, make_plots=True, **lc_kw)
            print('        [L4] gta.lightcurve() completed')
            # fermipy has no LC plotter of its own -- render the PNG here so
            # the artifact collector (and thus the UI) actually shows it.
            try:
                if _plot_lightcurve(lc, target_name, outdir):
                    print('        [L4] light-curve plot rendered')
            except Exception as e:
                print(f'        [L4] light-curve plot failed: '
                      f'{type(e).__name__}: {e}')
        except StopIteration:
            pass
        except Exception as e:
            print(f'        [L4] lightcurve skipped: {type(e).__name__}: {e}')

    # ---- Point-source (TS) map via psmap (only if requested) ----
    if 'psmap' in cfg and not scripted_products:
        try:
            with TimeoutContext(TSMAP_TIMEOUT, label='psmap'):
                gta.psmap(make_plots=True)
            print('        [L4] gta.psmap() completed')
        except Exception as e:
            print(f'        [L4] psmap skipped: {type(e).__name__}: {e}')

    # ---- Collect every PNG fermipy produced and copy it out ----
    try:
        seen = set()
        for root, _dirs, files in os.walk(collect_root or outdir):
            for fn in files:
                if not fn.lower().endswith('.png'):
                    continue
                if fn in seen:
                    continue
                src = os.path.join(root, fn)
                if since is not None:
                    try:
                        if os.path.getmtime(src) < since:
                            continue
                    except OSError:
                        continue
                seen.add(fn)
                dst = os.path.join(artifact_dir, fn)
                try:
                    shutil.copyfile(src, dst)
                    kind, caption = _classify_artifact(fn)
                    artifacts.append({
                        'filename': fn,
                        'kind': kind,
                        'caption': caption,
                    })
                except Exception:
                    pass
        # Stable, predictable ordering: SED first, then maps, then the rest.
        order = {'sed': 0, 'tsmap': 1, 'residmap': 2, 'lightcurve': 3,
                 'counts': 4, 'model': 5, 'plot': 9}
        artifacts.sort(key=lambda a: order.get(a['kind'], 9))
        result['artifacts'] = artifacts
        print(f'        [L4] collected {len(artifacts)} plot artifact(s)')
    except Exception as e:
        result['artifact_error'] = f'collect failed: {type(e).__name__}: {e}'


# ============================================================
# LEVEL 4 VALIDATION: gta.optimize() + gta.fit()
# ============================================================

def validate_level4(yaml_str, evfile, scfile, test_idx,
                    fit_timeout=DEFAULT_FIT_TIMEOUT, artifact_dir=None,
                    work_dir=None, incremental=False, roi_prefix='fit_model',
                    analysis_script=None, approved_digest=None,
                    config_aliases=None, stdout_path=None):
    """Level 4: Run gta.optimize() + gta.fit() and check science quality.

    Prerequisites: Level 3 (gta.setup()) must have already passed for this
    config. This function re-runs setup internally to get a live gta object
    (unless ``incremental`` is set, see below).

    Parameters
    ----------
    yaml_str : str
        The (already repaired) FermiPy YAML config string.
    evfile : str
        Path to the event file.
    scfile : str
        Path to the spacecraft file.
    test_idx : int
        Index into TEST_DATA_MAP (0=Mrk421, 1=Vela, 2=Crab).
    fit_timeout : int
        Maximum seconds for optimize+fit combined.
    work_dir : str, optional
        Persistent working directory to use instead of a throwaway temp
        directory. When given, the directory is NOT deleted when this
        function returns (needed so a later incremental run can reuse the
        saved ROI model / ltcube in it). When omitted, behaviour is
        unchanged from before: a temp directory is created and removed.
    incremental : bool
        If True, skip ``gta.setup()`` and ``gta.optimize()``/``gta.fit()``
        and instead reload a previously fitted ROI via
        ``gta.load_roi(roi_prefix)`` from ``work_dir``. Only the requested
        products (sed/tsmap/residmap/lightcurve/psmap) are (re)computed.
        Requires ``work_dir`` to already contain a snapshot written by a
        prior successful (non-incremental) call. On any failure to load,
        ``result['incremental_failed']`` is set True so the caller can fall
        back to a full run.
    roi_prefix : str
        Prefix used for ``gta.write_roi()`` / ``gta.load_roi()``.

    Returns
    -------
    dict with keys:
        level, optimize_ok, fit_ok, fit_quality, fit_status,
        target_ts, target_flux, target_flux_error,
        catalog_flux, flux_ratio, spectral_index,
        convergence_ok, ts_check, flux_check, index_check,
        level4_pass, error, traceback, elapsed_seconds,
        roi_sources, incremental_failed
    """
    result = {
        'level': 4,
        'optimize_ok': False,
        'fit_ok': False,
        'fit_quality': None,
        'fit_status': None,
        'target_ts': None,
        'target_flux': None,
        'target_flux_error': None,
        'target_flux_ul95': None,
        'target_eflux_ul95': None,
        'target_present': False,
        'detection_threshold_ts': 25.0,
        'detection_status': None,
        'catalog_flux': None,
        'flux_ratio': None,
        'spectral_index': None,
        'spectral_index_error': None,
        'convergence_ok': False,
        'ts_check': False,
        'flux_check': None,
        'catalog_check_applicable': False,
        'index_check': False,
        'level4_pass': False,
        'error': None,
        'traceback': '',
        'elapsed_seconds': 0,
        'n_free_params': None,
        'loglike': None,
        'sources_summary': [],
        'artifacts': [],
        'sed_data': None,
        'requested_products': [],
        'target_spectral_model': None,
        'target_warning': None,
        'roi_sources': [],
        'incremental_failed': False,
        'setup_ok': False,
        'setup_error': None,
        'setup_sources': [],
        'setup_n_sources': None,
        'approved_digest': approved_digest,
        'executed_python_sha256': None,
    }

    if not yaml_str or not yaml_str.strip():
        result['error'] = 'empty yaml'
        return result

    if not os.path.exists(evfile) or not os.path.exists(scfile):
        result['error'] = (
            f'data files not found: evfile={os.path.exists(evfile)}, '
            f'scfile={os.path.exists(scfile)}'
        )
        return result

    meta = get_test_meta(test_idx)
    target_name = meta.get('target', '')
    is_catalogued = meta.get('catalogued', True)
    catalog_ref = CATALOG_REFERENCE.get(target_name, {}) if is_catalogued else {}
    if is_catalogued and not catalog_ref and target_name:
        # Arbitrary catalog source: build the sanity-check reference from
        # the 4FGL catalog itself instead of the hardcoded demo table.
        try:
            import source_resolver
            catalog_ref = source_resolver.catalog_reference(target_name)
        except Exception as e:
            print(f"  Warning: no catalog reference for {target_name}: {e}")
            catalog_ref = {}

    if not target_name:
        result['error'] = f'no target defined for test_idx={test_idx}'
        return result

    owns_work_dir = work_dir is None
    if owns_work_dir:
        work_dir = tempfile.mkdtemp(prefix='fermipy_l4_')
    else:
        os.makedirs(work_dir, exist_ok=True)
    config_path = os.path.join(work_dir, 'config.yaml')

    t_start = time.time()

    try:
        # ---- Parse and patch config ----
        parsed = yaml.safe_load(yaml_str)
        if not isinstance(parsed, dict):
            result['error'] = 'yaml not a dict'
            return result

        if analysis_script:
            # The reviewed YAML is immutable. Relative output paths resolve
            # inside the isolated work directory through cwd, not a rewrite.
            config_path = os.path.join(
                os.path.dirname(analysis_script), 'final_config.yaml')
            with open(config_path) as stream:
                reviewed_yaml = stream.read()
            if reviewed_yaml != yaml_str.rstrip() + '\n':
                result['error'] = (
                    'final_config.yaml changed after approval; nothing ran')
                return result
            configured_outdir = (
                (parsed.get('fileio') or {}).get('outdir') or 'output')
            if os.path.isabs(configured_outdir):
                runtime_outdir = configured_outdir
            else:
                runtime_outdir = os.path.join(work_dir, configured_outdir)
            os.makedirs(runtime_outdir, exist_ok=True)
        else:
            # Legacy callers retain the historical sandbox-path normalization.
            if 'fileio' not in parsed:
                parsed['fileio'] = {}
            parsed['fileio']['outdir'] = os.path.join(work_dir, 'output')
            os.makedirs(parsed['fileio']['outdir'], exist_ok=True)
            parsed['fileio']['logfile'] = os.path.join(
                parsed['fileio']['outdir'], 'fermipy.log')
            if 'data' not in parsed:
                parsed['data'] = {}
            parsed['data']['evfile'] = evfile
            parsed['data']['scfile'] = scfile
            with open(config_path, 'w') as f:
                yaml.dump(parsed, f, default_flow_style=False)

        # ---- Instantiate, then either setup() (full) or load_roi() (incremental) ----
        from fermipy.gtanalysis import GTAnalysis

        # fermipy Logger.configure() is a no-op when the named logger already
        # has handlers, so a reused worker process keeps the first session's
        # file handler and this run's fileio.logfile is silently ignored.
        # Drop existing handlers so the log really goes to this run's outdir.
        _lg_name = (parsed.get('logging') or {}).get('prefix', '') + 'GTAnalysis'
        _lg = logging.getLogger(_lg_name)
        for _h in list(_lg.handlers):
            _lg.removeHandler(_h)
            try:
                _h.close()
            except Exception:
                pass

        script_product_results = None
        if analysis_script:
            # Execute the reviewed script exactly as written. Results come
            # from recording the GTAnalysis calls it makes, so the script
            # does not have to follow any output contract.
            from .script_runner import execute_script
            with open(analysis_script, 'rb') as stream:
                script_bytes = stream.read()
            result['executed_python_sha256'] = hashlib.sha256(
                script_bytes).hexdigest()
            with TimeoutContext(fit_timeout, label='approved analysis.py'):
                state = execute_script(
                    analysis_script, config_path, work_dir,
                    config_aliases=config_aliases or (),
                    stdout_path=stdout_path)
            result['script_calls'] = state['calls']
            result['script_stdout'] = state['stdout_tail']
            result['script_error'] = state['script_error']
            result['script_traceback'] = state['script_traceback']
            result['product_errors'] = state['product_errors']
            gta = state['gta']
            fit_result = state['fit_result']
            script_product_results = state['product_results']
            result['optimize_ok'] = bool(state['optimize_ok'])
            result['fit_called'] = bool(state['fit_called'])
            if incremental:
                result['fit_ok'] = bool(state['load_roi_ok'])
            else:
                result['fit_ok'] = isinstance(fit_result, dict)
            if gta is None or not state['setup_ok']:
                result['error'] = state['script_error'] or (
                    'analysis.py finished without setting up a GTAnalysis '
                    '(no gta.setup() or gta.load_roi() completed)')
                result['traceback'] = state['script_traceback'] or ''
                result['setup_error'] = result['error']
                return result
            result['setup_ok'] = True
            result['setup_n_sources'] = len(gta.roi.sources)
            result['setup_sources'] = [
                source.name for source in gta.roi.sources[:10]]
            if state['script_error']:
                result['error'] = f"analysis.py failed: {state['script_error']}"
                result['traceback'] = state['script_traceback'] or ''
        else:
            gta = GTAnalysis(config_path)

        if analysis_script:
            pass
        elif incremental:
            print(f"        [L4] Incremental mode: gta.load_roi('{roi_prefix}') "
                  f"(reusing the previous gta.setup()+optimize()+fit() instead of "
                  f"re-running them)...")
            try:
                gta.load_roi(roi_prefix)
            except Exception as e:
                result['error'] = f'gta.load_roi() failed: {type(e).__name__}: {str(e)[:300]}'
                result['incremental_failed'] = True
                result['elapsed_seconds'] = time.time() - t_start
                del gta
                return result
        else:
            print(f"        [L4] Running gta.setup()...")
            gta.setup()
            result['setup_ok'] = True
            result['setup_n_sources'] = len(gta.roi.sources)
            result['setup_sources'] = [
                source.name for source in gta.roi.sources[:10]]

        # ---- Free parameters for fitting ----
        # A reviewed script made its own modelling choices; the fixed
        # free/optimize/fit sequence below only applies to legacy callers.
        if analysis_script:
            try:
                target_name = gta.roi.get_source_by_name(target_name).name
                result['target_present'] = True
            except Exception:
                result['target_warning'] = (
                    f'The YAML target "{target_name}" is not in the ROI the '
                    f'script built, so no target fit results are reported.')
        else:
            # Free the target source
            print(f"        [L4] Freeing source parameters...")
            try:
                gta.free_source(target_name)
            except Exception:
                # Catalog aliases can differ slightly in whitespace/prefix. That
                # is the only safe fallback: fitting the brightest unrelated ROI
                # source would silently answer a different scientific question.
                roi_sources = list(gta.roi.sources)
                roi_source_names = [s.name for s in roi_sources]
                norm = lambda s: (s or '').replace('4FGL ', '').replace(' ', '').lower()
                target_key = norm(target_name)
                matched = False
                if is_catalogued:
                    for sname in roi_source_names:
                        if target_key and (target_key in norm(sname) or
                                           norm(sname) in target_key):
                            gta.free_source(sname)
                            target_name = sname
                            matched = True
                            break
                if not matched:
                    result['error'] = (
                        f'Requested target "{target_name}" is not present in the '
                        f'ROI model. No substitute source was fit. '
                        f'ROI sources: {roi_source_names[:10]}'
                    )
                    result['elapsed_seconds'] = time.time() - t_start
                    del gta
                    return result
            result['target_present'] = True

            # Free diffuse backgrounds
            try:
                gta.free_source('galdiff')
            except Exception:
                pass  # may not be named exactly 'galdiff'
            try:
                gta.free_source('isodiff')
            except Exception:
                pass

            # Free nearby bright sources (TS > 100, within 5 degrees)
            try:
                gta.free_sources(minmax_ts=[100, None], distance=5.0)
            except Exception:
                pass  # not critical if this fails

        if analysis_script:
            if fit_result is None and incremental:
                result['fit_ok'] = True
                result['convergence_ok'] = True
        elif incremental:
            # ROI was already fit in a previous run; loading it above already
            # restored the post-fit parameter values, so there is nothing to
            # optimize/fit here.
            print(f"        [L4] Incremental mode: reusing previous fit "
                  f"(gta.optimize()/gta.fit() skipped)")
            result['optimize_ok'] = True
            result['fit_ok'] = True
            fit_result = None
            result['convergence_ok'] = True
        else:
            # ---- Optimize ----
            print(f"        [L4] Running gta.optimize()...")
            with TimeoutContext(fit_timeout, label="optimize+fit"):
                try:
                    opt_result = gta.optimize()
                    result['optimize_ok'] = True
                    print(f"        [L4] gta.optimize() completed")
                except FitTimeoutError:
                    result['error'] = (
                        f'gta.optimize() timed out after {fit_timeout}s'
                    )
                    result['elapsed_seconds'] = time.time() - t_start
                    del gta
                    return result
                except Exception as e:
                    result['error'] = (
                        f'gta.optimize() failed: {type(e).__name__}: {str(e)[:300]}'
                    )
                    result['traceback'] = traceback.format_exc()[-500:]
                    result['elapsed_seconds'] = time.time() - t_start
                    # Try to continue with fit anyway -- optimize is optional
                    print(f"        [L4] optimize failed ({e}), attempting fit anyway...")

                # ---- Fit ----
                print(f"        [L4] Running gta.fit()...")
                try:
                    fit_result = gta.fit()
                    result['fit_ok'] = True
                    print(f"        [L4] gta.fit() completed")
                except FitTimeoutError:
                    result['error'] = (
                        f'gta.fit() timed out after {fit_timeout}s'
                    )
                    result['elapsed_seconds'] = time.time() - t_start
                    del gta
                    return result
                except Exception as e:
                    result['error'] = (
                        f'gta.fit() failed: {type(e).__name__}: {str(e)[:300]}'
                    )
                    result['traceback'] = traceback.format_exc()[-500:]
                    result['elapsed_seconds'] = time.time() - t_start
                    del gta
                    return result

        # ---- Extract fit metadata ----
        if fit_result is not None and isinstance(fit_result, dict):
            result['fit_quality'] = fit_result.get('fit_quality', None)
            result['fit_status'] = fit_result.get('fit_status', None)
            result['loglike'] = fit_result.get('loglike', None)
            result['n_free_params'] = fit_result.get('nfree', None)

            # Convergence: fit_quality >= 3 is good, fit_status == 0 is converged
            fit_quality = fit_result.get('fit_quality', 0)
            fit_status = fit_result.get('fit_status', -1)
            result['convergence_ok'] = (
                (fit_quality is not None and fit_quality >= 2)
                or (fit_status is not None and fit_status == 0)
            )
        else:
            # fit() returned None or unexpected type -- check gta state
            result['fit_quality'] = None
            result['fit_status'] = None
            # Still try to extract source info below

        # ---- Extract target source properties (post-fit) ----
        # IMPORTANT: After gta.fit(), we must use gta.get_src_model() to read
        # updated post-fit values. The src.todict() method returns cached
        # pre-fit values and will show ts=0, flux=0 for all sources.
        try:
            if analysis_script and not result['target_present']:
                raise LookupError(result['target_warning'])
            # get_src_model returns a dict with post-fit values
            src_model = gta.get_src_model(target_name)

            # Spectral model type (PowerLaw, LogParabola, PLSuperExpCutoff, ...)
            # so the UI can adapt the summary (e.g. hide a single "spectral
            # index" for curved spectra where it is not meaningful). Try the
            # src_model dict first, then the ROI source object as a fallback.
            stype = src_model.get('SpectrumType') or src_model.get('spectrum_type')
            if not stype:
                try:
                    stype = gta.roi[target_name]['SpectrumType']
                except Exception:
                    stype = None
            result['target_spectral_model'] = stype

            # TS value (post-fit)
            ts_val = src_model.get('ts', None)
            result['target_ts'] = float(ts_val) if ts_val is not None else None

            # Flux (integral photon flux, ph/cm2/s)
            flux_val = src_model.get('flux', None)
            if isinstance(flux_val, (list, tuple)):
                flux_err = flux_val[1] if len(flux_val) > 1 else None
                flux_val = flux_val[0] if len(flux_val) > 0 else None
            else:
                flux_err = src_model.get('flux_err', None)

            result['target_flux'] = float(flux_val) if flux_val is not None else None
            result['target_flux_error'] = float(flux_err) if flux_err is not None else None
            for result_key, model_key in (
                    ('target_flux_ul95', 'flux_ul95'),
                    ('target_eflux_ul95', 'eflux_ul95')):
                try:
                    value = float(src_model.get(model_key))
                    result[result_key] = value if math.isfinite(value) else None
                except (TypeError, ValueError):
                    result[result_key] = None

            # Spectral index from post-fit model
            spec_pars = src_model.get('spectral_pars', {})
            index_val = None
            index_err = None
            for key in ['Index', 'index', 'alpha', 'Index1']:
                if key in spec_pars:
                    par_info = spec_pars[key]
                    if isinstance(par_info, dict):
                        index_val = par_info.get('value', None)
                        index_err = par_info.get('error', None)
                    else:
                        index_val = par_info
                    break

            # FermiPy sometimes stores Index as negative
            if index_val is not None:
                index_val = abs(float(index_val))
            if index_err is not None:
                index_err = abs(float(index_err))

            result['spectral_index'] = index_val
            result['spectral_index_error'] = index_err

        except Exception as e:
            if analysis_script and not result['target_present']:
                pass  # reported as target_warning
            elif not result.get('error'):
                result['error'] = (
                    f'Could not extract source properties: '
                    f'{type(e).__name__}: {str(e)[:200]}'
                )
                result['traceback'] = traceback.format_exc()[-500:]

        # ---- Collect summary of top sources (post-fit) ----
        try:
            top_sources = []
            for src in gta.roi.sources[:20]:
                try:
                    sm = gta.get_src_model(src.name)
                    top_sources.append({
                        'name': src.name,
                        'ts': float(sm.get('ts', 0) or 0),
                        'flux': float(sm.get('flux', 0) if not isinstance(sm.get('flux'), (list, tuple)) else sm['flux'][0] if sm.get('flux') else 0),
                    })
                except Exception:
                    pass
            top_sources.sort(key=lambda s: s['ts'], reverse=True)
            top_sources = top_sources[:10]
            result['sources_summary'] = top_sources
        except Exception:
            pass

        # ---- Science quality checks ----
        # A low TS is a scientifically valid non-detection, not a pipeline
        # failure. Preserve the fit and expose its 95% profile-likelihood UL.
        if result['target_ts'] is not None:
            result['ts_check'] = result['target_ts'] >= 0
            if result['target_ts'] >= result['detection_threshold_ts']:
                result['detection_status'] = 'detected'
            elif result['target_ts'] > 0:
                result['detection_status'] = 'subthreshold'
            else:
                result['detection_status'] = 'not_detected'
        else:
            result['ts_check'] = False
            result['detection_status'] = 'unavailable'

        # 2. Flux within order-of-magnitude of catalog value
        catalog_flux = catalog_ref.get('flux_100mev_1tev', None)
        result['catalog_flux'] = catalog_flux
        result['catalog_check_applicable'] = catalog_flux is not None
        if result['target_flux'] is not None and catalog_flux is not None:
            try:
                if result['target_flux'] > 0 and catalog_flux > 0:
                    log_ratio = abs(
                        math.log10(result['target_flux'])
                        - math.log10(catalog_flux)
                    )
                    result['flux_ratio'] = result['target_flux'] / catalog_flux
                    tolerance = catalog_ref.get('flux_tolerance_log10', 1.5)
                    result['flux_check'] = log_ratio <= tolerance
                else:
                    result['flux_ratio'] = None
                    result['flux_check'] = False
            except (ValueError, ZeroDivisionError):
                result['flux_ratio'] = None
                result['flux_check'] = False
        else:
            result['flux_check'] = None

        # 3. Spectral index in expected range
        idx_min = catalog_ref.get('expected_index_min', 0.5)
        idx_max = catalog_ref.get('expected_index_max', 4.0)
        if result['spectral_index'] is not None:
            result['index_check'] = idx_min <= result['spectral_index'] <= idx_max
        else:
            result['index_check'] = False

        # ---- Overall Level 4 pass/fail ----
        if result['catalog_check_applicable']:
            # Preserve the existing catalog-source science sanity check.
            result['level4_pass'] = (
                result['fit_ok'] and result['ts_check']
                and bool(result['flux_check']))
        elif analysis_script and not result['target_present']:
            result['level4_pass'] = False
        else:
            # There is intentionally no catalog flux for a custom target.
            # Technical validity and a usable likelihood profile are enough;
            # TS determines detection status, not whether the run succeeded.
            result['level4_pass'] = (
                result['fit_ok'] and result['target_present']
                and result['convergence_ok'] and result['ts_check'])
        if analysis_script and result.get('script_error'):
            result['level4_pass'] = False

        # ---- Persist the fitted ROI model for future incremental runs ----
        # A later run whose core YAML sections (data/selection/binning/model/
        # components/gtlike) are unchanged can reload this snapshot with
        # gta.load_roi() instead of re-running gta.setup() + gta.optimize() +
        # gta.fit(), which is the whole point of ``work_dir``/``incremental``.
        if (result['fit_ok'] and not incremental
                and not result.get('script_error')):
            try:
                gta.write_roi(roi_prefix, make_plots=False)
                print(f"        [L4] gta.write_roi('{roi_prefix}') saved for incremental reuse")
            except Exception as e:
                print(f"        [L4] write_roi skipped: {type(e).__name__}: {e}")

        # ---- Generate visualisable output artifacts (plots + SED data) ----
        # Only worth doing once the fit succeeded. Fully guarded: a failure
        # here never changes the validation verdict above.
        if (result['fit_ok'] or analysis_script) and artifact_dir:
            try:
                outdir = parsed.get('fileio', {}).get('outdir', '')
                if analysis_script and not os.path.isabs(outdir):
                    outdir = os.path.join(work_dir, outdir)
                _generate_output_artifacts(
                    gta, target_name, parsed, outdir, artifact_dir, result,
                    product_results=script_product_results,
                    # A script may save its own plots anywhere in the work
                    # dir; only files from this run are collected.
                    collect_root=work_dir if analysis_script else None,
                    since=t_start if analysis_script else None,
                )
            except Exception as e:
                result['artifact_error'] = f'{type(e).__name__}: {str(e)[:200]}'

        del gta

    except FitTimeoutError:
        result['error'] = f'Overall timeout after {fit_timeout}s'
    except Exception as e:
        result['error'] = f'{type(e).__name__}: {str(e)[:500]}'
        result['traceback'] = traceback.format_exc()[-500:]
    finally:
        result['elapsed_seconds'] = time.time() - t_start
        # Only clean up a work directory we created ourselves; a caller-
        # supplied persistent work_dir (used so a later incremental run can
        # gta.load_roi() from it) must survive after this call returns.
        if owns_work_dir:
            shutil.rmtree(work_dir, ignore_errors=True)

    return result


# ============================================================
# COMBINED LEVEL 1-4 VALIDATION PIPELINE
# ============================================================

def validate_all_levels(yaml_str, test_idx, fit_timeout=DEFAULT_FIT_TIMEOUT,
                        artifact_dir=None):
    """Run Levels 1-4 validation in sequence, stopping on failure.

    Returns a dict with level1, level2, level3, level4 results.
    """
    meta = get_test_meta(test_idx)
    evfile = meta.get('evfile', '')
    scfile = meta.get('scfile', '')

    results = {
        'test_idx': test_idx,
        'target': meta.get('target', ''),
        'source_name': meta.get('name', ''),
    }

    # Level 1
    l1 = validate_level1(yaml_str)
    results['level1'] = l1
    if not l1.get('fermipy_load'):
        results['max_level_passed'] = 0
        return results

    # Level 2
    l2 = validate_level2(yaml_str)
    results['level2'] = l2
    if not l2.get('gta_init'):
        results['max_level_passed'] = 1
        return results

    # Level 3
    if not os.path.exists(evfile) or not os.path.exists(scfile):
        results['level3'] = {
            'level': 3, 'setup_ok': None,
            'error': 'data files not available'
        }
        results['max_level_passed'] = 2
        return results

    l3 = validate_level3(yaml_str, evfile, scfile)
    results['level3'] = l3
    if not l3.get('setup_ok'):
        results['max_level_passed'] = 2
        return results

    # Level 4
    print(f"      Levels 1-3 PASSED. Running Level 4 fit "
          f"(timeout={fit_timeout}s)...")
    l4 = validate_level4(yaml_str, evfile, scfile, test_idx,
                         fit_timeout=fit_timeout, artifact_dir=artifact_dir)
    results['level4'] = l4
    if l4.get('level4_pass'):
        results['max_level_passed'] = 4
    elif l4.get('fit_ok'):
        results['max_level_passed'] = 3  # fit ran but science checks failed
    else:
        results['max_level_passed'] = 3  # setup passed but fit failed

    return results


# ============================================================
# RE-VALIDATE EXISTING OUTPUTS WITH LEVEL 4
# ============================================================

def revalidate_with_level4(fit_timeout=DEFAULT_FIT_TIMEOUT):
    """Re-validate all existing generated outputs through Level 4.

    Reads the same improved_*.json files used by the Level 1-3 validator,
    applies the repair engine, then runs Level 4 on configs that pass Level 3.
    """
    print("\n" + "=" * 60)
    print("LEVEL 4 VALIDATION: Full Likelihood Fitting")
    print("=" * 60)
    print(f"  Fit timeout per source: {fit_timeout}s")
    print(f"  Timestamp: {datetime.now().isoformat()}")

    all_results = {}

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
                meta = TEST_DATA_MAP.get(test_idx, {})

                gen_yaml = extract_yaml(text)
                gen_py = extract_python(text)

                if not gen_yaml:
                    approach_results.append({
                        'test_idx': test_idx,
                        'has_yaml': False,
                        'has_python': bool(gen_py),
                        'success': False,
                        'error': 'no yaml extracted',
                        'max_level_passed': 0,
                    })
                    print(f"  {approach} test_{test_idx} "
                          f"({meta.get('name', '?')}): NO YAML")
                    continue

                # Repair the config first
                print(f"  {approach} test_{test_idx} "
                      f"({meta.get('name', '?')}):")
                final_yaml, val_results, repair_hist = (
                    validate_with_iterative_repair(gen_yaml, test_idx)
                )

                # Check if Level 3 passed
                l3_passed = any(
                    r.get('level3', {}).get('setup_ok', False)
                    for r in val_results
                )

                entry = {
                    'test_idx': test_idx,
                    'has_yaml': True,
                    'has_python': bool(gen_py),
                    'repair_history': repair_hist,
                    'n_repair_iterations': len(val_results),
                    'l3_passed': l3_passed,
                }

                if not l3_passed:
                    last_error = ''
                    if val_results:
                        last = val_results[-1]
                        l3 = last.get('level3', {})
                        last_error = (l3.get('error', '')[:100]
                                      if l3 else '')
                        if not last_error:
                            l2 = last.get('level2', {})
                            last_error = (l2.get('error', '')[:100]
                                          if l2 else '')
                    entry['success'] = False
                    entry['max_level_passed'] = 2
                    entry['level4'] = None
                    entry['error'] = last_error
                    print(f"    L3 FAIL: {last_error}")
                    approach_results.append(entry)
                    continue

                # Level 3 passed -- run Level 4
                evfile = meta.get('evfile', '')
                scfile = meta.get('scfile', '')

                print(f"    L3 PASS. Running Level 4 fit...")
                l4 = validate_level4(
                    final_yaml, evfile, scfile, test_idx,
                    fit_timeout=fit_timeout,
                )

                entry['level4'] = l4
                entry['success'] = l4.get('level4_pass', False)
                entry['max_level_passed'] = (
                    4 if l4.get('level4_pass', False) else 3
                )

                # Print detailed L4 results
                _print_l4_result(l4, meta)
                approach_results.append(entry)

            model_results[approach] = approach_results
        all_results[model_name] = model_results

    # Save results
    out_path = os.path.join(RESULTS_DIR, 'level4_validation.json')
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")

    # Print summary
    _print_summary(all_results)
    return all_results


# ============================================================
# VALIDATE A SINGLE CONFIG (for testing / standalone use)
# ============================================================

def validate_single_config(yaml_path_or_str, test_idx,
                           fit_timeout=DEFAULT_FIT_TIMEOUT):
    """Validate a single YAML config through all 4 levels.

    Parameters
    ----------
    yaml_path_or_str : str
        Path to a YAML file, or YAML string content.
    test_idx : int
        0=Mrk421, 1=Vela, 2=Crab
    fit_timeout : int
        Seconds for fit timeout.

    Returns
    -------
    dict with full validation results.
    """
    if os.path.isfile(yaml_path_or_str):
        with open(yaml_path_or_str) as f:
            yaml_str = f.read()
    else:
        yaml_str = yaml_path_or_str

    meta = get_test_meta(test_idx)
    print(f"Validating config for test_{test_idx} ({meta.get('name', '?')})...")

    # Repair first
    repaired_yaml, repairs = repair_config(yaml_str, test_idx)
    if repairs:
        print(f"  Applied {len(repairs)} repairs: {repairs[:5]}")

    if not repaired_yaml:
        print(f"  ERROR: repair produced empty yaml")
        return {'error': 'repair failed'}

    # Run all levels
    results = validate_all_levels(
        repaired_yaml, test_idx, fit_timeout=fit_timeout
    )

    if 'level4' in results:
        _print_l4_result(results['level4'], meta)

    return results


# ============================================================
# DISPLAY HELPERS
# ============================================================

def _print_l4_result(l4, meta):
    """Print formatted Level 4 results."""
    target_name = meta.get('name', '?')
    catalog_ref = CATALOG_REFERENCE.get(meta.get('target', ''), {})

    status = 'PASS' if l4.get('level4_pass') else 'FAIL'
    elapsed = l4.get('elapsed_seconds', 0)

    print(f"    L4 {status} ({elapsed:.1f}s)")

    if l4.get('error'):
        print(f"      Error: {l4['error'][:120]}")

    print(f"      Optimize: {'OK' if l4.get('optimize_ok') else 'FAIL'}  |  "
          f"Fit: {'OK' if l4.get('fit_ok') else 'FAIL'}")

    fq = l4.get('fit_quality')
    fs = l4.get('fit_status')
    print(f"      Fit quality={fq}  status={fs}  "
          f"convergence={'OK' if l4.get('convergence_ok') else 'FAIL'}")

    ts = l4.get('target_ts')
    print(f"      TS({target_name}) = {ts}"
          f"  {'[OK]' if l4.get('ts_check') else '[FAIL: TS <= 0]'}")

    flux = l4.get('target_flux')
    cat_flux = l4.get('catalog_flux')
    ratio = l4.get('flux_ratio')
    if flux is not None and cat_flux is not None:
        print(f"      Flux = {flux:.3e} ph/cm2/s  "
              f"(catalog: {cat_flux:.3e}, ratio: {ratio:.3f})"
              f"  {'[OK]' if l4.get('flux_check') else '[FAIL]'}")
    else:
        print(f"      Flux = {flux}  "
              f"(catalog: {cat_flux})  [FAIL: could not compare]")

    idx = l4.get('spectral_index')
    idx_err = l4.get('spectral_index_error')
    cat_idx = catalog_ref.get('catalog_index', '?')
    if idx is not None:
        err_str = f" +/- {idx_err:.3f}" if idx_err is not None else ""
        print(f"      Spectral index = {idx:.3f}{err_str}  "
              f"(catalog: {cat_idx})"
              f"  {'[OK]' if l4.get('index_check') else '[INFO: outside range]'}")
    else:
        print(f"      Spectral index = None  (catalog: {cat_idx})")


def _print_summary(all_results):
    """Print summary table for Level 4 validation."""
    print("\n" + "=" * 60)
    print("LEVEL 4 VALIDATION SUMMARY")
    print("=" * 60)

    grand_total = 0
    grand_l3 = 0
    grand_l4 = 0

    for model_name, model_results in all_results.items():
        print(f"\n{model_name}:")
        model_total = 0
        model_l3 = 0
        model_l4 = 0

        for approach, results in model_results.items():
            if not isinstance(results, list):
                continue
            n = len(results)
            n_l3 = sum(1 for r in results if r.get('l3_passed', False))
            n_l4 = sum(1 for r in results if r.get('success', False))
            model_total += n
            model_l3 += n_l3
            model_l4 += n_l4

            # Also show per-source detail
            detail_parts = []
            for r in results:
                tidx = r.get('test_idx', -1)
                tmeta = TEST_DATA_MAP.get(tidx, {})
                sname = tmeta.get('name', f't{tidx}')
                l4_data = r.get('level4')
                if l4_data and l4_data.get('level4_pass'):
                    detail_parts.append(f"{sname}:PASS")
                elif r.get('l3_passed'):
                    detail_parts.append(f"{sname}:L4-FAIL")
                else:
                    detail_parts.append(f"{sname}:L3-FAIL")

            detail_str = '  '.join(detail_parts) if detail_parts else ''
            print(f"  {approach:30s}: L3={n_l3}/{n}  L4={n_l4}/{n}"
                  f"  | {detail_str}")

        grand_total += model_total
        grand_l3 += model_l3
        grand_l4 += model_l4

        if model_total > 0:
            l3_pct = 100 * model_l3 / model_total
            l4_pct = 100 * model_l4 / model_total
            print(f"  {'TOTAL':30s}: L3={model_l3}/{model_total} ({l3_pct:.1f}%)  "
                  f"L4={model_l4}/{model_total} ({l4_pct:.1f}%)")

    if grand_total > 0:
        l3_pct = 100 * grand_l3 / grand_total
        l4_pct = 100 * grand_l4 / grand_total
        print(f"\nOVERALL: L3={grand_l3}/{grand_total} ({l3_pct:.1f}%)  "
              f"L4={grand_l4}/{grand_total} ({l4_pct:.1f}%)")


# ============================================================
# MAIN
# ============================================================

def main():
    diagnose_environment()

    fit_timeout = DEFAULT_FIT_TIMEOUT

    # Parse CLI arguments
    mode = 'revalidate'
    single_yaml = None
    single_test_idx = None

    args = sys.argv[1:]
    i = 0
    while i < len(args):
        arg = args[i]
        if arg == '--revalidate':
            mode = 'revalidate'
        elif arg == '--single':
            mode = 'single'
            if i + 2 < len(args):
                single_yaml = args[i + 1]
                single_test_idx = int(args[i + 2])
                i += 2
            else:
                print("Usage: --single <yaml_path_or_string> <test_idx>")
                sys.exit(1)
        elif arg == '--timeout':
            if i + 1 < len(args):
                fit_timeout = int(args[i + 1])
                i += 1
            else:
                print("Usage: --timeout <seconds>")
                sys.exit(1)
        elif arg == '--help':
            print(__doc__)
            print("\nUsage:")
            print("  python run_level4_validation.py [--revalidate] [--timeout SECONDS]")
            print("  python run_level4_validation.py --single <yaml_file> <test_idx>")
            print("\nOptions:")
            print("  --revalidate   Re-validate all existing LLM outputs (default)")
            print("  --single       Validate a single YAML config file")
            print("  --timeout N    Set fit timeout in seconds (default: 1800)")
            sys.exit(0)
        i += 1

    print(f"\nMode: {mode}")
    print(f"Fit timeout: {fit_timeout}s")

    if mode == 'revalidate':
        revalidate_with_level4(fit_timeout=fit_timeout)
    elif mode == 'single':
        if single_yaml and single_test_idx is not None:
            results = validate_single_config(
                single_yaml, single_test_idx, fit_timeout=fit_timeout
            )
            # Save single result too
            out_path = os.path.join(RESULTS_DIR, 'level4_single_result.json')
            with open(out_path, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            print(f"\nResult saved to {out_path}")
        else:
            print("ERROR: --single requires <yaml_path> <test_idx>")
            sys.exit(1)

    print('\n' + '=' * 60)
    print('Level 4 validation complete!')
    print('=' * 60)


if __name__ == '__main__':
    main()

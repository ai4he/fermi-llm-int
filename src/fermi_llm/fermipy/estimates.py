"""Runtime and bin-count estimates shown before a run starts.

These drive two user-visible guarantees: the estimated duration in the chat
reply, and the confirmation prompt before a long full-dataset run. They are
heuristics over the YAML, never measurements, so they stay cheap to call.
"""

from __future__ import annotations

import math
import os
import yaml

def _estimate_run_duration(yaml_str, full_data=False):
    """Heuristic, indicative wall-clock estimate for running this config.

    Level 3 (gtselect/ltcube/exposure via gta.setup) dominates and scales
    roughly with the selected livetime and ROI area; each optional product
    (SED, light curve, TS/residual map) adds its own fit/scan time on top.
    ``full_data`` adds the I/O cost of streaming the all-sky weekly photon
    files (~150 MB/week) through event selection, which bundled demo runs
    (pre-cut one-week files) do not pay.
    Returns None when the YAML cannot be parsed.
    """
    try:
        cfg = yaml.safe_load(yaml_str) or {}
    except Exception:
        return None
    if not isinstance(cfg, dict):
        return None

    sel = cfg.get('selection') or {}
    binning = cfg.get('binning') or {}

    def _num(v, default=None):
        try:
            return float(v)
        except (TypeError, ValueError):
            return default

    tmin, tmax = _num(sel.get('tmin')), _num(sel.get('tmax'))
    weeks = 4.0
    if tmin is not None and tmax is not None and tmax > tmin:
        weeks = (tmax - tmin) / (7 * 86400.0)
    weeks = min(max(weeks, 0.5), 750.0)

    roiwidth = _num(binning.get('roiwidth'), 10.0) or 10.0
    roi_factor = (roiwidth / 10.0) ** 2

    breakdown = []
    setup = (1.5 + 0.4 * weeks ** 0.9) * max(roi_factor, 0.3)
    if full_data:
        breakdown.append(('weekly all-sky event selection (I/O)',
                          max(1.0, 0.15 * weeks)))
    breakdown.append(('setup: data selection, ltcube, exposure', setup))
    breakdown.append(('likelihood fit', 1.5 * max(roi_factor, 0.5)))
    if 'sed' in cfg:
        breakdown.append(('SED', max(1.0, 0.35 * setup)))
    if 'lightcurve' in cfg:
        lc_cfg = cfg.get('lightcurve') or {}
        nbins = _num(lc_cfg.get('nbins'))
        if nbins is None:
            lc_binsz = _num(lc_cfg.get('binsz'))
            nbins = (weeks * 7 * 86400.0) / lc_binsz if lc_binsz else 8.0
        nbins = min(max(nbins, 2), 100)
        breakdown.append(('light curve (%d bins)' % round(nbins), 0.5 * nbins))
    if 'tsmap' in cfg:
        breakdown.append(('TS map', max(2.0, 0.6 * setup)))
    if 'residmap' in cfg:
        breakdown.append(('residual map', max(1.0, 0.3 * setup)))

    total = sum(m for _, m in breakdown)
    lo, hi = total * 0.6, total * 1.8

    def _fmt_span(m):
        if m >= 90:
            return f"{m / 60:.1f} h"
        return f"{max(1, int(round(m)))} min"

    parts = ', '.join(f"{name} ~{_fmt_span(m)}" for name, m in breakdown)
    text = (f"~{_fmt_span(lo)}–{_fmt_span(hi)} ({parts}). Indicative only — "
            f"scales mainly with the selected time range ({weeks:.1f} weeks) "
            f"and ROI width ({roiwidth:g}°).")
    return {
        'total_lo_min': round(lo, 1),
        'total_hi_min': round(hi, 1),
        'breakdown': [{'stage': n, 'minutes': round(m, 1)} for n, m in breakdown],
        'text': text,
    }


def _fit_timeout_for(test_idx):
    """Fit-stage timeout in seconds.

    Bundled demo runs keep the fixed default; full-dataset runs scale
    with the amount of selected data (capped at 4 h).
    """
    base = int(os.environ.get('FERMI_LLM_FIT_TIMEOUT', 1800))
    if isinstance(test_idx, dict):
        weeks = int(test_idx.get('n_weeks') or 0)
        return max(base, min(4 * 3600, 900 + 30 * weeks))
    return base


def _expected_sed_bin_count(yaml_text):
    """Return the SED bin count implied by the reviewed configuration."""
    try:
        cfg = yaml.safe_load(yaml_text) or {}
        sed_cfg = cfg.get('sed')
        if not isinstance(sed_cfg, dict):
            return None
        custom_edges = sed_cfg.get('loge_bins')
        if isinstance(custom_edges, list) and len(custom_edges) >= 2:
            return len(custom_edges) - 1
        selection = cfg.get('selection') or {}
        binning = cfg.get('binning') or {}
        # enumbins overrides binsperdec in FermiPy (gtanalysis.py).
        if binning.get('enumbins') is not None:
            return int(binning['enumbins'])
        emin = float(selection.get('emin'))
        emax = float(selection.get('emax'))
        binsperdec = float(binning.get('binsperdec', 8))
        if emin > 0 and emax > emin and binsperdec > 0:
            return int(round(binsperdec * math.log10(emax / emin)))
    except (TypeError, ValueError, yaml.YAMLError):
        pass
    return None


def _sed_bins_from_script_calls(expected_bins, script_calls):
    """Refine the expected SED bin count with the edges the script passed.

    A script that hands its own ``loge_bins`` to ``gta.sed()`` overrides the
    count the YAML implies, so the contract follows its edges.  The recorder
    stores keyword arguments as truncated ``repr`` text, so the edge count
    comes from the recorded ``kwarg_lens`` (a literal list is still accepted
    for call traces built by hand).
    """
    for call in script_calls or []:
        if call.get('method') != 'sed':
            continue
        edges = (call.get('kwargs') or {}).get('loge_bins')
        n_edges = (call.get('kwarg_lens') or {}).get('loge_bins')
        if isinstance(edges, (list, tuple)):
            n_edges = len(edges)
        if isinstance(n_edges, int) and n_edges >= 2:
            expected_bins = n_edges - 1
    return expected_bins


estimate_run_duration = _estimate_run_duration
fit_timeout_for = _fit_timeout_for
expected_sed_bin_count = _expected_sed_bin_count

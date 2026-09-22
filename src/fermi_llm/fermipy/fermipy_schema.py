"""Shared FermiPy option metadata for prompting and validation.

The installed FermiPy package is the source of truth. Small fallback sets keep
the web UI importable in lightweight development/test environments where
FermiPy is unavailable.
"""

from __future__ import annotations

import math
import os
import re
from typing import Dict, Optional, Set


_FALLBACK_KEYS: Dict[str, Set[str]] = {
    'sed': {
        'bin_index', 'cov_scale', 'free_background', 'free_pars',
        'free_radius', 'make_plots', 'ul_confidence', 'ul_ts_threshold',
        'use_local_index', 'write_fits', 'write_npy',
    },
    'lightcurve': {
        'binsz', 'free_background', 'free_params', 'free_radius',
        'free_sources', 'make_plots', 'multithread', 'nbins', 'npts',
        'outdir', 'save_bin_data', 'shape_ts_threshold', 'systematic',
        'time_bins', 'use_local_ltcube', 'use_scaled_srcmap',
        'write_fits', 'write_npy',
    },
}

def fermipy_option_keys(section: str) -> Set[str]:
    """Return keys accepted in the GTAnalysis YAML section."""
    keys = set(_FALLBACK_KEYS.get(section, set()))
    # Importing FermiPy without a configured FermiTools/CALDB environment can
    # abort in native code before Python can catch an exception.
    if os.environ.get('FERMI_DIR'):
        try:
            from fermipy import defaults
            options = getattr(defaults, section, {})
            if isinstance(options, dict):
                keys.update(options)
        except Exception:
            pass
    return keys


def normalize_analysis_yaml(yaml_str: str):
    """Remove only keys proven invalid by the shared FermiPy metadata."""
    if not yaml_str or not yaml_str.strip():
        return yaml_str, []
    try:
        import yaml
        config = yaml.safe_load(yaml_str)
    except Exception:
        return yaml_str, []
    if not isinstance(config, dict):
        return yaml_str, []

    changes = []
    for section in ('sed', 'lightcurve'):
        values = config.get(section)
        if not isinstance(values, dict):
            continue
        valid = fermipy_option_keys(section)
        for key in list(values):
            if key not in valid:
                del values[key]
                changes.append(f'removed invalid key {section}.{key}')
    if not changes:
        return yaml_str, []
    return yaml.dump(config, default_flow_style=False, sort_keys=False), changes


def fermipy_binning_prompt_rules() -> str:
    """Prompt fragment kept in one place for every generation path."""
    return """Energy/SED binning rules:
- binning.binsperdec is bins PER DECADE (FermiPy default: 8), NOT the total number of energy bins. FermiPy builds the analysis grid as round(binsperdec * log10(emax/emin)), and gta.sed() uses that grid unless it is given loge_bins.
- A request for "N energy bins" / "an SED with N bins" is a TOTAL. Translate it as follows, with ndec = log10(emax/emin):
  * If N / ndec is a whole number, write binning.binsperdec: N / ndec and pass no loge_bins. The analysis grid is then exactly the N bins requested. Example: 6 bins over 100 MeV - 100 GeV (3 decades) is binsperdec: 2.
  * Otherwise binsperdec cannot express N at all. Leave the analysis binning alone and pass N+1 logarithmically spaced edges spanning selection.emin..emax to the method: gta.sed(target, loge_bins=[...]) in log10(E/MeV). Example for 7 bins over 100 MeV - 100 GeV: gta.sed(target, loge_bins=[2.0, 2.428571, 2.857143, 3.285714, 3.714286, 4.142857, 4.571429, 5.0]).
- Never write the total into binsperdec directly: with emin=100, emax=100000, binsperdec: 7 gives 21 bins, not 7.
- gta.sed() snaps each loge_bins edge to the closest analysis bin edge (set_energy_range: "Input values will be rounded to the closest bin edge value"), so N bins that do not divide the analysis grid come out slightly uneven in width. The count is still N as long as N does not exceed the number of analysis bins.
- binning.enumbins sets the total number of ANALYSIS energy bins and overrides binsperdec. It re-bins the whole likelihood fit, so prefer loge_bins when only the SED sampling should change.
- sed.loge_bins is NOT a valid GTAnalysis YAML key; the edges belong in the gta.sed() call in the Python script.
- Do not use sed.num_bins; it is not a FermiPy SED option."""



def fermipy_default_fit_prompt_rules() -> str:
    """The fitting procedure to use when the user specifies none."""
    return """Fitting rules:
- When the user gives no fitting directions, the script must at least free the diffuse backgrounds and the target and then fit once:
    gta.free_source('galdiff')
    gta.free_source('isodiff')
    gta.free_source(target)
    gta.optimize()
    gta.fit()
  Free galdiff/isodiff only when the YAML declares model.galdiff / model.isodiff.
- When the user does give fitting directions, add them. If they contradict this default, follow the user: their directions win over the default in every case except one that cannot work (the validator reports those) or one that is a safety concern.
- Do not add extra freeing the user did not ask for, such as gta.free_sources(minmax_ts=...), on top of the default."""


# "7 energy bins", "an SED with 12 bins", "10 spectral bins".  The companion
# parser for light curves (_extract_lightcurve_binning in webapp/server.py)
# deliberately *skips* any sentence mentioning sed/spectr/energy, so the two
# never claim the same number.
_LC_PHRASE = r'light[\s-]?curve|\blc\b'
# (pattern, anchored).  An anchored pattern names the SED next to the count,
# so only the matched span has to be free of a light-curve phrase; an
# unanchored one relies on the surrounding sentence for its context.
_SED_BIN_PATTERNS = (
    (r'(\d{1,3})\s*(?:energy|spectral|logarithmic|log)\s*bins?\b', False),
    (r'(?:sed|spectrum|spectra)\b[^.!?]{0,40}?\b(\d{1,3})\s*bins?\b', True),
    (r'\b(\d{1,3})\s*bins?\b[^.!?]{0,40}?\b(?:sed|spectrum|spectra)\b', True),
)


def sed_bin_request(text: str) -> Optional[int]:
    """Return the total number of SED energy bins the text asks for.

    Returns None when the request leaves the SED binning unspecified.  Only a
    count stated in an SED/spectral/energy context is accepted, so a light
    curve's bin count is never misread as an SED one.
    """
    msg = (text or '').lower()
    if not msg:
        return None
    for pattern, anchored in _SED_BIN_PATTERNS:
        for match in re.finditer(pattern, msg):
            # Ignore a count that belongs to a light curve instead.
            scope = match.group(0)
            if not anchored:
                start = max(msg.rfind(c, 0, match.start()) for c in '.!?') + 1
                stops = [i for i in (msg.find(c, match.end()) for c in '.!?')
                         if i != -1]
                scope = msg[start:min(stops) if stops else len(msg)]
            if re.search(_LC_PHRASE, scope):
                continue
            count = int(match.group(1))
            if 2 <= count <= 200:
                return count
    return None


def sed_bin_plan(emin, emax, nbins) -> Optional[Dict[str, object]]:
    """How to honor a request for a TOTAL of ``nbins`` SED energy bins.

    ``binning.binsperdec`` is bins per decade, so it can carry the request
    only when ``nbins / log10(emax/emin)`` is a whole number; the analysis
    grid is then exactly the requested grid and ``gta.sed()`` uses it by
    default.  When it is not whole, ``binsperdec`` cannot express the total at
    all, and the count survives only as explicit logarithmically spaced edges
    passed to ``gta.sed(loge_bins=...)`` over the analysis energy range.

    Returns ``{'mode': 'binsperdec', 'nbins': N, 'binsperdec': int}`` or
    ``{'mode': 'loge_bins', 'nbins': N, 'loge_bins': [...]}`` (edges in
    log10(E/MeV)), or None when the inputs do not describe an energy range.

    Note that ``gta.sed()`` snaps every edge to the closest analysis bin edge
    (``set_energy_range``), so edges off the grid give slightly uneven bin
    widths; the count is preserved as long as N does not exceed the number of
    analysis bins.
    """
    try:
        emin = float(emin)
        emax = float(emax)
        nbins = int(nbins)
    except (TypeError, ValueError):
        return None
    if not (emin > 0 and emax > emin and nbins >= 2):
        return None

    logemin = math.log10(emin)
    logemax = math.log10(emax)
    ndec = logemax - logemin
    if ndec <= 0:
        return None

    exact = nbins / ndec
    whole = round(exact)
    if whole >= 1 and math.isclose(exact, whole, rel_tol=1e-9, abs_tol=1e-9):
        return {'mode': 'binsperdec', 'nbins': nbins, 'binsperdec': int(whole)}

    edges = [round(logemin + i * ndec / nbins, 6) for i in range(nbins + 1)]
    return {'mode': 'loge_bins', 'nbins': nbins, 'loge_bins': edges}

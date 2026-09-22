#!/usr/bin/env python3
"""Resolve arbitrary source names against the 4FGL catalog.

Maps a user-named source ("PG 1553+113", "Mkn 501", "PSR J1028-5819",
"4FGL J0428.6-3756") onto its catalog entry — exact 4FGL designation,
coordinates, class, and spectral parameters — so the pipeline can run on
any catalog source instead of only the three bundled demo sources.

The catalog FITS (gll_psc_v35.fit, 4FGL-DR4) ships inside fermipy's
package data; override its location with FERMI_LLM_CATALOG_FITS.

There is deliberately NO fallback: an unresolvable name returns
found=False and the caller must surface that to the user instead of
silently analyzing a different source.
"""

import os
import re
import glob
import threading

ASSOC_COLUMNS = (
    'ASSOC1', 'ASSOC2', 'ASSOC_GAM1', 'ASSOC_GAM2', 'ASSOC_GAM3',
    'ASSOC_TEV', 'ASSOC_FGL', 'ASSOC_FHL',
)

# Qualifier words that describe the source type but not its identity.
# Stripping them lets "Crab pulsar" match "Crab" in the catalog while
# preserving the hint for class-based tiebreaking.
QUALIFIERS = {
    'pulsar', 'nebula', 'remnant', 'source', 'region', 'field',
    'snr', 'pwn', 'blazar', 'quasar', 'galaxy', 'cluster',
}

# Qualifier → preferred 4FGL CLASS1 values for multi-hit disambiguation.
QUALIFIER_CLASS_HINTS = {
    'pulsar': {'PSR', 'MSP'},
    'nebula': {'PWN'},
    'remnant': {'SNR', 'spp'},
    'snr': {'SNR', 'spp'},
    'pwn': {'PWN'},
    'blazar': {'BLL', 'FSRQ', 'BCU', 'bll', 'fsrq', 'bcu'},
    'quasar': {'FSRQ', 'QSO', 'fsrq'},
    'galaxy': {'RDG', 'GAL', 'SEY', 'SBG', 'rdg', 'gal', 'sey', 'sbg'},
}

# Bidirectional aliases (same-group names are interchangeable during matching).
ALIAS_GROUPS = [
    ('markarian', 'mrk', 'mkn'),
    ('galactic center', 'galactic centre', 'sgr a*', 'sgr a star', 'sgr a'),
]

_lock = threading.Lock()
_catalog_rows = None      # list of dicts, one per catalog row
_extended_rows = None     # list of dicts from the ExtendedSources HDU
_alias_to_rows = None     # normalized alias → [row index, ...]
_dr3_names = None         # set of Source_Name in the DR3 catalog, or None


class SourceResolutionError(Exception):
    """A source name could not be resolved against the 4FGL catalog."""


def _fermipy_catalog_dir():
    """fermipy's data/catalogs dir, WITHOUT importing fermipy.

    Importing fermipy pulls in the Fermi science tools, which abort hard
    when FERMI_DIR/CALDB are not set — name resolution must not depend
    on that (the web server process never needs the science tools).
    """
    import importlib.util
    spec = importlib.util.find_spec('fermipy')
    if spec is None or not spec.submodule_search_locations:
        return None
    return os.path.join(next(iter(spec.submodule_search_locations)),
                        'data', 'catalogs')


def find_catalog_fits(version_prefix='gll_psc_v'):
    """Locate the newest 4FGL catalog FITS inside fermipy's package data."""
    env = os.environ.get('FERMI_LLM_CATALOG_FITS', '')
    if env:
        if not os.path.exists(env):
            raise SourceResolutionError(
                f'FERMI_LLM_CATALOG_FITS points to a missing file: {env}')
        return env
    cat_dir = _fermipy_catalog_dir()
    if not cat_dir:
        raise SourceResolutionError(
            'fermipy is not installed and FERMI_LLM_CATALOG_FITS is not set '
            '-- cannot locate the 4FGL catalog FITS')
    candidates = glob.glob(os.path.join(cat_dir, version_prefix + '*.fit'))
    versioned = []
    for c in candidates:
        m = re.search(r'gll_psc_v(\d+)\.fit$', c)
        if m:
            versioned.append((int(m.group(1)), c))
    if not versioned:
        raise SourceResolutionError(
            f'no gll_psc_v*.fit catalog found under {cat_dir}')
    return max(versioned)[1]


def _norm(s):
    if s is None:
        return ''
    if isinstance(s, bytes):
        s = s.decode('ascii', errors='ignore')
    return re.sub(r'\s+', ' ', str(s).strip().lower())


def _nospace(s):
    return s.replace(' ', '')


def _strip_catalog_prefix(q):
    for prefix in ('4fgl ', '4fgl-', '3fgl ', '2fgl ', '1fgl '):
        if q.startswith(prefix):
            return q[len(prefix):]
    return q


def _expand_aliases(s):
    out = {s}
    for group in ALIAS_GROUPS:
        for alias in group:
            if alias in s:
                for other in group:
                    if other != alias:
                        out.add(s.replace(alias, other))
    return out


def _query_variants(query):
    """Build a set of normalized forms to try against catalog values."""
    q = _norm(query)
    variants = {q, _nospace(q)} | _expand_aliases(q)

    tokens = q.split()
    stripped = ' '.join(t for t in tokens if t not in QUALIFIERS)
    if stripped and stripped != q:
        variants.add(stripped)
        variants.add(_nospace(stripped))
        variants |= _expand_aliases(stripped)

    for v in list(variants):
        variants.add(_nospace(v))
    return {v for v in variants if v}


def _query_class_hints(query):
    q = _norm(query)
    hints = set()
    for tok in q.split():
        if tok in QUALIFIER_CLASS_HINTS:
            hints |= QUALIFIER_CLASS_HINTS[tok]
    return hints


def _isnan(x):
    try:
        return x != x  # NaN != NaN
    except TypeError:
        return False


def _fval(row, key):
    v = row.get(key)
    if v is None or _isnan(v):
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _load_catalog():
    """Load the catalog once into plain-python rows (thread-safe, cached)."""
    global _catalog_rows, _extended_rows
    if _catalog_rows is not None:
        return _catalog_rows
    with _lock:
        if _catalog_rows is not None:
            return _catalog_rows
        from astropy.io import fits
        path = find_catalog_fits()
        wanted = (('Source_Name', 'RAJ2000', 'DEJ2000', 'GLON', 'GLAT',
                   'SpectrumType', 'CLASS1', 'Extended_Source_Name',
                   'Signif_Avg', 'PL_Index', 'PL_Flux_Density',
                   'Pivot_Energy') + ASSOC_COLUMNS)
        rows = []
        ext_rows = []
        with fits.open(path) as hdul:
            data = hdul[1].data
            colnames = set(data.columns.names)
            for r in data:
                rows.append({k: r[k] for k in wanted if k in colnames})
            if len(hdul) > 2:
                ext_data = hdul[2].data
                ext_cols = set(ext_data.columns.names)
                for er in ext_data:
                    ext_rows.append({k: er[k] for k in
                                     ('Source_Name', 'Model_Form',
                                      'Model_SemiMajor', 'Model_SemiMinor',
                                      'Spatial_Function', 'Spatial_Filename')
                                     if k in ext_cols})
        _extended_rows = ext_rows
        _catalog_rows = rows
        return _catalog_rows


def _alias_index():
    """normalized-nospace alias → list of catalog row indices (cached)."""
    global _alias_to_rows
    if _alias_to_rows is not None:
        return _alias_to_rows
    rows = _load_catalog()
    with _lock:
        if _alias_to_rows is not None:
            return _alias_to_rows
        index = {}

        def add(key, i):
            if key:
                index.setdefault(key, []).append(i)

        for i, r in enumerate(rows):
            sn = _norm(r.get('Source_Name'))
            add(_nospace(sn), i)
            add(_nospace(_strip_catalog_prefix(sn)), i)
            for col in ASSOC_COLUMNS:
                v = _norm(r.get(col))
                if not v:
                    continue
                for form in _expand_aliases(v):
                    add(_nospace(form), i)
            ext = _norm(r.get('Extended_Source_Name'))
            if ext:
                add(_nospace(ext), i)
        _alias_to_rows = index
        return _alias_to_rows


def _row_to_dict(row):
    ext_name = _norm(row.get('Extended_Source_Name')) and \
        str(row.get('Extended_Source_Name')).strip()
    ext_template = None
    if ext_name and _extended_rows:
        for er in _extended_rows:
            if str(er.get('Source_Name', '')).strip() == ext_name:
                ext_template = {
                    'form': str(er.get('Model_Form', '')).strip(),
                    'semi_major_deg': _fval(er, 'Model_SemiMajor'),
                    'semi_minor_deg': _fval(er, 'Model_SemiMinor'),
                    'spatial_function':
                        str(er.get('Spatial_Function', '')).strip() or None,
                    'spatial_filename':
                        str(er.get('Spatial_Filename', '')).strip() or None,
                }
                break
    return {
        'found': True,
        'source_name': str(row.get('Source_Name', '')).strip(),
        'ra': _fval(row, 'RAJ2000'),
        'dec': _fval(row, 'DEJ2000'),
        'glon': _fval(row, 'GLON'),
        'glat': _fval(row, 'GLAT'),
        'spectrum_type': str(row.get('SpectrumType', '')).strip(),
        'class1': str(row.get('CLASS1', '')).strip(),
        'assoc1': str(row.get('ASSOC1', '')).strip(),
        'assoc2': str(row.get('ASSOC2', '')).strip(),
        'assoc_tev': str(row.get('ASSOC_TEV', '')).strip(),
        'signif_avg': _fval(row, 'Signif_Avg'),
        'extended': bool(ext_name),
        'extended_template': ext_template,
    }


def _signif(row):
    return _fval(row, 'Signif_Avg') or 0.0


def _match_value(query_variants, catalog_value):
    cv = _norm(catalog_value)
    if not cv:
        return False
    cv_forms = {cv, _nospace(cv)}
    for q in query_variants:
        if q in cv_forms:
            return True
    for q in query_variants:
        if len(q) >= 4:
            for cv_form in cv_forms:
                if q in cv_form or cv_form in q:
                    return True
    return False


def resolve(query):
    """Resolve a source description to its 4FGL catalog entry.

    Returns {'found': True, 'source_name': '4FGL J....', 'ra': ..., ...}
    (with 'ambiguous'/'candidates' when several sources match) or
    {'found': False, 'query': ...} on a miss.
    """
    rows = _load_catalog()
    q = _norm(query)
    if not q:
        return {'found': False, 'query': query}
    q_jname = _strip_catalog_prefix(q)

    # Exact / prefix match on the 4FGL J-designation first.
    jname_hits = []
    for r in rows:
        sn_j = _strip_catalog_prefix(_norm(r.get('Source_Name')))
        if q_jname == sn_j:
            jname_hits = [r]
            break
    if not jname_hits and re.match(r'j\d{4}', q_jname):
        jname_hits = [r for r in rows
                      if _strip_catalog_prefix(
                          _norm(r.get('Source_Name'))).startswith(q_jname)]
        jname_hits.sort(key=lambda r: -_signif(r))
    if jname_hits:
        result = _row_to_dict(jname_hits[0])
        if len(jname_hits) > 1:
            result['ambiguous'] = True
            result['candidates'] = [
                str(h.get('Source_Name', '')).strip() for h in jname_hits[:5]]
        return result

    variants = _query_variants(query)
    class_hints = _query_class_hints(query)
    raw_hits = []
    for r in rows:
        for col in ASSOC_COLUMNS + ('Extended_Source_Name',):
            if _match_value(variants, r.get(col)):
                raw_hits.append(r)
                break
    hits = raw_hits
    if class_hints:
        preferred = [r for r in raw_hits
                     if str(r.get('CLASS1', '')).strip() in class_hints]
        hits = preferred if preferred else raw_hits
    hits.sort(key=lambda r: -_signif(r))

    if not hits:
        return {'found': False, 'query': query}

    result = _row_to_dict(hits[0])
    if len(hits) > 1:
        result['ambiguous'] = True
        result['candidates'] = [
            str(r.get('Source_Name', '')).strip() for r in hits[:5]]
    return result


# Words that commonly appear in analysis prompts and must never be treated
# as a source name on their own during free-text scanning.
_TEXT_SCAN_STOPWORDS = QUALIFIERS | {
    'analyze', 'analyse', 'analysis', 'spectral', 'spectrum', 'energy',
    'light', 'curve', 'lightcurve', 'flux', 'index', 'fermi', 'lat',
    'data', 'events', 'front', 'back', 'source', 'target', 'catalog',
}


def find_source_in_text(text):
    """Scan free text (a user prompt) for a catalog source name.

    Tries word n-grams (longest first) against the catalog alias index,
    so "compute the SED of PG 1553+113 over one year" finds PG 1553+113.
    Returns the resolve()-style dict of the best match plus
    'matched_text', or None when nothing in the text matches.
    """
    if not text:
        return None
    index = _alias_index()
    tokens = re.findall(r"[A-Za-z0-9.+*'-]+", text)
    if not tokens:
        return None

    best = None  # (ngram_len, signif, raw_text)
    for n in range(min(4, len(tokens)), 0, -1):
        for i in range(len(tokens) - n + 1):
            gram_tokens = tokens[i:i + n]
            raw = ' '.join(gram_tokens)
            qn = _norm(raw)
            if n == 1:
                tok = qn
                # Single words: require either a digit ("3c279") or a
                # reasonably specific name ("crab", "vela", "geminga").
                if tok in _TEXT_SCAN_STOPWORDS:
                    continue
                if not any(c.isdigit() for c in tok) and len(tok) < 4:
                    continue
                if tok.isdigit():
                    continue
            candidates = set()
            for form in _query_variants(raw):
                candidates.add(form)
            hit_rows = []
            for cand in candidates:
                hit_rows.extend(index.get(cand, []))
            if not hit_rows:
                continue
            rows = _load_catalog()
            top_signif = max(_signif(rows[j]) for j in set(hit_rows))
            key = (n, top_signif)
            if best is None or key > best[0]:
                best = (key, raw)
        if best is not None:
            break  # longest n-grams win; no need to try shorter ones

    if best is None:
        return None
    raw = best[1]
    result = resolve(raw)
    if not result.get('found'):
        return None
    result['matched_text'] = raw
    return result


def source_in_dr3(fgl_name):
    """True when the 4FGL name also exists in the DR3 catalog (gll_psc_v29).

    Used to pick model.catalogs for a resolved source: DR4-only sources
    are invisible to fermipy's '4FGL-DR3' ROI model. Returns True (the
    safe legacy default) when no DR3 catalog file can be found.
    """
    global _dr3_names
    if _dr3_names is None:
        with _lock:
            if _dr3_names is None:
                names = set()
                try:
                    from astropy.io import fits
                    cat_dir = _fermipy_catalog_dir() or ''
                    path = os.path.join(cat_dir, 'gll_psc_v29.fit')
                    if cat_dir and os.path.exists(path):
                        with fits.open(path) as hdul:
                            for r in hdul[1].data:
                                names.add(_nospace(_norm(r['Source_Name'])))
                except Exception:
                    names = set()
                _dr3_names = names
    if not _dr3_names:
        return True
    return _nospace(_norm(fgl_name)) in _dr3_names


def catalog_reference(fgl_name):
    """Build a Level-4 sanity-check reference dict for any catalog source.

    Mirrors the shape of run_level4_validation.CATALOG_REFERENCE. The
    100 MeV - 1 TeV integral photon flux is estimated from the catalog's
    power-law parameters (PL_Flux_Density at Pivot_Energy with PL_Index),
    which is an approximation for curved spectra — tolerances are
    correspondingly loose. Returns {} when the source is not found.
    """
    res = resolve(fgl_name)
    if not res.get('found'):
        return {}
    rows = _load_catalog()
    row = None
    target = _nospace(_norm(res['source_name']))
    for r in rows:
        if _nospace(_norm(r.get('Source_Name'))) == target:
            row = r
            break
    if row is None:
        return {}

    k = _fval(row, 'PL_Flux_Density')       # ph / cm2 / s / MeV at pivot
    e0 = _fval(row, 'Pivot_Energy')         # MeV
    gamma = _fval(row, 'PL_Index')
    flux = None
    if k and e0 and gamma and abs(gamma - 1.0) > 1e-3:
        e1, e2 = 100.0, 1e6                 # 100 MeV .. 1 TeV
        flux = (k * e0 / (1.0 - gamma)
                * ((e2 / e0) ** (1.0 - gamma) - (e1 / e0) ** (1.0 - gamma)))

    ref = {
        'name': res.get('assoc1') or res['source_name'],
        'flux_tolerance_log10': 2.0,
        'expected_ts_min': 9,
        'expected_ts_max': 1e9,
        'expected_index_min': 0.5,
        'expected_index_max': 5.0,
        'spectral_type': res.get('spectrum_type', ''),
        'reference_origin': ('4FGL catalog (power-law approximation over '
                             '100 MeV - 1 TeV)'),
    }
    if flux and flux > 0:
        ref['flux_100mev_1tev'] = flux
    if gamma:
        ref['catalog_index'] = gamma
        ref['expected_index_min'] = max(0.5, gamma - 1.5)
        ref['expected_index_max'] = min(5.0, gamma + 1.5)
    return ref


if __name__ == '__main__':
    import sys
    import json
    if len(sys.argv) < 2:
        print('usage: python source_resolver.py "<source name or free text>"')
        sys.exit(1)
    query = ' '.join(sys.argv[1:])
    out = resolve(query)
    if not out.get('found'):
        scanned = find_source_in_text(query)
        if scanned:
            out = scanned
    print(json.dumps(out, indent=2, default=str))

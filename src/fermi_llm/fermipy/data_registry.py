#!/usr/bin/env python3
"""Registry over the full Fermi-LAT weekly dataset.

Turns "analyze <any 4FGL source> from tmin to tmax" into concrete data
inputs for fermipy: an event-file list of the weekly all-sky photon
files covering the requested time range, plus the mission-merged
spacecraft file.

Expected layout under FERMI_LLM_FERMI_DATA_DIR (default
/workspace/fermillm/fermi-data):

    weekly/photon/lat_photon_weekly_wNNN_p305_v001.fits
    mission/spacecraft/lat_spacecraft_merged.fits

Weekly file boundaries are NOT uniform across the mission (data-taking
gaps shift them), so file selection reads TSTART/TSTOP from FITS
headers, located via binary search over the (time-sorted) file list —
O(log N) header reads per query, no pre-built index required.
"""

import os
import re
import glob
import threading

DEFAULT_DATA_DIR = '/workspace/fermillm/fermi-data'

# Time range used when the user does not specify one: the most recent
# year of available data.
DEFAULT_TIME_RANGE_DAYS = float(os.environ.get(
    'FERMI_LLM_DEFAULT_TIME_RANGE_DAYS', 365))

_lock = threading.Lock()
_header_cache = {}   # path → (tstart, tstop)
_files_cache = None


class DataRegistryError(Exception):
    """Base class for data-resolution failures (no silent fallbacks)."""


class SourceNotFoundError(DataRegistryError):
    pass


class NoDataError(DataRegistryError):
    pass


def data_root():
    return os.environ.get('FERMI_LLM_FERMI_DATA_DIR', DEFAULT_DATA_DIR)


def spacecraft_file():
    path = os.path.join(data_root(), 'mission', 'spacecraft',
                        'lat_spacecraft_merged.fits')
    if not os.path.exists(path):
        raise NoDataError(
            f'merged spacecraft file not found: {path} '
            f'(set FERMI_LLM_FERMI_DATA_DIR to the fermi-data root)')
    return path


def weekly_files(refresh=False):
    """Sorted list of weekly photon FITS paths (week order == time order)."""
    global _files_cache
    if _files_cache is not None and not refresh:
        return _files_cache
    photon_dir = os.path.join(data_root(), 'weekly', 'photon')
    files = sorted(
        f for f in glob.glob(os.path.join(photon_dir,
                                          'lat_photon_weekly_w*.fits'))
        if re.search(r'lat_photon_weekly_w\d+_p\d+_v\d+\.fits$', f))
    if not files:
        raise NoDataError(
            f'no weekly photon files found under {photon_dir} '
            f'(set FERMI_LLM_FERMI_DATA_DIR to the fermi-data root)')
    _files_cache = files
    return files


def week_time_range(path):
    """(TSTART, TSTOP) of one weekly file, header-read once per process."""
    cached = _header_cache.get(path)
    if cached:
        return cached
    from astropy.io import fits
    with fits.open(path) as hdul:
        hdr = hdul[1].header
        rng = (float(hdr['TSTART']), float(hdr['TSTOP']))
    with _lock:
        _header_cache[path] = rng
    return rng


def data_time_range():
    """(first TSTART, last TSTOP) across the whole weekly dataset."""
    files = weekly_files()
    return week_time_range(files[0])[0], week_time_range(files[-1])[1]


def _bisect_first_overlapping(files, tmin):
    """Index of the first file whose TSTOP > tmin (files are time-sorted)."""
    lo, hi = 0, len(files)
    while lo < hi:
        mid = (lo + hi) // 2
        if week_time_range(files[mid])[1] <= tmin:
            lo = mid + 1
        else:
            hi = mid
    return lo


def _bisect_last_overlapping(files, tmax):
    """Index one past the last file whose TSTART < tmax."""
    lo, hi = 0, len(files)
    while lo < hi:
        mid = (lo + hi) // 2
        if week_time_range(files[mid])[0] < tmax:
            lo = mid + 1
        else:
            hi = mid
    return lo


def select_weekly_files(tmin, tmax):
    """Weekly files overlapping the MET interval [tmin, tmax]."""
    if tmin >= tmax:
        raise NoDataError(f'invalid time range: tmin={tmin} >= tmax={tmax}')
    files = weekly_files()
    start = _bisect_first_overlapping(files, tmin)
    end = _bisect_last_overlapping(files, tmax)
    selected = files[start:end]
    if not selected:
        d0, d1 = data_time_range()
        raise NoDataError(
            f'no weekly photon data overlaps MET [{tmin:.0f}, {tmax:.0f}]; '
            f'available data covers MET [{d0:.0f}, {d1:.0f}]')
    return selected


def default_time_range(days=None):
    """Most recent `days` of available data, as a MET (tmin, tmax)."""
    if days is None:
        days = DEFAULT_TIME_RANGE_DAYS
    d0, d1 = data_time_range()
    return max(d0, d1 - days * 86400.0), d1


def write_evfile_list(files, dest_path):
    """Write a fermipy/gtselect-compatible event-file list (one per line)."""
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    with open(dest_path, 'w') as f:
        for p in files:
            f.write(p + '\n')
    return dest_path


def _normalized_source_name(value):
    return re.sub(r'[^a-z0-9.+-]', '', str(value or '').lower())


def _custom_source_for_target(cfg, query):
    """Return a validated model.sources entry matching the query.

    A non-catalog target is only executable when its position is explicit in
    the configuration. Keeping this information in model.sources makes the
    FermiPy configuration self-contained: ROIModel can center on
    selection.target even though that name is absent from 4FGL.
    """
    sources = (cfg.get('model') or {}).get('sources') or []
    if not isinstance(sources, list):
        return None
    qnorm = _normalized_source_name(query)
    for item in sources:
        if not isinstance(item, dict):
            continue
        if _normalized_source_name(item.get('name')) != qnorm:
            continue
        ra, dec = item.get('ra'), item.get('dec')
        if isinstance(ra, bool) or isinstance(dec, bool):
            return None
        if not isinstance(ra, (int, float)) or \
                not isinstance(dec, (int, float)):
            return None
        ra, dec = float(ra), float(dec)
        if not 0.0 <= ra < 360.0 or not -90.0 <= dec <= 90.0:
            return None
        spatial = str(item.get('SpatialModel') or 'PointSource')
        if spatial != 'PointSource':
            raise DataRegistryError(
                f'custom target "{query}" uses SpatialModel={spatial}; '
                f'the first custom-target implementation supports point '
                f'sources only')
        out = dict(item)
        out['name'] = str(item.get('name') or query).strip()
        out['ra'], out['dec'] = ra, dec
        out.setdefault('SpatialModel', 'PointSource')
        out.setdefault('SpectrumType', 'PowerLaw')
        out.setdefault('Index', 2.0)
        out.setdefault('Scale', 1000.0)
        out.setdefault('Prefactor', 1e-13)
        return out
    return None


def build_full_data_spec(yaml_str, list_dir, target=None):
    """Resolve an arbitrary-source config onto the full weekly dataset.

    Returns (spec, updated_yaml_str, notes):
      spec  — dict with the same keys as a TEST_DATA_MAP entry (evfile,
              scfile, target, name) plus ra/dec, mode='full', the applied
              tmin/tmax, n_weeks, and the catalog the source appears in.
      updated_yaml_str — the config with selection.target normalized to
              the exact 4FGL designation and selection.tmin/tmax filled
              in (defaulted to the most recent year when absent, clamped
              to data coverage otherwise).
      notes — human-readable list of what was decided, for progress logs.

    Raises SourceNotFoundError / NoDataError instead of falling back to
    a different source: a resolution failure must reach the user.
    """
    import yaml as _yaml
    from . import source_resolver

    try:
        cfg = _yaml.safe_load(yaml_str)
    except _yaml.YAMLError as e:
        raise DataRegistryError(f'config YAML does not parse: {e}')
    if not isinstance(cfg, dict):
        raise DataRegistryError('config YAML is not a mapping')
    sel = cfg.setdefault('selection', {})

    query = target or sel.get('target') or ''
    if not str(query).strip():
        raise SourceNotFoundError(
            'the configuration names no analysis target '
            '(selection.target is empty) and no source could be '
            'identified from the request')

    res = source_resolver.resolve(str(query))
    catalogued = bool(res.get('found'))
    custom_source = None
    if catalogued:
        target_name = res['source_name']
        target_ra, target_dec = res.get('ra'), res.get('dec')
        target_label = res.get('assoc1') or target_name
    else:
        custom_source = _custom_source_for_target(cfg, query)
    if not catalogued and custom_source is None:
        raise SourceNotFoundError(
            f'"{query}" does not match any source in the 4FGL catalog. '
            f'For a non-catalog target, provide explicit ICRS RA/Dec '
            f'coordinates so it can be added under model.sources.')
    if not catalogued:
        target_name = custom_source['name']
        target_ra, target_dec = custom_source['ra'], custom_source['dec']
        target_label = target_name

    notes = []
    if catalogued and res.get('ambiguous'):
        notes.append(
            f'"{query}" matched several catalog sources '
            f'({", ".join(res.get("candidates", [])[:5])}); using the most '
            f'significant: {res["source_name"]}')

    if str(sel.get('target') or '').strip() != target_name:
        kind = 'catalog designation' if catalogued else 'custom source name'
        notes.append(f'normalized target "{query}" -> {kind} '
                     f'"{target_name}"')
        sel['target'] = target_name
    # Coordinates are redundant for a catalog target but useful provenance;
    # for a custom source they are essential and prevent any downstream code
    # from silently centering on another object.
    if target_ra is not None and target_dec is not None:
        sel['ra'], sel['dec'] = float(target_ra), float(target_dec)
    if not catalogued:
        notes.append(
            f'using custom point source "{target_name}" at '
            f'RA={target_ra:.6f}, Dec={target_dec:.6f} (ICRS); 4FGL remains '
            f'the background-source catalog')

    d0, d1 = data_time_range()

    def _num(v):
        return float(v) if isinstance(v, (int, float)) else None

    tmin, tmax = _num(sel.get('tmin')), _num(sel.get('tmax'))
    if tmin is None or tmax is None or tmin >= tmax:
        tmin, tmax = default_time_range()
        notes.append(
            f'no (usable) time range in the request -- defaulting to the '
            f'most recent {DEFAULT_TIME_RANGE_DAYS:.0f} days of data: '
            f'MET [{tmin:.0f}, {tmax:.0f}]')
    else:
        c_tmin, c_tmax = max(tmin, d0), min(tmax, d1)
        if (c_tmin, c_tmax) != (tmin, tmax):
            notes.append(
                f'clamped requested time range [{tmin:.0f}, {tmax:.0f}] to '
                f'available data coverage [{c_tmin:.0f}, {c_tmax:.0f}]')
            tmin, tmax = c_tmin, c_tmax
        if tmin >= tmax:
            raise NoDataError(
                f'requested time range lies outside data coverage '
                f'MET [{d0:.0f}, {d1:.0f}]')
    sel['tmin'], sel['tmax'] = tmin, tmax

    files = select_weekly_files(tmin, tmax)
    evfile = write_evfile_list(
        files, os.path.join(list_dir, 'evfile_list.txt'))

    if catalogued:
        catalog = '4FGL-DR3' if source_resolver.source_in_dr3(target_name) \
            else '4FGL-DR4'
        if catalog == '4FGL-DR4':
            notes.append(f'{target_name} is not in 4FGL-DR3 -- the ROI model '
                         f'must use the 4FGL-DR4 catalog')
    else:
        configured = (cfg.get('model') or {}).get('catalogs') or []
        catalog = configured[0] if configured and \
            isinstance(configured[0], str) else '4FGL-DR3'

    spec = {
        'evfile': evfile,
        'scfile': spacecraft_file(),
        'target': target_name,
        'name': target_label,
        'ra': target_ra,
        'dec': target_dec,
        'mode': 'full',
        'catalog': catalog,
        'catalogued': catalogued,
        'target_origin': '4fgl' if catalogued else 'custom',
        'source_model': custom_source,
        'tmin': tmin,
        'tmax': tmax,
        'n_weeks': len(files),
        'class1': res.get('class1', '') if catalogued else '',
        'extended': bool(res.get('extended')) if catalogued else False,
    }
    if catalogued and res.get('extended'):
        notes.append(
            f'{target_name} is an extended source in the catalog; the ROI '
            f'model will use its extended template if the catalog provides '
            f'one')

    notes.append(f'selected {len(files)} weekly photon files covering '
                 f'MET [{tmin:.0f}, {tmax:.0f}]')

    updated_yaml = _yaml.dump(cfg, default_flow_style=False, sort_keys=False)
    return spec, updated_yaml, notes


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == '--coverage':
        d0, d1 = data_time_range()
        n = len(weekly_files())
        print(f'{n} weekly files, MET [{d0:.0f}, {d1:.0f}]')
    elif len(sys.argv) > 2 and sys.argv[1] == '--select':
        tmin, tmax = float(sys.argv[2]), float(sys.argv[3])
        for f in select_weekly_files(tmin, tmax):
            print(f)
    else:
        print('usage: python data_registry.py --coverage | '
              '--select <tmin> <tmax>')

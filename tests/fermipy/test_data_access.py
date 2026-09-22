#!/usr/bin/env python3
"""Tests for source_resolver + data_registry (no fermipy runtime needed).

Run inside the fermi-llm conda env (astropy required, FERMI_DIR not):

    python tests/fermipy/test_data_access.py

Needs the weekly dataset under FERMI_LLM_FERMI_DATA_DIR (default
/workspace/fermillm/fermi-data) for the registry half; the resolver half
only needs fermipy's bundled 4FGL catalog FITS.
"""
import os
import sys
import tempfile

SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from fermi_llm.fermipy import source_resolver as sr
from fermi_llm.fermipy import data_registry as dr

failures = []


def check(label, cond, detail=''):
    status = 'ok' if cond else 'FAIL'
    print(f'  [{status}] {label}' + (f' -- {detail}' if detail and not cond else ''))
    if not cond:
        failures.append(label)


print('== source_resolver.resolve ==')
for query, expected in [
    ('Mkn 501', '4FGL J1653.8+3945'),
    ('Mrk 501', '4FGL J1653.8+3945'),
    ('Markarian 501', '4FGL J1653.8+3945'),
    ('PG 1553+113', '4FGL J1555.7+1111'),
    ('Crab pulsar', '4FGL J0534.5+2200'),
    ('4FGL J0428.6-3756', '4FGL J0428.6-3756'),
    ('NGC 1068', '4FGL J0242.6-0000'),
    ('Geminga', '4FGL J0633.9+1746'),
    ('LS I +61 303', '4FGL J0240.5+6113'),
]:
    r = sr.resolve(query)
    check(f'resolve({query!r}) -> {expected}',
          r.get('found') and r['source_name'] == expected,
          f'got {r.get("source_name")}')
r = sr.resolve('Planet Nine')
check('resolve miss returns found=False, no fallback', r.get('found') is False)

print('== source_resolver.find_source_in_text ==')
for text, expected in [
    ('Compute the SED of PG 1553+113 over the last year', '4FGL J1555.7+1111'),
    ('analyze 3C 279 and make a light curve', '4FGL J1256.1-0547'),
    ('spectral analysis of the vela pulsar please', '4FGL J0835.3-4510'),
    ('please analyze J0534.5+2200 with front events', '4FGL J0534.5+2200'),
]:
    h = sr.find_source_in_text(text)
    check(f'text scan finds {expected}',
          h is not None and h['source_name'] == expected,
          f'got {h and h.get("source_name")}')
for text in ['make me a nice SED of some bright source',
             'compute the spectrum between 100 MeV and 300 GeV']:
    check(f'no false positive: {text[:40]!r}',
          sr.find_source_in_text(text) is None)

print('== source_resolver.catalog_reference ==')
ref = sr.catalog_reference('4FGL J1104.4+3812')
check('reference has flux and index',
      ref.get('flux_100mev_1tev', 0) > 0 and 'catalog_index' in ref)

print('== data_registry ==')
try:
    files = dr.weekly_files()
except dr.NoDataError as e:
    print(f'  [skip] weekly dataset not available here: {e}')
    sys.exit(1 if failures else 0)

d0, d1 = dr.data_time_range()
check(f'coverage sane ({len(files)} files, {(d1-d0)/86400/365.25:.1f} yr)',
      len(files) > 100 and d1 > d0)

tmin, tmax = dr.default_time_range()
sel = dr.select_weekly_files(tmin, tmax)
check(f'default range selects ~1 yr of files ({len(sel)})',
      45 <= len(sel) <= 60)
check('first file overlaps tmin', dr.week_time_range(sel[0])[1] > tmin)
check('last file overlaps tmax', dr.week_time_range(sel[-1])[0] < tmax)
i0 = files.index(sel[0])
check('file before selection does not overlap',
      i0 == 0 or dr.week_time_range(files[i0 - 1])[1] <= tmin)

with tempfile.TemporaryDirectory() as td:
    spec, new_yaml, notes = dr.build_full_data_spec(
        'selection:\n  target: PG 1553+113\n  emin: 100\n', td)
    check('spec resolves to 4FGL J1555.7+1111',
          spec['target'] == '4FGL J1555.7+1111')
    check('spec mode/full with evfile list',
          spec['mode'] == 'full' and os.path.isfile(spec['evfile']))
    check('yaml gained tmin/tmax', 'tmin:' in new_yaml and 'tmax:' in new_yaml)
    n_listed = len(open(spec['evfile']).read().split())
    check('evfile list length == n_weeks', n_listed == spec['n_weeks'])

    custom_yaml = """
selection:
  target: Candidate Alpha
  emin: 100
model:
  catalogs: [4FGL-DR3]
  sources:
    - name: Candidate Alpha
      ra: 150.123
      dec: -20.456
      SpatialModel: PointSource
      SpectrumType: PowerLaw
      Index: 2.0
      Scale: 1000.0
      Prefactor: 1.0e-13
"""
    custom, custom_yaml_out, custom_notes = dr.build_full_data_spec(
        custom_yaml, td)
    check('custom source is accepted without a 4FGL match',
          custom['target'] == 'Candidate Alpha'
          and custom['catalogued'] is False)
    check('custom source coordinates are preserved',
          custom['ra'] == 150.123 and custom['dec'] == -20.456)
    check('custom source model reaches the run spec',
          custom['source_model']['SpatialModel'] == 'PointSource')
    check('custom YAML keeps model.sources',
          'Candidate Alpha' in custom_yaml_out and 'sources:' in custom_yaml_out)

    for bad_yaml, exc in [
        ('selection:\n  target: Planet Nine\n', dr.SourceNotFoundError),
        ('binning:\n  roiwidth: 10\n', dr.SourceNotFoundError),
    ]:
        try:
            dr.build_full_data_spec(bad_yaml, td)
            check(f'{exc.__name__} raised', False, 'no exception')
        except exc:
            check(f'{exc.__name__} raised', True)

try:
    dr.select_weekly_files(1e8, 2e8)
    check('NoDataError for pre-mission range', False, 'no exception')
except dr.NoDataError:
    check('NoDataError for pre-mission range', True)

print()
if failures:
    print(f'FAILED ({len(failures)}): ' + '; '.join(failures))
    sys.exit(1)
print('ALL DATA-ACCESS TESTS PASSED')

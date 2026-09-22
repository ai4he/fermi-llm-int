"""Regression tests for Master/Validator diffuse-model consistency."""

import os
import sys

import yaml


PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
SRC_DIR = os.path.join(PROJECT_DIR, 'src')
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from fermi_llm.fermipy.diffuse_models import normalize_diffuse_yaml
from fermi_llm.components.models import prompting as model_manager


EXPECTED_ISO = {
    3: 'iso_P8R3_SOURCE_V3_v1.txt',
    1: 'iso_P8R3_SOURCE_V3_FRONT_v1.txt',
    32: 'iso_P8R3_SOURCE_V3_PSF3_v1.txt',
}


def _relative_yaml(evtype):
    return f"""selection:
  evtype: {evtype}
gtlike:
  irfs: P8R3_SOURCE_V3
model:
  galdiff: gll_iem_v07.fits
  isodiff: {EXPECTED_ISO[evtype]}
"""


def _assert_canonical(config, evtype):
    model = config['model']
    assert isinstance(model['galdiff'], list)
    assert isinstance(model['isodiff'], list)
    assert len(model['galdiff']) == 1
    assert len(model['isodiff']) == 1
    assert os.path.isabs(model['galdiff'][0])
    assert os.path.isabs(model['isodiff'][0])
    assert os.path.exists(model['galdiff'][0])
    assert os.path.exists(model['isodiff'][0])
    assert os.path.basename(model['galdiff'][0]) == 'gll_iem_v07.fits'
    assert os.path.basename(model['isodiff'][0]) == EXPECTED_ISO[evtype]


def test_normalizer_maps_all_supported_event_types_and_is_idempotent():
    for evtype in EXPECTED_ISO:
        normalized, changed = normalize_diffuse_yaml(_relative_yaml(evtype))
        assert changed == ['galdiff', 'isodiff']
        _assert_canonical(yaml.safe_load(normalized), evtype)
        normalized_again, changed_again = normalize_diffuse_yaml(normalized)
        assert normalized_again == normalized
        assert changed_again == []


def test_prompt_schema_and_few_shot_use_absolute_list_values():
    train = [{
        'prompt': 'Example task',
        'response': {'yaml': _relative_yaml(32), 'script': 'x = 1'},
    }]
    prompt = model_manager.build_prompt('Analyze a PSF3 target', train,
                                        n_shots=1)
    assert 'galdiff: list' in prompt
    assert '/galdiffuse/gll_iem_v07.fits' in prompt
    assert '/galdiffuse/iso_P8R3_SOURCE_V3_PSF3_v1.txt' in prompt
    assert 'galdiff: gll_iem_v07.fits' not in prompt


def test_template_master_generates_canonical_diffuse_values():
    from fermi_llm.components.intent.rule_based import MasterAgent

    agent = MasterAgent(None)
    for evtype in EXPECTED_ISO:
        analysis = {
            'target': '4FGL J1104.4+3812',
            'energy_range': None,
            'event_type': evtype,
            'analyses': [],
        }
        yaml_str, _ = agent.generate_config('test', analysis)
        _assert_canonical(yaml.safe_load(yaml_str), evtype)


def test_validator_does_not_repair_already_canonical_diffuse_values():
    from fermi_llm.fermipy import run_execution_validated as validator

    meta = validator.get_test_meta(0)
    config = {
        'logging': {'verbosity': 2, 'chatter': 2},
        'fileio': {'outdir': 'output', 'logfile': 'test'},
        'data': {'evfile': meta['evfile'], 'scfile': meta['scfile']},
        'binning': {'roiwidth': 10, 'binsz': 0.1, 'binsperdec': 8},
        'selection': {
            'emin': 1000, 'emax': 100000, 'zmax': 105,
            'target': meta['target'], 'radius': 15,
            'evclass': 128, 'evtype': 3,
            'filter': 'DATA_QUAL>0 && LAT_CONFIG==1',
        },
        'gtlike': {
            'edisp': True, 'edisp_disable': ['isodiff'],
            'irfs': 'P8R3_SOURCE_V3',
        },
        'model': {'catalogs': ['4FGL-DR4']},
    }
    canonical_yaml, _ = normalize_diffuse_yaml(
        yaml.safe_dump(config, sort_keys=False))
    _, repairs = validator.repair_config(canonical_yaml, 0)
    assert not any('isodiff' in item or 'galdiff' in item
                   for item in repairs), repairs


if __name__ == '__main__':
    import traceback

    tests = [obj for name, obj in sorted(globals().items())
             if name.startswith('test_') and callable(obj)]
    failed = 0
    for test in tests:
        try:
            test()
            print(f'  PASS: {test.__name__}')
        except Exception:                                 # noqa: BLE001
            failed += 1
            print(f'  FAIL: {test.__name__}')
            traceback.print_exc()
    print(f'RESULT: {len(tests) - failed}/{len(tests)} passed')
    sys.exit(1 if failed else 0)

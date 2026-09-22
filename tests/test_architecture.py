#!/usr/bin/env python3
"""Tests for the plugin architecture itself.

These are the guarantees the integration rules promise to other institutes:
components stack in priority order, a plugin can replace or disable a core
component, a broken plugin cannot stop the platform, and the example plugins
in ``examples/plugins`` actually load.

Run:  python tests/test_architecture.py
"""

import os
import sys
import tempfile
import traceback

import conftest  # noqa: F401  (bootstraps sys.path and the context)

from fermi_llm.core import kinds
from fermi_llm.core.config import load_settings
from fermi_llm.core.context import AppContext, build_context
from fermi_llm.core.contracts import ChatTurn, ValidationResult, check_contract
from fermi_llm.core.errors import ContractError, DuplicateComponent
from fermi_llm.core.hooks import HookBus
from fermi_llm.core.loader import LoadReport, load_paths
from fermi_llm.core.registry import Registry

EXAMPLES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        'examples', 'plugins')


def _fresh_registry():
    return Registry()


def _ctx(registry):
    settings = load_settings()
    return AppContext(settings=settings, registry=registry, hooks=HookBus())


# ---------------------------------------------------------------- registry

def test_components_stack_in_priority_order():
    reg = _fresh_registry()
    for name, priority in (('c', 300), ('a', 100), ('b', 200)):
        reg.register(kinds.GUARDRAIL, name, lambda ctx, n=name: n,
                     priority=priority)
    assert [r.name for r in reg.active(kinds.GUARDRAIL)] == ['a', 'b', 'c']


def test_duplicate_registration_is_refused_with_guidance():
    reg = _fresh_registry()
    reg.register(kinds.EXPORTER, 'notebook', lambda ctx: None, source='core')
    try:
        reg.register(kinds.EXPORTER, 'notebook', lambda ctx: None,
                     source='otherlab')
    except DuplicateComponent as exc:
        assert 'replaces' in str(exc)
    else:
        raise AssertionError('duplicate registration should be refused')


def test_replacement_supersedes_the_core_component():
    reg = _fresh_registry()
    reg.register(kinds.EXPORTER, 'notebook', lambda ctx: 'core', source='core')
    reg.register(kinds.EXPORTER, 'notebook_v2', lambda ctx: 'plugin',
                 replaces=('notebook',), source='otherlab')
    active = [r.name for r in reg.active(kinds.EXPORTER)]
    assert active == ['notebook_v2']
    states = {(d['name']): d['state'] for d in reg.diagnostics()}
    assert states['notebook'] == 'replaced'
    assert states['notebook_v2'] == 'active'


def test_disabling_a_component_leaves_it_visible():
    reg = _fresh_registry()
    reg.register(kinds.GUARDRAIL, 'full_data_defaults', lambda ctx: None)
    reg.disable(kinds.GUARDRAIL, 'full_data_defaults')
    assert reg.active(kinds.GUARDRAIL) == []
    assert [d['state'] for d in reg.diagnostics()] == ['disabled']
    reg.enable(kinds.GUARDRAIL, 'full_data_defaults')
    assert len(reg.active(kinds.GUARDRAIL)) == 1


def test_component_with_unmet_requirement_is_blocked_not_fatal():
    reg = _fresh_registry()
    reg.register(kinds.EXPORTER, 'needs_base', lambda ctx: None,
                 requires=('base_exporter',))
    assert reg.active(kinds.EXPORTER) == []
    assert [d['state'] for d in reg.diagnostics()] == ['blocked']


def test_singleton_selection_prefers_configuration_then_priority():
    reg = _fresh_registry()
    reg.register(kinds.EXECUTION_BACKEND, 'local', lambda ctx: 'local',
                 priority=100)
    reg.register(kinds.EXECUTION_BACKEND, 'cluster', lambda ctx: 'cluster',
                 priority=200)
    assert reg.selected(kinds.EXECUTION_BACKEND, None).name == 'local'
    assert reg.selected(kinds.EXECUTION_BACKEND, 'cluster').name == 'cluster'


# ---------------------------------------------------------------- contracts

def test_contract_violation_names_the_missing_member():
    class Incomplete:
        name = 'oops'

    try:
        check_contract(kinds.EXPORTER, Incomplete())
    except ContractError as exc:
        assert 'media_type' in str(exc) and 'docs/module-types.md' in str(exc)
    else:
        raise AssertionError('an incomplete component should be rejected')


def test_building_a_stack_skips_a_component_that_raises():
    reg = _fresh_registry()

    def broken(ctx):
        raise RuntimeError('boom')

    class Fine:
        name = 'fine'
        stage = 'yaml'

        def __init__(self, ctx):
            pass

        def validate(self, artifact):
            return ValidationResult(ok=True, stage='yaml')

    reg.register(kinds.VALIDATOR, 'broken', broken, priority=100)
    reg.register(kinds.VALIDATOR, 'fine', Fine, priority=200)
    ctx = _ctx(reg)
    built = ctx.build_stack(kinds.VALIDATOR)
    assert len(built) == 1 and isinstance(built[0], Fine)


# ------------------------------------------------------------------ loader

def test_failing_plugin_is_reported_and_skipped():
    reg = _fresh_registry()
    ctx = _ctx(reg)
    report = LoadReport()
    with tempfile.TemporaryDirectory() as tmp:
        with open(os.path.join(tmp, 'broken.py'), 'w') as handle:
            handle.write('raise RuntimeError("bad plugin")\n')
        with open(os.path.join(tmp, 'good.py'), 'w') as handle:
            handle.write(
                'from fermi_llm.core import kinds\n'
                'def register(registry, ctx=None):\n'
                '    registry.register(kinds.EXPORTER, "good",\n'
                '                      lambda c: None, source="test")\n')
        load_paths([tmp], ctx, report, reg)
    assert [f['source'].endswith('broken.py') for f in report.failures] == [True]
    assert 'good' in reg.names(kinds.EXPORTER)


def test_example_plugins_load_and_register():
    reg = _fresh_registry()
    ctx = _ctx(reg)
    report = LoadReport()
    load_paths([EXAMPLES], ctx, report, reg)
    assert report.failures == [], report.failures
    assert 'sed_csv' in reg.names(kinds.EXPORTER)
    assert 'echo' in reg.names(kinds.MODEL_PROVIDER)
    assert 'spectrum_panel' in reg.names(kinds.UI_EXTENSION)
    assert 'midnight' in reg.names(kinds.SKIN)
    assert 'strict_energy_range' in reg.names(kinds.VALIDATOR)


def test_example_validator_blocks_an_out_of_band_energy_range():
    reg = _fresh_registry()
    ctx = _ctx(reg)
    load_paths([EXAMPLES], ctx, LoadReport(), reg)
    validator = reg.create(kinds.VALIDATOR, 'strict_energy_range', ctx)
    ok = validator.validate({'yaml': 'selection:\n  emin: 100\n  emax: 100000\n'})
    assert ok.ok
    bad = validator.validate({'yaml': 'selection:\n  emin: 1\n  emax: 100000\n'})
    assert not bad.ok and 'floor' in bad.errors[0]


def test_example_model_provider_serves_a_generation():
    reg = _fresh_registry()
    ctx = _ctx(reg)
    load_paths([EXAMPLES], ctx, LoadReport(), reg)
    provider = reg.create(kinds.MODEL_PROVIDER, 'echo', ctx)
    spec = list(provider.list_models())[0]
    assert spec.id == 'echo-demo'

    from fermi_llm.core.contracts import GenerationRequest
    result = provider.generate(GenerationRequest(
        model_id=spec.id, spec=spec, user_prompt='anything'))
    assert 'selection:' in result.yaml and 'GTAnalysis' in result.python


# -------------------------------------------------------------------- hooks

def test_hooks_run_in_order_and_survive_a_failing_listener():
    bus = HookBus()
    seen = []

    bus.on('run.started', lambda **kw: seen.append('second'), priority=200)
    bus.on('run.started', lambda **kw: seen.append('first'), priority=100)

    def explode(**kwargs):
        raise RuntimeError('listener failure')

    bus.on('run.started', explode, priority=150, name='explode')
    bus.emit('run.started', session=None, run_id='x')
    assert seen == ['first', 'second']
    assert bus.errors and bus.errors[0][0] == 'run.started'


# ------------------------------------------------------- live platform view

def test_live_context_has_one_component_of_every_required_kind():
    ctx = conftest.context()
    snapshot = ctx.registry.snapshot()
    for kind in (kinds.MODEL_PROVIDER, kinds.GUARDRAIL, kinds.VALIDATOR,
                 kinds.EXECUTION_BACKEND, kinds.SESSION_STORE,
                 kinds.AUTH_PROVIDER, kinds.DATA_SOURCE, kinds.EXPORTER,
                 kinds.KNOWLEDGE_SOURCE, kinds.UI_EXTENSION, kinds.SKIN,
                 kinds.INTENT_ANALYZER, kinds.PIPELINE_STAGE):
        assert snapshot.get(kind), f'no active component for {kind}'
    assert ctx.load_report.failures == [], ctx.load_report.failures


def test_guardrail_chain_is_ordered_and_documented():
    ctx = conftest.context()
    names = [g.name for g in ctx.build_stack(kinds.GUARDRAIL)]
    assert names == ['edit_merge', 'target_enforcement', 'target_preservation',
                     'product_reconciliation', 'full_data_defaults',
                     'diffuse_paths']
    for registration in ctx.registry.active(kinds.GUARDRAIL):
        assert registration.metadata.get('label'), registration.name


def test_a_guardrail_can_be_disabled_without_breaking_the_chain():
    from fermi_llm.services.chat import run_guardrails
    ctx = conftest.context()
    ctx.registry.disable(kinds.GUARDRAIL, 'diffuse_paths')
    ctx.invalidate(kinds.GUARDRAIL)
    try:
        names = [g.name for g in ctx.build_stack(kinds.GUARDRAIL)]
        assert 'diffuse_paths' not in names
        turn = ChatTurn(session=None, user_message='no target here',
                        yaml='selection:\n  target: 4FGL J1104.4+3812\n',
                        analysis={})
        assert run_guardrails(turn) is turn
    finally:
        ctx.registry.enable(kinds.GUARDRAIL, 'diffuse_paths')
        ctx.invalidate(kinds.GUARDRAIL)


TESTS = [obj for name, obj in sorted(globals().items())
         if name.startswith('test_') and callable(obj)]


def main():
    failures = []
    for test in TESTS:
        try:
            test()
            print(f'  PASS: {test.__name__}')
        except Exception as exc:                          # noqa: BLE001
            failures.append((test.__name__, exc))
            print(f'  FAIL: {test.__name__}: {exc}')
            traceback.print_exc()
    print('=' * 70)
    print(f'RESULT: {len(TESTS) - len(failures)}/{len(TESTS)} passed')
    return 1 if failures else 0


if __name__ == '__main__':
    sys.exit(main())

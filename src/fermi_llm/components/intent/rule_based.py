"""The rule-based analyzer (the former ``MasterAgent``).

It reads a request without an LLM: the target, the energy range, the time
range and which products were asked for. Two jobs depend on it — the
``template`` model generates entirely from this plan, and every other model's
output is checked against it by the guardrails.

Registered as the default ``intent_analyzer``. A lab that wants an LLM-based
planner registers its own and selects it in ``plugins.toml``.
"""

from __future__ import annotations

import json
import os
import re
import yaml
from datetime import datetime

from ...core import kinds
from ...core.registry import REGISTRY
from ...core.runtime import RUNTIME
from ...fermipy.diffuse_models import normalize_diffuse_config
from ...fermipy.targets import target_is_bundled as _target_is_bundled


class MasterAgent:
    """Generates FermiPy configs with explainability."""

    name = 'rule_based'

    def __init__(self, ctx=None):
        self.ctx = ctx

    @property
    def knowledge(self):
        """Every active knowledge source, stacked."""
        if self.ctx is None:
            return []
        from ...core import kinds
        return self.ctx.build_stack(kinds.KNOWLEDGE_SOURCE)

    def analyze_prompt(self, prompt):
        """Analyze user prompt and return planning/rationale."""
        analysis = {
            'target': None,
            'energy_range': None,
            'event_type': None,
            'analyses': [],
            'rationale': [],
            'decisions': [],
        }

        prompt_lower = prompt.lower()

        # Detect target
        target_match = re.search(r'4FGL\s+J[\d.+-]+', prompt)
        if target_match:
            analysis['target'] = target_match.group(0)
        elif 'crab' in prompt_lower:
            analysis['target'] = '4FGL J0534.5+2200'
            analysis['rationale'].append('Identified Crab Nebula -> using catalog name 4FGL J0534.5+2200 (pulsar component)')
        elif 'vela' in prompt_lower:
            analysis['target'] = '4FGL J0835.3-4510'
            analysis['rationale'].append('Identified Vela pulsar -> 4FGL J0835.3-4510')
        elif 'markarian 421' in prompt_lower or 'mrk 421' in prompt_lower \
                or 'mrk421' in prompt_lower or 'mkn 421' in prompt_lower:
            analysis['target'] = '4FGL J1104.4+3812'
            analysis['rationale'].append('Identified Markarian 421 -> 4FGL J1104.4+3812')
        else:
            # Arbitrary sources: scan the prompt against the 4FGL catalog
            # (designations + association names). No fallback -- when
            # nothing matches, the analysis simply has no target and the
            # pipeline will refuse to run rather than substituting a
            # different source.
            try:
                from ...fermipy import source_resolver
                hit = source_resolver.find_source_in_text(prompt)
            except Exception as e:
                hit = None
                analysis['rationale'].append(
                    f'Source resolution unavailable ({type(e).__name__}: '
                    f'{str(e)[:120]})')
            if hit:
                analysis['target'] = hit['source_name']
                analysis['resolved_source'] = {
                    'target': hit['source_name'],
                    'name': hit.get('assoc1') or hit['source_name'],
                    'ra': hit.get('ra'),
                    'dec': hit.get('dec'),
                    'class1': hit.get('class1'),
                    'matched_text': hit.get('matched_text'),
                }
                assoc = f" ({hit['assoc1']})" if hit.get('assoc1') else ''
                analysis['rationale'].append(
                    f'Resolved "{hit.get("matched_text")}" -> '
                    f'{hit["source_name"]}{assoc} from the 4FGL catalog '
                    f'(RA={hit["ra"]:.3f}, Dec={hit["dec"]:.3f})')
                if hit.get('ambiguous'):
                    analysis['rationale'].append(
                        'Several catalog sources matched: '
                        + ', '.join(hit.get('candidates', [])[:5])
                        + ' -- picked the most significant; name the exact '
                          '4FGL designation to override')

        # Detect energy range
        energy_match = re.findall(r'(\d+)\s*(MeV|GeV|TeV)', prompt, re.IGNORECASE)
        if energy_match:
            energies = []
            for val, unit in energy_match:
                mev = float(val)
                if unit.lower() == 'gev':
                    mev *= 1000
                elif unit.lower() == 'tev':
                    mev *= 1e6
                energies.append(mev)
            if len(energies) >= 2:
                analysis['energy_range'] = (min(energies), max(energies))
                analysis['rationale'].append(f'Energy range: {min(energies):.0f} - {max(energies):.0f} MeV')

        # Detect event type
        if 'psf3' in prompt_lower:
            analysis['event_type'] = 32
            analysis['rationale'].append('PSF3 event type selected -> evtype=32, best angular resolution')
            analysis['decisions'].append('Using PSF3-specific isotropic model: iso_P8R3_SOURCE_V3_PSF3_v1.txt')
        elif 'front only' in prompt_lower or 'front event' in prompt_lower or 'evtype 1' in prompt_lower:
            analysis['event_type'] = 1
            analysis['rationale'].append('FRONT-only events selected -> evtype=1, better PSF')
            analysis['decisions'].append('Using FRONT-specific isotropic model: iso_P8R3_SOURCE_V3_FRONT_v1.txt')
        else:
            analysis['event_type'] = 3
            analysis['rationale'].append('Using FRONT+BACK events (default) -> evtype=3')

        # Detect requested analyses.  Users rarely spell out "SED": phrasings
        # like "extract a spectrum" must count as an SED request.  A bare
        # mention of "spectrum" is NOT enough, though — "the spectrum should
        # be a log-parabola model" is a model spec, not a product request —
        # so spectrum/spectra only counts alongside a produce-verb in the
        # same sentence.  \bsed\b (not a substring test) so "used"/"based"
        # don't trigger it.
        sed_requested = (
            re.search(r'\bsed\b', prompt_lower) is not None
            or 'spectral energy' in prompt_lower
            or 'spectral analysis' in prompt_lower
            or re.search(
                r'\b(extract\w*|produc\w*|comput\w*|generat\w*|deriv\w*|'
                r'measur\w*|obtain\w*|creat\w*|plot\w*|make|show)\b'
                r'[^.!?]{0,60}?\b(spectrum|spectra)\b', prompt_lower) is not None)
        if sed_requested:
            analysis['analyses'].append('SED (Spectral Energy Distribution)')
        if 'ts map' in prompt_lower or 'tsmap' in prompt_lower:
            analysis['analyses'].append('TS Map')
        if 'residual' in prompt_lower or 'residmap' in prompt_lower:
            analysis['analyses'].append('Residual Map')
        if 'light curve' in prompt_lower or 'lightcurve' in prompt_lower:
            analysis['analyses'].append('Light Curve')
        if 'extension' in prompt_lower or 'spatial extent' in prompt_lower:
            analysis['analyses'].append('Extension Test')
        if 'localize' in prompt_lower or 'localization' in prompt_lower:
            analysis['analyses'].append('Source Localization')
        if 'psmap' in prompt_lower:
            analysis['analyses'].append('PS Map')

        # RAG context
        rag_results = []
        for source in self.knowledge:
            rag_results.extend(source.search(prompt))
        if rag_results:
            analysis['rag_context'] = [r['title'] for r in rag_results]

        return analysis

    def generate_config(self, prompt, analysis=None):
        """Generate YAML and Python based on prompt analysis."""
        if analysis is None:
            analysis = self.analyze_prompt(prompt)

        # Few-shot examples come from the configured data directory, so a
        # deployment can ship its own without editing this component.
        data_dir = (RUNTIME.settings.data_dir if RUNTIME.ctx is not None
                    else os.path.join(os.getcwd(), 'data'))
        with open(os.path.join(data_dir, 'train.json')) as f:
            train_data = json.load(f)

        best_ex = train_data[0]
        if analysis.get('target'):
            for ex in train_data:
                if analysis['target'] in ex.get('prompt', ''):
                    best_ex = ex
                    break

        yaml_config = self._build_yaml(prompt, analysis, best_ex)
        python_script = self._build_python(prompt, analysis, best_ex)

        return yaml_config, python_script

    def _build_yaml(self, prompt, analysis, reference_ex):
        """Build YAML config from analyzed prompt."""
        ref_yaml = reference_ex['response']['yaml']
        try:
            config = yaml.safe_load(ref_yaml)
        except Exception:
            config = {}

        if 'selection' not in config:
            config['selection'] = {}

        if analysis.get('target'):
            config['selection']['target'] = analysis['target']
        if analysis.get('energy_range'):
            config['selection']['emin'] = analysis['energy_range'][0]
            config['selection']['emax'] = analysis['energy_range'][1]
        if analysis.get('event_type'):
            config['selection']['evtype'] = analysis['event_type']

        # The reference example's absolute time range is meaningless for a
        # non-bundled target (it can span most of the mission, silently
        # inflating a demo prompt into a multi-hour run). analyze_prompt
        # never extracts times from the prompt, so any tmin/tmax here came
        # from the training example — drop them and let the full-data
        # default (most recent year, made explicit in the chat reply) apply.
        if analysis.get('target') and not _target_is_bundled(analysis['target']):
            config['selection'].pop('tmin', None)
            config['selection'].pop('tmax', None)

        config['selection'].setdefault('evclass', 128)
        config['selection'].setdefault('zmax', 105)
        config['selection'].setdefault('radius', 15)

        evtype = config['selection'].get('evtype', 3)
        config.setdefault('gtlike', {})
        config['gtlike']['irfs'] = 'P8R3_SOURCE_V3'
        config['gtlike']['edisp'] = True
        config['gtlike']['edisp_disable'] = ['isodiff']

        config.setdefault('model', {})
        normalize_diffuse_config(config)
        config['model'].setdefault('catalogs', ['4FGL-DR3'])
        config['model'].setdefault('src_radius', 15)
        config['model'].setdefault('src_roiwidth', 15)

        config.setdefault('data', {})
        config['data']['evfile'] = '<path_to_evfile>'
        config['data']['scfile'] = '<path_to_scfile>'

        config.setdefault('fileio', {})
        config['fileio']['outdir'] = 'output'
        # Name the logfile after the analysis target, not after whatever
        # source the reference training example happened to use.
        _log_tgt = (analysis.get('target') or 'analysis').replace(' ', '_').replace('/', '_')
        config['fileio']['logfile'] = f'{_log_tgt}.log'

        config.setdefault('logging', {})
        config['logging']['verbosity'] = 3
        config['logging']['chatter'] = 3

        config.setdefault('binning', {})
        config['binning'].setdefault('roiwidth', 10)
        config['binning'].setdefault('binsz', 0.1)
        config['binning'].setdefault('binsperdec', 8)

        for a in analysis.get('analyses', []):
            if 'SED' in a:
                config['sed'] = {'make_plots': True, 'write_fits': True}
            elif 'TS Map' in a:
                config['tsmap'] = {'make_plots': True, 'write_fits': True}
            elif 'Residual' in a:
                config['residmap'] = {'make_plots': True, 'write_fits': True}
            elif 'Light Curve' in a:
                config['lightcurve'] = {'make_plots': True}
            elif 'PS Map' in a:
                config['psmap'] = {'make_plots': True}

        return yaml.dump(config, default_flow_style=False, sort_keys=False)

    def _build_python(self, prompt, analysis, reference_ex):
        """Build Python script from analyzed prompt."""
        target = analysis.get('target', "'<TARGET>'")
        lines = [
            "from fermipy.gtanalysis import GTAnalysis",
            "",
            "gta = GTAnalysis('config.yaml')",
            "gta.setup()",
            "",
            f"target = '{target}'",
            "",
            "# Free background components",
            "gta.free_source('galdiff')",
            "gta.free_source('isodiff')",
            "",
            "# Free the target source",
            "gta.free_source(target)",
            "",
            "# Free nearby bright sources",
            "gta.free_sources(minmax_ts=[100, None], distance=3.0)",
            "",
            "# Optimize and fit",
            "gta.optimize()",
            "gta.fit()",
        ]

        for a in analysis.get('analyses', []):
            lines.append("")
            if 'SED' in a:
                lines.append("# Compute Spectral Energy Distribution")
                lines.append("gta.sed(target)")
            elif 'TS Map' in a:
                lines.append("# Generate TS Map")
                lines.append("tsmap_model = {'Index': 2.0, 'SpatialModel': 'PointSource'}")
                lines.append("gta.tsmap('tsmap', model=tsmap_model)")
            elif 'Residual' in a:
                lines.append("# Generate Residual Map")
                lines.append("residmap_model = {'Index': 2.0, 'SpatialModel': 'PointSource'}")
                lines.append("gta.write_model_map(model_name='model_0')")
                lines.append("gta.residmap('model_0', model=residmap_model)")
            elif 'Light Curve' in a:
                lines.append("# Generate Light Curve")
                lines.append("gta.lightcurve(target, nbins=10)")
            elif 'Extension' in a:
                lines.append("# Test Spatial Extension")
                lines.append("gta.extension(target, free_background=True, make_plots=True)")
            elif 'Localization' in a:
                lines.append("# Localize Source")
                lines.append("gta.localize(target, free_background=True, make_plots=True)")
            elif 'PS Map' in a:
                lines.append("# Generate PS Map")
                lines.append("gta.write_model_map(model_name='model_0')")
                lines.append("gta.psmap(cmap='ccube_00.fits', mmap='mcube_model_0_00.fits')")

        return "\n".join(lines)

    # -- contract entry point ---------------------------------------------
    def analyze(self, message, context=None):
        """:class:`~fermi_llm.core.contracts.IntentAnalyzer` entry point."""
        return self.analyze_prompt(message)

REGISTRY.register(kinds.INTENT_ANALYZER, 'rule_based', MasterAgent,
                  priority=100, source='core',
                  metadata={'label': 'Rule-based prompt analyzer'})

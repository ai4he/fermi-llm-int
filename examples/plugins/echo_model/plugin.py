"""Example: plugging in another lab's model backend.

``EchoProvider`` is a stand-in for a real service: it returns a fixed
configuration instead of calling anything. Replace :meth:`generate` with an
HTTP call and this is a complete integration — the model appears in the
picker, guardrails apply to its output, and the review/run flow is unchanged.

What it demonstrates: the ``model_provider`` contract, ``ModelSpec``, and
availability reporting (a backend whose service is down should be listed but
not selectable).
"""

from __future__ import annotations

from fermi_llm.core import kinds
from fermi_llm.core.contracts import GenerationResult, ModelSpec
from fermi_llm.components.models.base import BaseProvider

EXAMPLE_YAML = """\
logging: {verbosity: 3, chatter: 3}
fileio: {outdir: output}
binning: {roiwidth: 10, binsz: 0.1, binsperdec: 8}
selection:
  emin: 1000.0
  emax: 1000000.0
  zmax: 90
  target: 4FGL J1104.4+3812
  evclass: 128
  evtype: 3
gtlike: {edisp: true, edisp_disable: [isodiff], irfs: P8R3_SOURCE_V3}
model: {src_radius: 15, src_roiwidth: 15, catalogs: [4FGL-DR4]}
sed: {make_plots: true, write_fits: true}
"""

EXAMPLE_PYTHON = """\
import os
from fermipy.gtanalysis import GTAnalysis

gta = GTAnalysis(os.environ['FERMI_LLM_CONFIG_PATH'])
gta.setup()
gta.optimize()
gta.fit()
gta.sed('4FGL J1104.4+3812', make_plots=True)
"""


class EchoProvider(BaseProvider):
    """A model backend that always answers with the same example analysis."""

    backend = 'echo'
    MODELS = {
        'echo-demo': {
            'name': 'Echo (example plugin)',
            'group': 'Example plugins',
            'backend': 'echo',
            'description': 'Returns a fixed Mrk 421 SED configuration.',
            'requires_gpu': False,
            'recommended': False,
        },
    }

    def availability(self, spec: ModelSpec):
        return True, 'available', 'Example plugin; no service required'

    def generate(self, request) -> GenerationResult:
        request.report('api_call', 'Echo provider: returning the example files')
        return GenerationResult(
            yaml=EXAMPLE_YAML, python=EXAMPLE_PYTHON,
            raw_text='(echo provider)',
            metadata={'method': 'echo', 'has_yaml': True, 'has_python': True})

    def complete(self, spec, prompt, **kwargs) -> str:
        return ''


def register(registry, ctx=None):
    registry.register(kinds.MODEL_PROVIDER, 'echo', EchoProvider,
                      priority=800, source='example:echo_model',
                      metadata={'label': 'Echo (example)'})

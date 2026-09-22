"""The curated demo prompts shown under the chat box.

Each entry names the model it was tuned for, so the picker can grey out a
sample whose backend is unavailable on this machine. A deployment that wants
its own examples registers another ``ui_extension`` with ``samples``; they
are concatenated in priority order.
"""

from __future__ import annotations

from ...core import kinds
from ...core.registry import REGISTRY

DEMO_SAMPLES = [
    # --- Clemson RCD vLLM (LoRA-tuned for FermiPy) ---
    # No local GPU required; calls the remote OpenAI-compatible endpoint.
    {
        'id': 'mrk421_lora_clemson',
        'title': 'Mrk 421 — FermiPy LoRA (Clemson)',
        'subtitle': 'Task-specific LoRA on Qwen3.6-27B — remote vLLM',
        'prompt': 'Perform spectral analysis of Markarian 421 between 1 GeV and 1 TeV, compute the spectral energy distribution',
        'model': 'qwen3.6-27b-lora-fermipy-clemson',
        'model_label': 'Qwen3.6-27B + FermiPy LoRA (Clemson vLLM)',
        'category': 'local',
        'target': 'Mrk 421',
        'icon': 'cloud',
        'speed': 'fast',
        'description': 'Fine-tuned for FermiPy generation. Runs on the Clemson RCD vLLM server — no local GPU required.',
    },
    {
        'id': 'vela_lora_clemson',
        'title': 'Vela Pulsar — FermiPy LoRA (Clemson)',
        'subtitle': 'Pulsar analysis via fine-tuned remote model',
        'prompt': 'Analyze the Vela pulsar (4FGL J0835.3-4510) with FRONT+BACK SOURCE-class events between 100 MeV and 100 GeV. Compute the spectral energy distribution.',
        'model': 'qwen3.6-27b-lora-fermipy-clemson',
        'model_label': 'Qwen3.6-27B + FermiPy LoRA (Clemson vLLM)',
        'category': 'local',
        'target': 'Vela',
        'icon': 'cloud',
        'speed': 'fast',
        'description': 'Same fine-tuned model on the Vela pulsar. Tests the LoRA on a different source class than Mrk 421.',
    },

    # --- Local model samples (Qwen3.5-35B-A3B) ---
    # Mrk 421 (57KB data) and Crab (34KB data) run fast; Vela (2.8MB) is slower
    {
        'id': 'mrk421_spectral_local',
        'title': 'Mrk 421 Spectral Analysis',
        'subtitle': 'GeV-TeV blazar analysis -- fast execution (~2 min)',
        'prompt': 'Perform spectral analysis of Markarian 421 between 1 GeV and 1 TeV, compute the spectral energy distribution',
        'model': 'qwen3.5-35b-a3b',
        'model_label': 'Qwen3.5-35B-A3B (Local GPU)',
        'category': 'local',
        'target': 'Mrk 421',
        'icon': 'local',
        'speed': 'fast',
        'description': 'Bright BL Lac object. The MoE model achieves 100% structural validity with schema-guided prompting. Small dataset = fast execution.',
    },
    {
        'id': 'crab_front_local',
        'title': 'Crab Nebula FRONT Events',
        'subtitle': 'FRONT-only calibration source -- fast execution (~2 min)',
        'prompt': 'Analyze the Crab Nebula with FRONT only events, perform spectral fitting and compute the SED',
        'model': 'qwen3.5-35b-a3b',
        'model_label': 'Qwen3.5-35B-A3B (Local GPU)',
        'category': 'local',
        'target': 'Crab',
        'icon': 'local',
        'speed': 'fast',
        'description': 'Canonical gamma-ray calibration source. Tests evtype=1 isodiff matching and the deterministic repair pipeline. Smallest dataset.',
    },
    {
        'id': 'vela_psf3_local',
        'title': 'Vela Pulsar PSF3 Analysis',
        'subtitle': 'High angular resolution pulsar -- longer execution (~5-10 min)',
        'prompt': 'Study the Vela pulsar using PSF3 events for best angular resolution, compute SED',
        'model': 'qwen3.5-35b-a3b',
        'model_label': 'Qwen3.5-35B-A3B (Local GPU)',
        'category': 'local',
        'target': 'Vela',
        'icon': 'local',
        'speed': 'slow',
        'description': 'Brightest persistent gamma-ray source. PSF3 event type tests IRF/isodiff matching. Larger dataset = longer execution but more impressive results.',
    },
    # --- Gemini API samples ---
    {
        'id': 'mrk421_sed_gemini',
        'title': 'Mrk 421 High-Energy SED',
        'subtitle': 'Cloud API blazar -- fastest demo (~1-2 min)',
        'prompt': 'Analyze Markarian 421 at high energies from 1 GeV to 1 TeV, compute the spectral energy distribution',
        'model': 'gemini-2.5-flash-lite',
        'model_label': 'Gemini 2.5 Flash Lite (Cloud API)',
        'category': 'gemini',
        'target': 'Mrk 421',
        'icon': 'cloud',
        'speed': 'fast',
        'description': 'Cloud API with 20x faster inference. Small dataset + fast model = quickest end-to-end demo. No GPU required.',
    },
    {
        'id': 'crab_analysis_gemini',
        'title': 'Crab Nebula Spectral Fit',
        'subtitle': 'Cloud API calibration source -- fast (~1-2 min)',
        'prompt': 'Analyze the Crab Nebula, perform spectral fitting and compute the spectral energy distribution',
        'model': 'gemini-2.5-flash-lite',
        'model_label': 'Gemini 2.5 Flash Lite (Cloud API)',
        'category': 'gemini',
        'target': 'Crab',
        'icon': 'cloud',
        'speed': 'fast',
        'description': 'Comprehensive Crab Nebula analysis via cloud API. Demonstrates schema-guided generation with correct evclass, evtype, and catalog parameters.',
    },
    {
        'id': 'vela_standard_gemini',
        'title': 'Vela Pulsar Standard Analysis',
        'subtitle': 'Cloud API pulsar -- longer execution (~5-10 min)',
        'prompt': 'Perform a standard spectral analysis of the Vela pulsar with FRONT+BACK events and compute the SED',
        'model': 'gemini-2.5-flash-lite',
        'model_label': 'Gemini 2.5 Flash Lite (Cloud API)',
        'category': 'gemini',
        'target': 'Vela',
        'icon': 'cloud',
        'speed': 'slow',
        'description': 'Standard FRONT+BACK analysis of the brightest gamma-ray pulsar. Gemini achieves 100% Level 3 with deterministic repairs. Larger dataset.',
    },
]


class DemoSamplesExtension:
    """Contributes the built-in sample prompts."""

    name = 'demo_samples'
    label = 'Built-in demo prompts'

    def __init__(self, ctx):
        self.ctx = ctx

    def manifest(self):
        return {'name': self.name, 'label': self.label, 'assets': [],
                'samples': DEMO_SAMPLES}

    def samples(self):
        return DEMO_SAMPLES


REGISTRY.register(kinds.UI_EXTENSION, 'demo_samples', DemoSamplesExtension,
                  priority=100, source='core',
                  metadata={'label': DemoSamplesExtension.label})

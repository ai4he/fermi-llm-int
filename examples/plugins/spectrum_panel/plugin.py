"""Example: a new panel in the interface.

Adds a "Run summary" panel to the results column. The manifest names the
assets to load; the loader injects them and calls the panel's ``render``.

What it demonstrates: the ``ui_extension`` contract and the frontend plugin
API (``window.FermiLLM.registerPanel`` / ``on``), which is how a lab adds a
viewer without forking the page.
"""

from __future__ import annotations

import os

from fermi_llm.core import kinds

ASSETS = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'assets')


class SpectrumPanelExtension:
    name = 'spectrum_panel'
    label = 'Run summary panel (example)'

    def __init__(self, ctx):
        self.ctx = ctx

    def manifest(self):
        return {
            'name': self.name,
            'label': self.label,
            'assets': [f'/static/plugins/{self.name}/panel.js',
                       f'/static/plugins/{self.name}/panel.css'],
        }

    def asset_root(self):
        return ASSETS


def register(registry, ctx=None):
    registry.register(kinds.UI_EXTENSION, 'spectrum_panel',
                      SpectrumPanelExtension, priority=500,
                      source='example:spectrum_panel',
                      metadata={'label': SpectrumPanelExtension.label})

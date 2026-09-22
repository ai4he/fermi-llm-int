"""Skins: the stylesheet(s) the page loads.

``classic`` is the look the app shipped with. A skin replaces or layers on
that: register another ``skin`` component and select it with
``skin = "..."`` in ``plugins.toml`` (or ``FERMI_LLM_SKIN``). The page always
requests ``/api/ui/skin.css``, so switching does not touch the HTML.
"""

from __future__ import annotations

import os

from ...core import kinds
from ...core.registry import REGISTRY


class ClassicSkin:
    """The default Fermi-LLM look."""

    name = 'classic'
    label = 'Classic'

    def __init__(self, ctx):
        self.ctx = ctx

    def stylesheets(self):
        return [os.path.join(self.ctx.settings.static_dir, 'skins',
                             'classic.css')]

    def asset_root(self):
        return os.path.join(self.ctx.settings.static_dir, 'skins')


REGISTRY.register(kinds.SKIN, 'classic', ClassicSkin, priority=100,
                  source='core', metadata={'label': ClassicSkin.label})

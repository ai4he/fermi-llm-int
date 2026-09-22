"""Example: re-skinning the interface.

Layers a dark palette on top of the classic stylesheet by declaring both
files, so the skin only has to state what differs. A full replacement skin
returns just its own file.

What it demonstrates: the ``skin`` contract and singleton selection — the
page always asks for ``/api/ui/skin.css``, and ``plugins.toml`` decides
which skin answers.
"""

from __future__ import annotations

import os

from fermi_llm.core import kinds

ASSETS = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'assets')


class MidnightSkin:
    name = 'midnight'
    label = 'Midnight (example)'

    def __init__(self, ctx):
        self.ctx = ctx

    def stylesheets(self):
        classic = os.path.join(self.ctx.settings.static_dir, 'skins',
                               'classic.css')
        return [classic, os.path.join(ASSETS, 'midnight.css')]

    def asset_root(self):
        return ASSETS


def register(registry, ctx=None):
    registry.register(kinds.SKIN, 'midnight', MidnightSkin, priority=200,
                      source='example:midnight_skin',
                      metadata={'label': MidnightSkin.label})

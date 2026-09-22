"""Frontend-facing components: skins, panels and demo prompts.

The browser never hardcodes these. ``/api/ui/extensions`` lists what is
active and ``core/plugins.js`` injects it, so a lab's panel is a plugin
directory plus one registration — no change to the page.
"""

from . import demo_samples, skins  # noqa: F401

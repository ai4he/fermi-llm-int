"""Deterministic guardrails applied to every model reply.

A language model writing a FermiPy configuration slips in predictable ways:
it drops a section it was told to keep, substitutes a source from its
training data, or omits a product the user asked for. Each guardrail fixes
exactly one of those failures, in a fixed order, and explains in the chat
what it changed.

Guardrails stack. A lab that has its own consistency rule registers another
one; a deployment that disagrees with ours disables it by name in
``plugins.toml``. Core never hardcodes the chain — see
``fermi_llm.services.chat``.
"""

from . import (diffuse_paths, edit_merge, full_data, products,
               target_rules)  # noqa: F401

"""Model providers: everything that can turn a request into YAML + Python.

Each backend is its own module under ``providers/`` and registers itself as a
``model_provider`` component. The platform never imports a provider directly;
it asks :class:`~fermi_llm.components.models.service.ModelService`.
"""

from . import providers  # noqa: F401  (imports register the components)

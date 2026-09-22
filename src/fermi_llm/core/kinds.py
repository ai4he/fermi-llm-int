"""The extension points of the platform.

Every pluggable part of Fermi-LLM is registered under one of these kinds.
Adding a *new kind* is a core change and needs a documented contract in
``fermi_llm.core.contracts`` plus a section in ``docs/module-types.md``;
adding a new *component* of an existing kind is a plugin and needs neither.
"""

MODEL_PROVIDER = 'model_provider'
GUARDRAIL = 'guardrail'
INTENT_ANALYZER = 'intent_analyzer'
VALIDATOR = 'validator'
PIPELINE_STAGE = 'pipeline_stage'
EXECUTION_BACKEND = 'execution_backend'
SESSION_STORE = 'session_store'
AUTH_PROVIDER = 'auth_provider'
DATA_SOURCE = 'data_source'
EXPORTER = 'exporter'
IMPORTER = 'importer'
KNOWLEDGE_SOURCE = 'knowledge_source'
UI_EXTENSION = 'ui_extension'
SKIN = 'skin'
API_EXTENSION = 'api_extension'

ALL_KINDS = (
    MODEL_PROVIDER, GUARDRAIL, INTENT_ANALYZER, VALIDATOR, PIPELINE_STAGE,
    EXECUTION_BACKEND, SESSION_STORE, AUTH_PROVIDER, DATA_SOURCE, EXPORTER,
    IMPORTER, KNOWLEDGE_SOURCE, UI_EXTENSION, SKIN, API_EXTENSION,
)

#: Kinds where every registered component is used together (order matters).
STACKABLE_KINDS = frozenset({
    GUARDRAIL, VALIDATOR, PIPELINE_STAGE, EXPORTER, IMPORTER, UI_EXTENSION,
    API_EXTENSION, KNOWLEDGE_SOURCE, DATA_SOURCE, MODEL_PROVIDER,
    AUTH_PROVIDER,
})

#: Kinds where exactly one component is active at a time (selected by config).
SINGLETON_KINDS = frozenset({SESSION_STORE, EXECUTION_BACKEND, SKIN,
                             INTENT_ANALYZER})

"""Contracts for every extension point.

These are the only shapes core code depends on. A component may subclass the
base class here or simply provide the same attributes/methods: everything is
duck-typed at runtime and checked by :func:`check_contract`, so a plugin never
has to import a heavy dependency just to satisfy a type.

Each contract documents what core guarantees to pass in and what it does with
the return value. Keep changes to these classes backwards compatible: plugins
outside this repository depend on them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

from .errors import ContractError
from . import kinds

# ============================================================
# Payloads passed across component boundaries
# ============================================================


@dataclass
class ModelSpec:
    """One selectable entry in the model picker."""

    id: str
    name: str
    backend: str
    group: str = 'Models'
    description: str = ''
    recommended: bool = False
    requires_gpu: bool = False
    supports_semantic_intent: bool = False
    options: Dict[str, Any] = field(default_factory=dict)

    def to_public_dict(self, available: bool) -> Dict[str, Any]:
        return {
            'id': self.id, 'name': self.name, 'group': self.group,
            'backend': self.backend, 'description': self.description,
            'recommended': self.recommended,
            'requires_gpu': self.requires_gpu,
            'available': available,
        }


@dataclass
class GenerationRequest:
    """What the chat endpoint asks a model provider for."""

    model_id: str
    spec: ModelSpec
    user_prompt: str
    context: Optional[Dict[str, Any]] = None      # {prompt, yaml, python}
    temperature: float = 0.1
    n_shots: int = 3
    include_intent: bool = False
    progress: Optional[Callable[[str, str], None]] = None
    thread_id: Optional[str] = None               # conversational backends
    train_examples: List[Dict[str, Any]] = field(default_factory=list)
    prompt_builder: Optional[Callable[..., str]] = None
    extras: Dict[str, Any] = field(default_factory=dict)

    def report(self, stage: str, detail: str) -> None:
        if self.progress:
            self.progress(stage, detail)

    def build_prompt(self, **overrides) -> str:
        """The standard prompt for this request.

        Providers that want the platform's schema + few-shot prompt call
        this; one with its own format ignores it. ``prompt_builder`` is
        injected by the model service, so a deployment can replace prompt
        assembly without touching any provider.
        """
        if self.prompt_builder is None:                   # pragma: no cover
            return self.user_prompt
        return self.prompt_builder(self, **overrides)


@dataclass
class GenerationResult:
    """What a model provider returns: files plus whatever the UI should show."""

    yaml: str = ''
    python: str = ''
    raw_text: str = ''
    semantic_intent: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ChatTurn:
    """State threaded through the guardrail chain for one chat message."""

    session: Any
    user_message: str
    yaml: str = ''
    python: str = ''
    analysis: Optional[Dict[str, Any]] = None
    semantic_intent: Optional[Dict[str, Any]] = None
    response_parts: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)

    def note(self, guardrail: str, detail: str) -> None:
        """Record that a guardrail changed something (shown in the UI log)."""
        self.notes.append(f'{guardrail}: {detail}')


@dataclass
class ValidationResult:
    ok: bool
    stage: str
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    repairs: List[str] = field(default_factory=list)
    data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExecutionJob:
    """A reviewed analysis ready to run."""

    session_dir: str
    run_id: str
    yaml_text: str
    python_text: str
    work_dir: str
    config_path: str
    script_path: str
    timeout: int
    env: Dict[str, str] = field(default_factory=dict)
    options: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExecutionResult:
    ok: bool
    status: str = 'complete'
    error: Optional[str] = None
    data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExportArtifact:
    filename: str
    media_type: str
    path: Optional[str] = None
    content: Optional[bytes] = None
    headers: Dict[str, str] = field(default_factory=dict)


@dataclass
class Identity:
    """A signed-in user, as returned by an auth provider."""

    subject: str
    email: str = ''
    name: str = ''
    picture: str = ''
    provider: str = ''
    claims: Dict[str, Any] = field(default_factory=dict)


# ============================================================
# Contracts
# ============================================================


class ModelProvider:
    """Turns a prompt into YAML + Python.

    Implement this to plug in a lab's own model, a hosted API, or a local
    runtime. ``list_models`` is called once at startup and whenever the model
    picker is refreshed; ``generate`` is called per chat message.
    """

    backend: str = ''

    def list_models(self) -> Sequence[ModelSpec]:
        raise NotImplementedError

    def is_available(self, spec: ModelSpec) -> bool:
        """False hides the entry from the picker (missing key, no GPU, ...)."""
        return True

    def generate(self, request: GenerationRequest) -> GenerationResult:
        raise NotImplementedError

    def availability(self, spec: ModelSpec):
        """``(available, status, detail)`` shown in the model picker."""
        return self.is_available(spec), 'available', ''

    def is_loaded(self, model_id: str) -> bool:
        """True while weights are resident (local backends only)."""
        return False

    def complete(self, spec: ModelSpec, prompt: str, **kwargs) -> str:
        """Plain text completion, used by review/validation agents."""
        raise NotImplementedError

    def shutdown(self) -> None:
        """Release GPUs, subprocesses or sockets. Always safe to call."""


class Guardrail:
    """A deterministic transform applied to a chat turn after generation.

    Guardrails are stacked in ``priority`` order and each one receives the
    turn produced by the previous one. They must be pure with respect to
    anything outside the turn, and must never raise for ordinary bad model
    output: return the turn unchanged instead.
    """

    name: str = ''

    def applies(self, turn: ChatTurn) -> bool:
        return True

    def apply(self, turn: ChatTurn) -> ChatTurn:
        raise NotImplementedError


class IntentAnalyzer:
    """Extracts a structured analysis plan from the user's message."""

    def analyze(self, message: str, context: Optional[Dict[str, Any]] = None
                ) -> Dict[str, Any]:
        raise NotImplementedError


class Validator:
    """Checks an artifact before it is allowed to run.

    ``stage`` groups validators: ``'yaml'`` runs on the configuration,
    ``'script'`` on the Python driver, ``'run'`` on a finished run. Only
    validators returning ``ok=False`` with entries in ``errors`` block a run;
    warnings are surfaced to the user.
    """

    name: str = ''
    stage: str = 'yaml'

    def validate(self, artifact: Dict[str, Any]) -> ValidationResult:
        raise NotImplementedError


class PipelineStage:
    """One step of the run pipeline (review, execute, collect products...)."""

    name: str = ''

    def run(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError


class ExecutionBackend:
    """Where a reviewed analysis actually runs.

    The default runs a fresh interpreter on this host. A cluster or remote
    execution plugin implements the same two methods.
    """

    name: str = ''

    def execute(self, job: ExecutionJob) -> ExecutionResult:
        raise NotImplementedError

    def supports(self, job: ExecutionJob) -> bool:
        return True


class SessionStore:
    """Persistence for tasks. The default writes one directory per session."""

    def create(self, session_id: str) -> str: ...
    def load(self, session_id: str) -> Optional[Dict[str, Any]]: ...
    def save(self, session_id: str, data: Dict[str, Any]) -> None: ...
    def delete(self, session_id: str) -> None: ...
    def list_ids(self) -> List[str]: ...
    def path(self, session_id: str) -> str: ...


class AuthProvider:
    """A way to sign in. ``guest`` is always present; others are optional."""

    id: str = ''

    def public_config(self) -> Dict[str, Any]:
        """What the frontend needs to render the button (never secrets)."""
        return {}

    def is_configured(self) -> bool:
        return True

    def verify(self, credential: str) -> Identity:
        raise NotImplementedError


class DataSource:
    """Resolves a target name into the photon/spacecraft files to analyse."""

    name: str = ''

    def handles(self, target: str, spec: Dict[str, Any]) -> bool:
        raise NotImplementedError

    def resolve(self, target: str, spec: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError


class Exporter:
    """Produces a downloadable artifact from a session."""

    name: str = ''
    label: str = ''
    media_type: str = 'application/octet-stream'

    def available(self, session: Any) -> bool:
        return True

    def build(self, session: Any) -> ExportArtifact:
        raise NotImplementedError


class Importer:
    """Reads an external file into a session (the inverse of an exporter)."""

    name: str = ''
    label: str = ''
    accepts: Sequence[str] = ()

    def load(self, payload: bytes, filename: str) -> Dict[str, Any]:
        raise NotImplementedError


class KnowledgeSource:
    """Documentation searched while planning an analysis (RAG)."""

    name: str = ''

    def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        raise NotImplementedError


class UIExtension:
    """Frontend code loaded by the browser: panels, chat widgets, viewers.

    The manifest is served from ``/api/ui/extensions`` and the loader in
    ``web/static/core/plugins.js`` injects the listed assets. Assets are
    served from the directory returned by :meth:`asset_root`.
    """

    name: str = ''

    def manifest(self) -> Dict[str, Any]:
        raise NotImplementedError

    def asset_root(self) -> Optional[str]:
        return None


class Skin:
    """A stylesheet replacing or layering on the default look."""

    name: str = ''
    label: str = ''

    def stylesheets(self) -> List[str]:
        raise NotImplementedError

    def asset_root(self) -> Optional[str]:
        return None


class APIExtension:
    """Extra HTTP routes. Return a FastAPI ``APIRouter`` from :meth:`router`."""

    name: str = ''
    prefix: str = ''

    def router(self, ctx: Any):
        raise NotImplementedError


# ============================================================
# Runtime checking
# ============================================================

REQUIRED_MEMBERS: Dict[str, Sequence[str]] = {
    kinds.MODEL_PROVIDER: ('list_models', 'generate'),
    kinds.GUARDRAIL: ('name', 'apply'),
    kinds.INTENT_ANALYZER: ('analyze',),
    kinds.VALIDATOR: ('name', 'stage', 'validate'),
    kinds.PIPELINE_STAGE: ('name', 'run'),
    kinds.EXECUTION_BACKEND: ('name', 'execute'),
    kinds.SESSION_STORE: ('load', 'save', 'path'),
    kinds.AUTH_PROVIDER: ('id', 'verify'),
    kinds.DATA_SOURCE: ('name', 'handles', 'resolve'),
    kinds.EXPORTER: ('name', 'media_type', 'build'),
    kinds.IMPORTER: ('name', 'load'),
    kinds.KNOWLEDGE_SOURCE: ('name', 'search'),
    kinds.UI_EXTENSION: ('name', 'manifest'),
    kinds.SKIN: ('name', 'stylesheets'),
    kinds.API_EXTENSION: ('name', 'router'),
}


def check_contract(kind: str, instance: Any) -> None:
    """Raise :class:`ContractError` if ``instance`` misses required members.

    Called by the loader for every component it constructs, so a typo in a
    plugin is reported with the plugin's name instead of an AttributeError
    deep inside a request.
    """
    missing = [m for m in REQUIRED_MEMBERS.get(kind, ())
               if not hasattr(instance, m)]
    if missing:
        raise ContractError(
            f'{type(instance).__name__} does not satisfy the {kind} contract; '
            f'missing: {", ".join(missing)}. See docs/module-types.md.')

# Module types

Fifteen extension points. Each section says what the kind is for, the
contract, where the platform calls it, and which shipped component to read as
an example.

Constants live in `fermi_llm.core.kinds`; contracts in
`fermi_llm.core.contracts`. A component may subclass the contract class or
just provide the same members — everything is duck-typed and checked at load
time by `check_contract`.

| Kind | Constant | Stacks? | Ships with |
|---|---|---|---|
| Model provider | `MODEL_PROVIDER` | yes | gemini, openai, codex, vllm ×3, huggingface, template |
| Guardrail | `GUARDRAIL` | yes (ordered) | 6 consistency rules |
| Intent analyzer | `INTENT_ANALYZER` | one active | rule_based |
| Validator | `VALIDATOR` | yes (by stage) | fermipy_level1/2, script_static |
| Pipeline stage | `PIPELINE_STAGE` | yes (ordered) | review, execute |
| Execution backend | `EXECUTION_BACKEND` | one active | local_subprocess |
| Session store | `SESSION_STORE` | one active | filesystem |
| Auth provider | `AUTH_PROVIDER` | yes | google, guest |
| Data source | `DATA_SOURCE` | yes (first match) | bundled_lat, weekly_archive |
| Exporter | `EXPORTER` | yes | run_bundle, notebook, llm_log, fermipy_log, artifacts_zip |
| Importer | `IMPORTER` | yes | — (contract ready) |
| Knowledge source | `KNOWLEDGE_SOURCE` | yes | rag_docs |
| UI extension | `UI_EXTENSION` | yes | demo_samples |
| Skin | `SKIN` | one active | classic |
| API extension | `API_EXTENSION` | yes | — (contract ready) |

---

## model_provider

**For:** any way of turning a request into YAML + Python — a hosted API, a
local runtime, a fine-tune, a rule-based generator, another lab's service.

```python
class MyProvider:
    backend = 'mylab'

    def list_models(self) -> Sequence[ModelSpec]: ...
    def availability(self, spec) -> tuple[bool, str, str]: ...   # optional
    def generate(self, request: GenerationRequest) -> GenerationResult: ...
    def complete(self, spec, prompt, **kwargs) -> str: ...       # optional
    def is_loaded(self, model_id) -> bool: ...                   # optional
    def shutdown(self) -> None: ...                              # optional
```

- `list_models` is the picker. One entry per selectable model.
- `availability` returns `(available, status, detail)`; the UI greys out an
  entry and shows `detail` ("No API key configured", "Requires GPU").
- `generate` receives `request.build_prompt()` — the platform's schema and
  few-shot prompt — or may ignore it entirely and use `request.user_prompt`.
- `complete` is the free-form call the script-review agent uses. Return
  `None`-equivalent behaviour (raise `NotImplementedError`) if your backend
  cannot answer arbitrary prompts; the platform falls back to heuristics.
- Subclass `components.models.base.BaseProvider` to get the catalogue and
  reply parsing for free (a typical provider is then ~40 lines).

**Read:** `components/models/providers/gemini.py`,
`examples/plugins/echo_model/`.

## guardrail

**For:** a deterministic correction applied to a model's output before the
user sees it.

```python
class MyGuardrail:
    name = 'mylab_check'

    def applies(self, turn: ChatTurn) -> bool: ...
    def apply(self, turn: ChatTurn) -> ChatTurn: ...
```

The `ChatTurn` carries the session, the user's message, the current YAML and
Python, the analysis plan and the reply being assembled. Append to
`turn.response_parts` to tell the user what you changed, and call
`turn.note(name, detail)` so it shows in the transparency log.

Rules: never raise (return the turn unchanged instead), never call a model,
and only change what your rule is about.

**Read:** `components/guardrails/target_rules.py`.

## intent_analyzer

**For:** turning a message into a structured plan (target, energy range, time
range, products). One active at a time.

```python
class MyAnalyzer:
    def analyze(self, message, context=None) -> dict: ...
    def generate_config(self, prompt, analysis=None) -> tuple[str, str]: ...
```

`generate_config` is what the `template` model uses to produce files without
an LLM; keep it if you replace the analyzer.

**Read:** `components/intent/rule_based.py`.

## validator

**For:** anything that must be true before a run starts. Stages: `yaml`
(configuration), `script` (the Python driver), `run` (a finished run).

```python
class MyValidator:
    name = 'mylab_policy'
    stage = 'yaml'

    def validate(self, artifact: dict) -> ValidationResult: ...
```

`ok=False` with `errors` blocks the run; `warnings` are shown. Site policy
(energy bands, allowed catalogs, data-use rules) belongs here.

**Read:** `components/validators/fermipy_levels.py`,
`examples/plugins/strict_energy_validator/`.

## pipeline_stage

**For:** a step in the run pipeline itself — a cost estimate, an approval
gate, a post-run archival step.

```python
class MyStage:
    name = 'mylab_gate'
    def run(self, ctx: dict) -> dict: ...
```

Stages are ordered by priority; core registers `review` (100) and
`execute` (200).

## execution_backend

**For:** where an approved analysis actually runs. One active at a time.

```python
class MyBackend:
    name = 'slurm'
    def supports(self, job: ExecutionJob) -> bool: ...
    def execute(self, job: ExecutionJob) -> ExecutionResult: ...
```

`job.options` is the Level-4 keyword payload; return the result dict in
`ExecutionResult.data`. This is the seam for a cluster, a container, or a
remote machine that holds the data.

**Read:** `components/execution/local_subprocess.py`.

## session_store

**For:** where tasks live. One active at a time.

```python
class MyStore:
    def create(self, session_id) -> str: ...
    def load(self, session_id) -> dict | None: ...
    def save(self, session_id, data) -> None: ...
    def delete(self, session_id) -> None: ...
    def list_ids(self) -> list[str]: ...
    def path(self, session_id) -> str: ...
```

`path` must return a real directory: runs write their working files there.
A database-backed store still needs a scratch directory.

**Read:** `components/storage/filesystem.py`.

## auth_provider

**For:** a way to sign in. Guest is always available, so nothing in core may
require a signed-in user.

```python
class MyAuth:
    id = 'institution_sso'
    def is_configured(self) -> bool: ...
    def public_config(self) -> dict: ...      # never secrets
    def verify(self, credential: str) -> Identity: ...
```

**Read:** `components/auth/google.py`.

## data_source

**For:** resolving a target to the photon/spacecraft files to analyse. Sources
are tried in priority order; the first whose `handles` returns True wins.

```python
class MySource:
    name = 'mylab_archive'
    def handles(self, target: str, spec: dict) -> bool: ...
    def resolve(self, target: str, spec: dict) -> dict: ...
```

`resolve` returns `{'mode', 'test_idx', 'yaml', 'notes', 'resolved_source'}`.
`test_idx` is either a bundled index or a dict describing the resolved data.

**Read:** `components/datasources/weekly_archive.py`.

## exporter

**For:** a downloadable artifact built from a session. Appears automatically
in `/api/session/{id}/exports` and in the UI.

```python
class MyExporter:
    name = 'my_format'
    label = 'My format (.abc)'
    media_type = 'application/x-abc'
    def available(self, session) -> bool: ...
    def build(self, session) -> ExportArtifact: ...
```

Return bytes in `content` or a path in `path`; set
`headers={'x-cleanup': path}` for a temporary file the API should delete
after sending.

**Read:** `components/exporters/notebook.py`,
`examples/plugins/csv_exporter/`.

## importer

**For:** reading an external file into a session (the inverse of an
exporter): another tool's configuration, a saved notebook, a batch of
prompts.

```python
class MyImporter:
    name = 'my_format'
    accepts = ('.abc',)
    def load(self, payload: bytes, filename: str) -> dict: ...
```

## knowledge_source

**For:** documentation consulted while planning. All active sources are
searched and the hits merged.

```python
class MyDocs:
    name = 'mylab_docs'
    def search(self, query: str, top_k: int = 3) -> list[dict]: ...
```

Each hit is `{'title', 'content', 'keywords'}`.

**Read:** `components/knowledge/rag_docs.py`.

## ui_extension

**For:** frontend code: panels, viewers, chat widgets, extra demo prompts.

```python
class MyPanel:
    name = 'mylab_viewer'
    def manifest(self) -> dict: ...          # {'assets': [...], 'panels': [...]}
    def asset_root(self) -> str | None: ...  # served at /static/plugins/<name>/
    def samples(self) -> list: ...           # optional demo prompts
```

In the browser: `window.FermiLLM.registerPanel({id, title, slot, render})`
and `window.FermiLLM.on('run:finished', fn)`.

**Read:** `examples/plugins/spectrum_panel/`.

## skin

**For:** the look. One active at a time; the page always requests
`/api/ui/skin.css`.

```python
class MySkin:
    name = 'mylab'
    def stylesheets(self) -> list[str]: ...   # absolute paths, concatenated
    def asset_root(self) -> str | None: ...
```

Return `[classic_css, my_overrides_css]` to layer on the default, or just
your own file to replace it.

**Read:** `examples/plugins/midnight_skin/`.

## api_extension

**For:** your own HTTP endpoints — a lab's job queue, a status probe, an
integration webhook.

```python
class MyAPI:
    name = 'mylab_api'
    prefix = '/api/mylab'
    def router(self, ctx):
        from fastapi import APIRouter
        router = APIRouter()

        @router.get('/status')
        async def status():
            return {'ok': True}

        return router
```

A failing extension is skipped and reported; the rest of the app still
serves.

---

## Hooks (not a kind, but the other way to plug in)

When you want to *observe* rather than replace:

```python
@ctx.hooks.on('run.finished', name='mylab.metrics')
def record(session=None, result=None, **_):
    ...
```

Events: `app.configured`, `app.shutdown`, `session.created`,
`session.loaded`, `session.saved`, `chat.request`, `chat.generated`,
`chat.completed`, `guardrail.applied`, `run.review`, `run.started`,
`run.progress`, `run.finished`, `export.built`, `model.selected`.

A listener that raises is recorded in `bus.errors` and never affects the
request.

**Read:** `examples/plugins/run_logger/`.

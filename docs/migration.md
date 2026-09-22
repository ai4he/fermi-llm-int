# Migration from the monolith

The previous deployment was two large modules — `webapp/server.py` (~4,800
lines) and `webapp/model_manager.py` (~1,700) — plus a flat `fermipy_pipeline/`
package. This repository is the same behaviour, decomposed. If you are
porting a patch, a fork or a local customisation, this is the map.

## Where things went

| Was | Is now |
|---|---|
| `webapp/server.py` routes | `src/fermi_llm/api/*.py` (one router per domain) |
| `chat()` endpoint body | `src/fermi_llm/services/chat.py` |
| YAML consistency helpers (`_merge_edit_yaml`, `_enforce_requested_target`, `_reconcile_planned_products`, `_apply_full_data_defaults`, …) | `components/guardrails/yamlops.py`, wrapped by one guardrail each |
| `MasterAgent` | `components/intent/rule_based.py` |
| `FermiPyRAG` | `components/knowledge/rag_docs.py` |
| `AnalysisSession` | `core/session.py` (+ `components/storage/filesystem.py`) |
| `write_progress` / `write_chat_progress` | `core/progress.py` |
| `_prepare_run_review`, `_build_analysis_python` | `components/pipelines/review.py` |
| `run_pipeline_isolated` | `components/pipelines/execute.py` |
| notebook / zip / log builders | `components/exporters/*.py` |
| auth helpers, Google verification | `components/auth/{tokens,google}.py` |
| `DEMO_SAMPLES` | `components/ui/demo_samples.py` |
| `_yaml_target`, `_SOURCE_ALIASES`, `_test_idx_for_target` | `fermipy/targets.py` |
| `_estimate_run_duration`, `_fit_timeout_for` | `fermipy/estimates.py` |
| `MODEL_REGISTRY` + `ModelManager` | `components/models/providers/*.py` + `components/models/service.py` |
| `build_prompt`, `extract_yaml`, intent schema | `components/models/prompting.py` |
| backend endpoints and key paths | `components/models/settings.py` |
| `fermipy_pipeline/*.py` | `src/fermi_llm/fermipy/*.py` |
| `webapp/static/app.js` | `src/fermi_llm/web/static/core/app.js` |
| `webapp/static/style.css` | `src/fermi_llm/web/static/skins/classic.css` |
| `webapp/static/index.html` | `src/fermi_llm/web/static/index.html` |

`tests/compat.py` encodes this map in code: it resolves an old name to the
module that owns it now, and its error message lists where it looked.

```python
from tests import compat
compat.server().module_of('_merge_edit_yaml')
# 'fermi_llm.components.guardrails.yamlops'
```

## What stayed exactly the same

- **Every HTTP endpoint**, including the download URLs. The frontend is the
  same application, so existing links, scripts and bookmarks keep working.
- **Session format.** Tasks written by the monolith load unchanged; the
  sessions directory is compatible in both directions.
- **Environment variables.** Same names, same defaults; an existing
  `configs/env.sh` needs no edits.
- **The two-click review contract**, the script validator, the isolated
  runner, the incremental-fit reuse, the guardrail behaviour.
- **`webapp/server.py` as an entry point.** It now just starts the app.

## What changed on purpose

| Change | Why | Impact |
|---|---|---|
| The stylesheet is served from `/api/ui/skin.css` | skins are a component now | a fork that patched `index.html` for CSS should register a skin instead |
| `app.js` moved to `/static/core/app.js` | room for `core/plugins.js` and plugin assets | update a hardcoded asset path if you have one |
| New endpoints: `/api/platform`, `/api/ui/extensions`, `/api/session/{id}/exports`, `/api/session/{id}/export/{name}` | introspection and pluggable exports | additive |
| `ModelManager` no longer exists | each backend owns its code; `ModelService` routes | call `ctx.models` (same `list_models` / `generate` / `complete`) |
| Analysis modules import as a package | they were importable only via `sys.path` | `from fermi_llm.fermipy import script_validator` |
| Subprocess entry is `python -m fermi_llm.fermipy.script_runner` | package-relative imports | transparent unless you invoked it directly |

## Porting a local patch

1. Find the function in the table (or with `compat.server().module_of(...)`).
2. Ask whether the patch is really a *component*: a new rule, format, model
   or panel should become a plugin, not an edit. See
   [plugin-quickstart.md](plugin-quickstart.md).
3. If it genuinely belongs in core, apply it there and add a test.
4. Run the suites in [testing.md](testing.md).

Patches to `server.py` that added a route, a model or a consistency rule are
exactly the ones that should become plugins — that is what the split was for.

## Behavioural fixes made during the split

These are real changes, not refactoring, and they are in this repository:

- **PS map inputs.** The generated script now writes the model cube and
  passes `cmap`/`mmap`/`emin`/`emax`, instead of calling `gta.psmap()` with
  an empty count-map path (which always failed).
- **Per-product error isolation.** A failing product (PS map, light curve…)
  is recorded in `product_errors` and skipped; it no longer discards a
  completed fit. The monolith's generated script had no error handling, so
  one failed product threw away a 20-minute run.
- **OpenAI structured envelope.** Parsing the `{intent, yaml, python}`
  envelope lives in the provider that asks for it.

# CLAUDE.md

Guidance for Claude Code (and other coding agents) working in this
repository. Read this first; it says where things are and which rules are
not negotiable.

## What this is

A modular web platform that turns a plain-English request into a FermiPy
analysis: generate YAML + `analysis.py`, correct them deterministically,
review them, run exactly what was approved, present the results.

The architecture is the product here. Every capability is a **component**
registered against a central registry; core only sequences them. If you are
about to edit core to add a feature, stop — it is almost certainly a plugin.

## Read before changing anything

| Task | Read |
|---|---|
| orienting from zero | `docs/getting-started.md` |
| adding any capability | `docs/module-types.md`, `docs/plugin-quickstart.md` |
| anything that will be shared with other institutes | `docs/integration-rules.md` (R1–R12) |
| integrating another codebase | `docs/agent-playbook.md` |
| understanding the flow | `docs/architecture.md` |
| finding moved code | `docs/migration.md`, `tests/compat.py` |
| testing | `docs/testing.md` |

## Layout

```
src/fermi_llm/
  core/         registry, contracts, hooks, config, loader, context, runtime
  api/          one router per domain — thin, no logic
  services/     orchestration (chat flow, run lifecycle)
  components/   every pluggable module, one package per kind
  fermipy/      the analysis layer (schema, repair, validation, runner)
  web/static/   core app, plugin loader, skins
plugins/        local drop-in plugins (gitignored)
examples/plugins/  five working examples
tests/          architecture, component and end-to-end suites
```

Two invariants:

1. `core/` never imports from `components/`.
2. A component imports core, contracts and `fermipy` — never another
   component. It reaches other components through `ctx`.

## Environment and commands

Everything runs in the `fermi-llm` conda env (FermiPy + ScienceTools);
`python` is not on the bare PATH.

```bash
source /opt/miniconda3/etc/profile.d/conda.sh && conda activate fermi-llm
export PYTHONPATH="$PWD/src:$PWD/tests"

./scripts/run_webapp.sh                     # :8765 (FERMI_LLM_PORT to change)
python tests/test_architecture.py           # plugin architecture — seconds
python tests/test_luna_features.py          # 71 component tests — no network
python tests/fermipy/test_data_access.py    # resolver + weekly registry
python tests/test_e2e.py --url http://localhost:8899   # real run, ~12 min
```

There is no linter and no build step. The frontend is vanilla JS served
as-is — keep it that way.

## Rules that are not negotiable

- **The approved files are what runs.** The first `run_pipeline` POST returns
  a review plus an approval token bound to a content digest; the second runs
  those exact bytes. Nothing may rewrite the YAML or the script after
  approval, and no analysis code may execute outside the execution backend.
- **Guardrails are deterministic and explain themselves.** They never call a
  model, never raise, and append a user-visible note when they change
  something.
- **A failing plugin never takes the platform down.** Loader and stack
  builders record the failure and continue; `/api/platform` reports it.
- **Secrets stay in `configs/`** (gitignored), are referenced by path, are
  never logged, and are scrubbed from analysis subprocess environments.
- **Write only inside the session directory** you were given.

## Where things live (common lookups)

| Looking for | File |
|---|---|
| chat flow | `services/chat.py` |
| the guardrail chain | `components/guardrails/` (ordered 100–500) |
| YAML editing helpers | `components/guardrails/yamlops.py` |
| review / approval | `components/pipelines/review.py` |
| worker-side run | `components/pipelines/execute.py` |
| model backends | `components/models/providers/` |
| prompt assembly | `components/models/prompting.py` |
| FermiPy repair and validation | `fermipy/run_execution_validated.py` |
| Level 4 + products | `fermipy/run_level4_validation.py` |
| script review / runner | `fermipy/script_validator.py`, `fermipy/script_runner.py` |
| target aliases, bundled data | `fermipy/targets.py` |
| frontend | `web/static/core/app.js`, `core/plugins.js`, `skins/classic.css` |

Several guardrails and UI behaviours exist because astrophysics reviewers
asked for them; that feedback and its rationale live in the private
`ai4he/fermi-testbed` repository (`FEEDBACK.md`). Before "simplifying"
something that looks redundant, check there or ask the maintainers — the
reason is usually a real reviewer complaint.

## When you add a component

```python
from fermi_llm.core import kinds

def register(registry, ctx=None):
    registry.register(kinds.VALIDATOR, 'mylab_policy', PolicyValidator,
                      priority=250, source='mylab:policy',
                      metadata={'label': 'Energy policy', 'stage': 'yaml'})
```

Then: `curl -s localhost:8765/api/platform` to confirm it is `active`, and a
test in the style of `tests/test_architecture.py`.

## Reporting back

State what you changed, what you ran, and what the output was. If a test
fails and you could not fix it, say so with the output — do not describe a
change as working because it should.

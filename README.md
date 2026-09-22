# Fermi-LLM

A web platform that turns a plain-English request into a real Fermi-LAT
analysis: it writes the FermiPy configuration and the Python driver, checks
them, runs exactly what you approved, and shows you the result — with every
step visible.

It is built to be extended by the community. Every model, rule, validator,
data source, export format, panel and skin is a module you can add, stack,
replace or switch off without forking the platform.

```
 your words  ─▶  plan  ─▶  model  ─▶  guardrails  ─▶  review  ─▶  run  ─▶  results
                  │         │            │              │          │
            intent_analyzer │       guardrail       validator  execution_backend
                      model_provider                             data_source
```

## Quick start

```bash
git clone https://github.com/ai4he/fermi-llm-int.git
cd fermi-llm-int
./scripts/setup.sh                    # conda env with FermiPy + ScienceTools
./scripts/run_webapp.sh               # http://localhost:8765
```

With no credentials configured it still runs: guest mode, the rule-based
`template` model, and the three bundled demonstration sources (Mrk 421, the
Vela pulsar, the Crab). Add an API key in `configs/` to enable a real model —
see [docs/deployment.md](docs/deployment.md).

## What it does

- **Chat to configuration.** Describe the analysis; get a FermiPy YAML and a
  matching `analysis.py`. Follow-up messages edit the existing files instead
  of starting over.
- **Deterministic guardrails.** Six checks correct the failures language
  models reliably make — a dropped section, a substituted source, a missing
  product, unpinned diffuse models — and each one explains what it changed.
- **Review before anything runs.** The first click on *Run Pipeline* shows
  the exact files that would execute, with an approval token bound to their
  content. The second click runs those bytes. Edit anything in between and
  the token is void.
- **Real execution.** The approved script runs in an isolated process with
  resource limits, against real LAT data, with live progress.
- **Reproducible output.** Every run yields the final YAML, the runnable
  Python, a manifest with both digests, the figures, a Jupyter notebook and
  the logs.

## Extending it

Adding a module is one file and one registration:

```python
from fermi_llm.core import kinds

def register(registry, ctx=None):
    registry.register(kinds.EXPORTER, 'mylab_sed_csv', SEDCsvExporter,
                      priority=500, source='mylab')
```

Drop it in `plugins/`, or ship it as a pip package with a
`fermi_llm.plugins` entry point. It shows up in the app — no core change, no
frontend change, no fork.

Fifteen extension points:

| | |
|---|---|
| `model_provider` | any LLM, API, local runtime or fine-tune |
| `guardrail` | a consistency rule applied to model output |
| `intent_analyzer` | how a request becomes a plan |
| `validator` | safety and site policy before a run |
| `pipeline_stage` | an extra step in review or execution |
| `execution_backend` | local, cluster, container, remote |
| `data_source` | your archive, catalog or simulated data |
| `session_store` | where tasks live |
| `auth_provider` | institutional sign-in |
| `exporter` / `importer` | file formats in and out |
| `knowledge_source` | documentation the planner consults |
| `ui_extension` | panels, viewers, widgets, demo prompts |
| `skin` | how the app looks |
| `api_extension` | your own endpoints |

Working examples of five of them are in
[`examples/plugins/`](examples/plugins/). `GET /api/platform` shows what is
plugged in right now, including anything that failed to load.

## Documentation

| | |
|---|---|
| [architecture.md](docs/architecture.md) | how it fits together, with diagrams |
| [module-types.md](docs/module-types.md) | the contract for each extension point |
| [integration-rules.md](docs/integration-rules.md) | the rules a module must follow |
| [plugin-quickstart.md](docs/plugin-quickstart.md) | your first module in ten minutes |
| [agent-playbook.md](docs/agent-playbook.md) | integrating an existing codebase (for agents) |
| [testing.md](docs/testing.md) | the three test layers |
| [deployment.md](docs/deployment.md) | install, configure, run as a service |
| [migration.md](docs/migration.md) | where everything moved from the monolith |
| [CONTRIBUTING.md](CONTRIBUTING.md) | how to contribute |

## Status

The platform runs real analyses today. The modular structure is new: the
contracts are documented and tested, and they will grow by addition —
existing fields keep their meaning (see rule R11).

Built by [AI4HE](https://github.com/ai4he) with the Clemson high-energy
astrophysics group. Contributions from the FermiPy community are the point of
this repository, not a bonus.

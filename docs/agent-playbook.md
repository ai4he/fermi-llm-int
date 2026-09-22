# Agent playbook: integrating an existing codebase

This is written for a coding agent (Claude Code or similar) asked to take
another lab's FermiPy web client and integrate it into this platform. A human
doing the same job can follow it unchanged.

Read [integration-rules.md](integration-rules.md) first; it is the contract.
This document is the procedure.

---

## Ground rules for this job

1. **Nothing in `src/fermi_llm/core/` or `src/fermi_llm/components/` changes.**
   If you believe it must, stop and report why — that is a platform decision,
   not an integration step.
2. **Work in one plugin package per source project**, containing one
   component per capability.
3. **Port behaviour, not structure.** The other codebase's file layout is not
   evidence about which extension point its features belong to.
4. **Every step ends with a test that runs.** No "should work".

---

## Step 1 — Inventory what the other codebase does

Produce a table before writing any code. For each capability: what it does
for the user, which files implement it, what it depends on (services, data,
credentials, GPU), and which kind it maps to.

```
| Capability            | Their files              | Needs            | Kind              |
|-----------------------|--------------------------|------------------|-------------------|
| Llama-3 backend       | llm/llama_client.py      | HTTP, token file | model_provider    |
| ROI preview plot      | web/roi_plot.js, api.py  | run outputs      | ui_extension      |
| DESY archive reader   | data/desy.py             | /mnt/desy        | data_source       |
| "no ULs" policy check | checks/policy.py         | —                | validator         |
| Batch submit          | batch/submit.py          | Slurm            | execution_backend |
```

Anything that does not map to a kind in [module-types.md](module-types.md) is
a finding to report, not something to force.

## Step 2 — Decide stack vs replace, per capability

For each row, decide and record why:

- **Stack** (default): their feature runs alongside ours. Most model
  providers, validators, exporters, panels and knowledge sources.
- **Replace**: only when both cannot be active at once (execution backend,
  session store, skin, intent analyzer), or when theirs is strictly better
  and the deployment wants exactly one. Use `replaces=(...)` so reverting is
  a config change.
- **Drop**: their code duplicates a core component with no behavioural
  difference. Say so explicitly in the report; do not port it silently.

## Step 3 — Create the plugin package

```
otherlab_fermi/
├── README.md              # kinds, settings, host requirements
├── pyproject.toml         # entry point: fermi_llm.plugins
├── otherlab_fermi/
│   ├── __init__.py
│   ├── plugin.py          # register(registry, ctx) — the only entry point
│   ├── models.py          # one module per component
│   ├── data.py
│   ├── validators.py
│   └── assets/            # frontend files, if any
└── tests/
```

`plugin.py` registers everything:

```python
from fermi_llm.core import kinds
from . import data, models, validators


def register(registry, ctx=None):
    registry.register(kinds.MODEL_PROVIDER, 'otherlab_llama',
                      models.LlamaProvider, priority=300,
                      source='otherlab:llama')
    registry.register(kinds.DATA_SOURCE, 'otherlab_desy',
                      data.DESYArchive, priority=200,
                      source='otherlab:desy')
    registry.register(kinds.VALIDATOR, 'otherlab_policy',
                      validators.PolicyCheck, priority=250,
                      source='otherlab:policy')
```

## Step 4 — Port one capability at a time

For each, in this order:

1. Copy their implementation into your module, unchanged where possible.
2. Wrap it in the contract class. The wrapper translates types and handles
   errors; it does not reimplement their logic.
3. Delete their glue: HTTP routing, session handling, config loading, logging
   setup, threading. The platform provides all of it, and keeping theirs is
   what creates two ways to do everything.
4. Write the test. Load the plugin directory into a throwaway `Registry`,
   build the component, exercise it with fixed inputs.
5. Run `python tests/test_architecture.py` and the ported suite.

Stop after each capability and check `/api/platform`: your component should
be `active`, and `plugin_failures` empty.

## Step 5 — Reconcile overlaps honestly

When both codebases do the same job, the answer is *not* "keep both silently".

- Two model providers: keep both. Models stack; users pick.
- Two guardrails for the same failure: keep ours, port theirs only if it
  catches a case ours misses, and say which case in the docstring.
- Two execution backends: keep both registered, select in `plugins.toml`,
  document the trade-off in your README.
- Their frontend vs ours: port their *panels* as UI extensions. Do not port
  their page shell; this platform's page is the host.

## Step 6 — Verify like a deployment, not like a unit test

```bash
# 1. plugin loads, nothing else broke
python tests/test_architecture.py
python tests/test_luna_features.py

# 2. the app starts with the plugin present
FERMI_LLM_PORT=8899 ./scripts/run_webapp.sh &
curl -s localhost:8899/api/platform | python -m json.tool | head -40

# 3. a real analysis still runs end to end
#    (chat → review → approve → result; see docs/testing.md)
```

A plugin that passes its own tests but breaks the run pipeline is not
integrated.

## Step 7 — Report

Deliver, in writing:

- the inventory table with the decision per capability;
- what you dropped and why;
- anything that did not fit an extension point (a platform gap);
- how to enable/disable each new component;
- test output, verbatim, including failures you could not fix.

---

## Sanity checks an agent gets wrong

- **Do not** create a new `kinds` constant to make something fit. New kinds
  are platform changes with contracts and docs.
- **Do not** import one component from another (R7). Use `ctx`.
- **Do not** modify YAML or the script after approval (R8), or add a code
  path that runs analysis outside the execution backend.
- **Do not** add `FERMI_LLM_*` environment variables; namespace yours.
- **Do not** leave `sys.path` manipulation, `print` debugging, or a second
  logging configuration in ported code.
- **Do** keep their comments explaining *why* a workaround exists. Those are
  the expensive part of the other codebase.

## Prompt to start from

> Read `docs/integration-rules.md`, `docs/module-types.md` and
> `docs/agent-playbook.md` in this repository. Then inventory the codebase at
> `<path>` following Step 1, and report the table with your stack/replace
> decision per capability before writing any code.

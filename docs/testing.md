# Testing

Three layers. Run the first two on every change; the third before a release
or after touching the run pipeline.

```bash
source /opt/miniconda3/etc/profile.d/conda.sh && conda activate fermi-llm
export PYTHONPATH="$PWD/src:$PWD/tests"
```

## 1. Architecture tests — seconds

```bash
python tests/test_architecture.py
```

Covers the guarantees the integration rules promise: priority ordering,
`replaces`/`requires`/disable semantics, contract checking, a failing plugin
being skipped rather than fatal, the hook bus, and the example plugins
loading and working. Run this after any change to `core/`, and when you add
a plugin.

## 2. Component tests — a minute, no network

```bash
python tests/test_luna_features.py          # 71 tests
python tests/fermipy/test_data_access.py    # 4FGL resolver + weekly registry
python tests/fermipy/test_diffuse_models.py
```

`test_luna_features.py` is the regression suite that predates the modular
split: model backends (mocked I/O), the review flow, the script validator and
runner, guardrails, notebook export, auth. It reaches the code through
`tests/compat.py`, which maps the old monolith names to their modules — also
useful as a "where did this go?" index.

Add a new test by appending the function to the `UNIT_TESTS` list at the
bottom; there is no auto-discovery.

## 3. End-to-end — ~12 minutes, needs a running server

```bash
FERMI_LLM_PORT=8899 ./scripts/run_webapp.sh &
python tests/test_e2e.py --url http://localhost:8899 --model template
```

Drives the real HTTP API: create a task, generate, review, approve, run a
real FermiPy analysis on the bundled Mrk 421 data, then download every
available export. It asserts the SED bin contract (what the configuration
implies is what the run produced) and that the approval flow refuses to run
anything before approval.

Use `--model gemini-2.5-flash-lite` (or another configured backend) to
include real generation; `template` needs no credentials.

## Testing a plugin

Load it into a throwaway registry — never the process-wide one:

```python
import conftest
from fermi_llm.core import kinds
from fermi_llm.core.loader import LoadReport, load_paths
from fermi_llm.core.registry import Registry

def test_my_plugin():
    registry, report = Registry(), LoadReport()
    load_paths(['/path/to/plugin/parent'], conftest.context(), report, registry)
    assert report.failures == []
    component = registry.create(kinds.EXPORTER, 'mylab_sed_csv',
                                conftest.context())
    ...
```

`tests/test_architecture.py` has complete examples, including one that
exercises the example validator and one that drives the example model
provider.

## What a change should be tested against

| You changed | Run |
|---|---|
| `core/` | architecture + component tests |
| a guardrail or validator | component tests, plus a case in `test_luna_features.py` |
| the run pipeline or the runner | all three layers |
| a model provider | component tests; `--live` for real API calls |
| the frontend | component tests (they assert on the served assets) + open the app |
| a plugin | architecture tests + your own |

## Notes

- Tests write sessions to `FERMI_LLM_SESSIONS_DIR` (a temp directory by
  default via `tests/conftest.py`), never to a deployment's data.
- `test_luna_features.py --live` makes real API calls and needs a server on
  `:8765`; it is opt-in for that reason.
- There is no linter and no build step. Match the surrounding style.

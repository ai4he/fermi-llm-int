# Plugin quickstart

Ten minutes from nothing to a module running in the app. If you have not
installed or run the platform yet, start with
[getting-started.md](getting-started.md).

## 1. Pick the extension point

Answer "what am I adding?" and read that section of
[module-types.md](module-types.md):

| I want to… | Kind |
|---|---|
| use our own model / a different LLM | `model_provider` |
| add a consistency rule to model output | `guardrail` |
| enforce a site policy before runs | `validator` |
| run analyses on our cluster | `execution_backend` |
| read our data archive | `data_source` |
| add a download format | `exporter` |
| add a panel, viewer or widget | `ui_extension` |
| restyle the app | `skin` |
| add an endpoint | `api_extension` |
| record runs somewhere | a hook, not a component |

## 2. Write it

A plugin is a directory with a `plugin.py` exposing
`register(registry, ctx)`:

```
mylab_csv/
└── plugin.py
```

```python
"""Export the fitted SED as CSV."""

import csv, io
from fermi_llm.core import kinds
from fermi_llm.core.contracts import ExportArtifact


class SEDCsvExporter:
    name = 'mylab_sed_csv'
    label = 'SED table (.csv)'
    media_type = 'text/csv'

    def __init__(self, ctx):
        self.ctx = ctx

    def available(self, session) -> bool:
        result = session.pipeline_result or {}
        return bool(((result.get('level4') or {}).get('sed_data') or {}).get('e_ctr'))

    def build(self, session) -> ExportArtifact:
        sed = (session.pipeline_result or {})['level4']['sed_data']
        buf = io.StringIO()
        writer = csv.writer(buf)
        writer.writerow(['e_ctr_MeV', 'e2dnde'])
        for energy, flux in zip(sed['e_ctr'], sed['e2dnde']):
            writer.writerow([energy, flux])
        return ExportArtifact(filename=f'{session.session_id}_sed.csv',
                              media_type=self.media_type,
                              content=buf.getvalue().encode())


def register(registry, ctx=None):
    registry.register(kinds.EXPORTER, 'mylab_sed_csv', SEDCsvExporter,
                      priority=500, source='mylab:csv',
                      metadata={'label': SEDCsvExporter.label})
```

## 3. Load it

Either drop it in the local directory:

```bash
cp -r mylab_csv /path/to/fermi-llm-int/plugins/
./scripts/run_webapp.sh
```

…or point the platform at where it already lives:

```bash
export FERMI_LLM_PLUGIN_PATH=/opt/mylab/fermi_modules
```

…or ship it as a package other institutes can `pip install`:

```toml
# mylab-fermi/pyproject.toml
[project.entry-points."fermi_llm.plugins"]
mylab_csv = "mylab_fermi.csv_exporter"
```

## 4. Check it loaded

```bash
curl -s localhost:8765/api/platform | python -m json.tool | grep -A4 mylab
```

Every component appears with a state: `active`, `replaced`, `disabled` or
`blocked`. A plugin that failed to import is listed under
`plugin_failures` with its traceback — the app still runs.

## 5. Test it

```python
# tests/test_mylab_csv.py
import conftest                                   # bootstraps a context
from fermi_llm.core import kinds
from fermi_llm.core.loader import LoadReport, load_paths
from fermi_llm.core.registry import Registry


def test_plugin_registers_and_builds():
    registry = Registry()
    ctx = conftest.context()
    report = LoadReport()
    load_paths(['/path/to/mylab_csv/..'], ctx, report, registry)
    assert report.failures == []
    assert 'mylab_sed_csv' in registry.names(kinds.EXPORTER)
```

`tests/test_architecture.py` has working examples of this pattern, including
building a component and exercising it.

## Common tasks

**Replace a core component**

```python
registry.register(kinds.EXPORTER, 'mylab_notebook', MyNotebook,
                  replaces=('notebook',), source='mylab')
```

**Run before a core guardrail** — core uses 100/200/300…; pick 250.

**Turn something off** — `configs/plugins.toml`:

```toml
[components]
disable = ["guardrail:full_data_defaults"]
```

**Read settings**

```toml
[settings."exporter:mylab_sed_csv"]
include_upper_limits = true
```

```python
opts = ctx.component_settings(kinds.EXPORTER, self.name)
```

**React to runs without changing anything**

```python
def register(registry, ctx=None):
    @ctx.hooks.on('run.finished', name='mylab.notify')
    def notify(session=None, result=None, **_):
        ...
```

## Rules

Before opening a pull request, read
[integration-rules.md](integration-rules.md) — especially namespacing (R2),
failing small (R5), and not reaching into other components (R7).

# Getting started

Everything you need for your first hour: install it, run an analysis,
configure it for your machine, and add your first module.

If you only want to *use* a running instance, read part 2 and stop there.

---

## 1. Install

You need Linux and conda. The setup script creates the `fermi-llm`
environment with FermiPy and the Fermi ScienceTools in it.

```bash
git clone https://github.com/ai4he/fermi-llm-int.git
cd fermi-llm-int
./scripts/setup.sh                 # ~15 minutes, mostly conda
```

Then start it:

```bash
./scripts/run_webapp.sh            # http://localhost:8765
```

Open the page. With no configuration at all you get:

- guest mode (no sign-in),
- the `template` model, which writes configurations from rules instead of an
  LLM,
- the three bundled demonstration sources — **Mrk 421**, the **Vela pulsar**
  and the **Crab** — with their photon files included in the checkout.

That is enough to run a complete, real analysis. Everything else is opt-in.

> Running on a shared machine where someone else already uses 8765? Use
> `FERMI_LLM_PORT=8899 ./scripts/run_webapp.sh`.

---

## 2. Run your first analysis

**Step 1 — describe it.** In the chat box:

> Perform a spectral analysis of Markarian 421 between 1 GeV and 1 TeV and
> compute the spectral energy distribution.

You get a FermiPy YAML configuration and a matching `analysis.py`, plus a
note of anything the platform corrected — a missing SED section, a source
name that did not match the 4FGL catalog, diffuse-model paths that had to be
pinned to this machine's files. Those notes are the point: you can see what
was changed and why.

**Step 2 — refine it.** Keep talking to it:

> Use 6 energy bins per decade and add a light curve with 10 bins.

Follow-up messages *edit* the existing files. Sections you did not mention
are preserved.

**Step 3 — review.** Click **Run Pipeline**. Nothing runs yet. You get the
exact final YAML and the exact `analysis.py` that would execute, after
repairs and validation, with:

- a list of every deterministic repair applied,
- the script review (safety, FermiPy API use, call order, agreement with the
  YAML),
- the expected number of SED bins,
- downloads of both files, so you can inspect or run them yourself.

Read them. Edit them in the panels if you want — editing invalidates the
approval, and you review again.

**Step 4 — approve and run.** Click **Run Pipeline** a second time. Now the
exact reviewed bytes run, in an isolated process, with live progress. A long
run asks for confirmation of the estimated duration first.

**Step 5 — take the results.** When it finishes you can download:

| | |
|---|---|
| `final_config.yaml` | what actually ran |
| `analysis.py` | runnable standalone, reproduces the analysis |
| `analysis_files.zip` | both, plus a manifest with SHA-256 digests |
| All outputs | figures, FITS products, the FermiPy working files |
| Notebook | prompt, config, script and figures in one `.ipynb` |
| Logs | the model side and the FermiPy side, separately |

**If something goes wrong:** the run's status and error are shown in the
results panel, the per-stage progress log is kept with the task, and a
product that failed (say a PS map) is reported without discarding the fit
that succeeded.

---

## 3. Configure it for your site

### Enable a real model

Drop a credential file into `configs/` and restart. Each file enables one
backend; a backend without its file stays visible in the picker, greyed out,
with the reason.

| File | Enables |
|---|---|
| `openai_key.txt` | ChatGPT Luna models (OpenAI API) |
| `gemini_keys.json` | Gemini models (one or more keys, rotated) |
| `clemson_vllm_key.txt` | the Clemson RCD vLLM endpoint, incl. the FermiPy LoRA |
| `google_oauth.json` | Google Sign-In, so tasks follow a user across devices |

`.example` templates for all of them are in `configs/`. Real files are
gitignored — never commit one.

### Point it at your data

Bundled demo data works out of the box. For any other 4FGL source you need
the weekly all-sky archive:

```bash
cp configs/env.sh.example configs/env.sh
# then, in configs/env.sh:
export FERMI_LLM_FERMI_DATA_DIR=/path/to/weekly/photon/files
```

Analyses of non-bundled targets then resolve the source through the 4FGL
catalog, build the file list and show you the real paths and time range
*before* you approve a run.

### Choose which modules run

Copy the template and edit:

```bash
cp configs/plugins.toml.example configs/plugins.toml
```

```toml
[components]
# turn something off
disable = ["guardrail:full_data_defaults"]

# pick a singleton
execution_backend = "local_subprocess"
skin = "classic"

# configure a component
[settings."validator:mylab_energy_policy"]
min_emin_mev = 100.0
```

To see the effect, open `/api/platform` (or `curl` it): every component with
its state — `active`, `replaced`, `disabled` or `blocked` — plus any plugin
that failed to load, with its traceback. That endpoint is the answer to
"why isn't my feature showing up?".

### Run it as a service

See [deployment.md](deployment.md) for a systemd unit, nginx and HTTPS, and
the full list of environment variables.

---

## 4. Extend it

The reason this repository exists. Adding a capability does not mean forking
the platform: you write a module and register it.

### Decide what you are adding

| You want to… | Module kind |
|---|---|
| use your own model or LLM service | `model_provider` |
| fix a mistake models keep making | `guardrail` |
| enforce a policy before runs | `validator` |
| run analyses on your cluster | `execution_backend` |
| read your own data archive | `data_source` |
| add a download format | `exporter` |
| add a panel, plot or widget | `ui_extension` |
| restyle the interface | `skin` |
| add an HTTP endpoint | `api_extension` |
| record or forward what happens | a hook |

Full list with contracts: [module-types.md](module-types.md).

### Write it

One file. This adds a CSV download of the fitted SED:

```python
# plugins/mylab_csv/plugin.py
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
        sed = ((session.pipeline_result or {}).get('level4') or {}).get('sed_data') or {}
        return bool(sed.get('e_ctr'))

    def build(self, session) -> ExportArtifact:
        sed = session.pipeline_result['level4']['sed_data']
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
                      priority=500, source='mylab')
```

Restart. The download appears in the results panel. No core file changed, no
frontend change.

### Load it from somewhere else

```bash
export FERMI_LLM_PLUGIN_PATH=/opt/mylab/fermi_modules   # your own directory
```

or ship it so colleagues can `pip install` it:

```toml
[project.entry-points."fermi_llm.plugins"]
mylab_csv = "mylab_fermi.csv_exporter"
```

### Stack, replace, remove

- **Stack:** register another component of the same kind — guardrails,
  validators, exporters, panels and data sources all run together, ordered by
  `priority` (core uses 100, 200, 300… so there is room between).
- **Replace:** `replaces=("notebook",)` takes over from a core component,
  which stays registered but inactive — so reverting is a config change.
- **Remove:** `disable = ["kind:name"]` in `configs/plugins.toml`.

### Test it

```bash
export PYTHONPATH="$PWD/src:$PWD/tests"
python tests/test_architecture.py      # the platform's own guarantees
python tests/test_luna_features.py     # component regression suite
```

`tests/test_architecture.py` shows how to load a plugin directory into a
throwaway registry and exercise the component — copy that pattern for yours.

### Five working examples

[`examples/plugins/`](../examples/plugins/) has one of each, each about a
page long and commented:

| | |
|---|---|
| `csv_exporter` | a new download format |
| `echo_model` | a model backend |
| `strict_energy_validator` | a site policy check |
| `spectrum_panel` | a panel plus frontend assets |
| `midnight_skin` | a dark skin layered on the default |
| `run_logger` | hooks, without changing anything |

Copy one into `plugins/` and restart to see it appear.

---

## Where to go next

| | |
|---|---|
| [architecture.md](architecture.md) | how the pieces fit, with diagrams |
| [module-types.md](module-types.md) | the contract for every extension point |
| [integration-rules.md](integration-rules.md) | the rules a shared module must follow |
| [plugin-quickstart.md](plugin-quickstart.md) | a shorter path for your second module |
| [agent-playbook.md](agent-playbook.md) | integrating an existing codebase, for agents |
| [testing.md](testing.md) | the three test layers |
| [deployment.md](deployment.md) | services, nginx, every setting |
| [CONTRIBUTING.md](../CONTRIBUTING.md) | sending it upstream |

Stuck? `GET /api/platform` first — it usually answers the question. Then open
an issue with its output.

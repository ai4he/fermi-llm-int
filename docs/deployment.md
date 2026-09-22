# Deployment

## Requirements

- Linux, Python 3.10+
- The `fermi-llm` conda environment (FermiPy + Fermi ScienceTools). Create it
  with `./scripts/setup.sh`; `environment.yml` pins it.
- Optional: a GPU for local models, an API key for a hosted model, the weekly
  LAT archive for non-bundled targets.

Nothing above is needed to *start* the app: with no credentials it runs in
guest mode with the `template` model and the three bundled demo sources.

## Install

```bash
git clone https://github.com/ai4he/fermi-llm-int.git
cd fermi-llm-int
./scripts/setup.sh                 # conda env + dependencies
cp configs/env.sh.example configs/env.sh
cp configs/plugins.toml.example configs/plugins.toml
```

Credentials are files in `configs/`, all gitignored:

| File | Enables |
|---|---|
| `openai_key.txt` | the ChatGPT Luna models |
| `gemini_keys.json` | the Gemini models |
| `clemson_vllm_key.txt` | the Clemson RCD vLLM endpoint |
| `google_oauth.json` (or `client_secret_*.json`) | Google Sign-In |
| `auth_secret.txt` | session cookies (generated on first run) |

A backend without its key file is listed in the picker but greyed out with
the reason — nothing else breaks.

## Run

```bash
./scripts/run_webapp.sh                      # :8765
FERMI_LLM_PORT=8769 ./scripts/run_webapp.sh  # another instance
```

Or as a module: `python -m fermi_llm.app`. Or installed: `pip install -e .`
then `fermi-llm`.

## As a service (systemd user unit)

```ini
[Unit]
Description=Fermi-LLM web app (port 8765)
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
WorkingDirectory=/opt/fermi-llm-int
Environment=FERMI_LLM_PORT=8765
ExecStart=/bin/bash -lc 'source /opt/miniconda3/etc/profile.d/conda.sh && conda activate fermi-llm && exec ./scripts/run_webapp.sh'
Restart=on-failure
RestartSec=5
KillMode=mixed

[Install]
WantedBy=default.target
```

```bash
systemctl --user daemon-reload
systemctl --user enable --now fermi-llm
loginctl enable-linger "$USER"     # survive logout / reboot
```

`deploy/` has the nginx reverse-proxy and HTTPS scripts.

## Configuration

Environment variables (see `configs/env.sh.example` for the full list):

| Variable | Default | Meaning |
|---|---|---|
| `FERMI_LLM_HOST` / `FERMI_LLM_PORT` | `0.0.0.0` / `8765` | bind address |
| `FERMI_LLM_PROJECT_DIR` | the checkout | root for data and configs |
| `FERMI_LLM_SESSIONS_DIR` | `webapp/sessions` | where tasks live |
| `FERMI_LLM_PLUGIN_PATH` | `plugins` | extra plugin directories (`:`-separated) |
| `FERMI_LLM_SKIN` | `classic` | active skin |
| `FERMI_LLM_EXECUTION_BACKEND` | `local_subprocess` | where analyses run |
| `FERMI_LLM_DEFAULT_MODEL` | `gpt-5.6-luna` | pre-selected model |
| `FERMI_LLM_MAX_ACTIVE_PIPELINES` | `4` | concurrent runs |
| `FERMI_LLM_CONFIRM_THRESHOLD_MIN` | `60` | confirm runs estimated longer |
| `FERMI_LLM_FIT_TIMEOUT` | `1800` | per-fit seconds |
| `FERMI_LLM_FERMI_DATA_DIR` | — | weekly LAT archive |
| `FERMI_LLM_SCRIPT_MAX_MEMORY_GB` / `_FILE_GB` | `64` / `20` | analysis rlimits |

Components are selected in `configs/plugins.toml`
([plugins.toml.example](../configs/plugins.toml.example)).

## Multiple instances

Instances are independent: separate port, separate `FERMI_LLM_SESSIONS_DIR`,
separate plugin set. Point two instances at the same sessions directory only
if you want them to share tasks — they will, including runs in flight.

A staging instance on another port with
`FERMI_LLM_PLUGIN_PATH=/path/to/candidate-plugins` is the recommended way to
try a new module before it reaches users.

## Health checks

| Endpoint | Tells you |
|---|---|
| `GET /` | the app is serving |
| `GET /api/versions` | FermiPy / ScienceTools / Python versions |
| `GET /api/models` | which backends are usable right now |
| `GET /api/platform` | every component's state, and any plugin that failed |

`GET /api/platform` is the first thing to read when a feature is missing:
the component is either `disabled`, `replaced`, `blocked` by a missing
requirement, or in `plugin_failures` with its traceback.

## Upgrading

```bash
git pull
./scripts/setup.sh --update        # only if environment.yml changed
python tests/test_architecture.py
python tests/test_luna_features.py
systemctl --user restart fermi-llm
```

Sessions are forward-compatible: a task written by an older version loads,
and missing fields take their defaults.

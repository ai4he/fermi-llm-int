# Contributing

Contributions from the FermiPy community are what this repository is for. If
something in your analysis workflow is tedious, the fix probably belongs
here as a module — and most modules are one file.

## Ways to contribute

| | Where it goes |
|---|---|
| A new model, format, panel, validator, data source… | a plugin — [plugin-quickstart.md](docs/plugin-quickstart.md) |
| A fix to an existing component | that component |
| A new extension point | an issue first; it changes contracts |
| Documentation, examples, better error messages | always welcome |
| A bug report | an issue with the output of `GET /api/platform` |

If you are unsure whether something is a plugin or a core change: if another
institute could reasonably not want it, it is a plugin.

If you have not run the app yet, start with
[docs/getting-started.md](docs/getting-started.md) — it covers installing,
running an analysis and writing a first module.

## Setting up

```bash
git clone https://github.com/ai4he/fermi-llm-int.git
cd fermi-llm-int
./scripts/setup.sh
source /opt/miniconda3/etc/profile.d/conda.sh && conda activate fermi-llm
export PYTHONPATH="$PWD/src:$PWD/tests"
python tests/test_architecture.py        # should pass before you change anything
```

Run the app on a spare port while you work:

```bash
FERMI_LLM_PORT=8899 ./scripts/run_webapp.sh
```

## Before you open a pull request

1. **Read [integration-rules.md](docs/integration-rules.md)** if you are
   adding a module. The checklist at the end is what reviewers use.
2. **Tests pass**, and your change has one:
   ```bash
   python tests/test_architecture.py
   python tests/test_luna_features.py
   ```
   Anything touching the run pipeline also needs
   `python tests/test_e2e.py --url http://localhost:8899`.
3. **No new dependency** without saying why in the PR. The deployment is a
   conda environment on a shared machine; the frontend has no build step and
   no framework.
4. **Explain the why in the code.** This codebase documents reasons, not
   mechanics: a comment that says what a line does is noise, one that says
   why a source name is corrected before generation is the point.

## Style

- Match the file you are editing. Python is plain and explicit; four-space
  indents; docstrings on every public class and non-obvious function.
- Frontend: vanilla JS, no bundler, namespaced ids and classes for anything
  a plugin adds.
- Names are for scientists, not implementers: a panel is a "task", not a
  "session object", in anything a user reads.
- Never commit credentials. `configs/` is gitignored apart from
  `.example` templates; check `git status` before committing.

## Reviewing

Reviewers check, in order: does it follow the integration rules; does it
break the approval contract; is it tested; would another deployment be able
to turn it off. A module that cannot be disabled is a design problem, not a
feature.

## Reporting a problem

Include:

- what you asked the app to do, and what happened;
- the output of `GET /api/platform` (it shows every active component and any
  plugin that failed to load);
- the task id, if a run is involved — the session directory has the full
  progress log, the executed YAML and the script;
- versions from `GET /api/versions`.

Please do not paste credentials or private data; the logs can contain the
prompts you sent.

## Code of conduct

Be straightforward and kind. Assume the person on the other side is a
scientist with a deadline, because they usually are.

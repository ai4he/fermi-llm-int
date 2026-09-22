# Integration rules

These are the rules a module must follow to be part of this platform. They
are written to be read by a person or by a coding agent, and they are the
contract between institutes: follow them and your module works in everyone's
deployment, including in combination with modules you have never seen.

Companion documents: [module-types.md](module-types.md) for the contract of
each kind, [plugin-quickstart.md](plugin-quickstart.md) to build one in ten
minutes, [agent-playbook.md](agent-playbook.md) for integrating an existing
codebase.

---

## R1 — A module registers; it never patches

Everything a module adds enters through the registry:

```python
from fermi_llm.core import kinds

def register(registry, ctx=None):
    registry.register(kinds.EXPORTER, 'my_format', MyExporter,
                      priority=500, source='mylab:my_format',
                      metadata={'label': 'My format (.abc)'})
```

Never import a core module to monkey-patch it, and never edit a file under
`src/fermi_llm/core/` or `src/fermi_llm/components/` to make your module fit.
If you cannot express what you need through a contract, that is a gap in the
contracts — open an issue, do not work around it.

## R2 — Names are namespaced and permanent

A component name is a public identifier: deployments disable it by name,
other modules require it by name, `/api/platform` reports it.

- Prefix anything institute-specific: `mylab_spectrum_viewer`, not `viewer`.
- Use lowercase with underscores.
- `source=` says who ships it: `"mylab:my_format"`.
- Renaming a released component is a breaking change. Register the new name
  with `replaces=("old_name",)` instead.

## R3 — One module, one job

A module does one thing at one extension point. If your integration adds a
model *and* a panel *and* an exporter, that is three components in one
plugin package — not one component doing three jobs. It is what lets a
deployment take the panel and skip the model.

## R4 — Declare how you relate to what is already there

| You want to | Declare |
|---|---|
| run alongside existing components | just a `priority` |
| take over from one | `replaces=("their_name",)` |
| require another module | `requires=("their_name",)` |
| run before/after a specific one | a `priority` between theirs |

Core priorities are round numbers (100, 200, 300…) so there is always room
between them. Never silently duplicate a core component's name: the registry
refuses it and tells you to use `replaces`.

## R5 — Fail small

A module must not be able to take the platform down.

- Raising during `register()` or construction disables *your* module; the
  platform still starts and reports the failure at `/api/platform`.
- Inside a request, catch what you can handle and return a neutral result.
  A guardrail that cannot parse the YAML returns the turn unchanged; an
  exporter that has nothing to export returns `available() == False`.
- Never call `sys.exit`, never spawn a process the platform cannot see
  (use the `execution_backend` contract), never block the event loop for
  more than a moment — long work belongs in a thread or a worker.

## R6 — Configuration comes from the context

Read settings from `ctx.component_settings(kind, name)`, which is fed by
`configs/plugins.toml`:

```toml
[settings."validator:mylab_energy_policy"]
min_emin_mev = 100.0
```

```python
options = ctx.component_settings(kinds.VALIDATOR, self.name)
self.min_emin = float(options.get('min_emin_mev', 50.0))
```

Environment variables that belong to your module are prefixed with your
namespace (`MYLAB_...`). Do not add new `FERMI_LLM_*` variables: those are
the platform's.

Secrets are files in `configs/`, referenced by path, never committed, and
never logged. Anything matching a credential pattern is scrubbed from the
environment of an analysis subprocess — do not defeat that.

## R7 — Do not reach into another component

Ask the context:

```python
models = ctx.models                       # the model service
store = ctx.sessions                      # active session store
backend = ctx.executor                    # active execution backend
sources = ctx.build_stack(kinds.DATA_SOURCE)
```

Importing `fermi_llm.components.models.providers.openai_api` from your
exporter couples you to a component a deployment may have replaced. The only
safe imports are `fermi_llm.core.*`, `fermi_llm.core.contracts`, and — for
analysis code — `fermi_llm.fermipy.*`.

## R8 — Respect the review contract

The platform's central promise is that **the files the user approved are the
files that run**. A module may:

- propose a change *before* approval (that is what the review stage is for),
- refuse a run (a validator returning errors),
- observe what ran (hooks, exporters).

A module may **not** modify the YAML or the script after approval, execute
analysis code outside the execution backend, or bypass the approval token.
A change here is a platform-level decision, not a plugin's.

## R9 — Keep the frontend build-free and scoped

- No bundler, no framework requirement: plain JS and CSS, served as-is.
- A UI extension declares its assets in its manifest; the loader injects
  them. Do not edit `index.html`.
- Namespace your DOM ids and CSS classes (`mylab-…`), and scope styles to
  your panel. A skin is the only component allowed to restyle the app, and
  it does that through the CSS variables `classic.css` defines.
- Use `window.FermiLLM.on(...)` for events and `registerPanel` for panels.
  Do not reach into core functions; they are not a public API.

## R10 — Ship tests and documentation with the module

A module is not integrated until:

1. it has a test that proves it registers and does its job
   (`tests/test_architecture.py` shows the pattern, including loading a
   plugin directory into a throwaway registry);
2. every public class carries a docstring saying what it is for and what it
   assumes;
3. the README of your plugin package says which kinds it registers, which
   settings it reads, and what it needs from the host (FermiPy version, GPU,
   network access, credentials).

## R11 — Compatibility is a promise

Contracts in `fermi_llm/core/contracts.py` are versioned by behaviour, not
by a number: fields get added, never removed or repurposed. If you need a
field that does not exist, propose it — a pull request that adds an optional
field with a default is easy to accept; one that changes a meaning is not.

When the platform must break a contract, the change lands with: a migration
note in `docs/migration.md`, a deprecation window in which both shapes work,
and an entry in the release notes.

## R12 — Data stays where the deployment put it

Write only inside the session directory you are given
(`session.session_dir`) or a path from `ctx.settings`. Never write into the
checkout, never assume a shared filesystem, never hardcode `/tmp` (use
`tempfile`). A module that writes outside its session breaks multi-user
deployments and makes tasks non-portable.

---

## Checklist before you open a pull request

- [ ] One job per component; namespaced names; `source=` set.
- [ ] Registers through `register(registry, ctx)`; no patching, no core edits.
- [ ] `priority` chosen deliberately; `replaces`/`requires` declared.
- [ ] Failure paths return neutral results; nothing raises out of a request.
- [ ] Settings via `ctx.component_settings`; secrets via `configs/`, ignored by git.
- [ ] No imports from other components; only core, contracts and `fermipy`.
- [ ] Approval contract untouched.
- [ ] Frontend assets namespaced, declared in the manifest, no build step.
- [ ] Tests included and passing: `python tests/test_architecture.py`.
- [ ] README documents kinds, settings and host requirements.

# Local plugin directory

Anything placed here is loaded at startup (a `.py` file, or a directory with
a `plugin.py`), after the built-in components and installed packages. Use it
for site-specific modules you do not intend to publish.

```bash
cp -r examples/plugins/csv_exporter plugins/
./scripts/run_webapp.sh          # "SED table (.csv)" now appears in downloads
```

The contents are gitignored on purpose: this directory is per-deployment. A
plugin meant for other institutes belongs in its own repository (see
`docs/plugin-quickstart.md`), not here.

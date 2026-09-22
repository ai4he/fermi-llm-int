"""Every output file a run produced, as one archive.

Large intermediates (photon files, event lists) are skipped: the archive is
meant to be shareable, and regenerating an ``ft1`` file is cheaper than
mailing it. The final YAML, the script and the manifest are added at the
root so the archive doubles as a reproducibility bundle.
"""

from __future__ import annotations

import os
import re
import tempfile
import zipfile

from ...core import kinds
from ...core.contracts import ExportArtifact
from ...core.registry import REGISTRY
from .run_bundle import ensure_session_run_exports as _ensure_session_run_exports

# Files never bundled into the downloadable artifacts.zip: raw event/photon
# FITS files (the analysis inputs, not outputs -- also frequently large).
_ZIP_EXCLUDE_NAME_PATTERNS = ('_ph.fits', 'ft1', '_events.fits')

# Any single file over this size is left out of the zip regardless of type
# (this is what keeps a large exposure/livetime cube out, but does not
# single it out -- any other oversized product is excluded the same way).
_ZIP_MAX_FILE_BYTES = 100 * 1024 * 1024  # 100 MB


def _zip_should_include(fname, fsize):
    low = fname.lower()
    if any(p in low for p in _ZIP_EXCLUDE_NAME_PATTERNS):
        return False
    if fsize > _ZIP_MAX_FILE_BYTES:
        return False
    return True


class ArtifactsZipExporter:
    name = 'artifacts_zip'
    label = 'All outputs (.zip)'
    media_type = 'application/zip'

    def __init__(self, ctx):
        self.ctx = ctx

    def available(self, session) -> bool:
        outdir = os.path.join(session.session_dir, 'fermipy_workdir', 'output')
        return os.path.isdir(outdir)

    def build(self, session) -> ExportArtifact:
        session_id = session.session_id
        _ensure_session_run_exports(session)

        outdir = os.path.join(session.session_dir, 'fermipy_workdir', 'output')
        if not os.path.isdir(outdir):
            raise FileNotFoundError(
                "No run output directory found for this session")

        # Resolve once so every walked entry can be checked to still be inside
        # it (defends against a symlink escaping the session's output dir).
        outdir_real = os.path.realpath(outdir)

        tmp_fd, tmp_path = tempfile.mkstemp(suffix='.zip', dir=session.session_dir)
        os.close(tmp_fd)
        try:
            with zipfile.ZipFile(tmp_path, mode='w', compression=zipfile.ZIP_DEFLATED) as zf:
                for root, _dirs, files in os.walk(outdir_real):
                    for fn in files:
                        fpath = os.path.realpath(os.path.join(root, fn))
                        if not (fpath == outdir_real or fpath.startswith(outdir_real + os.sep)):
                            continue  # path traversal / symlink escape guard
                        try:
                            fsize = os.path.getsize(fpath)
                        except OSError:
                            continue
                        if not _zip_should_include(fn, fsize):
                            continue
                        arcname = os.path.relpath(fpath, outdir_real)
                        try:
                            zf.write(fpath, arcname)
                        except Exception:
                            pass
            # Also include the flat artifacts/ dir (plots + roi_sources.json)
            # copied out during the run, so a user gets plots even if the same-
            # named file was elsewhere excluded from the outdir walk above.
            artifacts_dir = os.path.join(session.session_dir, 'artifacts')
            if os.path.isdir(artifacts_dir):
                with zipfile.ZipFile(tmp_path, mode='a', compression=zipfile.ZIP_DEFLATED) as zf:
                    existing = set(zf.namelist())
                    for fn in os.listdir(artifacts_dir):
                        fpath = os.path.join(artifacts_dir, fn)
                        if not os.path.isfile(fpath):
                            continue
                        arcname = os.path.join('artifacts', fn)
                        if arcname in existing:
                            continue
                        try:
                            fsize = os.path.getsize(fpath)
                        except OSError:
                            continue
                        if not _zip_should_include(fn, fsize):
                            continue
                        try:
                            zf.write(fpath, arcname)
                        except Exception:
                            pass

            # Put the final validated configuration and its matching runnable
            # driver at the root of the existing all-outputs archive as well.
            pipeline_result = session.pipeline_result or {}
            latest_run_id = pipeline_result.get('run_id')
            if latest_run_id and re.fullmatch(r'[A-Za-z0-9_-]+', latest_run_id):
                run_dir = os.path.join(session.session_dir, 'runs', latest_run_id)
                with zipfile.ZipFile(tmp_path, mode='a',
                                     compression=zipfile.ZIP_DEFLATED) as zf:
                    existing = set(zf.namelist())
                    for fn in ('final_config.yaml', 'analysis.py', 'manifest.json'):
                        fpath = os.path.join(run_dir, fn)
                        if fn not in existing and os.path.isfile(fpath):
                            zf.write(fpath, fn)
        except Exception:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            raise

        # The archive is a temporary file; the API layer deletes it after
        # the response is sent (see ExportArtifact.headers['x-cleanup']).
        return ExportArtifact(
            filename=f'{session_id}_artifacts.zip',
            media_type=self.media_type,
            path=tmp_path,
            headers={'x-cleanup': tmp_path})


REGISTRY.register(kinds.EXPORTER, 'artifacts_zip', ArtifactsZipExporter,
                  priority=400, source='core',
                  metadata={'label': ArtifactsZipExporter.label})

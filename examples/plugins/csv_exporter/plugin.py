"""Example: a new download format, in one file.

Adds "SED as CSV" to the downloads of any finished run. Copy this directory
into ``plugins/`` (or point ``FERMI_LLM_PLUGIN_PATH`` at it), restart, and
the entry appears in the UI — no core change, no frontend change.

What it demonstrates: the ``exporter`` contract, and that a component only
needs the objects it actually uses (here: the session).
"""

from __future__ import annotations

import csv
import io

from fermi_llm.core import kinds
from fermi_llm.core.contracts import ExportArtifact


class SEDCsvExporter:
    name = 'sed_csv'
    label = 'SED table (.csv)'
    media_type = 'text/csv'

    def __init__(self, ctx):
        self.ctx = ctx

    def _sed(self, session):
        result = session.pipeline_result or {}
        return ((result.get('level4') or {}).get('sed_data')) or {}

    def available(self, session) -> bool:
        return bool(self._sed(session).get('e_ctr'))

    def build(self, session) -> ExportArtifact:
        sed = self._sed(session)
        buffer = io.StringIO()
        writer = csv.writer(buffer)
        writer.writerow(['e_ctr_MeV', 'e2dnde', 'e2dnde_err', 'ts',
                         'is_upper_limit'])
        centres = sed.get('e_ctr') or []
        for index, energy in enumerate(centres):
            def at(key):
                values = sed.get(key) or []
                return values[index] if index < len(values) else ''
            writer.writerow([energy, at('e2dnde'), at('e2dnde_err'),
                             at('ts'), at('is_ul')])
        return ExportArtifact(
            filename=f'fermillm_{session.session_id}_sed.csv',
            media_type=self.media_type,
            content=buffer.getvalue().encode())


def register(registry, ctx=None):
    registry.register(kinds.EXPORTER, 'sed_csv', SEDCsvExporter,
                      priority=500, source='example:csv_exporter',
                      metadata={'label': SEDCsvExporter.label})

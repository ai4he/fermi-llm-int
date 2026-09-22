"""Local documentation search over ``webapp/rag_docs/``.

A deliberately simple keyword index: the deployment has no vector database
and the corpus is small. Swap in a real retriever by registering another
``knowledge_source`` — the analyzer queries every active source and merges
the hits, so sources stack rather than replace.
"""

from __future__ import annotations

import json
import os
import re

from ...core import kinds
from ...core.registry import REGISTRY
from ...fermipy.diffuse_models import diffuse_prompt_rules
from ...fermipy.fermipy_schema import fermipy_binning_prompt_rules


class FermiPyRAG:
    """Simple RAG over FermiPy documentation and API reference."""

    name = 'rag_docs'

    def __init__(self, ctx=None):
        self.ctx = ctx
        self.rag_dir = (ctx.settings.rag_dir if ctx is not None
                        else os.path.join(os.getcwd(), 'rag_docs'))
        self.docs = []
        self._load_builtin_docs()
        self._load_external_docs()

    def _load_builtin_docs(self):
        """Load built-in FermiPy API reference."""
        self.docs.append({
            'title': 'FermiPy YAML Configuration Schema',
            'content': """FermiPy GTAnalysis YAML Configuration:

Required sections:
- logging: {verbosity: int (2-3), chatter: int (2-3)}
- fileio: {outdir: str, logfile: str}
- data: {evfile: str (path to photon FITS), scfile: str (path to spacecraft FITS), ltcube: str (optional)}
- binning: {roiwidth: float (deg, typically 10), binsz: float (deg, 0.08-0.1), binsperdec: positive number (default 8)}
- selection: {emin: float (MeV), emax: float (MeV), zmax: float (90-105), target: str (4FGL name), radius: float (deg), tmin: float (MET), tmax: float (MET), evclass: int (128=SOURCE), evtype: int, filter: str}
- gtlike: {edisp: bool, edisp_disable: list, irfs: str (P8R3_SOURCE_V3)}
- model: {src_radius: float, src_roiwidth: float, galdiff: list, isodiff: list, catalogs: list}

Optional sections: sed, lightcurve, psmap, tsmap, residmap, plotting, extension, localize

""" + diffuse_prompt_rules() + """
""" + fermipy_binning_prompt_rules() + """
""",
            'keywords': ['yaml', 'configuration', 'config', 'schema', 'section', 'evtype', 'irfs', 'isodiff']
        })
        self.docs.append({
            'title': 'FermiPy Python API Reference',
            'content': """FermiPy GTAnalysis Python API:

Initialization:
  from fermipy.gtanalysis import GTAnalysis
  gta = GTAnalysis('config.yaml')
  gta.setup()  # Runs gtselect, gtmktime, gtltcube, gtsrcmaps

Source manipulation:
  gta.free_source(name)  # Free parameters of a source
  gta.free_sources(minmax_ts=[lo,hi], distance=d, pars='norm')
  gta.delete_source(name)
  gta.add_source(name, dict)  # Add new source with spectral/spatial model

Fitting:
  gta.optimize()  # Quick optimization
  gta.fit(optimizer='MINUIT', tol=1e-3)  # Full likelihood fit

Analysis:
  gta.sed(target)  # Spectral energy distribution
  gta.lightcurve(target, nbins=N)  # Light curve
  gta.tsmap(prefix, model=dict)  # TS map
  gta.residmap(prefix, model=dict)  # Residual map
  gta.psmap(cmap=str, mmap=str)  # PS map
  gta.extension(target)  # Test spatial extension
  gta.localize(target)  # Localize source position
  gta.find_sources(prefix, model=dict, sqrt_ts_threshold=float)

Output:
  gta.write_model_map(model_name=str)
  gta.write_roi(prefix, make_plots=bool)
""",
            'keywords': ['api', 'python', 'script', 'setup', 'fit', 'sed', 'tsmap', 'free_source', 'localize']
        })
        self.docs.append({
            'title': 'Common Fermi-LAT Analysis Patterns',
            'content': """Common analysis patterns:

Standard spectral analysis:
1. gta.setup()
2. Free background (galdiff, isodiff) and target
3. Free bright nearby sources (minmax_ts=[100, None], distance=3.0)
4. gta.optimize() then gta.fit()
5. gta.sed(target) for spectral energy distribution

Extension testing:
1. First do standard fit
2. gta.extension(source, free_background=True, make_plots=True)

Source localization:
1. First do standard fit
2. loc = gta.localize(target, free_background=True)
3. Optionally: gta.delete_source(target), gta.add_source(target, new_spec)

TS/Residual maps:
  model = {'Index': 2.0, 'SpatialModel': 'PointSource'}
  gta.tsmap('tsmap', model=model)
  gta.residmap('residmap', model=model)

Important parameters:
- evclass=128 (SOURCE), evtype=3 (FRONT+BACK), evtype=32 (PSF3), evtype=1 (FRONT)
- Common catalogs: 4FGL-DR3, 4FGL-DR4
- tmin/tmax in MET (Mission Elapsed Time) seconds
- emin/emax in MeV
""",
            'keywords': ['pattern', 'workflow', 'analysis', 'spectral', 'extension', 'localize', 'tsmap']
        })

    def _load_external_docs(self):
        """Load any user-provided documentation from rag_docs/."""
        if not os.path.isdir(self.rag_dir):
            return
        for fname in os.listdir(self.rag_dir):
            fpath = os.path.join(self.rag_dir, fname)
            if fname.endswith('.txt') or fname.endswith('.md'):
                try:
                    with open(fpath) as f:
                        content = f.read()
                    self.docs.append({
                        'title': fname,
                        'content': content,
                        'keywords': fname.replace('.', ' ').replace('_', ' ').split()
                    })
                except Exception:
                    pass

    def search(self, query, top_k=3):
        """Simple keyword-based search over docs."""
        query_lower = query.lower()
        scored = []
        for doc in self.docs:
            score = 0
            for kw in doc['keywords']:
                if kw.lower() in query_lower:
                    score += 2
            for word in query_lower.split():
                if len(word) > 3 and word in doc['content'].lower():
                    score += 1
            if score > 0:
                scored.append((score, doc))
        scored.sort(key=lambda x: -x[0])
        return [doc for _, doc in scored[:top_k]]


def _make_rag(ctx):
    return FermiPyRAG(ctx)


REGISTRY.register(kinds.KNOWLEDGE_SOURCE, 'rag_docs', _make_rag,
                  priority=100, source='core',
                  metadata={'label': 'Local FermiPy documentation'})

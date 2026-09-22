"""Prompt assembly and response parsing.

Everything here is provider-independent: the FermiPy schema block, few-shot
selection from ``data/train.json``, the semantic-intent envelope, and the
extraction of YAML/Python out of a model's reply. A provider that wants the
standard prompt calls ``request.build_prompt()``; one with its own format
(a fine-tune, a chat service) simply ignores it.

Replacing this module is a supported extension: register a component of kind
``model_provider`` that builds its own prompt, or ship a plugin that
monkey-patches nothing and instead wraps :class:`ModelService`.
"""

import json
import os
import re
from typing import Optional

from ...fermipy.diffuse_models import diffuse_prompt_rules
from ...fermipy.fermipy_schema import (fermipy_binning_prompt_rules,
                                       fermipy_default_fit_prompt_rules,
                                       normalize_analysis_yaml)
from ...fermipy.diffuse_models import normalize_diffuse_yaml

_PRODUCT_NAMES = (
    'sed', 'lightcurve', 'tsmap', 'residmap', 'psmap',
    'extension', 'localization',
)

SEMANTIC_INTENT_MODELS = frozenset({
    'gpt-5.6-luna',
    'gpt-5.6-luna-codex',
})

DEFAULT_MODEL_ID = os.environ.get(
    'FERMI_LLM_DEFAULT_MODEL', 'gpt-5.6-luna').strip() or 'gpt-5.6-luna'

def _semantic_intent_schema():
    """Strict JSON schema shared by prompting, the API, and unit tests."""
    return {
        'type': 'object',
        'properties': {
            'mode': {'type': 'string', 'enum': ['create', 'edit']},
            'summary': {'type': 'string'},
            'preserve_unspecified': {'type': 'boolean'},
            'target': {
                'type': 'object',
                'properties': {
                    'action': {
                        'type': 'string',
                        'enum': ['set', 'keep', 'remove', 'unspecified'],
                    },
                    'query': {'type': 'string'},
                    'evidence': {'type': 'string'},
                    'ra_deg': {'type': ['number', 'null']},
                    'dec_deg': {'type': ['number', 'null']},
                    'position_evidence': {'type': 'string'},
                    'spatial_model': {
                        'type': 'string',
                        'enum': ['point', 'gaussian', 'disk', 'template',
                                 'unspecified'],
                    },
                },
                'required': [
                    'action', 'query', 'evidence', 'ra_deg', 'dec_deg',
                    'position_evidence', 'spatial_model',
                ],
                'additionalProperties': False,
            },
            'products': {
                'type': 'array',
                'items': {
                    'type': 'object',
                    'properties': {
                        'name': {'type': 'string', 'enum': list(_PRODUCT_NAMES)},
                        'action': {
                            'type': 'string',
                            'enum': ['add', 'remove', 'update', 'keep'],
                        },
                        'evidence': {'type': 'string'},
                    },
                    'required': ['name', 'action', 'evidence'],
                    'additionalProperties': False,
                },
            },
            'selection_changes': {
                'type': 'array',
                'items': {
                    'type': 'object',
                    'properties': {
                        'path': {'type': 'string'},
                        'action': {
                            'type': 'string',
                            'enum': ['set', 'remove', 'reset', 'keep'],
                        },
                        # A JSON-encoded scalar/list/object keeps this schema
                        # strict while allowing arbitrary future FermiPy keys.
                        'value_json': {'type': 'string'},
                        'evidence': {'type': 'string'},
                    },
                    'required': ['path', 'action', 'value_json', 'evidence'],
                    'additionalProperties': False,
                },
            },
            'ambiguities': {
                'type': 'array',
                'items': {'type': 'string'},
            },
        },
        'required': [
            'mode', 'summary', 'preserve_unspecified', 'target',
            'products', 'selection_changes', 'ambiguities',
        ],
        'additionalProperties': False,
    }

SEMANTIC_GENERATION_SCHEMA = {
    'type': 'object',
    'properties': {
        'intent': _semantic_intent_schema(),
        'yaml': {'type': 'string'},
        'python': {'type': 'string'},
    },
    'required': ['intent', 'yaml', 'python'],
    'additionalProperties': False,
}

FERMIPY_SCHEMA = """FermiPy YAML Configuration Schema (STRICT - follow exactly):

REQUIRED sections:
- logging: {verbosity: int (2-3), chatter: int (2-3)}
- fileio: {outdir: str (always "output"), logfile: str}
- data: {evfile: str, scfile: str, ltcube: str (optional, omit if not given)}
- binning: {roiwidth: float (typically 10), binsz: float (0.08-0.1), binsperdec: positive number (default 8)}
- selection: {emin: float (MeV), emax: float (MeV), zmax: float (90-105), target: str, ra: float (deg, optional), dec: float (deg, optional), radius: float (15-20), tmin: float (MET seconds), tmax: float (MET seconds), evclass: int, evtype: int, filter: str or null}
- gtlike: {edisp: bool, edisp_disable: list, irfs: str}
- model: {src_radius: float, src_roiwidth: float, galdiff: list, isodiff: list, catalogs: list, sources: list (optional)}

CRITICAL RULES:
1. evclass MUST be one of: 128 (SOURCE), 256 (ULTRACLEAN)
2. evtype: 3 (FRONT+BACK), 1 (FRONT), 32 (PSF3)
""" + diffuse_prompt_rules() + """
5. tmin/tmax in MET seconds
6. Catalog target: use its exact 4FGL name (e.g. '4FGL J1104.4+3812')
7. Non-4FGL target: never invent a catalog name. Set selection.target to the
   user name, set selection.ra/dec, and include a matching model.sources entry.
   Coordinates must come explicitly from the user.
""" + fermipy_binning_prompt_rules() + """

""" + fermipy_default_fit_prompt_rules() + """

FermiPy Python API:
  from fermipy.gtanalysis import GTAnalysis
  gta = GTAnalysis('config.yaml') -> gta.setup()
  gta.free_source(name), gta.free_sources(minmax_ts=[lo,hi], distance=d)
  gta.optimize(), gta.fit()
  gta.sed(), gta.tsmap(), gta.residmap(), gta.lightcurve(), gta.localize(), gta.extension()
  gta.sed(name, loge_bins=[...]) accepts custom aligned edges in Python only
"""

def build_prompt(user_prompt, train_examples, n_shots=3, context=None,
                 include_intent=False, structured_envelope=False):
    """Build a schema-guided prompt for FermiPy code generation.

    ``context`` (optional) carries the current session's prior prompt and the
    existing YAML/Python so a follow-up message edits the existing
    configuration instead of regenerating it from scratch. It is a dict with
    optional keys: ``prompt``, ``yaml``, ``python``.
    """
    parts = [
        "You are a FermiPy expert. Generate EXACTLY one YAML configuration and one Python script.",
        "",
        FERMIPY_SCHEMA,
        "",
        # The Validation Agent executes this script as written (after review),
        # so these rules mirror its static checks (script_validator.py).
        "The Python script is executed exactly as written after review:",
        "- Create GTAnalysis with a relative config file name such as 'config.yaml' "
        "(the reviewed YAML is provided under that name) and call gta.setup() first.",
        "- Implement every analysis step the user asked for, in the requested order, "
        "with the requested options, and add nothing significant that was not requested.",
        "- Write files only to relative paths (they resolve inside the analysis working "
        "directory); FermiPy's own products are written to gta.outdir.",
        "- Use only fermipy, numpy, scipy, astropy, matplotlib, yaml, os, pathlib, json and "
        "math. No subprocess, network access, eval/exec, or file deletion.",
        "",
    ]

    if include_intent:
        parts.extend([
            "Before generating files, interpret the user's instruction semantically.",
            "Do not rely on literal section names: understand paraphrases, negation,",
            "corrections, and compound edits. For edits, mark unmentioned settings as",
            "keep/unspecified and set preserve_unspecified=true. Evidence must be a",
            "short exact phrase from the user's latest instruction.",
            "Product names are: sed, lightcurve, tsmap, residmap, psmap, extension, localization.",
            "Selection paths use dotted YAML paths such as selection.emin or selection.tmax.",
            "value_json is a JSON-encoded value, or an empty string when no value applies.",
            "If a requested target is a common name, place that exact name in target.query;",
            "the application will resolve it against the 4FGL catalog.",
            "For a target outside 4FGL, copy explicit ICRS RA/Dec coordinates from the",
            "latest user instruction into target.ra_deg/dec_deg and quote the exact",
            "coordinate phrase in position_evidence. Never infer or invent coordinates;",
            "use null/null and explain the missing position in ambiguities when absent.",
            "Use spatial_model=point unless the user explicitly supplies an extended",
            "morphology; never guess an extension or template.",
            "The YAML and Python must implement the same interpreted instruction.",
            "",
        ])

    if structured_envelope:
        parts.extend([
            "Return the response using the provided JSON schema with fields intent, yaml, and python.",
            "Put raw YAML and raw Python in their string fields; do not use markdown fences.",
            "Return no explanation outside the structured response.",
        ])
    else:
        output_items = "intent JSON, YAML, and Python" if include_intent else "YAML and Python"
        parts.extend([
            f"Output ONLY the {output_items} in markdown code blocks. No explanations.",
            "Format:",
        ])
        if include_intent:
            parts.extend([
                "### Semantic Intent:",
                "```json",
                "<intent JSON matching the requested fields>",
                "```",
            ])
        parts.extend([
            "### YAML Configuration:",
            "```yaml",
            "<yaml>",
            "```",
            "### Python Script:",
            "```python",
            "<python>",
            "```",
        ])

    if n_shots > 0 and train_examples:
        parts.append("\n## Examples:\n")
        for j, ex in enumerate(train_examples[:n_shots]):
            parts.append(f"--- Example {j+1} ---")
            parts.append(f"Task: {ex['prompt'][:300]}...")
            yaml_str = ex['response']['yaml']
            yaml_str, _ = normalize_analysis_yaml(yaml_str)
            yaml_str, _ = normalize_diffuse_yaml(yaml_str)
            script = ex['response']['script']
            if yaml_str.strip():
                parts.append(f"### YAML Configuration:\n```yaml\n{yaml_str}\n```")
            parts.append(f"### Python Script:\n```python\n{script}\n```\n")

    # Conversational edit context: when the user is iterating on an existing
    # configuration, show the current files and instruct the model to apply
    # the new request as a modification, preserving everything else.
    if context and (context.get('yaml') or context.get('python')):
        parts.append("\n--- CURRENT SESSION (edit, do not start over) ---")
        if context.get('prompt'):
            parts.append(f"The configuration below was produced for this earlier request:\nTask: {context['prompt']}")
        if context.get('yaml'):
            parts.append(f"### Current YAML Configuration:\n```yaml\n{context['yaml']}\n```")
        if context.get('python'):
            parts.append(f"### Current Python Script:\n```python\n{context['python']}\n```")
        # Continuity: outcome of the last pipeline run for this task, so a
        # follow-up can fix a failure or expand on real results instead of
        # regenerating blindly.
        if context.get('run_outcome'):
            parts.append(
                "The current configuration was last EXECUTED with this outcome:\n"
                f"{context['run_outcome']}\n"
                "Use this outcome to inform your edit: if it failed, fix the "
                "specific cause; if the user is expanding the analysis, build on "
                "these results and keep the already-fitted setup unchanged so the "
                "pipeline can reuse it. Do not undo earlier auto-repairs."
            )
        parts.append(
            "Apply the user's new instruction below as an INCREMENTAL EDIT to "
            "the current YAML and Python above. Preserve all earlier choices "
            "(target, energy range, event selection, requested products) that "
            "the new instruction does not explicitly change. Output the full "
            "updated YAML and Python."
        )

    parts.append("--- YOUR TASK ---")
    parts.append(f"Task: {user_prompt}")
    parts.append("\nGenerate the YAML configuration and Python script now:")

    return "\n".join(parts)

def extract_yaml(text):
    m = re.search(r'```yaml\s*\n(.*?)\n```', text, re.DOTALL)
    return m.group(1).strip() if m else ''

def extract_python(text):
    m = re.search(r'```python\s*\n(.*?)\n```', text, re.DOTALL)
    return m.group(1).strip() if m else ''

def _normalise_semantic_intent(value):
    """Return a conservative, well-shaped intent dict or ``None``.

    API Structured Outputs already enforce the schema.  This second boundary
    is needed for the Codex markdown path and protects the server from acting
    on invented operation names or malformed partial JSON.
    """
    if not isinstance(value, dict):
        return None
    target = value.get('target')
    products = value.get('products')
    changes = value.get('selection_changes')
    ambiguities = value.get('ambiguities')
    if not isinstance(target, dict) or not isinstance(products, list) or \
            not isinstance(changes, list) or not isinstance(ambiguities, list):
        return None
    if value.get('mode') not in ('create', 'edit'):
        return None
    if target.get('action') not in ('set', 'keep', 'remove', 'unspecified'):
        return None

    ra = target.get('ra_deg')
    dec = target.get('dec_deg')
    if isinstance(ra, bool) or not isinstance(ra, (int, float)) or \
            not 0.0 <= float(ra) < 360.0:
        ra = None
    else:
        ra = float(ra)
    if isinstance(dec, bool) or not isinstance(dec, (int, float)) or \
            not -90.0 <= float(dec) <= 90.0:
        dec = None
    else:
        dec = float(dec)
    # A half-position is unusable and usually indicates a malformed model
    # response. Treat the pair atomically so downstream code cannot center an
    # ROI on an accidental default coordinate.
    if (ra is None) != (dec is None):
        ra = dec = None
    spatial_model = str(target.get('spatial_model') or 'unspecified').lower()
    if spatial_model not in ('point', 'gaussian', 'disk', 'template',
                             'unspecified'):
        spatial_model = 'unspecified'

    clean_products = []
    for item in products:
        if not isinstance(item, dict):
            continue
        if item.get('name') not in _PRODUCT_NAMES or \
                item.get('action') not in ('add', 'remove', 'update', 'keep'):
            continue
        clean_products.append({
            'name': item['name'],
            'action': item['action'],
            'evidence': str(item.get('evidence') or ''),
        })

    clean_changes = []
    for item in changes:
        if not isinstance(item, dict):
            continue
        path = str(item.get('path') or '').strip()
        action = item.get('action')
        if not path or action not in ('set', 'remove', 'reset', 'keep'):
            continue
        clean_changes.append({
            'path': path,
            'action': action,
            'value_json': str(item.get('value_json') or ''),
            'evidence': str(item.get('evidence') or ''),
        })

    return {
        'mode': value['mode'],
        'summary': str(value.get('summary') or ''),
        'preserve_unspecified': bool(value.get('preserve_unspecified', True)),
        'target': {
            'action': target['action'],
            'query': str(target.get('query') or ''),
            'evidence': str(target.get('evidence') or ''),
            'ra_deg': ra,
            'dec_deg': dec,
            'position_evidence': str(target.get('position_evidence') or ''),
            'spatial_model': spatial_model,
        },
        'products': clean_products,
        'selection_changes': clean_changes,
        'ambiguities': [str(x) for x in ambiguities if str(x).strip()],
    }

def extract_semantic_intent(text):
    """Extract the optional intent block emitted by the Codex Luna path."""
    m = re.search(r'```json\s*\n(.*?)\n```', text or '', re.DOTALL)
    if not m:
        return None
    try:
        value = json.loads(m.group(1))
    except (TypeError, ValueError):
        return None
    return _normalise_semantic_intent(value)

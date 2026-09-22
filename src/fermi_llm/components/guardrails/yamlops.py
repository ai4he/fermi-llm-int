"""YAML editing primitives shared by the guardrails.

Every function here is pure: YAML text in, YAML text (plus a description of
what changed) out. That is what lets the guardrail chain be reordered,
extended or partially disabled without any of the pieces knowing about each
other.

A plugin that needs to adjust configurations should reuse these rather than
re-parsing YAML, so the platform keeps one definition of "insert a section",
"set a selection key" and "carry a dropped section over".
"""

from __future__ import annotations

import json
import math
import os
import re
import yaml

from ...core.runtime import RUNTIME
from ...fermipy.targets import (_SOURCE_ALIASES, bundled_idx_for_target,
                                coords_for_target, norm_name,
                                target_is_bundled, yaml_target)

_PRODUCT_SECTIONS = ('sed', 'lightcurve', 'tsmap', 'residmap', 'psmap')


_SEMANTIC_PRODUCT_LABELS = {
    'sed': 'SED (Spectral Energy Distribution)',
    'lightcurve': 'Light Curve',
    'tsmap': 'TS Map',
    'residmap': 'Residual Map',
    'psmap': 'PS Map',
    'extension': 'Extension Test',
    'localization': 'Source Localization',
}


# Maps analyze_prompt() plan labels to the YAML product section that gates
# their execution in the pipeline (see _generate_output_artifacts).
_PLAN_TO_YAML_SECTION = [
    ('SED', 'sed', {'make_plots': True, 'write_fits': True}),
    ('TS Map', 'tsmap', {'make_plots': True, 'write_fits': True}),
    ('Residual', 'residmap', {'make_plots': True, 'write_fits': True}),
    ('Light Curve', 'lightcurve', {'make_plots': True}),
    ('PS Map', 'psmap', {'make_plots': True}),
]


_NEGATION_WORDS = ('remove', 'without', 'drop', 'delete', 'exclude', 'no longer', "don't")


_LC_UNIT_SECONDS = {'day': 86400.0, 'week': 604800.0,
                    'month': 2629746.0, 'year': 31557600.0}


def _set_section_key(yaml_str, section, key, value):
    """Set <section>.<key> in YAML text, preserving formatting.

    Replaces the existing line, or inserts one right after ``<section>:``;
    appends the section block if none exists.
    """
    lines = yaml_str.splitlines()
    in_section = False
    sec_line_idx = None
    for i, line in enumerate(lines):
        if not line.strip() or line.lstrip().startswith('#'):
            continue
        if not line[0].isspace():
            in_section = line.split(':')[0].strip() == section
            if in_section:
                sec_line_idx = i
            continue
        if in_section and line.strip().startswith(f'{key}:'):
            indent = line[:len(line) - len(line.lstrip())]
            lines[i] = f"{indent}{key}: {value}"
            return '\n'.join(lines) + ('\n' if yaml_str.endswith('\n') else '')
    if sec_line_idx is not None:
        lines.insert(sec_line_idx + 1, f"  {key}: {value}")
        return '\n'.join(lines) + ('\n' if yaml_str.endswith('\n') else '')
    return yaml_str.rstrip() + f"\n{section}:\n  {key}: {value}\n"


def _set_selection_key(yaml_str, key, value):
    return _set_section_key(yaml_str, 'selection', key, value)


def _append_into_yaml_section(yaml_str, section, kv):
    """Insert key/value pairs right under an existing top-level block-style
    section header, preserving the original YAML text (no re-dump).  Returns
    (yaml_str, True) on success; an inline-style section (`lightcurve: {...}`)
    is left untouched and reported as False."""
    lines = yaml_str.splitlines()
    for i, line in enumerate(lines):
        if re.match(re.escape(section) + r':\s*(#.*)?$', line):
            ins = []
            for k, v in kv.items():
                if isinstance(v, float) and v.is_integer():
                    v = int(v)
                ins.append(f'  {k}: {v}')
            lines[i + 1:i + 1] = ins
            return ('\n'.join(lines)
                    + ('\n' if yaml_str.endswith('\n') else '')), True
    return yaml_str, False


def _semantic_product_actions(intent):
    """Return ``{product: action}`` from a validated semantic intent."""
    if not isinstance(intent, dict):
        return {}
    out = {}
    for item in intent.get('products') or []:
        if isinstance(item, dict) and item.get('name') in _SEMANTIC_PRODUCT_LABELS:
            out[item['name']] = item.get('action')
    return out


def _semantic_path_actions(intent):
    """Return normalized dotted-path operations from semantic intent."""
    if not isinstance(intent, dict):
        return {}
    out = {}
    for item in intent.get('selection_changes') or []:
        if not isinstance(item, dict):
            continue
        path = str(item.get('path') or '').strip().lower()
        if path:
            out[path] = item.get('action')
    return out


def _parse_explicit_icrs_position(text):
    """Parse decimal-degree RA/Dec from a user-supplied evidence phrase."""
    number = r'([+-]?(?:\d+(?:\.\d*)?|\.\d+))'
    match = re.search(
        r'\bra(?:j2000)?\s*[:=]?\s*' + number
        + r'\s*(?:deg(?:rees?)?|d)?\s*[,;/]?\s*'
          r'\bdec(?:j2000)?\s*[:=]?\s*' + number
        + r'\s*(?:deg(?:rees?)?|d)?\b',
        str(text or ''), re.IGNORECASE)
    if not match:
        return None
    try:
        ra, dec = float(match.group(1)), float(match.group(2))
    except (TypeError, ValueError):
        return None
    if not 0.0 <= ra < 360.0 or not -90.0 <= dec <= 90.0:
        return None
    return ra, dec


def _target_change_requested(new_target, user_message):
    """Did the user's message plausibly ask for ``new_target``?

    True when the message mentions the new target: exact/alias match against
    the bundled-source alias table, or any alphanumeric fragment (>= 3 chars)
    of the target designation appearing in the message.
    """
    msg = norm_name(user_message)
    if not msg:
        return False
    # Only revert clearly-unrequested swaps: any wording that suggests the
    # user is redirecting the analysis ("switch to...", "analyze X instead",
    # "change the target/source...") counts as a requested change, even when
    # we can't resolve the common name to the 4FGL designation.
    if re.search(r'\b(switch|change|swap|instead|rather|another|different|'
                 r'target|source|object|analyz|look at)\w*\b',
                 (user_message or '').lower()):
        return True
    tnorm = norm_name(new_target)
    if tnorm and tnorm in msg:
        return True
    # Alias table: if the new target is a known source and the message uses
    # any of its other names ("crab nebula" for 4FGL J0534.5+2200).
    for aliases in _SOURCE_ALIASES.values():
        anorm = [norm_name(a) for a in aliases]
        if any(tnorm and (a in tnorm or tnorm in a) for a in anorm):
            return any(a in msg for a in anorm)
    # Fallback: digit/letter fragments of the designation ("1553", "j1555.7").
    frags = [f for f in re.findall(r'[a-z]*\d[\d.+-]*', tnorm) if len(f) >= 3]
    return any(f in msg for f in frags)


def _extract_lightcurve_binning(user_message):
    """Pull an explicitly requested light-curve binning out of the prompt.

    Returns {'nbins': N} for "a lightcurve with 30 bins", {'binsz': seconds}
    for "a bin of 1 week" / "weekly bins", or None when the request leaves
    binning unspecified.  Nothing else in the generation path extracts
    numeric parameters from the prompt, so without this the executed config
    silently falls back to the validator's default bin count even though the
    (display-only) Python script shows the right call.
    """
    msg = (user_message or '').lower()
    if not msg:
        return None
    lc = r'(?:light[\s-]?curve|\blc\b)'
    nb = r'(\d{1,3})\s*(?:time\s*)?bins?\b'
    # Bin count stated in the same sentence as the light-curve request
    # ("a lightcurve with 30 bins"), so an SED/energy bin count elsewhere
    # in the prompt is not misread as LC binning.
    m = (re.search(lc + r'[^.!?]*?\b' + nb, msg)
         or re.search(r'\b' + nb + r'[^.!?]*?' + lc, msg))
    if m and int(m.group(1)) >= 2:
        return {'nbins': int(m.group(1))}
    # Time-width bins: "a bin of 1 week", "1 week time bins", "2-day bins".
    num = r'(\d+(?:\.\d+)?)'
    unit = r'(day|week|month|year)'
    m = (re.search(r'\b' + num + r'[\s-]*' + unit + r's?\b[^.!?]{0,30}?\bbin', msg)
         or re.search(r'\bbins?\b[^.!?]{0,30}?\b' + num + r'[\s-]*' + unit + r's?\b', msg))
    if m:
        return {'binsz': float(m.group(1)) * _LC_UNIT_SECONDS[m.group(2)]}
    # Adverb form: "daily/weekly/monthly lightcurve", "binned weekly".
    adv = r'\b(daily|weekly|monthly|yearly)\b'
    m = (re.search(adv + r'[^.!?]{0,40}?(?:\bbin|' + lc + r')', msg)
         or re.search(lc + r'[^.!?]{0,40}?' + adv, msg))
    if m:
        unit_word = {'daily': 'day', 'weekly': 'week',
                     'monthly': 'month', 'yearly': 'year'}[m.group(1)]
        return {'binsz': _LC_UNIT_SECONDS[unit_word]}
    # Bare "with 30 bins": accept only when its sentence isn't about an
    # SED/spectral product instead.
    for m in re.finditer(nb, msg):
        start = max(msg.rfind('.', 0, m.start()), msg.rfind('!', 0, m.start()),
                    msg.rfind('?', 0, m.start())) + 1
        stops = [i for i in (msg.find(c, m.end()) for c in '.!?') if i != -1]
        sentence = msg[start:min(stops) if stops else len(msg)]
        if int(m.group(1)) >= 2 and not re.search(r'\bsed\b|spectr|energy',
                                                  sentence):
            return {'nbins': int(m.group(1))}
    return None


def _analysis_from_semantic_intent(intent, fallback=None, user_message=''):
    """Translate Luna's semantic interpretation into the existing UI plan.

    Natural-language meaning comes from the model.  Catalog identity remains a
    deterministic lookup so an association name cannot silently become a
    hallucinated 4FGL designation.
    """
    analysis = {
        'target': None,
        'energy_range': None,
        'event_type': None,
        'analyses': [],
        'rationale': [],
        'decisions': [],
        'semantic_intent': intent,
    }
    if not isinstance(intent, dict):
        return fallback or analysis

    summary = str(intent.get('summary') or '').strip()
    if summary:
        analysis['rationale'].append(summary)

    target = intent.get('target') or {}
    if target.get('action') == 'set' and str(target.get('query') or '').strip():
        query = str(target['query']).strip()
        position_evidence = str(target.get('position_evidence') or '').strip()
        evidence_ok = bool(position_evidence) and \
            position_evidence.lower() in (user_message or '').lower()
        verified_position = _parse_explicit_icrs_position(
            position_evidence) if evidence_ok else None
        ra, dec = verified_position or (None, None)
        try:
            from ...fermipy import source_resolver
            hit = source_resolver.resolve(query)
        except Exception as e:
            hit = {'found': False}
            analysis['rationale'].append(
                f'Source resolution unavailable ({type(e).__name__}: {str(e)[:120]})')
        if hit.get('found'):
            analysis['target'] = hit['source_name']
            analysis['resolved_source'] = {
                'target': hit['source_name'],
                'name': hit.get('assoc1') or hit['source_name'],
                'ra': hit.get('ra'),
                'dec': hit.get('dec'),
                'class1': hit.get('class1'),
                'matched_text': query,
            }
            assoc = f" ({hit['assoc1']})" if hit.get('assoc1') else ''
            analysis['rationale'].append(
                f'Resolved "{query}" -> {hit["source_name"]}{assoc} from the 4FGL catalog')
            if hit.get('ambiguous'):
                analysis.setdefault('ambiguities', []).append(
                    'Multiple catalog sources matched: '
                    + ', '.join(hit.get('candidates', [])[:5]))
        elif ra is not None and dec is not None and evidence_ok and \
                target.get('spatial_model') in ('point', 'unspecified'):
            analysis['target'] = query
            analysis['custom_target'] = {
                'name': query,
                'ra': float(ra),
                'dec': float(dec),
                'spatial_model': 'point',
                'position_evidence': position_evidence,
            }
            analysis['resolved_source'] = {
                'target': query, 'name': query,
                'ra': float(ra), 'dec': float(dec), 'class1': '',
                'matched_text': query, 'catalogued': False,
            }
            analysis['rationale'].append(
                f'Using non-4FGL point source "{query}" at '
                f'RA={float(ra):.6f}, Dec={float(dec):.6f} (ICRS)')
        else:
            reason = (
                'explicit coordinates were not verified in the latest '
                'instruction' if ra is None or dec is None or not evidence_ok
                else 'an explicit extended-source morphology is not yet supported')
            analysis.setdefault('ambiguities', []).append(
                f'Could not resolve requested target "{query}" in the 4FGL '
                f'catalog and {reason}')

    actions = _semantic_product_actions(intent)
    analysis['analyses'] = [
        _SEMANTIC_PRODUCT_LABELS[name]
        for name, action in actions.items()
        if action in ('add', 'update')
    ]
    for ambiguity in intent.get('ambiguities') or []:
        text = str(ambiguity).strip()
        if text:
            analysis.setdefault('ambiguities', []).append(text)
    return analysis


def _append_analysis_summary(response_parts, analysis):
    """Render the plan after the authoritative interpretation is known."""
    if analysis.get('rationale'):
        response_parts.append("\n**Planning & Rationale:**")
        response_parts.extend(f"- {r}" for r in analysis['rationale'])
    if analysis.get('decisions'):
        response_parts.append("\n**Key Decisions:**")
        response_parts.extend(f"- {d}" for d in analysis['decisions'])
    if analysis.get('analyses'):
        response_parts.append(
            f"\n**Planned Analyses:** {', '.join(analysis['analyses'])}")
    if analysis.get('ambiguities'):
        response_parts.append("\n**Needs review:**")
        response_parts.extend(f"- {a}" for a in analysis['ambiguities'])


def _append_expected_outcome(response_parts, analysis):
    response_parts.append(
        "\n**Expected Outcome:** The configuration will set up a FermiPy GTAnalysis pipeline that:")
    response_parts.append("1. Selects events matching your criteria (energy, angle, time range)")
    response_parts.append("2. Builds a model of the region of interest from the 4FGL catalog")
    response_parts.append("3. Performs likelihood fitting of the model to the data")
    for i, item in enumerate(analysis.get('analyses') or [], 4):
        response_parts.append(f"{i}. Computes {item}")


def _build_run_outcome_summary(session):
    """Compact summary of this task's LAST pipeline run, fed back to the model
    on the next message so a follow-up can *fix* or *expand* the analysis using
    what actually happened — fit results, errors, auto-repairs, products —
    instead of regenerating from scratch. Returns '' when there is no prior run.

    This is the continuity signal for the stateless API/local backends (which
    have no server-side memory of the run), and it also benefits the Codex
    backend, whose conversation never saw the pipeline results (the pipeline
    runs outside Codex).
    """
    res = getattr(session, 'pipeline_result', None)
    if not isinstance(res, dict) or not res:
        return ''

    lines = []
    status = res.get('status') or ('complete' if res.get('success') else 'unknown')
    run_mode = res.get('run_mode')
    lines.append(f"Status: {status}"
                 + (f" (run_mode: {run_mode})" if run_mode else ""))

    # Failure details first — this is what a "fix it" follow-up needs most.
    if not res.get('success'):
        if res.get('error'):
            lines.append(f"Error: {str(res['error'])[:400]}")
        tb = res.get('traceback')
        if tb:
            # last, most-relevant line of the traceback
            tail = [l for l in str(tb).splitlines() if l.strip()]
            if tail:
                lines.append(f"Traceback (last line): {tail[-1][:200]}")

    # Fit results (present on success/partial) — useful for "expand" follow-ups.
    l4 = res.get('level4') or {}
    if isinstance(l4, dict) and l4:
        def _fmt(v):
            if isinstance(v, float):
                return f"{v:.4g}"
            return v
        fit_bits = []
        for key, label in (('fit_ok', 'fit_ok'), ('fit_quality', 'fit_quality'),
                           ('target_ts', 'TS'), ('target_flux', 'flux'),
                           ('spectral_index', 'index'),
                           ('convergence_ok', 'converged')):
            v = l4.get(key)
            if v is not None:
                fit_bits.append(f"{label}={_fmt(v)}")
        if fit_bits:
            lines.append("Fit: " + ", ".join(str(b) for b in fit_bits))
        checks = [f"{c}={l4.get(c)}" for c in ('ts_check', 'flux_check')
                  if l4.get(c) is not None]
        if checks:
            lines.append("Validation checks: " + ", ".join(checks))
        arts = [a.get('name') if isinstance(a, dict) else str(a)
                for a in (l4.get('artifacts') or [])]
        arts = [a for a in arts if a]
        if arts:
            lines.append("Products computed: " + ", ".join(arts[:10]))

    # Auto-repairs the validator applied — so the model doesn't re-introduce a
    # mistake that was already corrected, and can build on the repaired config.
    rh = res.get('repair_history') or []
    if rh:
        n = len(rh)
        last = rh[-1] if isinstance(rh[-1], dict) else {}
        repairs = last.get('repairs') or last.get('reason') or last.get('error')
        detail = f" (last: {str(repairs)[:200]})" if repairs else ""
        lines.append(f"Auto-repairs applied over {n} iteration(s){detail}")

    if res.get('target_unmatched'):
        lines.append("WARNING: the target was NOT matched in the 4FGL catalog "
                     "on the last run — verify the exact 4FGL source name.")

    return "\n".join(lines).strip()


def _merge_edit_yaml(new_yaml, prev_yaml, user_message, semantic_intent=None):
    """Merge a conversational-edit generation with the previous YAML.

    Models asked to "edit, preserving everything else" routinely drop
    sections (target, sed, even whole core blocks) they were told to keep.
    Treat the new generation as authoritative only for what it CONTAINS:
    top-level sections and selection sub-keys present in the previous YAML
    but missing from the new one are carried over — except product sections
    the user's message asks to remove.

    Returns (yaml_str, carried_over_names).
    """
    try:
        new_cfg = yaml.safe_load(new_yaml) or {}
        prev_cfg = yaml.safe_load(prev_yaml) or {}
    except Exception:
        return new_yaml, []
    if not isinstance(new_cfg, dict) or not isinstance(prev_cfg, dict):
        return new_yaml, []

    msg = (user_message or '').lower()
    negation = any(w in msg for w in _NEGATION_WORDS)
    product_actions = _semantic_product_actions(semantic_intent)
    path_actions = _semantic_path_actions(semantic_intent)
    target_action = ((semantic_intent or {}).get('target') or {}).get('action')
    carried = []

    for key, val in prev_cfg.items():
        if key in new_cfg:
            continue
        if semantic_intent:
            if product_actions.get(key) == 'remove' or \
                    path_actions.get(key) in ('remove', 'reset'):
                continue
        elif key in _PRODUCT_SECTIONS and negation and key in msg:
            continue  # legacy explicit removal request
        new_yaml = new_yaml.rstrip() + '\n\n' + yaml.dump(
            {key: val}, default_flow_style=False, sort_keys=False)
        carried.append(key)

    new_sel = new_cfg.get('selection')
    prev_sel = prev_cfg.get('selection')
    if isinstance(new_sel, dict) and isinstance(prev_sel, dict):
        for k, v in prev_sel.items():
            if k not in new_sel:
                action = path_actions.get(f'selection.{str(k).lower()}')
                if semantic_intent and action in ('remove', 'reset'):
                    continue
                if semantic_intent and k == 'target' and \
                        target_action in ('set', 'remove'):
                    continue
                new_yaml = _set_selection_key(new_yaml, k, v)
                carried.append(f'selection.{k}')
    return new_yaml, carried


def _enforce_requested_target(yaml_str, requested_target):
    """Force selection.target (and known coordinates) to the source the user
    named in their message.

    The LoRA sometimes omits the target or substitutes a favourite source
    from its training data; the user's own words are authoritative. Returns
    (yaml_str, previous_target_or_None, changed: bool).
    """
    if not requested_target or not yaml_str.strip():
        return yaml_str, None, False
    try:
        cfg = yaml.safe_load(yaml_str) or {}
    except Exception:
        return yaml_str, None, False
    if not isinstance(cfg, dict):
        return yaml_str, None, False
    current = (cfg.get('selection') or {}).get('target') or ''
    if current and norm_name(current) == norm_name(requested_target):
        return yaml_str, None, False
    # Same physical source under another alias? Then leave it alone.
    if current:
        cn, rn = norm_name(current), norm_name(requested_target)
        for aliases in _SOURCE_ALIASES.values():
            anorm = [norm_name(a) for a in aliases]
            if any(a in cn or cn in a for a in anorm) and \
               any(a in rn or rn in a for a in anorm):
                return yaml_str, None, False
    yaml_str = _set_selection_key(yaml_str, 'target', requested_target)
    coords = coords_for_target(requested_target)
    if coords:
        yaml_str = _set_selection_key(yaml_str, 'ra', coords[0])
        yaml_str = _set_selection_key(yaml_str, 'dec', coords[1])
    return yaml_str, (current or None), True


def _ensure_custom_target(yaml_str, custom_target):
    """Inject a validated non-catalog point source into a FermiPy config.

    The model may suggest spectral parameters, but identity and position come
    from server-validated semantic intent. A marker lets conversational edits
    replace the previous custom target instead of leaving a degenerate source.
    """
    if not yaml_str.strip() or not isinstance(custom_target, dict):
        return yaml_str
    try:
        cfg = yaml.safe_load(yaml_str) or {}
    except Exception:
        return yaml_str
    if not isinstance(cfg, dict):
        return yaml_str
    name = str(custom_target.get('name') or 'CustomTarget').strip()[:120]
    ra, dec = custom_target.get('ra'), custom_target.get('dec')
    if not name or not isinstance(ra, (int, float)) or \
            not isinstance(dec, (int, float)):
        return yaml_str

    selection = cfg.setdefault('selection', {})
    selection['target'] = name
    selection['ra'], selection['dec'] = float(ra), float(dec)
    model = cfg.setdefault('model', {})
    sources = model.get('sources')
    if not isinstance(sources, list):
        sources = []

    existing, kept = {}, []
    target_key = norm_name(name)
    for source in sources:
        if not isinstance(source, dict):
            continue
        if norm_name(source.get('name')) == target_key:
            existing.update(source)
            continue
        if source.get('fermillm_custom_target'):
            continue
        kept.append(source)
    existing.update({
        'name': name, 'ra': float(ra), 'dec': float(dec),
        'SpatialModel': 'PointSource',
        'fermillm_custom_target': True,
    })
    existing.setdefault('SpectrumType', 'PowerLaw')
    existing.setdefault('Index', 2.0)
    existing.setdefault('Scale', 1000.0)
    existing.setdefault('Prefactor', 1e-13)
    model['sources'] = kept + [existing]
    return yaml.dump(cfg, default_flow_style=False, sort_keys=False)


def _preserve_target_on_edit(new_yaml, prev_yaml, user_message,
                             semantic_intent=None):
    """Revert an unrequested target change made during a conversational edit.

    Follow-up generations sometimes swap selection.target (and ra/dec) to an
    unrelated source even though the user's message never mentioned one.
    When that happens, restore the previous target and coordinates in the
    YAML text (formatting-preserving line replacement).

    Returns (yaml_str, reverted_from_or_None, prev_target).
    """
    try:
        new_cfg = yaml.safe_load(new_yaml) or {}
        prev_cfg = yaml.safe_load(prev_yaml) or {}
    except Exception:
        return new_yaml, None, None
    new_sel = new_cfg.get('selection') or {}
    prev_sel = prev_cfg.get('selection') or {}
    new_target = new_sel.get('target') or ''
    prev_target = prev_sel.get('target') or ''
    if not new_target or not prev_target:
        return new_yaml, None, prev_target
    if norm_name(new_target) == norm_name(prev_target):
        return new_yaml, None, prev_target
    if semantic_intent:
        target_action = ((semantic_intent.get('target') or {}).get('action'))
        if target_action in ('set', 'remove'):
            return new_yaml, None, prev_target
    elif _target_change_requested(new_target, user_message):
        return new_yaml, None, prev_target

    # Unrequested change: restore target and, when present in both, ra/dec.
    lines = new_yaml.splitlines()
    out = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith('target:'):
            indent = line[:len(line) - len(line.lstrip())]
            out.append(f"{indent}target: {prev_target}")
        elif stripped.startswith('ra:') and prev_sel.get('ra') is not None:
            indent = line[:len(line) - len(line.lstrip())]
            out.append(f"{indent}ra: {prev_sel['ra']}")
        elif stripped.startswith('dec:') and prev_sel.get('dec') is not None:
            indent = line[:len(line) - len(line.lstrip())]
            out.append(f"{indent}dec: {prev_sel['dec']}")
        else:
            out.append(line)
    return '\n'.join(out) + ('\n' if new_yaml.endswith('\n') else ''), new_target, prev_target


def _reconcile_planned_products(yaml_str, analyses, user_message='',
                                semantic_intent=None):
    """Ensure every planned analysis has its gating YAML section.

    LLM-generated configs sometimes announce a product in the plan (e.g.
    "Planned Analyses: SED") but omit the corresponding top-level YAML section;
    since product execution is gated on those sections, the product silently
    never runs. Append a default section for each planned-but-missing product,
    preserving the original YAML text (sections are appended, not re-dumped).

    Returns (yaml_str, injected_section_names, notes); `notes` are extra
    user-visible consistency messages (e.g. explicit light-curve binning
    from the prompt applied to the config).
    """
    if not yaml_str.strip() or not analyses:
        return yaml_str, [], []
    msg = (user_message or '').lower()
    if not semantic_intent and any(w in msg for w in _NEGATION_WORDS):
        # The user is asking to take something away; don't second-guess the
        # generated config by re-adding sections. Legacy fallback only: Luna's
        # semantic product actions handle mixed add/remove requests precisely.
        return yaml_str, [], []
    try:
        cfg = yaml.safe_load(yaml_str)
    except Exception:
        return yaml_str, [], []
    if not isinstance(cfg, dict):
        return yaml_str, [], []

    injected = []
    additions = []
    notes = []
    lc_binning = _extract_lightcurve_binning(user_message)
    product_actions = _semantic_product_actions(semantic_intent)

    def _fmt_binning(b):
        return ', '.join(
            '`%s: %s`' % (k, int(v) if isinstance(v, float) and v.is_integer()
                          else v)
            for k, v in b.items())

    for label, section, default_spec in _PLAN_TO_YAML_SECTION:
        requested = any(label in a for a in analyses)
        if semantic_intent:
            requested = product_actions.get(section) in ('add', 'update')
        if requested and section not in cfg:
            spec = dict(default_spec)
            if section == 'lightcurve' and lc_binning:
                spec.update(lc_binning)
                notes.append(
                    f"the light-curve binning you asked for "
                    f"({_fmt_binning(lc_binning)}) was written into the "
                    f"added `lightcurve` section — without it the run "
                    f"would fall back to the validator's 6-bin default.")
            additions.append(yaml.dump({section: spec},
                                       default_flow_style=False, sort_keys=False))
            injected.append(section)
    if additions:
        yaml_str = yaml_str.rstrip() + '\n\n' + '\n'.join(additions)
    # The user's explicit binning also wins when the model DID emit a
    # lightcurve section but dropped nbins/binsz from it.
    if (lc_binning and isinstance(cfg.get('lightcurve'), dict)
            and not ({'nbins', 'binsz'} & set(cfg['lightcurve']))):
        yaml_str, ok = _append_into_yaml_section(yaml_str, 'lightcurve',
                                                 lc_binning)
        if ok:
            notes.append(
                f"set {_fmt_binning(lc_binning)} in the `lightcurve` "
                f"section from your request — the generated config left "
                f"the light-curve binning unspecified, which would have "
                f"fallen back to the 6-bin default.")
    return yaml_str, injected, notes


def _apply_full_data_defaults(yaml_str, response_parts, session_dir,
                              user_message=''):
    """Make the actual data inputs and defaults explicit in the config.

    Generated YAML often carries invented data paths mimicking the
    training examples (``./data/..._file_list.txt``); the pipeline always
    overrides them at run time, but showing fiction to the user invites
    confusion. Rewrite data.evfile/scfile to the real paths the run will
    use, and for arbitrary (non-bundled) sources with no usable time
    range, write the default (most recent year of available data) into
    selection.tmin/tmax so the user sees -- and can edit -- exactly what
    will run.

    When the user asked for *all* the data, the model-emitted tmin/tmax
    are overridden with the full on-disk coverage: the LLM computes "now"
    from its own knowledge cutoff, so its tmax lands years in the past and
    silently drops the newest data. Explicit ranges are clamped to the
    coverage actually on disk.
    """
    target = yaml_target(yaml_str)
    if not target:
        return yaml_str

    bundled_idx = bundled_idx_for_target(target)
    if bundled_idx is not None:
        # Bundled demo source: show the real bundled files and clamp the
        # time range to what they actually contain (the reference example
        # often carries a multi-year range that the run-time repair would
        # clamp anyway -- showing it here just inflates the estimate).
        try:
            from run_execution_validated import (TEST_DATA_MAP,
                                                 get_fits_time_range)
            meta = TEST_DATA_MAP.get(bundled_idx, {})
            if meta.get('evfile') and os.path.exists(meta['evfile']):
                yaml_str = _set_section_key(yaml_str, 'data', 'evfile',
                                            meta['evfile'])
                yaml_str = _set_section_key(yaml_str, 'data', 'scfile',
                                            meta['scfile'])
                info = get_fits_time_range(meta['evfile'])
                tstart, tstop = info.get('tstart'), info.get('tstop')
                cfg = yaml.safe_load(yaml_str) or {}
                sel = (cfg.get('selection') or {})
                tmin, tmax = sel.get('tmin'), sel.get('tmax')
                inside = (isinstance(tmin, (int, float))
                          and isinstance(tmax, (int, float))
                          and tstart <= tmin < tmax <= tstop)
                if tstart and tstop and not inside:
                    yaml_str = _set_selection_key(yaml_str, 'tmin', tstart)
                    yaml_str = _set_selection_key(yaml_str, 'tmax', tstop)
                    response_parts.append(
                        f"\n**Time range:** clamped to the bundled demo "
                        f"data for {meta.get('name', target)} "
                        f"(MET {tstart:.0f}-{tstop:.0f}, one week).")
        except Exception:
            pass
        return yaml_str

    try:
        cfg = yaml.safe_load(yaml_str) or {}
    except Exception:
        return yaml_str
    sel = (cfg.get('selection') or {})

    try:
        from ...fermipy import data_registry
        d_tmin, d_tmax = data_registry.default_time_range()
        d_start, d_end = data_registry.data_time_range()
        default_days = int(data_registry.DEFAULT_TIME_RANGE_DAYS)
        scfile = data_registry.spacecraft_file()
    except Exception as e:
        response_parts.append(
            f"\n**Data coverage warning:** could not read the weekly data "
            f"registry ({type(e).__name__}: {str(e)[:150]}); data paths and "
            f"the time range will be resolved when the pipeline runs.")
        return yaml_str

    tmin, tmax = sel.get('tmin'), sel.get('tmax')
    have_range = (isinstance(tmin, (int, float))
                  and isinstance(tmax, (int, float)) and tmin < tmax)
    all_data = re.search(
        r'\b(?:all (?:the |of the |available )?data\b(?!\s*qual)|'
        r'(?:entire|full|whole|complete)\s+(?:data\s*set|dataset|mission|'
        r'archive)\b)', (user_message or '').lower())
    if all_data:
        # "all available data": force the full on-disk coverage. The
        # model-emitted tmax reflects its stale notion of "now" (knowledge
        # cutoff) and silently drops the newest years of data.
        off = (not have_range or abs(tmin - d_start) > 604800.0
               or abs(tmax - d_end) > 604800.0)
        if off:
            note = (f"; the generated config had MET {tmin:.0f}-{tmax:.0f}, "
                    f"which would have missed the most recent "
                    f"{max(0.0, (d_end - tmax)) / 86400.0:.0f} days"
                    if have_range else "")
            yaml_str = _set_selection_key(yaml_str, 'tmin', round(d_start, 1))
            yaml_str = _set_selection_key(yaml_str, 'tmax', round(d_end, 1))
            response_parts.append(
                f"\n**Time range:** you asked for all available data -- "
                f"using the full on-disk coverage MET {d_start:.0f}-{d_end:.0f}"
                f" ({(d_end - d_start) / 86400.0 / 365.25:.1f} years{note}).")
            tmin, tmax = d_start, d_end
            have_range = True
    elif have_range:
        # Clamp an explicit range to the coverage actually on disk.
        new_tmin = min(max(tmin, d_start), d_end)
        new_tmax = max(min(tmax, d_end), d_start)
        if (new_tmin, new_tmax) != (tmin, tmax) and new_tmin < new_tmax:
            yaml_str = _set_selection_key(yaml_str, 'tmin', round(new_tmin, 1))
            yaml_str = _set_selection_key(yaml_str, 'tmax', round(new_tmax, 1))
            response_parts.append(
                f"\n**Time range:** the requested MET {tmin:.0f}-{tmax:.0f} "
                f"extends beyond the data on disk -- clamped to the "
                f"available coverage (MET {new_tmin:.0f}-{new_tmax:.0f}).")
            tmin, tmax = new_tmin, new_tmax
    if not have_range:
        tmin, tmax = d_tmin, d_tmax
        yaml_str = _set_selection_key(yaml_str, 'tmin', round(tmin, 1))
        yaml_str = _set_selection_key(yaml_str, 'tmax', round(tmax, 1))
        response_parts.append(
            f"\n**Time range:** none specified -- defaulting to the most "
            f"recent {default_days} days of available data "
            f"(MET {tmin:.0f}-{tmax:.0f}). Edit `selection.tmin`/`tmax` "
            f"to analyze a different interval; longer ranges take "
            f"proportionally longer to run.")

    # Real data inputs: the run generates an event-file list of the weekly
    # all-sky photon files overlapping [tmin, tmax] at this exact path.
    evfile_list = os.path.join(session_dir, 'evfile_list.txt')
    yaml_str = _set_section_key(yaml_str, 'data', 'evfile', evfile_list)
    yaml_str = _set_section_key(yaml_str, 'data', 'scfile', scfile)
    try:
        from ...fermipy import data_registry
        n_weeks = len(data_registry.select_weekly_files(tmin, tmax))
        response_parts.append(
            f"\n**Data files:** this run selects from the full weekly "
            f"dataset -- {n_weeks} all-sky photon files covering the chosen "
            f"time range (listed in `data.evfile` at run time) plus the "
            f"mission-merged spacecraft file.")
    except Exception:
        pass
    return yaml_str

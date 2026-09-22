"""One chat message, from prompt to reviewed configuration.

The flow, unchanged from the monolith but now assembled from components:

1. the intent analyzer turns the message into a plan;
2. every knowledge source is searched for relevant documentation;
3. the selected model generates YAML + Python (in a worker thread, so a slow
   local model cannot freeze the event loop);
4. a model that returns a semantic intent overrides the rule-based plan;
5. an empty reply falls back to template generation;
6. the guardrail stack runs in priority order;
7. the reply, the files and the estimate are saved on the session.

Steps 1, 2, 3 and 6 are all plugin boundaries — this module only sequences
them.
"""

from __future__ import annotations

import asyncio
import os
from datetime import datetime
from typing import Any, Dict, Optional

from ..core import kinds
from ..core.contracts import ChatTurn
from ..core.progress import write_chat_progress
from ..core.runtime import RUNTIME
from ..components.guardrails.yamlops import (_append_analysis_summary,
                                             _append_expected_outcome,
                                             _build_run_outcome_summary)
from ..components.models.prompting import SEMANTIC_INTENT_MODELS
from ..fermipy.estimates import estimate_run_duration
from ..fermipy.targets import target_is_bundled, yaml_target


def guardrail_stack():
    """Active guardrails, in the order they will run."""
    return RUNTIME.ctx.build_stack(kinds.GUARDRAIL)


def run_guardrails(turn: ChatTurn) -> ChatTurn:
    """Apply every guardrail that claims the turn.

    A guardrail that raises is skipped and reported: a broken consistency
    rule must never cost the scientist the generated configuration.
    """
    for guardrail in guardrail_stack():
        try:
            if not guardrail.applies(turn):
                continue
            before = (turn.yaml, turn.python)
            turn = guardrail.apply(turn)
            RUNTIME.hooks.emit('guardrail.applied', session=turn.session,
                               name=guardrail.name,
                               changed=(turn.yaml, turn.python) != before)
        except Exception as exc:                          # noqa: BLE001
            turn.note(getattr(guardrail, 'name', 'guardrail'),
                      f'skipped after error: {type(exc).__name__}: {exc}')
    return turn


async def handle_message(session, message: str, selected_model: str) -> Dict[str, Any]:
    """Run one chat turn and return the API payload."""
    ctx = RUNTIME.ctx
    models = RUNTIME.models

    # Clear previous chat progress so the stream shows only this turn.
    progress_file = os.path.join(session.session_dir, 'chat_progress.jsonl')
    if os.path.exists(progress_file):
        os.remove(progress_file)

    session.chat_history.append({
        'role': 'user', 'content': message,
        'timestamp': datetime.now().isoformat(), 'model': selected_model,
    })
    if not getattr(session, 'title', ''):
        session.title = (message or '').strip()[:100]

    RUNTIME.hooks.emit('chat.request', session=session, message=message,
                       model_id=selected_model)

    # ---- 1. plan ----
    write_chat_progress(session.session_dir, 'analyzing',
                        'Analyzing prompt — identifying target, analyses, '
                        'and parameters...')
    rule_analysis = RUNTIME.agent.analyze(message)
    analysis = rule_analysis
    semantic_mode = selected_model in SEMANTIC_INTENT_MODELS
    write_chat_progress(session.session_dir, 'analyzed',
                        f'Prompt analysis complete — '
                        f'{len(analysis.get("analyses", []))} analyses planned')

    response_parts = []
    if not semantic_mode:
        _append_analysis_summary(response_parts, analysis)

    # ---- 2. documentation ----
    write_chat_progress(session.session_dir, 'rag',
                        'Searching FermiPy documentation (RAG)...')
    hits = []
    for source in RUNTIME.knowledge:
        try:
            hits.extend(source.search(message))
        except Exception:                                 # noqa: BLE001
            continue
    if hits:
        write_chat_progress(session.session_dir, 'rag_done',
                            f'Found {len(hits)} relevant documentation sections')
        response_parts.append(
            f"\n**Reference Documentation:** Consulted {len(hits)} "
            f"documentation sections: {', '.join(h['title'] for h in hits)}")
    else:
        write_chat_progress(session.session_dir, 'rag_done',
                            'No additional documentation matched')

    if not semantic_mode:
        _append_expected_outcome(response_parts, analysis)

    # ---- 3. generation ----
    gen_meta: Optional[dict] = None
    yaml_str = python_str = ''
    edit_context = None
    if (getattr(session, 'current_yaml', '') or '').strip() or \
       (getattr(session, 'current_python', '') or '').strip():
        edit_context = {
            'prompt': getattr(session, 'current_prompt', '') or '',
            'yaml': session.current_yaml or '',
            'python': session.current_python or '',
            # Continuity: what the last run actually did, so a follow-up can
            # fix or extend it instead of starting over.
            'run_outcome': _build_run_outcome_summary(session),
        }

    if selected_model != 'template':
        response_parts.append(
            f"\n**Model:** Using {selected_model} for code generation...")
        write_chat_progress(session.session_dir, 'model_start',
                            f'Starting code generation with {selected_model}...')

        def _progress(stage, detail):
            write_chat_progress(session.session_dir, stage, detail)

        try:
            yaml_str, python_str, gen_meta = await asyncio.to_thread(
                models.generate, selected_model, message, temperature=0.1,
                n_shots=3, progress_cb=_progress, context=edit_context,
                codex_thread_id=getattr(session, 'codex_thread_id', None),
            )
            if gen_meta:
                if gen_meta.get('codex_thread_id'):
                    session.codex_thread_id = gen_meta['codex_thread_id']
                response_parts.append(
                    f"**Generation:** {gen_meta.get('gen_time', '?')}s, "
                    f"YAML={'found' if gen_meta.get('has_yaml') else 'missing'}, "
                    f"Python={'found' if gen_meta.get('has_python') else 'missing'}")
        except Exception as exc:                          # noqa: BLE001
            error_msg = str(exc)[:200]
            write_chat_progress(session.session_dir, 'model_error',
                                f'Model error: {error_msg}')
            response_parts.append(f"\n**Model Error:** {error_msg}")
            response_parts.append("Falling back to template generation...")
            yaml_str = python_str = ''

    # ---- 4. semantic intent overrides the rule-based plan ----
    semantic_intent = (gen_meta or {}).get('semantic_intent')
    if semantic_mode:
        if semantic_intent:
            from ..components.guardrails.yamlops import _analysis_from_semantic_intent
            analysis = _analysis_from_semantic_intent(
                semantic_intent, fallback=rule_analysis, user_message=message)
            write_chat_progress(
                session.session_dir, 'semantic_intent',
                'Luna semantic intent accepted as the authoritative request '
                'interpretation')
        else:
            analysis = rule_analysis
            analysis.setdefault('decisions', []).append(
                'Luna did not return a valid semantic-intent envelope; '
                'using the legacy rule-based interpretation as a safe fallback')
        _append_analysis_summary(response_parts, analysis)
        _append_expected_outcome(response_parts, analysis)
    if gen_meta:
        analysis['gen_meta'] = gen_meta

    # ---- 5. template fallback ----
    if not yaml_str.strip():
        write_chat_progress(session.session_dir, 'template',
                            'Generating configuration using template (no LLM)...')
        response_parts.append(
            "\nGenerating YAML configuration and Python script (template)...")
        yaml_str, python_str = RUNTIME.agent.generate_config(message, analysis)
        write_chat_progress(session.session_dir, 'template_done',
                            'Template generation complete')

    # ---- 6. guardrails ----
    turn = ChatTurn(session=session, user_message=message, yaml=yaml_str,
                    python=python_str, analysis=analysis,
                    semantic_intent=semantic_intent,
                    response_parts=response_parts,
                    metadata={'edit_context': edit_context,
                              'model_id': selected_model})
    RUNTIME.hooks.emit('chat.generated', session=session, turn=turn)
    turn = run_guardrails(turn)
    yaml_str, python_str = turn.yaml, turn.python
    response_parts = turn.response_parts

    # ---- 7. persist and estimate ----
    session.current_yaml = yaml_str
    session.current_python = python_str
    session.current_prompt = message

    target = yaml_target(yaml_str)
    full_data = bool(target) and not target_is_bundled(target)
    duration_est = estimate_run_duration(yaml_str, full_data=full_data)
    if duration_est:
        response_parts.append(
            f"\n**Estimated run time:** {duration_est['text']}")
        threshold = float(ctx.settings.confirm_threshold_min)
        if full_data and duration_est['total_hi_min'] > threshold:
            response_parts.append(
                f"\n**Note:** this exceeds {threshold:.0f} minutes, so "
                f"Run Pipeline will ask you to confirm the estimated "
                f"duration before starting.")

    response_parts.append(
        "\n**Status:** Configuration generated. You can review and edit the "
        "YAML and Python panels, then click **Run Pipeline** to execute.")
    write_chat_progress(session.session_dir, 'complete',
                        'Configuration generated successfully')

    assistant_msg = "\n".join(response_parts)
    session.chat_history.append({
        'role': 'assistant', 'content': assistant_msg,
        'timestamp': datetime.now().isoformat(), 'analysis': analysis,
    })
    session.save()
    RUNTIME.hooks.emit('chat.completed', session=session, turn=turn)

    return {
        'response': assistant_msg,
        'yaml': yaml_str,
        'python': python_str,
        'analysis': analysis,
        'gen_meta': gen_meta,
        'duration_estimate': duration_est,
        'guardrails': turn.notes,
    }

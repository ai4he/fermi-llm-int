"""JSON helpers shared by the API layer."""

from __future__ import annotations

def _sanitize_for_json(obj):
    """Recursively replace NaN/Inf with None for JSON serialization."""
    import math
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_for_json(v) for v in obj]
    return obj


sanitize_for_json = _sanitize_for_json

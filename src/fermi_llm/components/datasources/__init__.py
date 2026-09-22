"""Where the photons come from.

Data sources are tried in priority order; the first one that claims the
target resolves it. That is the seam for an institute with its own archive,
a different catalog, or simulated data.
"""

from . import bundled, weekly_archive  # noqa: F401


def resolve_data(ctx, target, yaml_text, session_dir):
    """Ask each active data source, in order, to resolve ``target``."""
    from ...core import kinds
    spec = {'yaml': yaml_text, 'session_dir': session_dir}
    for source in ctx.build_stack(kinds.DATA_SOURCE):
        if source.handles(target, spec):
            resolved = source.resolve(target, spec)
            resolved.setdefault('source', source.name)
            return resolved
    raise RuntimeError(f'no data source could resolve target {target!r}')

"""The run pipeline: review, then execute exactly what was reviewed.

Two modules, one per half:

``review``
    Repairs and validates the configuration, reviews the model's script and
    stages the exact files for approval. Nothing runs here.
``execute``
    What a worker process runs once the user approves: validation of the
    approved bytes, execution through the active backend, then products and
    science checks.

Both are registered as pipeline stages so the platform view can show the
order, and so a plugin can insert a stage (a pre-flight cost estimate, an
institute's approval step) between them.
"""

from ...core import kinds
from ...core.registry import REGISTRY
from . import execute, review  # noqa: F401


class ReviewStage:
    name = 'review'
    label = 'Repair, validate and stage files for approval'

    def __init__(self, ctx):
        self.ctx = ctx

    def run(self, ctx):
        return review.prepare_run_review(ctx['session'])


class ExecuteStage:
    name = 'execute'
    label = 'Run the approved analysis in a worker process'

    def __init__(self, ctx):
        self.ctx = ctx

    def run(self, ctx):
        return execute.run_pipeline_isolated(**ctx['job'])


REGISTRY.register(kinds.PIPELINE_STAGE, 'review', ReviewStage,
                  priority=100, source='core',
                  metadata={'label': ReviewStage.label})
REGISTRY.register(kinds.PIPELINE_STAGE, 'execute', ExecuteStage,
                  priority=200, source='core',
                  metadata={'label': ExecuteStage.label})

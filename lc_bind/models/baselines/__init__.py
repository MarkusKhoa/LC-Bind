"""Published-baseline reproduction pipeline."""

def run(paths, **overrides):
    from lc_bind.models.baselines.pipeline import run as _run

    return _run(paths, **overrides)

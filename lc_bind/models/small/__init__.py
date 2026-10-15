"""Small-molecule binding pipeline."""

def run(paths, **overrides):
    from lc_bind.models.small.pipeline import run as _run

    return _run(paths, **overrides)

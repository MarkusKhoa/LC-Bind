"""Small stand-ins for Colab display helpers."""

from __future__ import annotations


def display(obj=None, *args, **kwargs):
    """Print a dataframe or any other object. Extra Colab arguments are ignored."""
    if obj is None:
        return
    try:
        import pandas as pd

        if isinstance(obj, pd.DataFrame):
            with pd.option_context(
                "display.max_columns",
                40,
                "display.width",
                160,
                "display.max_rows",
                40,
            ):
                print(obj.to_string())
            return
        if isinstance(obj, pd.Series):
            print(obj.to_string())
            return
    except Exception:
        pass
    print(obj)

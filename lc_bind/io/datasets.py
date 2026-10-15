"""Load the canonical project tables. File names match the notebooks."""

from __future__ import annotations

import ast
from pathlib import Path

import pandas as pd

from lc_bind.paths import TEST300_CSV, TEST46_CSV, TRAIN_CSV, ProjectPaths

LABELS = [
    "metal_binding",
    "nuclear_binding",
    "small_binding",
    "any_ligand_binding_sites",
]


def parse_list(value):
    if isinstance(value, str):
        return list(ast.literal_eval(value))
    return list(value)


def load_binding_table(path, labels=None, check_label=None):
    """Load one grouped binding-site CSV.

    ``check_label`` validates that one label column matches the sequence
    length. Pass ``None`` to check every label column that is present.
    """
    path = Path(path)
    df = pd.read_csv(path)
    df["prot_id"] = df["prot_id"].astype(str)
    if "sequence" in df.columns:
        df["sequence"] = df["sequence"].astype(str)
    use_labels = list(labels) if labels is not None else list(LABELS)
    for column in use_labels:
        if column in df.columns:
            df[column] = df[column].apply(parse_list)
    if not df.prot_id.is_unique:
        raise RuntimeError(f"{path}: duplicate prot_id")
    if "sequence" in df.columns:
        columns = [check_label] if check_label else [
            column for column in use_labels if column in df.columns
        ]
        for row in df.itertuples():
            for column in columns:
                labels_row = getattr(row, column)
                if len(labels_row) != len(row.sequence):
                    raise RuntimeError(
                        f"{path}/{row.prot_id}/{column}: "
                        f"{len(labels_row)} != {len(row.sequence)}"
                    )
    return df


def load_canonical_splits(paths: ProjectPaths, check_label=None):
    """Return Dev raw train (1014), TestSet300, and TestSetNew46."""
    train = load_binding_table(paths.data_root / TRAIN_CSV, check_label=check_label)
    test300 = load_binding_table(paths.data_root / TEST300_CSV, check_label=check_label)
    test46 = load_binding_table(paths.data_root / TEST46_CSV, check_label=check_label)
    return {
        "train": train,
        "TestSet300": test300,
        "TestSetNew46": test46,
    }


def read_fasta(path):
    """Read a FASTA file into ``{id: sequence}``."""
    records = {}
    name = None
    chunks: list[str] = []
    with open(path) as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if name is not None:
                    records[name] = "".join(chunks)
                name = line[1:].split()[0]
                chunks = []
            else:
                chunks.append(line)
    if name is not None:
        records[name] = "".join(chunks)
    return records


def read_id_list(path):
    return [
        line.strip()
        for line in Path(path).read_text().splitlines()
        if line.strip()
    ]


def load_official_folds(fold_root):
    """Load ``ids_split1.txt`` ... ``ids_split5.txt`` from a directory."""
    fold_root = Path(fold_root)
    folds = {}
    order = []
    seen = set()
    for k in range(1, 6):
        path = fold_root / f"ids_split{k}.txt"
        ids = read_id_list(path)
        overlap = seen.intersection(ids)
        if overlap:
            raise RuntimeError(f"Fold overlap: {sorted(overlap)[:10]}")
        folds[k] = ids
        order.extend(ids)
        seen.update(ids)
    return folds, order


def subset_dev(train_df, official_order):
    """Order the raw train table to the official development IDs."""
    missing = set(official_order) - set(train_df.prot_id)
    if missing:
        raise RuntimeError(
            f"Official Dev IDs missing from project CSV: {sorted(missing)[:20]}"
        )
    lookup = train_df.set_index("prot_id", drop=False)
    return lookup.loc[list(official_order)].reset_index(drop=True)

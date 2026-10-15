"""Project paths. Drive locations are never hardcoded.

Data file names stay exactly as they are in the notebooks. Only the root
directories are configurable, via arguments or environment variables.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]

TRAIN_CSV = "development_set/full_grouped_train_binding_sites_df.csv"
TEST300_CSV = "development_set/full_grouped_test_binding_sites_df.csv"
TEST46_CSV = "independent_set/grouped_test_46_new_binding_sites.csv"

DEV_FASTA = "development_set/all.fasta"
TRAIN_335_FASTA = "development_set/Train_335.fa"
TEST_315_FASTA = "development_set/Test_315.fa"
TEST_60_FASTA = "development_set/Test_60.fa"
INDEP_FASTA = "independent_set/indep_set.fasta"

FOLD_IDS = "archived_data/ids_split{k}.txt"

RESIDUE_FILES = {
    "dev_metal": "development_set/binding_residues_2.5_metal.txt",
    "dev_nuclear": "development_set/binding_residues_2.5_nuclear.txt",
    "dev_small": "development_set/binding_residues_2.5_small.txt",
    "indep_metal": "independent_set/binding_residues_metal.txt",
    "indep_nuclear": "independent_set/binding_residues_nuclear.txt",
    "indep_small": "independent_set/binding_residues_small.txt",
}

RESULT_DIRS = {
    "anatomy": "ALL_BINDINGS_DATASET_ANATOMY_V1",
    "nuclear_audit": "NUCLEAR_DATASET_AUDIT_V1",
    "esm_association": "esm_feature_association",
    "metal_v1": "GEAR_ULTIMATE_METAL_BINDING",
    "metal_v2": "GEAR_METAL_ONLY_ULTIMATE_V2",
    "nuclear": "NUCLEAR_ULTIMATE_V1_DATA_DRIVEN_PATCH",
    "small": "SMALL_ULTIMATE_V1_DATA_DRIVEN_POCKET",
    "baselines": "baseline_reproduction_master_v6",
    "nuclear_xtask": "ADDIN_STANDALONE_NUCLEAR_XTASK",
    "small_xtask": "ADDIN_STANDALONE_SMALL_XTASK",
}


def _env_path(name: str) -> Path | None:
    raw = os.environ.get(name)
    if not raw:
        return None
    return Path(raw).expanduser().resolve()


@dataclass(frozen=True)
class ProjectPaths:
    """Roots for data, results, cache, and the optional gear_ultimate backend."""

    data_root: Path
    results_root: Path
    cache_root: Path
    gear_dir: Path | None = None
    alt_root: Path | None = None

    @property
    def repo_root(self) -> Path:
        """Directory that contains ``data/`` when data lives in the repo layout."""
        if self.data_root.name == "data":
            return self.data_root.parent
        return self.data_root.parent

    def data(self, *parts: str) -> Path:
        return self.data_root.joinpath(*parts)

    def results(self, *parts: str) -> Path:
        return self.results_root.joinpath(*parts)

    def result_dir(self, key: str) -> Path:
        path = self.results_root / RESULT_DIRS[key]
        path.mkdir(parents=True, exist_ok=True)
        return path

    def require_gear(self) -> Path:
        gear = self.gear_dir
        if gear is None or not (gear / "gear_ultimate").exists():
            raise FileNotFoundError(
                "GEAR backend not found. Pass --gear-dir or set LC_BIND_GEAR_DIR "
                "to the directory that contains the gear_ultimate package."
            )
        return gear

    def canonical_csv(self, name: str) -> Path:
        mapping = {
            "train": TRAIN_CSV,
            "test300": TEST300_CSV,
            "test46": TEST46_CSV,
        }
        path = self.data_root / mapping[name]
        if not path.exists():
            raise FileNotFoundError(path)
        return path


def resolve_paths(
    data_root: str | Path | None = None,
    results_root: str | Path | None = None,
    cache_root: str | Path | None = None,
    gear_dir: str | Path | None = None,
    alt_root: str | Path | None = None,
) -> ProjectPaths:
    data = Path(data_root).expanduser().resolve() if data_root else (
        _env_path("LC_BIND_DATA_ROOT") or (REPO_ROOT / "data")
    )
    results = Path(results_root).expanduser().resolve() if results_root else (
        _env_path("LC_BIND_RESULTS_ROOT") or (REPO_ROOT / "results")
    )
    cache = Path(cache_root).expanduser().resolve() if cache_root else (
        _env_path("LC_BIND_CACHE_ROOT") or (results / "cache")
    )
    if gear_dir:
        gear = Path(gear_dir).expanduser().resolve()
    else:
        gear = _env_path("LC_BIND_GEAR_DIR")
    if alt_root:
        alt = Path(alt_root).expanduser().resolve()
    else:
        alt = _env_path("LC_BIND_ALT_ROOT")
    results.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)
    return ProjectPaths(
        data_root=data,
        results_root=results,
        cache_root=cache,
        gear_dir=gear,
        alt_root=alt,
    )


def bind_session(paths: ProjectPaths, session) -> None:
    """Publish path roots on the shared session used by extracted stages."""
    session.clear()
    session.PATHS = paths
    session.BASE = paths.repo_root
    session.DATA_ROOT = paths.data_root
    session.RESULTS_ROOT = paths.results_root
    session.CACHE_ROOT = paths.cache_root
    session.GEAR_DIR = paths.gear_dir
    session.ALT_BASE = paths.alt_root or paths.repo_root
    session.MAMBA_BIN = Path(os.environ.get("LC_BIND_MAMBA_BIN", "micromamba"))

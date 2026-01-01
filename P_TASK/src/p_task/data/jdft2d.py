"""Utilities for downloading/structuring the Jdft2d exfoliation benchmark."""

from __future__ import annotations

import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
from jarvis.core.atoms import Atoms
from jarvis.db.figshare import data as jarvis_data
from pymatgen.io.cif import CifWriter

LOGGER = logging.getLogger(__name__)

# Name used by jarvis.db.figshare; "dft_2d" contains ~1.1k 2D entries.
DEFAULT_JDFT2D_DATASET = "dft_2d"
DEFAULT_CACHE_ROOT = Path("/home/pnn/2dgen/P_TASK/data/JARVIS/jdft2d_exfoliation/ache")
DEFAULT_METADATA_FILENAME = "jdft2d_exfoliation_metadata.csv"
DEFAULT_SPLIT_TEMPLATE = "splits/jdft2d_seed{seed}.json"
NA_VALUE = "na"

# Fields that are pulled directly from the JARVIS entries.
SELECTED_KEYS: Sequence[str] = (
    "jid",
    "formula",
    "spg_number",
    "spg_symbol",
    "atoms",
    "density",
)

# Additional derived columns that are computed locally for downstream loaders.
DERIVED_FIELDS: Sequence[Tuple[str, callable]] = (
    (
        "exfoliation_energy_ev_per_area",
        lambda entry, atoms: _first_available(
            entry,
            "exfoliation_energy_ev_per_area",
            "exfoliation_energy",
            "exfoliation_energy_ev",
        ),
    ),
    (
        "band_gap",
        lambda entry, atoms: _first_available(
            entry,
            "band_gap",
            "optb88vdw_bandgap",
            "mbj_bandgap",
            "hse_gap",
        ),
    ),
    (
        "pretty_formula",
        lambda entry, atoms: _first_available(
            entry,
            "pretty_formula",
            "formula",
            default=atoms.composition.reduced_formula if atoms else NA_VALUE,
        ),
    ),
    (
        "elements",
        lambda entry, atoms: _elements_string(atoms),
    ),
    (
        "spacegroup.number",
        lambda entry, atoms: _first_available(
            entry,
            "spacegroup.number",
            "spacegroup_number",
            "spg_number",
        ),
    ),
    ("cif", lambda entry, atoms: _atoms_to_cif_string(atoms)),
)

SplitFractions = Tuple[float, float, float]


def download_jdft2d_dataset(
    output_dir: Path | str = DEFAULT_CACHE_ROOT,
    *,
    dataset_name: str = DEFAULT_JDFT2D_DATASET,
    metadata_filename: str = DEFAULT_METADATA_FILENAME,
    split_template: str = DEFAULT_SPLIT_TEMPLATE,
    split_seed: int = 42,
    split_fracs: SplitFractions = (0.8, 0.1, 0.1),
    force: bool = False,
) -> Dict[str, Path]:
    """Download the Jdft2d entries, build metadata CSV + JSON split file.

    Parameters
    ----------
    output_dir:
        Root directory that should contain the downloaded metadata/splits.
    dataset_name:
        Name understood by :func:`jarvis.db.figshare.data` (defaults to ``jdft_2d``).
    metadata_filename:
        Filename used for the metadata CSV.
    split_template:
        Template for the JSON split file placed inside ``splits/``.
    split_seed:
        Random seed used for the default train/val/test split.
    split_fracs:
        Fractions for train/val/test (should sum to ~1).
    force:
        If ``True`` re-download even if cached files exist.

    Returns
    -------
    dict
        Mapping with ``metadata`` and ``split`` paths for downstream consumers.
    """

    root = Path(output_dir)
    raw_path = root / f"{dataset_name}.jsonl"
    metadata_path = root / metadata_filename
    split_path = root / split_template.format(seed=split_seed)
    root.mkdir(parents=True, exist_ok=True)
    split_path.parent.mkdir(parents=True, exist_ok=True)

    if metadata_path.exists() and split_path.exists() and not force:
        LOGGER.info("Dataset artifacts already present; skipping download.")
        return {"metadata": metadata_path, "split": split_path}

    if raw_path.exists() and not force:
        LOGGER.info("Loading cached Jdft2d entries from %s", raw_path)
        entries = _load_cached_entries(raw_path)
    else:
        LOGGER.info("Fetching entries for dataset '%s'", dataset_name)
        entries = jarvis_data(dataset_name)
        _write_jsonl(raw_path, entries)
        LOGGER.info("Cached raw entries to %s", raw_path)

    df = _build_metadata(entries)
    df.to_csv(metadata_path, index=False)
    LOGGER.info("Wrote metadata to %s", metadata_path)

    split_data = _build_split(df, split_seed, split_fracs)
    split_path.write_text(json.dumps(split_data, indent=2), encoding="utf-8")
    LOGGER.info("Wrote split file to %s", split_path)

    return {"metadata": metadata_path, "split": split_path}


def _build_metadata(entries: Sequence[Mapping[str, object]]) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for entry in entries:
        atoms_dict = entry.get("atoms")
        atoms = Atoms.from_dict(atoms_dict) if atoms_dict else None
        row = {key: entry.get(key, NA_VALUE) for key in SELECTED_KEYS}
        if atoms and _is_missing(row.get("density")):
            row["density"] = atoms.density
        for field, getter in DERIVED_FIELDS:
            row[field] = getter(entry, atoms)
        rows.append(row)

    ordered_columns = list(SELECTED_KEYS) + [
        field for field, _ in DERIVED_FIELDS if field not in SELECTED_KEYS
    ]
    return pd.DataFrame(rows, columns=ordered_columns)


def _build_split(df: pd.DataFrame, seed: int, fracs: SplitFractions) -> Dict[str, object]:
    num_rows = len(df)
    if num_rows == 0:
        raise ValueError("No entries retrieved for split generation.")

    if not np.isclose(sum(fracs), 1.0):
        LOGGER.warning("Split fractions sum to %.2f (expected 1.0)", sum(fracs))

    rng = np.random.default_rng(seed)
    indices = np.arange(num_rows)
    rng.shuffle(indices)

    n_train = int(fracs[0] * num_rows)
    n_val = int(fracs[1] * num_rows)
    n_test = num_rows - n_train - n_val

    train_idx = indices[:n_train]
    val_idx = indices[n_train : n_train + n_val]
    test_idx = indices[n_train + n_val :]

    def _select(idx: np.ndarray) -> List[str]:
        return df.iloc[idx]["jid"].astype(str).tolist()

    return {
        "seed": seed,
        "id_key": "jid",
        "fractions": {
            "train": fracs[0],
            "val": fracs[1],
            "test": fracs[2],
        },
        "splits": {
            "train": _select(train_idx),
            "val": _select(val_idx),
            "test": _select(test_idx),
        },
    }


def _first_available(entry: Mapping[str, object], *keys: str, default=NA_VALUE):
    for key in keys:
        if key in entry and not _is_missing(entry[key]):
            return entry[key]
    return default


def _is_missing(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, str) and value.lower() == NA_VALUE:
        return True
    return False


def _elements_string(atoms: Atoms | None) -> str:
    if not atoms:
        return NA_VALUE
    unique_elements = sorted(set(atoms.elements))
    return " ".join(unique_elements) if unique_elements else NA_VALUE


def _atoms_to_cif_string(atoms: Atoms | None) -> str:
    if atoms is None:
        return NA_VALUE

    structure = atoms.pymatgen_converter()
    if structure is not None:
        return str(CifWriter(structure))

    with tempfile.NamedTemporaryFile(mode="r+", suffix=".cif", delete=False) as tmp:
        atoms.write_cif(tmp.name)
        tmp.seek(0)
        cif_contents = tmp.read()
    os.unlink(tmp.name)
    return cif_contents


def _load_cached_entries(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _write_jsonl(path: Path, entries: Sequence[Mapping[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for entry in entries:
            json.dump(entry, f)
            f.write("\n")

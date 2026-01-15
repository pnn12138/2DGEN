from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Dict, List

import pandas as pd
from ase import io as ase_io
from ase.db import connect


OUTPUT_COLUMNS = [
    # Identifiers / basic attributes
    "material_id",
    "chemical_formula",
    "space_group_number",
    "space_group_symbol",
    # Energetics and stability
    "total_energy",
    "formation_energy",
    "energy_above_hull",
    "exfoliation_energy",
    # Structure
    "cif",
]


def atoms_to_cif(atoms) -> str:
    """Write atoms to a temporary CIF file and return the contents."""
    with NamedTemporaryFile(suffix=".cif", delete=False) as tmp:
        tmp_path = Path(tmp.name)

    try:
        ase_io.write(tmp_path, atoms, format="cif")
        return tmp_path.read_text().strip()
    finally:
        try:
            tmp_path.unlink()
        except FileNotFoundError:
            pass


def row_to_record(row: Any) -> Dict[str, Any]:
    """Convert an ASE db row into a flat record for CSV export."""
    atoms = row.toatoms()

    material_id = getattr(row, "uid", None) or getattr(row, "olduid", None) or row.id

    record: Dict[str, Any] = {
        "material_id": material_id,
        "chemical_formula": getattr(row, "formula", None)
        or atoms.get_chemical_formula(),
        "space_group_number": getattr(row, "number", None),
        "space_group_symbol": getattr(row, "international", None),
        "total_energy": getattr(row, "energy", None),
        "formation_energy": getattr(row, "hform", None),
        "energy_above_hull": getattr(row, "ehull", None),
        "exfoliation_energy": getattr(row, "E_B", None),
        "cif": atoms_to_cif(atoms),
    }
    return record


def export_c2db_to_csv(db_path: Path, output_path: Path) -> int:
    """Read the entire C2DB database and export selected fields to CSV."""
    records: List[Dict[str, Any]] = []
    with connect(db_path) as db:
        for row in db.select():
            records.append(row_to_record(row))

    df = pd.DataFrame.from_records(records, columns=OUTPUT_COLUMNS)
    df.to_csv(output_path, index=False)
    return len(df)


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    db_path = base_dir / "c2db.db"
    output_path = base_dir / "c2db_summary.csv"

    count = export_c2db_to_csv(db_path, output_path)
    print(f"Wrote {count} entries to {output_path}")


if __name__ == "__main__":
    main()

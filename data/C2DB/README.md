C2DB SQLite snapshot stored at `data/C2DB/c2db.db`.

Quick inspection
- Schema/keys preview: `python3 data/C2DB/test.py`
- CSV export with CIF column (requires ASE + pandas): `python3 data/C2DB/download.py`

Tables
- `systems`: core atomic data (positions, cell, energy, forces, etc.) plus `key_value_pairs` JSON.
- `keys`/`text_key_values`/`number_key_values`: auxiliary indexes for metadata keys.
- `species`: atomic species counts.
- `information`: DB-level metadata.

Metadata keys in `key_value_pairs` (83 distinct)
`A, A_u, E_B, J, J_u, N_nn, N_nn_u, P_spontaneous_norm, alphax, alphax_el, alphax_lat, alphay, alphay_el, alphay_lat, alphaz, alphaz_el, alphaz_lat, bravais_search, bravais_type, cbm, cbm_gw, cbm_hse, cbm_u, cod_id, dE_zx, dE_zx_u, dE_zy, dE_zy_u, dipz, doi, dyn_stab, efermi, efermi_gw, efermi_hse, efermi_u, ehull, emass_cbm, emass_vbm, evac, evacdiff, folder, gap, gap_dir, gap_dir_gw, gap_dir_hse, gap_dir_nosoc, gap_dir_u, gap_gw, gap_hse, gap_u, halfmetal_gap, halfmetal_gap_dir, halfmetal_gap_dir_hse, halfmetal_gap_hse, has_inversion_symmetry, hform, icsd_id, international, is_ferroelectric, is_magnetic, is_magnetic_u, label, lam, lam_u, layergroup, lgnum, magmom_u, minhessianeig, number, olduid, plasmafrequency_x, plasmafrequency_y, spin, spin_axis, spin_axis_u, spin_u, thickness, topology, uid, vbm, vbm_gw, vbm_hse, vbm_u`

Fields relevant to downstream filtering
- Material identifiers: `uid` (primary), `olduid` fallback; no `material_id` column exists in this snapshot (ASE stores an internal `unique_id` hash instead); chemical formula can be rebuilt via ASE from `numbers`.
- Symmetry: `international`, `number`, `layergroup`, `lgnum`.
- Energetics: `energy` (DFT total), `hform` (formation energy), `ehull` (energy above hull), `E_B` (exfoliation/binding energy), `free_energy` (if needed).
- Stability: `dyn_stab` (dynamical/phonon flag), `has_inversion_symmetry`, `topology`, `minhessianeig`.
- Magnetism: `is_magnetic`, `spin`, `spin_axis` (+ `_u` variants).
- Electronic gaps: `gap`, `gap_hse`, `gap_gw`, `gap_dir*`, `vbm*`, `cbm*`.

Sample entry (via `python3 data/C2DB/test.py`)
```
uid: 1InNaAs2S6-1
international: C2
number: 5
hform: -0.3464 eV/atom
ehull: 0.0840 eV/atom
dyn_stab: Yes
is_magnetic: False
gap: 1.91 eV
E_B: not present on this entry (use when available)
```

Notes for extraction
- `download.py` rebuilds a CIF string per entry using ASE (`row.toatoms()` → `ase.io.write`).
- If ASE is unavailable, you can inspect scalar metadata directly via `sqlite3` (see `test.py`).
- Total entries in this snapshot: 16,905.

#!/usr/bin/env python3

from pathlib import Path
import pandas as pd


PROJECT = Path(__file__).resolve().parents[1]
NEW_BASE = PROJECT / "qmmm_validation"
OLD_BASE = PROJECT.parent / "qmmm_validation_43" / "qmmm_validation"

OUT_TABLE = PROJECT / "results" / "tables" / "dataset_43_to_47_changes.tsv"
OUT_SUMMARY = PROJECT / "results" / "statistics" / "dataset_validation_summary.txt"


def main() -> None:
    new_csv = NEW_BASE / "barriers.csv"
    old_csv = OLD_BASE / "barriers.csv"
    pdb_dir = NEW_BASE / "complete_structures"

    if not new_csv.exists():
        raise FileNotFoundError(f"Missing new barriers file: {new_csv}")

    if not old_csv.exists():
        raise FileNotFoundError(f"Missing old barriers file: {old_csv}")

    if not pdb_dir.is_dir():
        raise FileNotFoundError(f"Missing complete structure directory: {pdb_dir}")

    new_df = pd.read_csv(new_csv)
    old_df = pd.read_csv(old_csv)

    required_column = "ligand_id"
    for label, df in [("old", old_df), ("new", new_df)]:
        if required_column not in df.columns:
            raise ValueError(
                f"{label} barriers.csv lacks required column: {required_column}"
            )

    old_ids = set(old_df[required_column].astype(str).str.strip())
    new_ids = set(new_df[required_column].astype(str).str.strip())
    pdb_ids = {path.stem for path in pdb_dir.glob("*.pdb")}

    all_ids = sorted(old_ids | new_ids)

    change_rows = []
    for ligand_id in all_ids:
        if ligand_id in new_ids and ligand_id not in old_ids:
            status = "added_after_successful_recomputation"
        elif ligand_id in old_ids and ligand_id in new_ids:
            status = "retained_from_original_43"
        elif ligand_id in old_ids and ligand_id not in new_ids:
            status = "removed_from_new_dataset"
        else:
            status = "unknown"

        change_rows.append(
            {
                "ligand_id": ligand_id,
                "in_old_43": ligand_id in old_ids,
                "in_new_47": ligand_id in new_ids,
                "has_complete_pdb": ligand_id in pdb_ids,
                "status": status,
            }
        )

    OUT_TABLE.parent.mkdir(parents=True, exist_ok=True)
    OUT_SUMMARY.parent.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(change_rows).to_csv(OUT_TABLE, sep="\t", index=False)

    missing_pdb = sorted(new_ids - pdb_ids)
    extra_pdb = sorted(pdb_ids - new_ids)
    added_ids = sorted(new_ids - old_ids)
    removed_ids = sorted(old_ids - new_ids)

    checks_pass = (
        len(old_df) == 43
        and len(new_df) == 47
        and len(old_ids) == 43
        and len(new_ids) == 47
        and len(pdb_ids) == 47
        and not missing_pdb
        and not extra_pdb
        and not removed_ids
    )

    lines = [
        "QMMM validation dataset integrity summary",
        "=========================================",
        f"Old barrier rows: {len(old_df)}",
        f"Old unique ligand IDs: {len(old_ids)}",
        f"New barrier rows: {len(new_df)}",
        f"New unique ligand IDs: {len(new_ids)}",
        f"Common ligand IDs: {len(old_ids & new_ids)}",
        f"Added ligand IDs: {len(added_ids)}",
        f"Removed ligand IDs: {len(removed_ids)}",
        f"Complete PDB files: {len(pdb_ids)}",
        f"CSV IDs without PDB: {len(missing_pdb)}",
        f"PDB IDs without CSV: {len(extra_pdb)}",
        "",
        "Added ligand IDs:",
        *[f"  {ligand_id}" for ligand_id in added_ids],
        "",
        "Removed ligand IDs:",
        *([f"  {ligand_id}" for ligand_id in removed_ids] or ["  None"]),
        "",
        "CSV IDs without complete PDB:",
        *([f"  {ligand_id}" for ligand_id in missing_pdb] or ["  None"]),
        "",
        "Complete PDB IDs without barrier:",
        *([f"  {ligand_id}" for ligand_id in extra_pdb] or ["  None"]),
        "",
        f"Dataset integrity: {'PASS' if checks_pass else 'CHECK REQUIRED'}",
    ]

    OUT_SUMMARY.write_text("\n".join(lines) + "\n")

    print("\n".join(lines))
    print(f"\nWrote table: {OUT_TABLE}")
    print(f"Wrote summary: {OUT_SUMMARY}")

    if not checks_pass:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3

from collections import Counter
from pathlib import Path
import pandas as pd


PROJECT = Path(__file__).resolve().parents[1]
PDB_DIR = PROJECT / "qmmm_validation" / "complete_structures"
BARRIERS_CSV = PROJECT / "qmmm_validation" / "barriers.csv"

OUT_TABLE = PROJECT / "results" / "tables" / "complete_structure_qc.tsv"
OUT_SUMMARY = PROJECT / "results" / "statistics" / "complete_structure_qc_summary.txt"


WATER_NAMES = {"HOH", "WAT", "TIP", "TIP3", "SOL"}


def classify_record(record: str, resname: str) -> str:
    if resname == "GTP":
        return "gtp"
    if resname in {"MG", "MG2"}:
        return "mg"
    if resname == "LIG":
        return "ligand"
    if resname in WATER_NAMES:
        return "water"
    if record == "ATOM":
        return "protein"
    return "other_hetero"


def inspect_pdb(pdb_path: Path) -> dict:
    counts = Counter()
    residue_names = Counter()
    malformed_atom_lines = 0

    with pdb_path.open(errors="replace") as handle:
        for raw_line in handle:
            if not raw_line.startswith(("ATOM  ", "HETATM")):
                continue

            if len(raw_line) < 54:
                malformed_atom_lines += 1
                continue

            record = raw_line[0:6].strip()
            resname = raw_line[17:20].strip()

            counts["total_atoms"] += 1
            counts[f"{record.lower()}_records"] += 1
            counts[classify_record(record, resname)] += 1
            residue_names[resname] += 1

    return {
        "ligand_id": pdb_path.stem,
        "total_atoms": counts["total_atoms"],
        "protein_atoms": counts["protein"],
        "gtp_atoms": counts["gtp"],
        "mg_atoms": counts["mg"],
        "ligand_atoms": counts["ligand"],
        "water_atoms": counts["water"],
        "other_hetero_atoms": counts["other_hetero"],
        "atom_records": counts["atom_records"],
        "hetatm_records": counts["hetatm_records"],
        "malformed_atom_lines": malformed_atom_lines,
        "has_protein": counts["protein"] > 0,
        "has_GTP": counts["gtp"] > 0,
        "has_MG": counts["mg"] > 0,
        "has_LIG": counts["ligand"] > 0,
        "residue_names": ";".join(sorted(name for name in residue_names if name)),
    }


def main() -> None:
    if not PDB_DIR.is_dir():
        raise FileNotFoundError(f"Missing directory: {PDB_DIR}")

    barrier_df = pd.read_csv(BARRIERS_CSV)
    expected_ids = set(barrier_df["ligand_id"].astype(str).str.strip())

    rows = [inspect_pdb(path) for path in sorted(PDB_DIR.glob("*.pdb"))]
    qc_df = pd.DataFrame(rows)

    OUT_TABLE.parent.mkdir(parents=True, exist_ok=True)
    OUT_SUMMARY.parent.mkdir(parents=True, exist_ok=True)

    qc_df.to_csv(OUT_TABLE, sep="\t", index=False)

    missing_protein = qc_df.loc[~qc_df["has_protein"], "ligand_id"].tolist()
    missing_gtp = qc_df.loc[~qc_df["has_GTP"], "ligand_id"].tolist()
    missing_mg = qc_df.loc[~qc_df["has_MG"], "ligand_id"].tolist()
    missing_lig = qc_df.loc[~qc_df["has_LIG"], "ligand_id"].tolist()
    malformed = qc_df.loc[
        qc_df["malformed_atom_lines"] > 0,
        ["ligand_id", "malformed_atom_lines"],
    ]

    parsed_ids = set(qc_df["ligand_id"])
    missing_files = sorted(expected_ids - parsed_ids)
    unexpected_files = sorted(parsed_ids - expected_ids)

    checks_pass = (
        len(qc_df) == 47
        and not missing_protein
        and not missing_gtp
        and not missing_mg
        and not missing_lig
        and malformed.empty
        and not missing_files
        and not unexpected_files
    )

    lines = [
        "Complete-structure QC summary",
        "=============================",
        f"Structures checked: {len(qc_df)}",
        f"Expected ligand IDs: {len(expected_ids)}",
        f"Atom-count minimum: {int(qc_df['total_atoms'].min())}",
        f"Atom-count maximum: {int(qc_df['total_atoms'].max())}",
        f"Atom-count mean: {qc_df['total_atoms'].mean():.2f}",
        f"Structures containing protein: {int(qc_df['has_protein'].sum())}/47",
        f"Structures containing GTP: {int(qc_df['has_GTP'].sum())}/47",
        f"Structures containing Mg: {int(qc_df['has_MG'].sum())}/47",
        f"Structures containing LIG: {int(qc_df['has_LIG'].sum())}/47",
        f"Structures with malformed atom lines: {len(malformed)}",
        f"Expected IDs without PDB: {len(missing_files)}",
        f"Unexpected PDB IDs: {len(unexpected_files)}",
        "",
        f"Structure QC: {'PASS' if checks_pass else 'CHECK REQUIRED'}",
    ]

    OUT_SUMMARY.write_text("\n".join(lines) + "\n")

    print("\n".join(lines))
    print(f"\nWrote table: {OUT_TABLE}")
    print(f"Wrote summary: {OUT_SUMMARY}")

    if not checks_pass:
        if missing_protein:
            print("Missing protein:", missing_protein)
        if missing_gtp:
            print("Missing GTP:", missing_gtp)
        if missing_mg:
            print("Missing Mg:", missing_mg)
        if missing_lig:
            print("Missing LIG:", missing_lig)
        if missing_files:
            print("Missing files:", missing_files)
        if unexpected_files:
            print("Unexpected files:", unexpected_files)
        raise SystemExit(1)


if __name__ == "__main__":
    main()

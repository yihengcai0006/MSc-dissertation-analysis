#!/usr/bin/env python3

from pathlib import Path
import subprocess
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parent.parent

STRUCTURE_DIR = ROOT / "qmmm_validation" / "complete_structures"
BARRIERS_FILE = ROOT / "qmmm_validation" / "barriers.csv"

FEATURE_SCRIPT = ROOT / "scripts" / "03_generate_features_complete.py"
REQUIRED_FEATURES = (
    ROOT / "model" / "dist_features_below_10_ang_list_new.npy"
)

FEATURE_DIR = ROOT / "features" / "per_ligand"
LOG_DIR = ROOT / "logs" / "feature_generation"
SUMMARY_PATH = (
    ROOT / "results" / "tables"
    / "feature_generation_summary_47.tsv"
)


def run_command(command: list[str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)

    with log_path.open("w") as log_handle:
        process = subprocess.run(
            command,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )

    return process.returncode


def main() -> None:
    FEATURE_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)

    barriers = pd.read_csv(BARRIERS_FILE)

    if "ligand_id" not in barriers.columns:
        raise ValueError("barriers.csv lacks ligand_id column")

    ligand_ids = (
        barriers["ligand_id"]
        .astype(str)
        .str.strip()
        .tolist()
    )

    if len(ligand_ids) != 47:
        raise ValueError(
            f"Expected 47 ligands, found {len(ligand_ids)}"
        )

    summary_rows = []

    for index, ligand_id in enumerate(ligand_ids, start=1):
        print(f"[{index:02d}/{len(ligand_ids)}] {ligand_id}")

        pdb_path = STRUCTURE_DIR / f"{ligand_id}.pdb"
        feature_path = FEATURE_DIR / f"{ligand_id}.parquet"
        log_path = LOG_DIR / f"{ligand_id}.log"

        row = {
            "ligand_id": ligand_id,
            "complete_pdb_exists": pdb_path.is_file(),
            "feature_file_exists_before": feature_path.is_file(),
            "return_code": None,
            "features_ok": False,
            "status": "",
        }

        if not pdb_path.is_file():
            row["status"] = "missing_complete_pdb"
            summary_rows.append(row)
            print("  FAIL: missing complete PDB")
            continue

        command = [
            sys.executable,
            str(FEATURE_SCRIPT),
            "--complex",
            str(pdb_path),
            "--molecule",
            ligand_id,
            "--output",
            str(feature_path),
            "--required-features",
            str(REQUIRED_FEATURES),
        ]

        return_code = run_command(command, log_path)
        row["return_code"] = return_code

        if return_code != 0:
            row["status"] = "feature_generation_failed"
            summary_rows.append(row)
            print(f"  FAIL: return code {return_code}")
            continue

        if not feature_path.is_file():
            row["status"] = "feature_file_missing_after_success"
            summary_rows.append(row)
            print("  FAIL: output file not found")
            continue

        row["features_ok"] = True
        row["status"] = "PASS"
        summary_rows.append(row)
        print("  PASS")

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(SUMMARY_PATH, sep="\t", index=False)

    print("\nFeature-generation summary")
    print("==========================")
    print(summary["status"].value_counts(dropna=False).to_string())
    print("\nWritten:", SUMMARY_PATH)

    passed = int((summary["status"] == "PASS").sum())
    print(f"Successful ligands: {passed}/{len(summary)}")

    if passed != len(summary):
        raise SystemExit(1)


if __name__ == "__main__":
    main()

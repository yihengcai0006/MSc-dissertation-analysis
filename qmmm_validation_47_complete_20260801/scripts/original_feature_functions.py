#!/usr/bin/env python3
"""Original feature functions extracted from provenance/pipeline_MD.py."""

from collections import defaultdict

import numpy as np
import pandas as pd
import MDAnalysis as mda
from MDAnalysis.analysis import distances

def compute_pp_dists_from_pdb(
    pdb_path: str,
    ligands=("GTP", "LIG"),
    radius=25.0,
    molid="Molecule"  # Default name, can be overridden
) -> pd.DataFrame:
    """
    Build a shell of protein residues within `radius` Å of any ligand in `ligands`,
    then compute pairwise minimum distances (Å) among those residues plus the ligands.
    Returns a single-row DataFrame (one row for this structure).
    """
    u = mda.Universe(pdb_path)

    # Build ligand selection string and take a protein shell around it
    lig_expr = " or ".join(f"resname {rn}" for rn in ligands)
    shell = u.select_atoms(f"protein and around {radius} ({lig_expr})").residues

    # Build per-residue selection expressions; include segid to avoid resid collisions
    res_exprs = []
    seen = set()
    for res in shell:
        seg = (res.segid or "").strip()
        expr = f"segid {seg} and resid {res.resid}" if seg else f"protein and resid {res.resid}"
        if expr not in seen:
            res_exprs.append(expr)
            seen.add(expr)

    # Add ligand selections that actually exist in this structure
    lig_exprs = []
    for rn in ligands:
        sel = u.select_atoms(f"resname {rn}")
        if sel.n_atoms == 0:
            continue
        lig_exprs.append(f"resname {rn}")

    # All selections and their pair combinations
    sel_exprs = res_exprs + lig_exprs
    if len(sel_exprs) < 2:
        raise ValueError("Not enough selections to form pairs (check ligands/residue shell).")

    pairs = [(a, b) for i, a in enumerate(sel_exprs) for b in sel_exprs[i+1:]]

    # Human-readable labels
    def _label(expr: str) -> str:
        if expr.startswith("resname "):
            return expr.replace("resname ", "")
        if expr.startswith("segid "):
            # segid X and resid N  ->  PROX_N
            parts = expr.split()
            seg = parts[1]
            resid = parts[-1]
            return f"PRO{seg}_{resid}"
        # fallback: protein and resid N -> PRON
        return "PRO" + expr.split()[-1]

    col_names = [f"{_label(a)}-{_label(b)}" for a, b in pairs]

    # Pre-create selections and compute distances (PBC-aware if unit cell present)
    sel_objs = {expr: u.select_atoms(expr) for expr in sel_exprs}
    dimensions = getattr(u, "dimensions", None)
    box = (
        dimensions
        if dimensions is not None
        and len(dimensions) >= 3
        and not np.allclose(dimensions[:3], 0)
        else None
    )

    # One row of results
    row = np.empty(len(pairs), dtype=np.float32)
    for j, (a, b) in enumerate(pairs):
        A, B = sel_objs[a], sel_objs[b]
        if A.n_atoms == 0 or B.n_atoms == 0:
            row[j] = np.nan
        else:
            row[j] = distances.distance_array(A.positions, B.positions, box=box).min()

    df = pd.DataFrame([np.round(row, 3)], columns=col_names)
    df.insert(0, "Molecule", molid)  # Add molecule ID as first column
    return df

def clean_protein_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove chain/segid letters from PRO* feature names.
    Example: PROA_4-PROB_5 -> PRO4-PRO5
    """
    def _clean_label(label: str) -> str:
        if not isinstance(label, str):
            return label
        parts = label.split("-")
        cleaned_parts = []
        for p in parts:
            if p.startswith("PRO"):
                # remove 'PROA_'/'PROB_' -> 'PRO' and digits
                cleaned_parts.append("PRO" + "".join(ch for ch in p if ch.isdigit()))
            else:
                cleaned_parts.append(p)
        return "-".join(cleaned_parts)

    df = df.copy()
    df.columns = [ _clean_label(c) for c in df.columns ]
    return df

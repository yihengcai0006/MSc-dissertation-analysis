# Update — complete structures provided, and what to expect

For Mingsong and Yiheng.  From Edina, 2026-07.

Thank you both — your validations were careful and agreed with each other
(Pearson r ≈ 0.05, compressed predictions). That result is **correct**. This
note (a) removes the structure workaround you both had to do, and (b) sets an
honest expectation for the rerun.

## 1. Complete structures are now provided

`complete_structures/<ligand_id>.pdb`  (47 ligands)

Each is the **assembled QM/MM complex** the barrier was actually computed from —
one PDB containing, in a single consistent coordinate frame:

- the full protein (RAS + NF1)
- **GTP and Mg** (correctly placed — no alignment needed)
- the docked ligand (resname `LIG`)

This is exactly what the old feature code
(`compute_pp_dists_from_pdb(..., ligands=("GTP","LIG"))`) expects. **Please rerun
the RF prediction using these files directly** — no complex reconstruction, no
GTP transfer by alignment. That removes any doubt that the earlier result was an
artifact of how the complex was rebuilt.

The set is also now **47 ligands** (the 4 previously QC-excluded ones were
recomputed successfully — see `barriers.csv`).

## 2. Expect the same result — and here is why (important)

I checked the model directly against these complete structures, and the near-zero
correlation is **not** fixable by better structures. The reason:

- The RF model puts **98.7% of its importance on protein–protein distances**
  (GTP 1.0%, ligand 0.3%).
- **~90% of that importance sits on features that are identical across all 47
  ligands** — e.g. its single most important feature `PRO1281–PRO1391` (25% of
  the whole model) has standard deviation **0.000 Å** across the set, in both the
  supplied protein AND the complete complex.
- Those residues are not among the ones our docking pipeline allows to move
  (only 11 sidechains flex: RAS 62,65,67,68,69,96,99 and NF1 1237,1241,1290,1402).
  So they sit at the same reference position for every ligand, the model's
  dominant inputs are constant, and it returns ~28.76 for everything.

The old model was trained on structures generated a different way (full solvated
MD equilibration), in which those residues varied per ligand. Our improved
barrier pipeline (flexible-sidechain docking + constrained QM/MM scan) does not
reproduce that variation — by design, and arguably correctly, since some of those
residues are far from the binding site and may reflect equilibration noise rather
than the chemistry.

## 3. What this means — the real conclusion

**The old model cannot be exercised by structures from the improved pipeline.**
This is a precise, reportable result, and it is more useful than "the model
failed": it tells us *why* it does not transfer.

The next step for the project is **to retrain a model on the new input strategy** —
using the barriers here as labels and features from the structures our pipeline
actually produces (where the varying quantities are the flexible sidechains, the
ligand contacts, and the reaction-centre geometry). A model trained this way will
key on features that genuinely differ between ligands, rather than on protein
motions the new pipeline does not generate.

So for the rerun: please confirm the correlation with the complete structures
(for the record), then treat this as the motivation and starting point for
retraining, not as an endpoint.

Questions → Edina.

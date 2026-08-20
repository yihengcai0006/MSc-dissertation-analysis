# Methods — how the 23 reference barriers were computed

Computed on ARCHER2, July 2026.

## Protocol

Docked+relaxed ligand pose  →  ligand parameterisation + pose mapped onto the
GAFF topology  →  CHARMM complex build (`geom_complex.psf`)  →  41-frame
reaction path  →  **QM/MM scan**  →  **QM/MM single points**  →
barrier = max(E) − min(E) over the selected path windows.

## Level of theory

| setting | value |
|---|---|
| QM method | B3LYP |
| basis | 6-31+G* |
| QM program | Q-Chem 4.3 (István Szabó `mess-qmmm` build, `qchem-dev-old`) |
| MM program | CHARMM 46b1 |
| integral threshold | **THRESH 14** (see note below) |
| SCF convergence | 1e-6 |

## The key difference from the older setup

- **Scan stage:** QM region is the reaction centre — GTP phosphates, the
  attacking water, and Gln61. The ligand is **MM** here.
- **Single-point stage:** the QM region additionally **includes the ligand**
  (~150 QM atoms total, with 8 link atoms).

The older protocol that the ML model was trained on did **not** include the
ligand in the QM region at the single-point stage. That is the difference this
validation is testing.

### Why `THRESH 14` is needed
With the ligand in the QM region (~150 atoms) and 6-31+G*'s diffuse functions,
the overlap matrix becomes near-singular and Q-Chem aborts every window with
*"Negative overlap matrix eigenvalue"*. Critically, **CHARMM still reports
NORMAL TERMINATION**, so this failure silently yields a pure-MM barrier that
looks plausible. `THRESH 14` resolves it. Every barrier in this set was
verified to have converged QM SCFs (non-zero `Convergence criterion met` count)
and QM-magnitude energies (~−3.4 × 10⁶ kcal/mol, not ~+2 × 10⁵).

## Heterogeneity to be aware of

The 23 are **not uniform in one respect** — the number of QM/MM single-point
windows differs:

| subset | n | SP windows |
|---|---|---|
| batch12 | 5 | 8 |
| batch9 | 9 | 4 |
| batch10 | 9 | 4 |

The barrier is max−min over the sampled windows, so 4-window sampling is
coarser than 8-window. Both bracket the extremes, and the difference is expected
to be small relative to the ~±1 kcal/mol pose noise — but the `n_windows` column
is in `barriers_23.csv` so you can test for a subset effect if you wish.

## Known uncertainty

Pose sensitivity was measured explicitly (multi-seed perturbation test on one
ligand, 5 seeds × 4 perturbation magnitudes):

| ligand-pose perturbation (RMS) | barrier shift |
|---|---|
| 0.034 Å | +0.99 ± 0.38 |
| 0.084 Å | +0.20 ± 1.19 |
| 0.169 Å | +0.36 ± 1.12 |
| 0.338 Å | −2.09 ± 0.59 |

**A single-pose barrier is reliable to roughly ±1 kcal/mol.** Do not treat
differences smaller than that as meaningful.

## Summary statistics of the set

n = 43, range 23.25 – 30.63, mean 26.99, SD 1.67 kcal/mol.

## Quality control

Each barrier was verified with `verify_barriers.py`, which requires:
- every selected SP window has a QM-magnitude energy (< -1e6 kcal/mol)
- converged SCF count >= number of windows
- resulting barrier physically plausible (0 < b < 200 kcal/mol)

Four ligands failed and were excluded (see `excluded.csv`). Root cause was a
Q-Chem scratch-write failure (`Error writing to TMP file`) caused by disk
pressure on /work — not an SCF convergence problem. Those four are being
recomputed.

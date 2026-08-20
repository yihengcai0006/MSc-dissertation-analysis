# Task brief — does the old ML model still hold up against the new QM/MM barriers?

For Mingsong and Yiheng.  Prepared by Edina, updated 2026-07-23.

Package location (bohr):
`/home/edina/shared/qmmm_validation/`

---

## Background

The existing ML model was trained on QM/MM barriers computed with the **older
setup, in which the ligand was NOT included in the QM region**. We have since
improved the protocol so that the **ligand is inside the QM region** during the
single-point step, which we believe gives more reliable barriers.

We now have **43 ligands** with barriers computed using the improved setup.
That raises a question worth answering directly, before we rely further on the
existing screening:

> **Does the old model still predict anything useful against the better reference?**

## The task

1. Take the **43 ligands** in `barriers.csv` (IDs, computed barriers, and the
   exact docked poses used are all provided here).
2. Run the **old ML code, unchanged**, to predict barriers for those same 43.
3. Compare predicted vs computed.

The point is *not* to make the model look good. A negative result is equally
valuable and equally reportable.

## What to report

- **Spearman (rank) and Pearson correlation, each with a confidence interval.**
  For screening, rank order matters more than absolute agreement.
- **A scatter plot** of predicted vs computed with the 1:1 line.
- **The systematic offset.** The two sets are at different levels of theory, so
  expect a shift in the mean. A constant offset is far less concerning than a
  scrambled ordering — please separate those two effects.

## Five things to be careful about

**1. Check for training-set leakage first.**
These ligands come from `batch0001`. If any were in the model's training data,
the comparison is circular and the correlation will be inflated. Please verify
this before anything else and report how many were excluded.

**2. n = 43 is still modest.**
Show confidence intervals rather than quoting point estimates.

**3. Range restriction.**
These span 23.25–30.63 kcal/mol (mean 26.99, SD 1.67) — a narrow window.
Range restriction mechanically attenuates correlation; factor that into the
interpretation.

**4. There is a hard ceiling set by label noise.**
We measured pose sensitivity directly: perturbing a ligand pose by ~0.03 Å RMS
shifts the computed barrier by ~1 kcal/mol. A single-pose barrier is good to
about **±1 kcal/mol**.

Against an observed SD of 1.67, that implies a reliability of roughly 0.64, so
**even a perfect predictor could only reach r ≈ 0.8**. Judge the result against
that ceiling, not against 1.0 — a correlation of ~0.5 would be quite respectable.

**5. Use the poses provided.**
The barrier is pose-dependent at the ~1 kcal/mol level, so re-docking or using a
different pose would not be a like-for-like test.

## Interpretation

- **Decent rank correlation** → the old screening retains practical value.
- **Near-zero or scrambled** → the old rankings do not transfer, and retraining
  on ligand-in-QM barriers becomes a priority.

## Contents of this package

| file | what it is |
|---|---|
| `TASK_BRIEF.md` | this document |
| `barriers.csv` | 43 ligand IDs + computed barriers (+ provenance columns) |
| `METHODS.md` | exact level of theory and protocol |
| `excluded.csv` | 4 ligands excluded for incomplete QM — do NOT use these |
| `poses/<molid>/<molid>.sdf` | the docked+relaxed ligand pose used |
| `poses/<molid>/rank1_protein.pdb` | the corresponding flexible-sidechain protein |

## Quality control applied

Every barrier here was verified to have **complete QM across all single-point
windows** — not merely "at least one converged SCF". Four ligands were rejected
for partial QM (see `excluded.csv`); three had obviously nonsense values
(~3.7e6 kcal/mol) but one, `PV-000233762504`, returned a perfectly
plausible-looking 28.53 despite only 3 of 4 windows having QM. That is why the
per-window check matters — please do not add those four back in.

Questions → Edina.

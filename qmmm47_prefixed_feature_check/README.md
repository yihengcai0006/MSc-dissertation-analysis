# QMMM47 Pre-Fixed Stable Feature Exploratory Check

## Overview

This repository contains an exploratory transfer check of three protein–protein structural features that were fixed in advance from the independent February 2026 100-ligand nested cross-validation analysis.

The purpose of this analysis is to test whether these three previously identified structural descriptors:

1. retain structural variation across 47 complete improved-QM/MM complexes; and
2. where variable, show any exploratory association with the improved-QM/MM activation barriers.

This is **not** a new feature-discovery analysis.

No features are selected using the 47-structure dataset, and no new classifier or predictive regression model is trained.

---

# Scientific Context

The February 2026 100-ligand analysis used barriers predicted by the existing random-forest (RF) model to define clear low/high groups.

The formal binary-classification dataset contained:

- 26 low-barrier-labelled molecules;
- 57 high-barrier-labelled molecules;
- 83 molecules in total after excluding 17 intermediate cases.

Nested cross-validation, permutation testing, and feature-selection stability analysis identified three particularly stable protein–protein features:

| Internal feature | Structural mapping | Protein–protein selection frequency | Combined-set selection frequency |
|---|---|---:|---:|
| `PRO62-PRO1279` | GLU62(RAS)–SER1279(NF1) | 25/25 = 1.00 | 25/25 = 1.00 |
| `PRO1275-PRO1276` | PHE1275(NF1)–ARG1276(NF1) | 22/25 = 0.88 | 21/25 = 0.84 |
| `PRO90-PRO92` | PHE90(RAS)–ASP92(RAS) | 21/25 = 0.84 | 20/25 = 0.80 |

These three features were therefore fixed **before** examining their relationship with the improved-QM/MM barriers.

The present project asks whether these stable structural descriptors of the old-RF-defined low/high behaviour transfer to the improved-QM/MM dataset.

---

# Important Interpretation Boundary

The February 2026 low/high labels were derived from **old RF-predicted barriers**.

They were:

- not experimentally measured barriers;
- not the improved-QM/MM barriers analysed here.

Therefore, the three pre-fixed features should be interpreted as structural descriptors associated with the behaviour of the old RF model.

This exploratory analysis does not establish that these features are causal determinants of catalytic activity or improved-QM/MM activation barriers.

---

# Dataset

The supplied validation dataset is stored under:

`qmmm_validation/`

The main inputs are:

`qmmm_validation/complete_structures/`

`qmmm_validation/poses/`

`qmmm_validation/barriers.csv`

`qmmm_validation/excluded.csv`

`qmmm_validation/METHODS.md`

`qmmm_validation/NOTE_complete_structures.md`

`qmmm_validation/TASK_BRIEF.md`

The analysis uses:

- 47 complete improved-QM/MM structures;
- 47 supplied improved-QM/MM activation barriers.

The barrier column is:

`barrier_kcal_per_mol`

The supplied barriers span approximately:

- minimum: 23.25 kcal mol^-1;
- maximum: 30.63 kcal mol^-1;
- mean: 27.065 kcal mol^-1;
- standard deviation: 1.650 kcal mol^-1.

---

# Feature Source and Provenance

The structural feature matrix was generated previously during the corrected validation using the 47 complete improved-QM/MM structures.

Original source:

`~/yihengcai_ucl_essay_work/qmmm_validation_47_complete_20260801/features/feature_matrix_47.parquet`

A local copy used by this project is stored at:

`data/existing_features/feature_matrix_47.parquet`

The source and provenance are recorded in:

`data/provenance/FEATURE_SOURCE.txt`

During the corrected 47-structure validation, the supplied complete structures were used directly.

The corrected workflow did not perform:

- structural alignment;
- GTP transfer;
- ligand insertion;
- complex reconstruction.

The present analysis therefore reuses the previously validated feature matrix rather than recomputing the 4,086 structural features.

---

# Pre-Fixed Features

Only the following three features are analysed:

- `PRO62-PRO1279`
- `PRO1275-PRO1276`
- `PRO90-PRO92`

Structural mapping from the original feature-generation implementation gives:

| Internal feature | Biological identity |
|---|---|
| `PRO62-PRO1279` | GLU62(RAS)–SER1279(NF1) |
| `PRO1275-PRO1276` | PHE1275(NF1)–ARG1276(NF1) |
| `PRO90-PRO92` | PHE90(RAS)–ASP92(RAS) |

The original protein–protein features are defined as the **minimum all-atom inter-residue distance** between the two residues.

They are not Cα–Cα distances.

---

# What This Analysis Does Not Do

This project deliberately does not:

- rank all 4,086 RF input features against the improved-QM/MM barriers;
- select new features using the 47-structure dataset;
- search for the most correlated descriptors;
- tune a predictive model;
- train a new random forest;
- train a classifier;
- train a predictive regression model.

The analysis is a **pre-fixed exploratory transfer check only**.

---

# Repository Structure

A simplified directory structure is:

`qmmm47_prefixed_feature_check/`

- `data/`
  - `existing_features/`
    - `feature_matrix_47.parquet`
    - `feature_matrix_47_preview.tsv`
  - `provenance/`
    - `FEATURE_SOURCE.txt`

- `qmmm_validation/`
  - `complete_structures/`
  - `poses/`
  - `barriers.csv`
  - `excluded.csv`
  - `METHODS.md`
  - `NOTE_complete_structures.md`
  - `TASK_BRIEF.md`

- `scripts/`
  - `01_audit_qmmm47_dataset.py`
  - `02_extract_prefixed_features.py`
  - `03_analyse_prefixed_features_vs_qmmm.py`

- `results/`
  - `figures/`
  - `logs/`
  - `tables/`

- `README.md`

---

# Environment

The scripts were run in the existing thesis analysis environment:

`conda activate thesis_features`

Main Python dependencies include:

- Python;
- NumPy;
- pandas;
- SciPy;
- Matplotlib;
- PyArrow.

---

# Reproduction

Run all commands from the repository root.

Change to the project directory:

`cd ~/yihengcai_ucl_essay_work/qmmm47_prefixed_feature_check`

Activate the analysis environment:

`conda activate thesis_features`

Then execute the scripts in order:

`python scripts/01_audit_qmmm47_dataset.py`

`python scripts/02_extract_prefixed_features.py`

`python scripts/03_analyse_prefixed_features_vs_qmmm.py`

The workflow is intentionally divided into three stages:

- Script 01 → dataset and provenance quality control;
- Script 02 → pre-fixed feature extraction and one-to-one merge;
- Script 03 → exploratory structural variation and barrier association analysis.

This separation makes the workflow easier to inspect and reproduce.

---

# Script 01 — Dataset Audit

Script:

`01_audit_qmmm47_dataset.py`

## Purpose

This script checks that:

- exactly 47 improved-QM/MM barrier rows are present;
- exactly 47 complete PDB structures are present;
- exactly 47 feature rows are present;
- ligand IDs match exactly between barriers, structures, and features;
- all three pre-fixed features are available;
- none of the three target features contains missing values.

No correlation analysis or model fitting is performed.

## QC Result

The audit found:

| Check | Result |
|---|---:|
| Barrier rows | 47 |
| Unique barrier IDs | 47 |
| PDB files | 47 |
| Unique PDB IDs | 47 |
| Feature rows | 47 |
| Unique feature IDs | 47 |

ID differences:

- features not in barriers: `[]`
- barriers not in features: `[]`
- structures not in barriers: `[]`
- barriers not in structures: `[]`

Final QC result:

`QC PASSED: exact 47/47 ID match across features, barriers and complete PDBs.`

The three target features are all present with no missing values:

| Feature | Missing values | Unique values |
|---|---:|---:|
| `PRO62-PRO1279` | 0 | 45 |
| `PRO1275-PRO1276` | 0 | 1 |
| `PRO90-PRO92` | 0 | 1 |

## Outputs

`results/tables/01_qmmm47_dataset_audit.tsv`

`results/logs/01_qmmm47_dataset_audit.log`

---

# Script 02 — Extract and Merge the Pre-Fixed Features

Script:

`02_extract_prefixed_features.py`

## Purpose

This script extracts only the three pre-fixed features from the existing corrected feature matrix and performs a one-to-one merge with the improved-QM/MM barrier table.

The merged dataset contains one row per ligand.

No feature selection is performed.

## Main Output

`results/tables/02_qmmm47_prefixed_feature_values.tsv`

This is the main 47-ligand analysis table used by Script 03.

---

# Structural Variation Across the 47 Structures

## GLU62(RAS)–SER1279(NF1)

Internal feature:

`PRO62-PRO1279`

Structural variation:

| Measure | Value |
|---|---:|
| Unique values | 45 |
| Minimum | 5.161 Å |
| Maximum | 7.547 Å |
| Range | 2.386 Å |
| Standard deviation | 0.509 Å |

This feature retains substantial variation across the 47 complete structures.

---

## PHE1275(NF1)–ARG1276(NF1)

Internal feature:

`PRO1275-PRO1276`

Structural variation:

| Measure | Value |
|---|---:|
| Unique values | 1 |
| Minimum | 1.347 Å |
| Maximum | 1.347 Å |
| Range | 0.000 Å |
| Standard deviation | 0.000 Å |

This feature is invariant across all 47 structures.

---

## PHE90(RAS)–ASP92(RAS)

Internal feature:

`PRO90-PRO92`

Structural variation:

| Measure | Value |
|---|---:|
| Unique values | 1 |
| Minimum | 3.103 Å |
| Maximum | 3.103 Å |
| Range | 0.000 Å |
| Standard deviation | 0.000 Å |

This feature is also invariant across all 47 structures.

## Outputs

`results/tables/02_qmmm47_prefixed_feature_values.tsv`

`results/logs/02_extract_prefixed_features.log`

---

# Script 03 — Exploratory Association with Improved-QM/MM Barriers

Script:

`03_analyse_prefixed_features_vs_qmmm.py`

## Purpose

For each of the three pre-fixed features, the script:

- quantifies structural variation;
- determines whether correlation analysis is mathematically meaningful;
- calculates Pearson correlation only for variable features;
- calculates Spearman correlation only for variable features;
- generates a scatter plot for testable features.

Invariant predictors are not passed to the correlation functions.

They are explicitly reported as:

`not_testable_invariant`

---

# Why Invariant Features Are Not Correlated

Pearson and Spearman correlation require variation in the predictor.

If every structure has exactly the same value, for example:

`3.103`

`3.103`

`3.103`

then there is no structural variation that can be associated with changes in the activation barrier.

For this reason:

- PHE1275(NF1)–ARG1276(NF1);
- PHE90(RAS)–ASP92(RAS);

are reported as invariant rather than assigned artificial correlation values.

---

# Main Results

## GLU62(RAS)–SER1279(NF1)

Internal feature:

`PRO62-PRO1279`

This is the only one of the three pre-fixed features that retains substantial structural variation across the 47 complete structures.

Structural variation:

- unique values: 45;
- range: 2.386 Å;
- standard deviation: 0.509 Å.

Its association with the improved-QM/MM barrier is:

### Pearson Correlation

- Pearson r = -0.093;
- p = 0.533.

### Spearman Correlation

- Spearman rho = -0.010;
- p = 0.949.

Both correlations are very close to zero.

Therefore, despite substantial variation in the GLU62(RAS)–SER1279(NF1) distance, there is no evident linear or monotonic association with the improved-QM/MM activation barrier in this 47-structure dataset.

---

## PHE1275(NF1)–ARG1276(NF1)

Internal feature:

`PRO1275-PRO1276`

Across all 47 structures:

`distance = 1.347 Å`

with:

- range = 0;
- standard deviation = 0.

Analysis status:

`not_testable_invariant`

No Pearson or Spearman correlation is calculated.

---

## PHE90(RAS)–ASP92(RAS)

Internal feature:

`PRO90-PRO92`

Across all 47 structures:

`distance = 3.103 Å`

with:

- range = 0;
- standard deviation = 0.

Analysis status:

`not_testable_invariant`

No Pearson or Spearman correlation is calculated.

---

# Thesis-Ready Result Summary

| Internal feature | Structural label | Variation in 47 structures | Pearson r | Spearman rho | Interpretation |
|---|---|---|---:|---:|---|
| `PRO62-PRO1279` | GLU62(RAS)–SER1279(NF1) | Variable; range = 2.386 Å | -0.093 | -0.010 | No evident association with improved-QM/MM barrier |
| `PRO1275-PRO1276` | PHE1275(NF1)–ARG1276(NF1) | Invariant at 1.347 Å | NA | NA | Not testable because there is no structural variation |
| `PRO90-PRO92` | PHE90(RAS)–ASP92(RAS) | Invariant at 3.103 Å | NA | NA | Not testable because there is no structural variation |

---

# Main Figure

The main exploratory figure is:

`results/figures/03_PRO62_PRO1279_vs_improved_qmmm_barrier.png`

Vector version:

`results/figures/03_PRO62_PRO1279_vs_improved_qmmm_barrier.svg`

The figure shows all 47 complete structures.

Axes:

- x-axis = GLU62(RAS)–SER1279(NF1) minimum inter-residue distance;
- y-axis = improved-QM/MM activation barrier.

The fitted line is included for descriptive visualisation only.

It is not a newly trained predictive regression model.

The nearly flat trend is consistent with the near-zero Pearson and Spearman correlations.

---

# Main Scientific Conclusion

The three most stable protein–protein descriptors identified in the old-RF-defined February 2026 classification do not transfer straightforwardly to the improved-QM/MM barrier dataset.

Two of the three descriptors:

- PHE1275(NF1)–ARG1276(NF1);
- PHE90(RAS)–ASP92(RAS);

are completely invariant across all 47 complete structures.

They therefore cannot explain variation in the improved-QM/MM barriers within this dataset.

The remaining descriptor:

`GLU62(RAS)–SER1279(NF1)`

retains substantial structural variation but shows negligible association with the improved-QM/MM activation barrier:

- Pearson r = -0.093;
- Spearman rho = -0.010.

This supports interpreting the stable features identified in the February old-RF-defined classification primarily as structural descriptors of old-model behaviour rather than as automatically transferable determinants of the improved-QM/MM activation barrier.

---

# Relationship to the Completed 47-Structure Old-RF Validation

The previous corrected validation of the existing RF model against the 47 complete improved-QM/MM structures produced weak agreement:

- Pearson r ≈ 0.139;
- Spearman rho ≈ 0.133.

The present pre-fixed feature-level exploratory check is consistent with that model-level result.

Together, the analyses show:

**weak transfer of the existing RF predictions**

plus

**weak or absent transfer of the most stable old-RF structural descriptors**

to the improved-QM/MM activation-barrier dataset.

---

# Interpretation Limitations

These results should not be interpreted as proving that GLU62, SER1279, PHE1275, ARG1276, PHE90, or ASP92 are biologically irrelevant.

The analysis establishes only that:

- two of the three pre-fixed descriptors are invariant in this 47-structure dataset; and
- the remaining variable descriptor shows no evident association with improved-QM/MM barrier variation.

The analysis is exploratory and uses:

`n = 47`

complete structures.

No causal inference is made.

The absence of transfer in this dataset does not demonstrate that the corresponding residues or structural relationships are unimportant in other molecular contexts.

---

# Key Result Files

## Dataset QC

`results/tables/01_qmmm47_dataset_audit.tsv`

## Merged 47-Ligand Values

`results/tables/02_qmmm47_prefixed_feature_values.tsv`

## Full Feature Statistics

`results/tables/03_qmmm47_prefixed_feature_statistics.tsv`

## Concise Thesis Summary

`results/tables/03_qmmm47_prefixed_feature_thesis_summary.tsv`

## Main Figure

`results/figures/03_PRO62_PRO1279_vs_improved_qmmm_barrier.png`

## Analysis Logs

`results/logs/01_qmmm47_dataset_audit.log`

`results/logs/02_extract_prefixed_features.log`

`results/logs/03_analyse_prefixed_features_vs_qmmm.log`

---

# Workflow Status

| Stage | Status |
|---|---|
| 01 Dataset audit | COMPLETE |
| 02 Pre-fixed feature extraction | COMPLETE |
| 03 Exploratory association analysis | COMPLETE |

The 47-complete-structure pre-fixed stable-feature exploratory check is complete.

---

# Code Organisation

The analysis is intentionally divided into three scripts with separate responsibilities.

## Script 01

`01_audit_qmmm47_dataset.py`

Purpose:

- input integrity;
- file counts;
- ligand-ID matching;
- feature availability;
- missing-value checks.

## Script 02

`02_extract_prefixed_features.py`

Purpose:

- extraction of the three pre-fixed features;
- one-to-one merge with the improved-QM/MM barrier table;
- structural-variation summary.

## Script 03

`03_analyse_prefixed_features_vs_qmmm.py`

Purpose:

- structural-variation assessment;
- determination of whether each feature is testable;
- Pearson and Spearman analysis for variable features;
- exploratory figure generation.

This separation avoids combining data validation, data preparation, and statistical interpretation into a single monolithic script.

The scripts use:

- explicit input checks;
- deterministic output paths;
- descriptive output names;
- fail-fast quality-control checks where appropriate.

---

# Reproducibility Notes

This analysis deliberately reuses the previously validated 47-structure feature matrix.

The 4,086 structural features are not regenerated during this project.

The analysis therefore separates:

1. the previously completed structural-feature generation;
2. extraction of the three pre-fixed features;
3. exploratory feature-level transfer analysis.

This prevents unintended changes to the corrected 47-structure feature-generation workflow.

---

# Important Dataset-Separation Rule

The three pre-fixed features originate from the February 2026 100-ligand structural-classifier analysis.

The 47 complete improved-QM/MM structures are used only for the independent exploratory transfer check.

The 47-structure dataset is not used to:

- retrain the February classifier;
- redefine the low/high labels;
- select replacement stable features;
- search over the complete 4,086-feature space.

This separation is important for avoiding information leakage between feature discovery and transfer evaluation.

---

# Final Interpretation

The exploratory transfer check gives two main findings.

First, two of the three pre-fixed stable features are completely invariant across the 47 complete improved-QM/MM structures.

Second, the remaining variable feature, GLU62(RAS)–SER1279(NF1), retains substantial structural variation but shows negligible correlation with the improved-QM/MM activation barriers.

Therefore, the structural descriptors that are reproducible within the old-RF-defined February low/high classification task do not show straightforward transfer to the improved-QM/MM activation-barrier target.

The results support a conservative interpretation:

the stable features are informative descriptors of structural patterns associated with the old RF model's behaviour, but they should not be treated as automatically transferable predictors or causal determinants of the improved-QM/MM activation barrier.
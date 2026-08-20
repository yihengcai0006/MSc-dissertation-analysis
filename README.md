# MSc Dissertation Analysis Repository

## Overview

This repository contains the computational analyses used in an MSc dissertation investigating the transferability and structural interpretation of an existing machine-learning model for QM/MM activation barriers.

The repository contains three related analysis workflows:

1. `thesis_features_analysis/`
   - structural classification and interpretation using the original 100-ligand dataset;

2. `qmmm_validation_47_complete_20260801/`
   - direct validation of the existing random-forest model using 47 complete structures with improved-QM/MM activation barriers;

3. `qmmm47_prefixed_feature_check/`
   - independent transfer analysis of three pre-fixed structural features in the same 47-structure improved-QM/MM dataset.

Each subproject contains its own detailed README describing the scripts, inputs, outputs, results, and interpretation.

---

## Repository Structure

`yihengcai_ucl_essay_work/`

- `thesis_features_analysis/`
- `qmmm_validation_47_complete_20260801/`
- `qmmm47_prefixed_feature_check/`
- `sample_data/`
- `README.md`

---

## Dataset Overview

Two main datasets are used in this repository and they are kept strictly separate.

### 1. Original 100-Ligand Structural Dataset

Used in:

`thesis_features_analysis/`

This is the original February 2026 structural-feature dataset.

It contains:

- 100 ligands;
- one structural-feature parquet file per ligand;
- approximately 4 GB of raw feature data in total;
- old random-forest-predicted activation barriers.

The old RF-predicted barriers define:

- 26 low-barrier ligands;
- 17 intermediate ligands;
- 57 high-barrier ligands.

The formal binary classifier excludes the 17 intermediate systems and therefore uses:

- 83 ligands;
- 26 low;
- 57 high.

These labels are derived from the existing RF model. They are not experimental or improved-QM/MM activation barriers.

Because the complete raw parquet dataset is approximately 4 GB, it is not duplicated in this GitHub repository.

A small example dataset is provided under:

`sample_data/thesis_features_example/`

The example data are intended only to demonstrate the feature format and cannot reproduce the full nested-CV analysis.

### 2. Improved-QM/MM Dataset

Used in:

- `qmmm_validation_47_complete_20260801/`
- `qmmm47_prefixed_feature_check/`

This dataset contains:

- 47 complete molecular structures;
- 47 improved-QM/MM activation barriers.

It is used for:

- direct validation of the existing RF model;
- independent evaluation of three pre-fixed structural features.

The 47-structure dataset is not used to redefine the 100-ligand classifier labels or to select new stable features.

---

## Main Scientific Results

The existing RF model shows limited transferability to the improved-QM/MM dataset.

For the 47 complete structures:

- Pearson correlation: `r = 0.139`;
- Spearman correlation: `rho = 0.133`;
- regression `R² = 0.019`.

The predictions are strongly compressed toward the mean.

In the separate 100-ligand structural-classification analysis, protein–protein distance features show the strongest discrimination between the old-RF-defined low and high groups.

Mean nested-CV balanced accuracy:

- Protein–ligand: `0.583 ± 0.118`;
- Protein–GTP: `0.594 ± 0.122`;
- Protein–protein: `0.643 ± 0.119`;
- Combined: `0.624 ± 0.145`;
- Dummy baseline: `0.500`.

Three stable protein–protein descriptors are identified:

- GLU62(RAS)–SER1279(NF1);
- PHE1275(NF1)–ARG1276(NF1);
- PHE90(RAS)–ASP92(RAS).

These features are interpreted as reproducible structural associations with the behaviour of the existing RF model rather than causal determinants of activation barriers.

---

## Data Availability and Reproducibility

The complete 100-ligand raw feature dataset is not included in the GitHub repository because the 100 parquet files occupy approximately 4 GB.

The repository instead contains:

- analysis scripts;
- completed result tables;
- nested-CV outputs;
- permutation-test outputs;
- feature-stability outputs;
- figure source data;
- dissertation figures;
- compact example data.

The original analysis scripts use repository-relative paths rather than user-specific absolute paths.

For full reproduction of the 100-ligand analysis, the complete raw parquet files should be restored under:

`thesis_features_analysis/data/raw/batch0100/features/`

The corresponding barrier file is expected at:

`thesis_features_analysis/data/raw/batch0100/barriers/ligand_summaries_20260223_2245.txt`

No script path changes are required once the complete data are restored.

---

## Environment

The local structural analyses were run in the conda environment:

`conda activate thesis_features`

Main Python dependencies include:

- NumPy;
- pandas;
- SciPy;
- Matplotlib;
- scikit-learn;
- PyArrow.

PyMOL was used separately for structural-context visualisation.

---

## Example Data

A compact example is provided at:

`sample_data/thesis_features_example/example_feature_values.tsv`

It contains six example ligands and the three stable protein–protein features.

This sample is provided for data-format inspection only.

It is not a replacement for the full 100-ligand dataset and should not be used to rerun the formal nested-CV, feature-selection stability, or label-permutation analyses.

---

## Recommended Reading Order

For a first-time reader:

1. `README.md`
2. `qmmm_validation_47_complete_20260801/README.md`
3. `thesis_features_analysis/README.md`
4. `qmmm47_prefixed_feature_check/README.md`

The detailed subproject READMEs contain the complete workflow, numerical results, scripts, and provenance information.

---

## Important Provenance Note

The original 100-ligand classifier dataset and the later 47-structure improved-QM/MM dataset represent different scientific tasks and must not be mixed.

The 100-ligand analysis characterises structural patterns associated with the existing RF model.

The 47-structure analyses evaluate transfer to an improved-QM/MM activation-barrier target.

This separation is maintained throughout the repository.
# Sample Data

## Purpose

This directory contains compact example inputs for inspection and demonstration
of the MSc dissertation code repository.

The complete February 2026 100-ligand structural-feature dataset contains
100 parquet files and occupies approximately 4 GB in total. The full raw
dataset is therefore not duplicated in this GitHub repository.

The sample files provided here are intended to:

- illustrate the expected input format;
- allow inspection of representative structural-feature values;
- demonstrate selected data-loading steps;
- provide a lightweight example without changing the original dissertation
  analysis inputs or results.

The sample files are not intended to reproduce the complete nested
cross-validation, label-permutation, or feature-selection analyses.

The completed numerical outputs used in the dissertation are retained in the
corresponding subproject `results/` directories.

---

## Contents

### `thesis_features_example/`

Compact example data extracted from the February 2026 100-ligand
structural-feature dataset.

The example contains only a small number of ligands and the three stable
protein–protein features carried forward for structural interpretation.

The complete raw parquet dataset is not included.

---

## Sample Generation

The structural-feature example was generated using:

`create_thesis_features_example.py`

Run from the repository root with:

`python sample_data/create_thesis_features_example.py`

The script reads the existing parquet files and writes a new compact TSV file
under `sample_data/`.

It does not modify the original parquet files or rerun any scientific
analysis.

---

## Analysis Environment

The local dissertation analyses were run under WSL2 Ubuntu using the Conda
environment:

`thesis_features`

The environment was activated with:

`conda activate thesis_features`

Main Python packages used across the structural-analysis workflows include:

- Python
- NumPy
- pandas
- SciPy
- scikit-learn
- Matplotlib
- PyArrow

PyMOL was used separately for structural-context visualisation.

A repository-level Conda environment description is provided in:

`environment_thesis_features.yml`

---

## Important Reproducibility Note

The sample-data directory is separate from the original dissertation analysis
directories.

No existing analysis script has been redirected to the sample files.

Creating the sample does not:

- rerun nested cross-validation;
- rerun feature selection;
- rerun label permutations;
- recalculate classifier performance;
- recalculate stable-feature statistics;
- modify completed result tables;
- overwrite original raw data.

The sample exists only for repository inspection and demonstration.
# Structural-Feature Example

## Overview

This directory contains a compact example extracted from the February 2026
100-ligand structural-feature dataset used in the dissertation.

The original dataset contains 100 ligand-level parquet files and occupies
approximately 4 GB in total.

For GitHub inspection, only six ligand records and three previously identified
stable protein–protein features are included here.

---

## Example File

`example_feature_values.tsv`

The file contains:

- 6 example ligands;
- ligand identifiers;
- three stable structural descriptors.

The columns are:

- `ligand_id`
- `PRO62-PRO1279`
- `PRO1275-PRO1276`
- `PRO90-PRO92`

The biological identities are:

| Internal feature | Biological identity |
|---|---|
| `PRO62-PRO1279` | GLU62(RAS)–SER1279(NF1) |
| `PRO1275-PRO1276` | PHE1275(NF1)–ARG1276(NF1) |
| `PRO90-PRO92` | PHE90(RAS)–ASP92(RAS) |

These features are defined as minimum all-atom inter-residue distances.

They are not C-alpha–C-alpha distances.

---

## Provenance

The example values were read directly from the existing February 2026 parquet
files located in the original dissertation project.

The sample-generation script selected the first six parquet files after
deterministic filename sorting and read only the three specified feature
columns.

The original parquet files were read only.

They were not modified.

---

## Generation Script

The sample was generated using:

`sample_data/create_thesis_features_example.py`

Run from the repository root with:

`python sample_data/create_thesis_features_example.py`

The generated output is:

`sample_data/thesis_features_example/example_feature_values.tsv`

---

## Scientific Analysis

This example is not a reduced version of the formal classifier.

It is not intended to reproduce:

- the 83-sample nested-CV classifier;
- the 25 outer-fold performance estimates;
- the 30 label permutations;
- feature-selection stability;
- dissertation-level statistical conclusions.

The formal scientific results remain in:

`thesis_features_analysis/results/`

The example is provided only to illustrate the structural-feature data format
and demonstrate safe reading of representative input values.

---

## Environment

The sample-generation script was run in the Conda environment:

`thesis_features`

Activated using:

`conda activate thesis_features`

The script requires:

- Python
- pandas
- PyArrow
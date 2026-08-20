# Structural Feature Analysis for the 100-Ligand Dataset

## Overview

This directory contains the structural-feature analysis used in the MSc dissertation project.

The workflow analyses structural patterns associated with the behaviour of an existing random-forest (RF) model in the original February 2026 100-ligand dataset.

The analysis includes:

- construction of the structural feature/barrier dataset;
- exploratory low/high feature analysis;
- definition of formal structural feature sets;
- repeated nested cross-validation;
- dummy-classifier evaluation;
- label-permutation testing;
- feature-selection stability analysis;
- quantitative interpretation of stable structural features;
- PyMOL structural-context mapping;
- dissertation-final figure regeneration.

The classifier analysis is based only on the original February 2026 100-ligand dataset.

The later August 2026 electrostatic workflow is a separate analysis and is not mixed with the classifier dataset.

The separate improved-QM/MM 47-structure validation and pre-fixed feature-transfer analyses are also maintained separately from this directory.

---

# Scientific Scope

The low/high labels used in this analysis are derived from activation barriers predicted by the existing RF workflow.

They are therefore:

- not experimentally measured activation barriers;
- not improved-QM/MM activation barriers;
- not independent ground-truth labels.

The complete 100-ligand dataset contains:

- 100 ligands;
- 26 low-barrier samples;
- 17 intermediate samples;
- 57 high-barrier samples.

The 17 intermediate systems are excluded from the formal binary classifier.

The final classification dataset therefore contains:

- 26 low samples;
- 57 high samples;
- **n = 83**.

The purpose of the classifier is therefore to identify structural patterns associated with the behaviour of the existing RF model.

The identified structural features should be interpreted as reproducible structural associations with old-model behaviour rather than as causal determinants of measured or improved-QM/MM activation barriers.

---

# Dataset Provenance

## February 2026 100-Ligand Dataset

This dataset is used for:

- structural feature assembly;
- low/high structural classification;
- nested cross-validation;
- dummy baseline;
- label-permutation testing;
- feature-selection stability;
- quantitative low/high comparison of stable features;
- PyMOL structural interpretation.

All Scripts 01–12 in this directory refer to this February dataset unless otherwise stated.

## Improved-QM/MM Dataset

The separate improved-QM/MM dataset contains 47 complete structures with improved-QM/MM activation barriers.

It is used elsewhere for:

- direct validation of the existing RF model;
- exploratory transfer analysis of the three pre-fixed stable features.

The improved-QM/MM dataset is **not** used to train the structural classifier or to search for new stable features in this directory.

---

# Analysis Workflow

## 01 — Build Master Feature/Barrier Table

Script:

`01_build_feature_barrier_table.py`

The 100 ligand feature files are matched one-to-one with the existing RF-predicted barriers.

The resulting master dataset contains:

- 100 ligands;
- 83,276 structural feature columns;
- no unmatched ligand IDs.

This master table forms the basis of the later structural analyses.

---

## 02 — Exploratory Low/High Feature Analysis

Script:

`02_rank_all_low_high_features.py`

All 83,276 structural distance features are analysed individually using descriptive and exploratory statistics.

These include:

- group means;
- group standard deviations;
- Cohen's d;
- Welch's t-test;
- Mann–Whitney test;
- Pearson correlation;
- global FDR correction.

This analysis is exploratory only.

It is **not** used to pre-select candidate features for the formal classifier.

One prominent protein–protein feature is:

`PRO62-PRO1279`

with approximately:

- high-minus-low distance = `-0.440 Å`;
- Cohen's d ≈ `1.28` in magnitude.

This feature is later independently recovered as the most stable protein–protein feature in the nested-CV analysis.

---

## 03 — Define Formal Structural Feature Sets

Script:

`03_build_model_feature_sets.py`

The existing RF model uses 4,086 structural distance features.

All 4,086 required features are recovered in the 100-ligand dataset.

The formal feature sets are defined without using the low/high labels.

| Internal feature-set ID | Dissertation terminology | Number of features |
|---|---|---:|
| `residue_ligand` | Protein–ligand | 21 |
| `residue_cofactor` | Protein–GTP | 83 |
| `residue_residue` | Protein–protein | 3,982 |
| `combined_all` | Combined | 4,086 |

The combined set is the exact union of the other three sets.

The formal candidate feature spaces are therefore defined independently of the observed low/high group differences.

---

## 04 — Exploratory Structural-Feature Figures

Script:

`04_plot_exploratory_structural_features.py`

Exploratory structural figures are generated for model-listed features.

These include low/high distributions and effect-size visualisations.

These figures are used only for exploratory structural interpretation.

They are not used to define the formal nested-CV candidate feature spaces.

---

# Formal Classification Analysis

## 05 — Repeated Nested Cross-Validation

Script:

`05_run_four_set_nested_cv.py`

A class-balanced logistic-regression classifier is evaluated using repeated nested cross-validation.

### Cross-Validation Design

- outer CV: 5 folds × 5 repeats;
- total outer evaluations: 25;
- inner CV: 5 folds;
- inner-CV optimisation of `SelectKBest` k;
- inner-CV optimisation of logistic-regression C.

All data-dependent preprocessing remains inside the cross-validation pipeline:

`median imputation → variance filtering → SelectKBest → standardisation → class-balanced logistic regression`

Balanced accuracy is used as the primary performance measure.

The prior dummy classifier is evaluated using the same outer splits.

### Main Nested-CV Results

| Feature set | Balanced accuracy | ROC AUC | MCC |
|---|---:|---:|---:|
| Protein–ligand | 0.583 ± 0.118 | 0.614 ± 0.163 | 0.166 ± 0.236 |
| Protein–GTP | 0.594 ± 0.122 | 0.618 ± 0.113 | 0.177 ± 0.231 |
| **Protein–protein** | **0.643 ± 0.119** | **0.712 ± 0.132** | **0.289 ± 0.235** |
| Combined | 0.624 ± 0.145 | 0.698 ± 0.140 | 0.244 ± 0.279 |
| Dummy prior | 0.500 | 0.500 | 0.000 |

The protein–protein feature set gives the highest observed mean balanced accuracy.

The classifier performance is interpreted as a **modest structural separation**, not as a highly accurate classifier.

---

## 06 — Feature-Selection Stability

Script:

`06_analyse_feature_stability.py`

Feature-selection stability is calculated from the features already selected in the 25 outer-training models.

No classifier is refitted during this analysis.

A small reproducible protein–protein feature core is identified.

| Internal feature | Biological identity | Protein–protein frequency | Combined frequency |
|---|---|---:|---:|
| `PRO62-PRO1279` | GLU62(RAS)–SER1279(NF1) | 25/25 = 1.00 | 25/25 = 1.00 |
| `PRO1275-PRO1276` | PHE1275(NF1)–ARG1276(NF1) | 22/25 = 0.88 | 21/25 = 0.84 |
| `PRO90-PRO92` | PHE90(RAS)–ASP92(RAS) | 21/25 = 0.84 | 20/25 = 0.80 |

Only three of the 3,982 protein–protein candidate features reach a selection frequency of at least 0.80.

These three features are therefore carried forward for quantitative and structural interpretation.

### Structural Identities

- GLU62(RAS)–SER1279(NF1): inter-protein RAS–NF1 feature;
- PHE90(RAS)–ASP92(RAS): local intra-RAS feature;
- PHE1275(NF1)–ARG1276(NF1): local intra-NF1 feature.

---

## 06b — Dissertation-Final Feature-Stability Figure

Script:

`06_plot_feature_stability_final.py`

This script is a **presentation-only final-pass script**.

It reads the already completed figure-source table:

`results/batch0100/stability/figure_source_data/06_residue_residue_stability_source.tsv`

It does not:

- refit any classifier;
- rerun nested cross-validation;
- rerun feature selection;
- recalculate selection frequencies;
- recalculate coefficient-direction consistency;
- recalculate Jaccard similarity;
- modify completed stability TSV files;
- overwrite the original stability figure.

The scientific values displayed in the final figure are unchanged.

The script changes only figure presentation, including:

- larger figure dimensions;
- larger axis labels;
- larger tick labels;
- larger numerical annotations;
- thesis-consistent terminology.

Final dissertation-formatted outputs:

- `results/batch0100/stability/figures/top15_residue_residue_selection_frequency_final.png`
- `results/batch0100/stability/figures/top15_residue_residue_selection_frequency_final.svg`

Run independently with:


python scripts/06_plot_feature_stability_final.py






## 07 — Label-Permutation Test

Script:

`07_run_label_permutation_test.py`

Thirty complete label permutations were performed.

For each permutation, the entire nested-CV procedure was rerun, including:

- preprocessing;
- feature selection;
- hyperparameter optimisation;
- outer-fold evaluation.

### Main Permutation Results

| Feature set | Observed BA | Null mean | Null maximum | Null ≥ observed | Empirical p |
|---|---:|---:|---:|---:|---:|
| Protein–ligand | 0.583 | 0.507 | 0.623 | 3/30 | 0.129 |
| Protein–GTP | 0.594 | 0.492 | 0.582 | 0/30 | 0.032 |
| **Protein–protein** | **0.643** | **0.485** | **0.570** | **0/30** | **0.032** |
| Combined | 0.624 | 0.488 | 0.564 | 0/30 | 0.032 |

For the protein–protein classifier, none of the 30 permuted-label analyses reached the observed balanced accuracy.

With 30 permutations, the finite-sample empirical value for zero exceedances is:

`1 / 31 = 0.032`

This value has limited resolution and should be interpreted as a summary of the sampled null distribution rather than as a high-precision p-value.

---

## 08 — Classifier Result Figures

Script:

`08_plot_classifier_results.py`

This script packages and visualises the completed classifier and permutation results.

It does not refit any model.

### Nested-CV Performance

Main original figure:

`08A_nested_cv_balanced_accuracy`

This figure shows:

- four structural feature sets;
- 25 outer-fold scores per feature set;
- observed mean balanced accuracy;
- dummy baseline.

### Label Permutation

Main original figure:

`08B_label_permutation_balanced_accuracy`

This figure shows:

- 30 complete permutation scores per feature set;
- observed nested-CV performance;
- permutation exceedance counts.

Cross-analysis consistency checks confirmed that the Script 05 nested-CV means and Script 07 observed values agree.

---

## 08b — Dissertation-Final Classifier Figures

Script:

`08_plot_classifier_results_final.py`

This script is a **presentation-only final-pass script**.

It reads the completed nested-CV and permutation outputs and does not rerun any scientific analysis.

It does not:

- refit classifiers;
- rerun nested cross-validation;
- rerun feature selection;
- retune hyperparameters;
- rerun label permutations;
- modify completed TSV files;
- overwrite original classifier figures.

The displayed terminology is changed only for dissertation presentation.

| Internal ID | Final displayed label |
|---|---|
| `residue_ligand` | Protein–ligand |
| `residue_cofactor` | Protein–GTP |
| `residue_residue` | Protein–protein |
| `combined_all` | Combined |

Final outputs:

- `08A_nested_cv_balanced_accuracy_final.png`
- `08A_nested_cv_balanced_accuracy_final.svg`
- `08B_label_permutation_balanced_accuracy_final.png`
- `08B_label_permutation_balanced_accuracy_final.svg`

Run independently with:

`python scripts/08_plot_classifier_results_final.py`

---

# Quantitative Structural Interpretation

## 09 — Stable-Feature Low/High Distributions

Script:

`09_analyse_stable_feature_distributions.py`

This script quantifies the original February low/high distributions of the three stable protein–protein features.

The same 83 clear classifier samples are used:

- 26 low;
- 57 high.

The three features were already fixed by the nested-CV stability analysis before this step.

They were therefore not selected again using the observed low/high distance differences.

### Main Results

| Feature | Low mean (Å) | High mean (Å) | Low–high shift (Å) | Cohen's d |
|---|---:|---:|---:|---:|
| GLU62(RAS)–SER1279(NF1) | 7.662 | 7.221 | +0.440 | 1.28 |
| PHE1275(NF1)–ARG1276(NF1) | 1.346 | 1.344 | +0.002 | 1.00 |
| PHE90(RAS)–ASP92(RAS) | 3.189 | 3.075 | +0.114 | 0.98 |

All three raw distances are larger in the old-RF-defined low group.

The largest absolute distance difference is observed for:

`GLU62(RAS)–SER1279(NF1)`

The other two features show smaller absolute structural changes.

### Original Figure

`results/batch0100/structural_interpretation/figures/09_stable_feature_low_high_distributions.png`

---

## 09b — Thesis-Ready Stable-Feature Summary

Script:

`09b_build_stable_feature_summary.py`

This script combines the quantitative low/high results with the feature-selection stability results.

Main output:

`results/batch0100/structural_interpretation/tables/09b_stable_feature_thesis_summary.tsv`

This table links:

- biological residue identity;
- internal feature ID;
- selection frequency;
- coefficient-direction consistency;
- low/high distance statistics.

---

## 09c — Dissertation-Final Stable-Feature Distribution Figure

Script:

`09_plot_stable_feature_distributions_final.py`

This script is a **presentation-only final-pass script**.

It reads the completed Script 09 outputs:

- `results/batch0100/structural_interpretation/tables/09_stable_feature_values.tsv`
- `results/batch0100/structural_interpretation/tables/09_stable_feature_group_statistics.tsv`

It does not:

- rerun stable-feature analysis;
- rerun nested cross-validation;
- re-select features;
- recalculate group means;
- recalculate standard deviations;
- recalculate Cohen's d;
- recalculate Welch tests;
- recalculate Mann–Whitney tests;
- modify completed Script 09 TSV files;
- overwrite the original figure.

The scientific values shown in the final figure are unchanged.

Presentation changes include:

- vertical panel layout;
- larger figure dimensions;
- larger axis labels;
- larger tick labels;
- larger annotations;
- prominent biological residue identities;
- retention of internal `PRO...` identifiers as secondary labels;
- removal of `batch0100` from the formal figure title;
- dissertation-consistent protein–protein terminology.

Final outputs:

- `results/batch0100/structural_interpretation/figures/09_stable_feature_low_high_distributions_final.png`
- `results/batch0100/structural_interpretation/figures/09_stable_feature_low_high_distributions_final.svg`

Run independently with:

`python scripts/09_plot_stable_feature_distributions_final.py`

---

# PyMOL Structural Interpretation

## 10 — Select Potential Representative Structures

Script:

`10_select_pymol_representatives.py`

Potential representative low/high molecules were selected objectively as those closest to the corresponding February group medians.

The original February PDB coordinates for these specific representative molecules were no longer retained.

Therefore:

- the representative IDs were recorded;
- the missing February PDBs were not reconstructed;
- August rerun structures were not substituted;
- the representatives were not used for low/high PyMOL rendering.

This preserves the provenance of the February classifier dataset.

---

## 11–12 — PyMOL Structural-Context Mapping

Original scripts:

- `11_pymol_stable_feature_overview_final.pml`
- `12_pymol_stable_feature_closeups_final.pml`

These scripts generate reproducible PyMOL structural-context figures.

A retained February-era 100-ligand test-pocket structure with the same RAS–NF1 residue numbering is used for structural illustration.

This reference structure is:

- not one of the 83 classifier samples;
- not a representative low/high structure;
- not used to measure the quantitative low/high differences.

The quantitative results come from the February feature dataset analysed in Script 09.

### Structural-Context Outputs

The PyMOL workflow generates:

- a whole-complex structural overview;
- a GLU62(RAS)–SER1279(NF1) close-up;
- a PHE90(RAS)–ASP92(RAS) close-up;
- a PHE1275(NF1)–ARG1276(NF1) close-up.

### Feature Interpretation

#### GLU62(RAS)–SER1279(NF1)

An inter-protein RAS–NF1 structural descriptor.

This feature shows:

- the highest selection stability;
- the largest absolute low/high distance difference.

#### PHE90(RAS)–ASP92(RAS)

A local intra-RAS structural descriptor.

It shows a smaller absolute low/high structural shift than GLU62(RAS)–SER1279(NF1).

#### PHE1275(NF1)–ARG1276(NF1)

A local intra-NF1 descriptor involving consecutive residues.

Its absolute low/high distance shift is very small.

It is therefore interpreted conservatively as a highly constrained local geometric descriptor.

### Distance Definition

The classifier features are defined as:

**minimum all-atom inter-residue distances**

They are not Cα–Cα distances.

Any dashed Cα–Cα lines shown in the PyMOL overview are schematic localisation guides only.

They are not the numerical classifier features.

---

## 11–12b — No-Label PyMOL Figures for Dissertation Formatting

Final presentation scripts:

- `11_pymol_stable_feature_overview_nolabel.pml`
- `12_pymol_stable_feature_closeups_nolabel.pml`

These scripts were added during the final dissertation figure-formatting pass.

They retain:

- the same reference structure;
- the same residue selections;
- the same protein cartoons;
- the same residue colours;
- the same ligand/GTP context;
- the same structural interpretation.

They remove PyMOL's automatic residue-name text labels.

This allows larger manual arrows and labels to be added during final dissertation figure preparation.

These scripts do not:

- rerun any classifier;
- rerun feature selection;
- modify any feature table;
- recalculate structural statistics;
- change the stable-feature identities;
- overwrite the earlier `_clean` images or sessions.

New image outputs include:

- `PyMOL_01_stable_feature_overview_nolabel.png`
- `PyMOL_02_GLU62_SER1279_closeup_nolabel.png`
- `PyMOL_03_PHE90_ASP92_closeup_nolabel.png`
- `PyMOL_S01_PHE1275_ARG1276_closeup_nolabel.png`

Corresponding PyMOL session files are also saved with `_nolabel` filenames.

These files are presentation-only variants.

The scientific content is unchanged.

---

# Main Scientific Conclusions

The structural-distance analysis gives four main conclusions.

## 1. Protein–Protein Features Show the Strongest Structural Separation

The protein–protein feature set gives the highest observed nested-CV balanced accuracy:

`0.643 ± 0.119`

This is higher than:

- Protein–ligand: `0.583 ± 0.118`;
- Protein–GTP: `0.594 ± 0.122`;
- Combined: `0.624 ± 0.145`.

---

## 2. The Structural Signal Is Reproducible but Modest

The protein–protein classifier performs above the dummy baseline.

None of the 30 complete label permutations reaches the observed protein–protein balanced accuracy.

However, classification performance remains variable across outer folds.

The classifier should therefore be interpreted as showing modest structural separation rather than high predictive accuracy.

---

## 3. Three Protein–Protein Features Form a Stable Core

The three most stable descriptors are:

1. GLU62(RAS)–SER1279(NF1);
2. PHE1275(NF1)–ARG1276(NF1);
3. PHE90(RAS)–ASP92(RAS).

GLU62(RAS)–SER1279(NF1) is selected in all 25 protein–protein outer models.

---

## 4. The Stable Features Represent Different Structural Contexts

The three stable features do not represent one common type of geometry.

Instead:

- GLU62(RAS)–SER1279(NF1) represents inter-protein RAS–NF1 geometry;
- PHE90(RAS)–ASP92(RAS) represents local intra-RAS geometry;
- PHE1275(NF1)–ARG1276(NF1) represents local intra-NF1 geometry.

The largest absolute low/high structural difference is observed for GLU62(RAS)–SER1279(NF1).

---

# Recommended Numerical Outputs

## Nested Cross-Validation

`results/batch0100/nested_cv/final/nested_cv_summary.tsv`

## Label Permutation

`results/batch0100/permutation/final_perm0001_0030/permutation_test_summary.tsv`

## Classifier Summary

`results/batch0100/classifier_figures/tables/08_classifier_result_summary.tsv`

## Stable Feature Candidates

`results/batch0100/stability/tables/candidate_feature_stability.tsv`

## Stable-Feature Statistics

`results/batch0100/structural_interpretation/tables/09_stable_feature_group_statistics.tsv`

## Thesis-Ready Stable-Feature Summary

`results/batch0100/structural_interpretation/tables/09b_stable_feature_thesis_summary.tsv`

---

# Recommended Figures

## Nested-CV Performance

Original:

`results/batch0100/classifier_figures/figures/08A_nested_cv_balanced_accuracy.png`

Dissertation-final:

`results/batch0100/classifier_figures/figures/08A_nested_cv_balanced_accuracy_final.png`

## Label-Permutation Test

Original:

`results/batch0100/classifier_figures/figures/08B_label_permutation_balanced_accuracy.png`

Dissertation-final:

`results/batch0100/classifier_figures/figures/08B_label_permutation_balanced_accuracy_final.png`

## Feature-Selection Stability

Original:

`results/batch0100/stability/figures/top15_residue_residue_selection_frequency.png`

Dissertation-final:

`results/batch0100/stability/figures/top15_residue_residue_selection_frequency_final.png`

## Stable-Feature Distributions

Original:

`results/batch0100/structural_interpretation/figures/09_stable_feature_low_high_distributions.png`

Dissertation-final:

`results/batch0100/structural_interpretation/figures/09_stable_feature_low_high_distributions_final.png`

## PyMOL Structural Context

No-label dissertation versions:

- `results/batch0100/structural_interpretation/pymol/images/PyMOL_01_stable_feature_overview_nolabel.png`
- `results/batch0100/structural_interpretation/pymol/images/PyMOL_02_GLU62_SER1279_closeup_nolabel.png`
- `results/batch0100/structural_interpretation/pymol/images/PyMOL_03_PHE90_ASP92_closeup_nolabel.png`
- `results/batch0100/structural_interpretation/pymol/images/PyMOL_S01_PHE1275_ARG1276_closeup_nolabel.png`

---

# Reproducibility Notes

The dissertation-final plotting scripts are intentionally separated from the scientific-analysis scripts.

The following scripts change figure presentation only:

- `06_plot_feature_stability_final.py`
- `08_plot_classifier_results_final.py`
- `09_plot_stable_feature_distributions_final.py`

The following PyMOL scripts create no-label presentation variants:

- `11_pymol_stable_feature_overview_nolabel.pml`
- `12_pymol_stable_feature_closeups_nolabel.pml`

These scripts do not change the completed numerical analysis.

Original figures and original numerical outputs are retained.

---

# Electrostatic Analysis

The electrostatic-feature workflow is developed separately on Myriad.

It is not part of the completed structural-distance classifier analysis documented in this directory.

The electrostatic workflow should not be mixed with the February 100-ligand classifier dataset.

In the dissertation, the electrostatic work is treated as future work rather than as a completed Results analysis.

---

# Key Limitations

The main limitations of the structural-distance analysis are:

- only 83 clear low/high samples are used for formal classification;
- the low/high labels are derived from old RF-predicted barriers;
- 17 intermediate systems are excluded;
- only 30 label permutations are used, limiting empirical p-value resolution;
- the structural representation is based on minimum inter-residue distance descriptors;
- stable feature selection does not establish causality;
- stable feature selection in the old-RF-defined task does not establish transferability to improved-QM/MM barriers;
- the PyMOL structural-context reference is not one of the 83 classifier samples and is used for localisation only.

---

# Repository Structure

A simplified directory structure is:

`thesis_features_analysis/`

- `data/`
  - `raw/`

- `scripts/`
  - `01_build_feature_barrier_table.py`
  - `02_rank_all_low_high_features.py`
  - `03_build_model_feature_sets.py`
  - `04_plot_exploratory_structural_features.py`
  - `05_run_four_set_nested_cv.py`
  - `06_analyse_feature_stability.py`
  - `06_plot_feature_stability_final.py`
  - `07_run_label_permutation_test.py`
  - `08_plot_classifier_results.py`
  - `08_plot_classifier_results_final.py`
  - `09_analyse_stable_feature_distributions.py`
  - `09_plot_stable_feature_distributions_final.py`
  - `09b_build_stable_feature_summary.py`
  - `10_select_pymol_representatives.py`
  - `11_pymol_stable_feature_overview_final.pml`
  - `11_pymol_stable_feature_overview_nolabel.pml`
  - `12_pymol_stable_feature_closeups_final.pml`
  - `12_pymol_stable_feature_closeups_nolabel.pml`

- `results/`
  - `batch0100/`
    - `classifier_figures/`
    - `feature_sets/`
    - `figure_source_data/`
    - `figures/`
    - `nested_cv/`
    - `permutation/`
    - `stability/`
    - `structural_interpretation/`

- `README.md`


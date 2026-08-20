# batch0100 analysis status

Last updated: 2026-08-06

## Completed and frozen

### Script 01

- File: `scripts/01_build_feature_barrier_table.py`
- Status: COMPLETE / FROZEN
- Role: construction of the batch0100 master feature–barrier table
- Input feature files: 100 ligand parquet files
- Master structural features: 83,276
- Samples: 100
- Groups:
  - low: 26
  - middle: 17
  - high: 57
- Barrier source: old RF-predicted barriers
- Main output:
  - `results/batch0100/tables/features_with_barriers.parquet`

### Script 02

- File: `scripts/02_rank_all_low_high_features.py`
- Status: COMPLETE / FROZEN
- Role: exploratory univariate low-versus-high feature analysis
- Samples used:
  - low: 26
  - high: 57
  - total: 83
- Excluded samples:
  - middle: 17
- Analysed structural features: 83,276
- Quality-passing features: 62,410
- Statistical outputs:
  - descriptive group statistics
  - Cohen's d
  - Welch's t-test
  - Mann–Whitney U test
  - Pearson correlation with continuous old RF-predicted barriers
  - Benjamini–Hochberg FDR correction
- Interpretation:
  - exploratory analysis of old-model behaviour
  - not a formal classifier
  - not used to preselect features before nested CV

### Script 03

- File: `scripts/03_build_model_feature_sets.py`
- Status: COMPLETE / FROZEN
- Role: label-independent construction of the formal distance-feature sets
- Unique original RF model-listed distance features: 4,086
- Available in batch0100: 4,086
- Missing from batch0100: 0
- Duplicate model feature names: 0
- Formal feature sets:
  - residue_ligand: 21
  - residue_cofactor: 83
  - residue_residue: 3,982
  - combined_all: 4,086
- Integrity:
  - `21 + 83 + 3,982 = 4,086`
  - `combined_all` is the exact union of the three structural sets
- Leakage controls:
  - selection uses labels: False
  - selection uses script-02 ranking: False
  - global preprocessing applied: False
- Intended use:
  - candidate feature manifests for formal nested cross-validation

## Frozen archive

- Directory:
  - `archive/frozen_batch0100_distance_pipeline_20260806/`
- Integrity file:
  - `archive/frozen_batch0100_distance_pipeline_20260806/SHA256SUMS`
- SHA256 verification status:
  - all tracked scripts, summaries and feature manifests verified as OK

## In progress / pending

### Existing distance-analysis workflow

- Review and reorganise script 04
- Review and reorganise script 05
- Develop the formal four-set nested-CV script
- Run outer 5 folds × 10 repeats
- Run inner 5-fold hyperparameter selection
- Include class-balanced logistic regression
- Include dummy-classifier baseline
- Save outer-fold predictions and selected features
- Perform feature-selection stability analysis
- Perform label-permutation testing
- Prepare publication-quality figures
- Complete PyMOL structural interpretation

### Electrostatic workflow

- Electrostatic feature calculation on Myriad: IN PROGRESS
- Planned descriptors:
  - electrostatic potential at PG
  - electrostatic potential at O3B
  - electrostatic potential at the lytic-water oxygen
  - electrostatic potential at Mg2+
  - projected electric fields along breaking/forming bonds
  - Arg-finger–phosphate Coulomb terms
  - ligand–triphosphate Coulomb terms
- Electrostatic features will be analysed as a separate physically motivated feature set after generation and QC.

### Improved QM/MM structure workflow

- Corrected validation using complete improved QM/MM structures: COMPLETE
- Old-model validation role: COMPLETE
- Exploratory check of preselected batch0100 features: PENDING
- No new model will be trained on the small improved-QM/MM structure set.
- Only a small number of features fixed in advance from batch0100 will be examined.

## Scientific interpretation

The batch0100 labels are derived from old RF-predicted barriers.

Therefore, the batch0100 classifier characterises structural patterns
associated with the behaviour of the old model. It does not predict
measured barriers or directly establish determinants of the improved
QM/MM activation barriers.

The intermediate group is excluded from the formal low/high
classification. Identified structural features will be interpreted as
associations rather than causal determinants.

## Version policy

Scripts 01–03 and their frozen archive copies must not be edited.

If a genuine correction is required:

1. retain the current frozen version;
2. create a new version of the affected script;
3. document the reason for the change;
4. rerun every affected downstream analysis;
5. generate a new frozen archive and SHA256 record.

GitHub repository preparation will be completed later, after the main
experimental analyses and code organisation are finished.

# Stable residue-residue features: thesis-ready summary

All quantitative classifier-related values in this table come from the original February 2026 batch0100 dataset. The three features were fixed from nested-CV feature-selection stability before this descriptive summary was constructed.

| Feature | Residue pair | Structural context | RR stability | Combined stability | Low mean (A) | High mean (A) | High-low mean (A) | Cohen's d (low-high) | RR direction consistency | Interpretation note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| PRO62-PRO1279 | GLU62–SER1279 | cross-chain A-B residue pair | 1.00 | 1.00 | 7.662 | 7.221 | -0.440 | +1.28 | 1.000 | Highest-stability feature; largest raw low/high mean shift; priority candidate for RAS-NF1 structural-context interpretation. |
| PRO1275-PRO1276 | PHE1275–ARG1276 | consecutive-residue local backbone geometry | 0.88 | 0.84 | 1.346 | 1.344 | -0.002 | +1.00 | 0.955 | Large standardized effect but extremely small absolute distance shift; an August representative structure suggests a constrained peptide-backbone C-N geometry, so avoid describing this as a long-range interface contact. |
| PRO90-PRO92 | PHE90–ASP92 | local intrachain residue pair | 0.84 | 0.80 | 3.189 | 3.075 | -0.114 | +0.98 | 0.952 | Stable intrachain feature with a moderate absolute low/high distance shift; suitable as a secondary structural-context feature. |

## Direction conventions

- `High-low mean (A)` is the high-group mean minus the low-group mean. Negative values therefore mean that the raw distance is larger in the low group.
- Positive `Cohen's d (low-high)` means that the raw distance is larger in the low group.
- `RR direction consistency` is the dominant logistic-regression coefficient-direction fraction among outer folds in which the feature was selected.

## Interpretation boundary

The low/high comparisons are descriptive/post-hoc interpretations of features already identified by nested-CV stability. They should not be treated as independent confirmatory hypothesis tests. The primary evidence for classifier signal remains nested cross-validation, the dummy baseline, and the complete label-permutation test.

These features characterize structural associations with old-RF-derived low/high labels and are not established causal determinants of catalytic barriers or improved-QM/MM barriers.

Residue identities were established using the original feature-generation logic together with a retained August rerun structure. The original February final PDB structures were cleaned up, so exact February atom-pair identities cannot be recovered. August distances must therefore not be substituted for February classifier distances.

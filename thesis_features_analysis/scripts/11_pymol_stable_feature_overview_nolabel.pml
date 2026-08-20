reinitialize

# ============================================================
# Stable residue-residue structural-context overview
# NO AUTOMATIC RESIDUE TEXT LABELS
#
# Reference structure:
# PV-009860881948_post_prod_mini_final_snap1.pdb
#
# IMPORTANT:
# This retained February-era batch0100 test-pocket structure is
# used only for structural-context illustration.
# It is NOT one of the 83 February classifier samples and is NOT
# a representative low/high classifier structure.
#
# Quantitative classifier evidence comes from the February parquet
# dataset analysed in Scripts 05-09b.
#
# The classifier features are minimum all-atom inter-residue
# distances. The CA-CA dashed lines below are schematic visual
# guides only and are NOT the quantitative classifier distances.
#
# This script changes VISUAL PRESENTATION ONLY.
# No analysis or numerical result is recalculated.
# ============================================================


# ============================================================
# Load reference structure
# ============================================================

load data/raw/batch0100/structures/PV-009860881948_post_prod_mini_final_snap1.pdb, complex

hide everything


# ============================================================
# Protein
# ============================================================

show cartoon, complex and polymer.protein

# Chain A = RAS
# Chain B = NF1
color lightblue, complex and polymer.protein and chain A
color wheat, complex and polymer.protein and chain B

set cartoon_transparency, 0.12


# ============================================================
# Ligand and GTP
# ============================================================

select ligand, complex and resn LIG
select gtp, complex and resn GTP

show sticks, ligand
show sticks, gtp

color cyan, ligand
color orange, gtp

set stick_radius, 0.20, ligand
set stick_radius, 0.18, gtp


# ============================================================
# Stable residues
# ============================================================

# Chain A / RAS
select res62, complex and polymer.protein and chain A and resi 62
select res90, complex and polymer.protein and chain A and resi 90
select res92, complex and polymer.protein and chain A and resi 92

# Chain B / NF1
select res1275, complex and polymer.protein and chain B and resi 1275
select res1276, complex and polymer.protein and chain B and resi 1276
select res1279, complex and polymer.protein and chain B and resi 1279


# ============================================================
# QC only
# ============================================================

count_atoms res62
count_atoms res90
count_atoms res92
count_atoms res1275
count_atoms res1276
count_atoms res1279


# ============================================================
# Group the three stable features
# ============================================================

select feature_62_1279, res62 or res1279
select feature_90_92, res90 or res92
select feature_1275_1276, res1275 or res1276


# ============================================================
# Show target residues
# ============================================================

show sticks, feature_62_1279
show sticks, feature_90_92
show sticks, feature_1275_1276


# ============================================================
# Feature colours
#
# Keep these colours consistent with the close-up figures.
# ============================================================

color red, feature_62_1279
color blue, feature_90_92
color yellow, feature_1275_1276

set stick_radius, 0.26, feature_62_1279
set stick_radius, 0.26, feature_90_92
set stick_radius, 0.26, feature_1275_1276


# ============================================================
# Schematic localisation guides
#
# IMPORTANT:
# These are CA-CA lines for visual localisation only.
# They do NOT represent the minimum all-atom classifier feature.
# ============================================================

distance guide_62_1279, res62 and name CA, res1279 and name CA
distance guide_90_92, res90 and name CA, res92 and name CA
distance guide_1275_1276, res1275 and name CA, res1276 and name CA

color red, guide_62_1279
color blue, guide_90_92
color yellow, guide_1275_1276

set dash_width, 3
set dash_gap, 0.30
set dash_length, 0.30


# ============================================================
# Remove ALL automatic PyMOL text labels
# ============================================================

hide labels, all


# ============================================================
# Clean display
# ============================================================

hide everything, resn HOH
hide everything, resn POT
hide everything, resn CLA

bg_color white
set ray_opaque_background, on

set antialias, 2
set ray_trace_mode, 1
set ambient, 0.50
set specular, 0.20
set shininess, 15
set depth_cue, 0

set orthoscopic, on


# ============================================================
# Camera
# ============================================================

orient feature_62_1279 or feature_90_92 or feature_1275_1276 or ligand or gtp

zoom feature_62_1279 or feature_90_92 or feature_1275_1276 or ligand or gtp, 11


# ============================================================
# Export
#
# NEW filenames:
# original clean files are NOT overwritten.
# ============================================================

png results/batch0100/structural_interpretation/pymol/images/PyMOL_01_stable_feature_overview_nolabel.png, width=2200, height=1700, dpi=300, ray=1

save results/batch0100/structural_interpretation/pymol/sessions/PyMOL_01_stable_feature_overview_nolabel.pse
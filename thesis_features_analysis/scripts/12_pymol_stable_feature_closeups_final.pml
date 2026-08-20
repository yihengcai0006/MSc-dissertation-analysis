reinitialize

# ============================================================
# Clean thesis close-ups for stable residue-residue features
#
# Reference structure:
# PV-009860881948_post_prod_mini_final_snap1.pdb
#
# IMPORTANT:
# This retained February-era batch0100 test-pocket structure is
# used only for structural-context illustration.
# It is NOT one of the 83 classifier samples.
#
# Quantitative low/high differences are taken from Script 09 and
# are NOT measured from this displayed reference structure.
#
# No CA-CA guide is shown in the close-ups, because the classifier
# feature is defined as the minimum all-atom inter-residue distance.
# ============================================================


# ============================================================
# Load structure
# ============================================================

load data/raw/batch0100/structures/PV-009860881948_post_prod_mini_final_snap1.pdb, complex

hide everything


# ============================================================
# Protein
# ============================================================

show cartoon, complex and polymer.protein

color lightblue, complex and polymer.protein and chain A
color wheat, complex and polymer.protein and chain B

set cartoon_transparency, 0.18


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
# Residues
# ============================================================

select glu62, complex and polymer.protein and chain A and resi 62
select phe90, complex and polymer.protein and chain A and resi 90
select asp92, complex and polymer.protein and chain A and resi 92

select phe1275, complex and polymer.protein and chain B and resi 1275
select arg1276, complex and polymer.protein and chain B and resi 1276
select ser1279, complex and polymer.protein and chain B and resi 1279


# ============================================================
# General rendering
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

set label_size, 14
set label_color, black
set label_outline_color, white


# ============================================================
# FIGURE 2
# GLU62-SER1279
#
# February quantitative evidence:
# RR stability       = 25/25 = 1.00
# Combined stability = 25/25 = 1.00
# Low mean           = 7.662 A
# High mean          = 7.221 A
# High-low           = -0.440 A
# Cohen's d          = +1.28
#
# Main structural-context feature.
# ============================================================

hide sticks, all
hide labels, all

show cartoon, complex and polymer.protein

# Keep ligand and GTP for overall context
show sticks, ligand
show sticks, gtp

color cyan, ligand
color orange, gtp

# Show only the two target residues
show sticks, glu62 or ser1279
color red, glu62 or ser1279
set stick_radius, 0.32, glu62 or ser1279

label glu62 and name CA, "GLU62"
label ser1279 and name CA, "SER1279"

orient glu62 or ser1279
zoom glu62 or ser1279, 8

png results/batch0100/structural_interpretation/pymol/images/PyMOL_02_GLU62_SER1279_closeup_clean.png, width=2200, height=1700, dpi=300, ray=1


# ============================================================
# FIGURE 3
# PHE90-ASP92
#
# February quantitative evidence:
# RR stability       = 21/25 = 0.84
# Combined stability = 20/25 = 0.80
# Low mean           = 3.189 A
# High mean          = 3.075 A
# High-low           = -0.114 A
# Cohen's d          = +0.98
# ============================================================

hide sticks, all
hide labels, all

show cartoon, complex and polymer.protein

show sticks, ligand
show sticks, gtp

color cyan, ligand
color orange, gtp

show sticks, phe90 or asp92
color blue, phe90 or asp92
set stick_radius, 0.32, phe90 or asp92

label phe90 and name CA, "PHE90"
label asp92 and name CA, "ASP92"

orient phe90 or asp92
zoom phe90 or asp92, 7

png results/batch0100/structural_interpretation/pymol/images/PyMOL_03_PHE90_ASP92_closeup_clean.png, width=2200, height=1700, dpi=300, ray=1


# ============================================================
# SUPPLEMENTARY FIGURE
# PHE1275-ARG1276
#
# February quantitative evidence:
# RR stability       = 22/25 = 0.88
# Combined stability = 21/25 = 0.84
# Low mean           = 1.346 A
# High mean          = 1.344 A
# High-low           = -0.002 A
# Cohen's d          = +1.00
#
# Stable but with an extremely small absolute mean shift.
# Interpret as a highly constrained local geometric descriptor.
# ============================================================

hide sticks, all
hide labels, all

show cartoon, complex and polymer.protein

show sticks, ligand
show sticks, gtp

color cyan, ligand
color orange, gtp

show sticks, phe1275 or arg1276
color yellow, phe1275 or arg1276
set stick_radius, 0.32, phe1275 or arg1276

label phe1275 and name CA, "PHE1275"
label arg1276 and name CA, "ARG1276"

orient phe1275 or arg1276
zoom phe1275 or arg1276, 7

png results/batch0100/structural_interpretation/pymol/images/PyMOL_S01_PHE1275_ARG1276_closeup_clean.png, width=2200, height=1700, dpi=300, ray=1


# ============================================================
# Save session
# ============================================================

save results/batch0100/structural_interpretation/pymol/sessions/PyMOL_stable_feature_closeups_clean.pse

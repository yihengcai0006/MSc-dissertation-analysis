reinitialize

# ============================================================
# Clean thesis close-ups for stable residue-residue features
# NO AUTOMATIC RESIDUE TEXT LABELS
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
# No CA-CA guide is shown in the close-ups because the classifier
# feature is defined as the minimum all-atom inter-residue distance.
#
# This script changes VISUAL PRESENTATION ONLY.
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

# Chain A = RAS
# Chain B = NF1
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

# RAS / chain A
select glu62, complex and polymer.protein and chain A and resi 62
select phe90, complex and polymer.protein and chain A and resi 90
select asp92, complex and polymer.protein and chain A and resi 92

# NF1 / chain B
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

set orthoscopic, on

# Absolutely no automatic PyMOL residue labels
hide labels, all


# ============================================================
# FIGURE 2
# GLU62(RAS) -- SER1279(NF1)
#
# Inter-protein RAS--NF1 structural-context feature.
#
# Quantitative results are NOT recalculated here.
# ============================================================

hide sticks, all
hide labels, all

show cartoon, complex and polymer.protein

show sticks, ligand
show sticks, gtp

color cyan, ligand
color orange, gtp

show sticks, glu62 or ser1279
color red, glu62 or ser1279
set stick_radius, 0.32, glu62 or ser1279

orient glu62 or ser1279
zoom glu62 or ser1279, 8

png results/batch0100/structural_interpretation/pymol/images/PyMOL_02_GLU62_SER1279_closeup_nolabel.png, width=2200, height=1700, dpi=300, ray=1


# ============================================================
# FIGURE 3
# PHE90(RAS) -- ASP92(RAS)
#
# Local intra-RAS structural-context feature.
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

orient phe90 or asp92
zoom phe90 or asp92, 7

png results/batch0100/structural_interpretation/pymol/images/PyMOL_03_PHE90_ASP92_closeup_nolabel.png, width=2200, height=1700, dpi=300, ray=1


# ============================================================
# FIGURE 4
# PHE1275(NF1) -- ARG1276(NF1)
#
# Local intra-NF1 structural-context feature.
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

orient phe1275 or arg1276
zoom phe1275 or arg1276, 7

png results/batch0100/structural_interpretation/pymol/images/PyMOL_S01_PHE1275_ARG1276_closeup_nolabel.png, width=2200, height=1700, dpi=300, ray=1


# ============================================================
# Save NEW session
# ============================================================

save results/batch0100/structural_interpretation/pymol/sessions/PyMOL_stable_feature_closeups_nolabel.pse
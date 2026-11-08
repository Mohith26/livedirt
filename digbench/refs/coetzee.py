"""Reference data: Coetzee, Basson & Vermeer (2007) 2D bucket-filling benchmark.

Labels used throughout:
  [TEXT]      value stated verbatim in the paper's text / tables
  [DIGITIZED] value read off a published figure by pixel analysis (approximate)
  NOT FOUND   not in the accessible text (usually lives in Coetzee's thesis, refs [14],[15] of the paper)

Only paper 1 was accessible via Duke institutional access. Papers 2 and 3 are NOT accessible
(ScienceDirect: "Duke University does not subscribe to this content"; the Duke proxy link redirected
to the same paywalled page without offering access without a login). Nothing was taken from them
except the abstract of paper 2 (see PAPER2_ABSTRACT_FACTS).
"""

CITATION = (
    "Coetzee, C.J., Basson, A.H., Vermeer, P.A. (2007). Discrete and continuum modelling of "
    "excavator bucket filling. Journal of Terramechanics 44(2), 177-186."
)
DOI = "10.1016/j.jterra.2006.07.001"
PDF_LOCAL = "digbench/refs/coetzee2007.pdf (gitignored via refs/*.pdf)"

# ---------------------------------------------------------------- setup
SETUP = {
    "source": "[TEXT] Section 2 of the paper",
    "type": "2D (plane strain) experiment, bucket profile dragged between two glass panels",
    "panel_spacing_mm": 200,                      # [TEXT] "two glass panels 200 mm apart" = slice width
    "bucket_type": "dragline-type bucket (profile, with a tooth)",
    "bucket_total_length_mm": 350,                # [TEXT] "approximately 350 mm"
    "drag_velocity_mm_s": 10,                     # [TEXT] constant, horizontal
    "bucket_vertical_dof": "free to move vertically (effective weight set by linkage + counterweight)",
    "bucket_path": "NOT given numerically in the paper: path was recorded in the experiment and fed "
                   "as input to DEM/MPM (refs [14],[15] = Coetzee thesis)",
    "effective_bucket_weight": "NOT FOUND (adjustable; value not stated)",
    "lip_angle_deg": "NOT FOUND",
    "bucket_profile_dims": "NOT FOUND (only the ~350 mm total length)",
    "fill_depth_bed_depth": "NOT FOUND (experimental bed depth not stated). Bucket 'placed on top of the "
                            "material with the tooth engaged'",
    "total_drag_displacement_mm": 800,            # [TEXT]/[DIGITIZED] fig x-axis 0-800 mm
    "mpm_domain_m": (2.0, 0.7),                   # [TEXT] 2.0 x 0.7 m, 200x100 elements of 10x7 mm
    "mpm_elements": (200, 100),
    "dem_initial_state": "material rained from fixed height along the full test rig length "
                         "(height NOT FOUND)",
    "k0": "1 - sin(phi) initial earth-pressure coefficient (MPM)",
    "bucket_model": "rigid body",
    "material": "cohesionless corn (maize) grains, mean size ~10 mm",
}

# ---------------------------------------------------------------- material (Table 1)
# "measured" = obtained by Coetzee from direct shear + oedometer tests on the corn [TEXT]:
#   "Coetzee [14] used direct shear and oedometer tests to determine the corn properties".
# The paper does not give per-row provenance; rows flagged here are: measured (shear/oedometer),
# assumed/derived, or calibrated. DEM micro-properties were set so DEM direct shear matched
# measured friction angle within +-1 deg (conclusion section) -> i.e. calibrated, not measured.
MATERIAL = {
    "continuum_macro": {
        "youngs_modulus_E_MPa": 2.76,            # from oedometer/shear tests (paper: "obtained from")
        "poisson_nu": 0.2,                        # NOT stated whether measured or assumed
        "density_kg_m3": 778,
        "friction_angle_deg": 26,
        "dilatancy_angle_deg": 2,
        "cohesion_Pa": 0,
        "cosserat_shear_modulus_Gc_MPa": 2.3,     # Gc = 2G (polar MPM only)
        "characteristic_length_mm": 10,           # = mean grain size (polar only)
        "particle_moment_of_inertia_J33_m2": 12.5e-6,  # as printed in paper; 0.5*(0.5*l)^2 (units as printed)
        "friction_with_steel_deg": 14,
    },
    "dem_micro": {
        "particle_stiffness_kn_ks_kN_per_m": 420,
        "particle_density_kg_m3": 855,
        "particle_friction_mu": 0.1,
    },
    "provenance": (
        "[TEXT] macro properties from direct shear + oedometer tests (Coetzee thesis [14]); "
        "per-row measured-vs-assumed split NOT FOUND. DEM micro-properties are calibrated "
        "(DEM direct shear friction angle within +-1 deg of measured). Particle shape: DEM particles "
        "are 2D discs (inferred; the 'moment of inertia as cylinder' remark is for MPM) - not stated."
    ),
    "dem_constitutive": "no rolling resistance in the DEM package used [TEXT]",
    "mpm_constitutive": "Drucker-Prager (nonpolar), polar Drucker-Prager (Cosserat), plane strain",
}

# ---------------------------------------------------------------- measured curve [DIGITIZED]
# Source: Fig. 1 (draft force vs x1-displacement). Only a 378x303 px raster (the largest image
# Elsevier serves, gr1.jpg) is available; the curves are very noisy (+-10 N scatter).
# Calibration: plot frame left edge px 38 = 0 mm, right edge px 367 = 800 mm; bottom px 265 = 0 N,
# top px 5 = 250 N (y-axis ticks at 0..250 step 50, x ticks 0..800 step 100).
# Values are 25-mm-bin centre estimates of the black upper trace (the "Experiment" curve), obtained by
# 2-cluster k-means on dark pixels per bin separating it from the black lower trace ("Polar" MPM),
# after masking the in-plot label text/leader lines. Bins 637.5, 662.5, 687.5 are visual reads
# (leader-line crossing), lower confidence.
MEASURED = {
    "label": "[DIGITIZED] measured experimental draft force, Fig. 1",
    "displacement_mm": [12.5, 37.5, 62.5, 87.5, 112.5, 137.5, 162.5, 187.5, 212.5, 237.5, 262.5,
                        287.5, 312.5, 337.5, 362.5, 387.5, 412.5, 437.5, 462.5, 487.5, 512.5,
                        537.5, 562.5, 587.5, 612.5, 637.5, 662.5, 687.5, 712.5, 737.5, 762.5,
                        787.5],
    "force_N": [49, 61, 61, 60, 68, 68, 72, 74, 76, 75, 78, 91, 91, 86, 89, 92, 96, 106, 107,
                111, 121, 126, 136, 132, 170, 185, 195, 200, 218, 227, 238, 247],
    "uncertainty": {
        "pixel_scale": "1.04 px/N vertical, 0.411 px/mm horizontal (so +-1 px ~ +-1 N / +-2.4 mm)",
        "bin_value_sigma_N": "~ +-6 N within-bin spread; real curve noise amplitude ~ +-10-15 N",
        "systematic_N": "~ +-8 N for 300-600 mm (traces interleave); ~ +-10 N for 612-690 mm",
        "endpoint": "~250 N at 800 mm (curve touches the top frame)",
    },
    "trend_summary": "roughly linear-ish rise from ~50-60 N at start to ~250 N at 800 mm",
}

# Published numerical curves from the same figure [DIGITIZED, low confidence; bins with stray pixels
# from labels/black-line antialiasing removed, hence uneven spacing].
PUBLISHED_MODEL_CURVES_DIGITIZED = {
    "DEM": {
        "displacement_mm": [12.5, 37.5, 62.5, 87.5, 112.5, 137.5, 162.5, 187.5, 212.5, 237.5, 262.5,
                            287.5, 312.5, 337.5, 362.5, 387.5, 412.5, 462.5, 487.5, 562.5, 612.5,
                            662.5, 687.5, 712.5, 737.5, 762.5, 787.5],
        "force_N": [33, 32, 36, 41, 45, 47, 54, 55, 57, 60, 63, 66, 69, 71, 75, 80, 83, 92, 97,
                    111, 123, 134, 137, 144, 152, 156, 161],
    },
    "nonpolar_MPM": {
        "displacement_mm": [12.5, 37.5, 62.5, 87.5, 112.5, 137.5, 162.5, 187.5, 212.5, 237.5, 262.5,
                            287.5, 312.5, 337.5, 362.5, 387.5, 412.5, 462.5, 487.5, 562.5, 612.5,
                            662.5, 687.5, 712.5, 737.5, 762.5, 787.5],
        "force_N": [54, 42, 48, 53, 58, 64, 72, 69, 78, 82, 83, 89, 90, 96, 106, 106, 112, 118,
                    128, 146, 156, 168, 176, 184, 191, 199, 209],
    },
    "polar_MPM": {
        "note": "black lower trace, 2-cluster estimate, valid only up to ~590 mm; interleaved with "
                "experiment above that",
        "displacement_mm": [12.5, 37.5, 62.5, 87.5, 112.5, 137.5, 162.5, 187.5, 212.5, 237.5, 262.5,
                            287.5, 312.5, 337.5, 362.5, 387.5, 412.5, 437.5, 462.5, 487.5, 512.5,
                            537.5, 562.5],
        "force_N": [17, 24, 30, 30, 38, 38, 41, 40, 43, 47, 49, 56, 55, 60, 66, 71, 73, 82, 88,
                    89, 107, 114, 122],
    },
    "uncertainty": "+-8 to +-12 N (thick gray bands, +-5 N ripple); the nonpolar and DEM bands "
                   "were separated by 2-means so mislabelling is possible only where bands touch (none observed)",
}

# ---------------------------------------------------------------- published errors [TEXT]
PUBLISHED_ERRORS = {
    "nonpolar_MPM": "[TEXT] corresponds well to measured force up to 600 mm displacement; "
                    "under-predicts after; error 12% at 800 mm",
    "DEM": "[TEXT] predicted force is 20 N lower than measured up to 600 mm displacement",
    "polar_MPM": "[TEXT] lower than both the measured and the nonpolar force (no number given)",
    "conclusion": "[TEXT] all three predict the trend; nonpolar MPM most accurate quantitatively",
    "digitized_cross_check": {
        "DEM_minus_measured_N": "digitized ~ -15 to -25 N over 100-500 mm (e.g. 100-300 mm: ~ -22 N) "
                                 "-> consistent with '20 N lower'",
        "nonpolar_at_800mm": "digitized nonpolar ~ 209-215 N vs measured ~ 247-250 N -> 14-16 % low "
                             "(text says 12 %, i.e. ~220 N; difference is within the thick-band and "
                             "noise uncertainty, upper edge of nonpolar band reaches ~222 N)",
        "nonpolar_vs_measured_up_to_600mm": "digitized nonpolar ~ 5-10 N above measured at 300-500 mm; "
                                             "within noise -> 'corresponds well'",
    },
}

PAPER2_ABSTRACT_FACTS = {
    "source": "Coetzee & Els 2009, J. Terramechanics 46(5):217-227, DOI 10.1016/j.jterra.2009.05.003 "
              "(ABSTRACT ONLY; full text not accessible via Duke)",
    "text": "DEM predicted lower bucket drag forces than experiment but the general trend was "
            "accurately modelled; error in predicted drag force at end of filling = 20%; DEM predicted "
            "volume of material in bucket within +-6% at all stages.",
}

ACCESS_STATUS = {
    "paper1_coetzee2007": "full text + PDF saved",
    "paper2_coetzee_els_2009_bucket": "NOT ACCESSIBLE (Duke does not subscribe on ScienceDirect; Duke proxy link did not grant access)",
    "paper3_coetzee_els_2009_calibration": "NOT ACCESSIBLE (same)",
}

DIGITIZATION_NOTE = (
    "Digitized from the only available raster of Fig. 1 (378x303 px; the PDF page figure is the same "
    "resolution). Axis calibration from plot frame/ticks (see MEASURED['uncertainty']). The measured "
    "curve is a noisy load-cell trace; values are 25 mm-bin estimates of the black upper trace, with "
    "~+-6 N bin scatter, +-8-10 N systematic where black traces overlap, and visual reads for "
    "637-690 mm. Use these as a smooth reference with +-10 N tolerance, not as sample-level data. "
    "Fig. 1 is a single run; no repeat-run scatter is reported in the paper."
)


# =====================================================================================
# ADDED 2026-11-07 from Coetzee (2004) PhD thesis, Stellenbosch ("The modelling of granular flow
# using the particle-in-cell method"), file digbench/refs/coetzee2004_thesis.pdf (gitignored).
# Page numbers: "pN" = printed thesis page, "pdfN" = PDF page index (pdf = printed + 21).
# Provenance labels: [THESIS-TEXT] verbatim in thesis; [VECTOR-FIG] read from the vector
# drawing paths embedded in the PDF (exact plotted coordinates, only axis-calibration error
# ~ +-0.5 N / +-1 mm); [drawing-measured] read from a dimensioned/isometric drawing or a raster
# photo/sim panel by measurement (uncertainty stated); NOT FOUND = not in the thesis.
# The thesis itself only REPORTS Coetzee (2000) (MSc, Stellenbosch; not in hand) experiments:
# the experiment is a re-use of that rig, Chapter 6.2.
# =====================================================================================

SOURCE_PAGES = {
    "rig_description": "Sec 6.2.1, p90-92 (pdf111-113), Fig 6.1 (p91 rig), Fig 6.2 (p92 arm mechanism)",
    "bucket_drawing": "Fig 6.3, p93 (pdf114): isometric dimensioned drawing, dims in mm",
    "material": "Sec 6.2.2-6.2.3, p92-95 (pdf113-116), Table 6.1 (p95), Table 6.2 (p102 Cosserat)",
    "velocity_effect": "Sec 6.2.4, p95-96 (pdf116-117)",
    "blade_setup": "Sec 6.2.5, p96 (pdf117); blade height 350 mm stated p91 (pdf112)",
    "blade_forces_sim": "Fig 6.12 p103 (psi effect), Fig 6.13 p104 (depth effect), both PICCUS nonpolar, 0-40 mm displacement",
    "blade_measured": "Fig 6.14, p105 (pdf126): average draft force vs depth h (exp, DEM, PICCUS, Sokolovski); text p103-105",
    "bucket_setup": "Sec 6.2.6, p117 (pdf138)",
    "bucket_measured_force": "Fig 6.25, p118 (pdf139) -- vector drawing, digitized exactly",
    "bucket_photos_and_free_surfaces": "Figs 6.26-6.29 p119-122 (pdf140-143) experiment photos + DEM/nonpolar/polar; Fig 6.30 p123 free surfaces",
    "shear_zone_theory": "p124-125, Fig 6.31 (Rowlands 1991); PDR maps Figs 6.32-6.34 p127-129",
    "bucket_pressures": "Figs 6.35-6.36 p130 (max normal/shear stress on bucket inside at 800 mm)",
    "blade_published_errors": "text p104-105 (pdf125-126)",
    "bucket_published_errors": "text p117 (pdf138)",
    "sokolovski_verification_blade_sand": "Sec 4.7 p59-67 (pdf80-88): separate code-verification blade (sand-like MC material), NOT the corn experiment",
}

THESIS_SETUP = {
    "NOTE": "Thesis re-reports Coetzee (2000) experiments; many rig details are only in that MSc thesis (NOT in hand).",
    "rig": {
        "type": "2D test rig, profile dragged between two vertical glass panels",
        "glass_panels_mm": "2200 x 820 [THESIS-TEXT p90]",
        "panel_spacing_mm": "about 200, slightly adjustable [THESIS-TEXT p90]; bucket width 200 (Fig 6.3 dim)",
        "viewing_panel": "10 mm glass + 6 mm glass (replaceable, scratch protection) [p90]",
        "other_panel": "3 mm steel sheet + 6 mm glass (so wall friction is glass-like on both sides) [p90]",
        "profile_side_seals": "two teflon wipers (one each side) prevent grains passing between profile and glass [p92]",
        "drive": "stepper motor + ball screw moves trolley on linear ball bearings; profile attached to trolley via four-bar linkage [p90]",
        "velocity": "constant 10 mm/s in x1 for all blade and bucket experiments AND simulations [p96 Sec 6.2.5, p117]; rig can vary 0-100 mm/s, no significant effect on draft force found [p95]",
        "total_drag_mm": "800 (Fig 6.25 x-axis 0-800; photos at 100 mm steps to 800) [p118-122]",
        "inclination": "rig can tilt in 5 deg steps to 30 deg (not used for the reported tests; 0 deg implied, not stated explicitly)",
        "force_measurement": "three sets of strain gauges: x1-force, x2-force, moment about Om [p92]. NO LVDT mentioned anywhere.",
    },
    "vertical_constraint": {
        "bucket": "arm 2 free to move vertically; vertical motion determined by bucket weight and by effect of material flowing in [p117]. Counterweights at positions in Fig 6.2a cancel arm/linkage weight and allow changing effective bucket weight [p91-92].",
        "blade": "arm 2 locked (blade cannot move vertically) [p92]",
        "rotation": "none: four-bar/arm mechanism holds profile orientation (attack angle fixed relative to vertical arm 2; adjustable but value NOT STATED) [p92]",
        "vertical_velocity": "'found to be constant' in the experiment and prescribed as input to DEM and PICCUS [p117]; numeric value NOT STATED in text, see VERTICAL_PATH (drawing-measured).",
        "vertical_force_measured": "x2-force is measured by the rig but NO bucket x2 curve is published; blade x2 only from sim (Fig 6.12/6.13)",
    },
    "bucket_mass_effective_weight": "NOT FOUND (adjustable via counterweights, value never stated in the thesis)",
    "bed": {
        "material": "corn grains (maize), also wheat used by Coetzee 2000 (very similar properties); only corn reported here [p93]",
        "bed_depth_or_fill_height": "NOT FOUND for the bucket experiment. Sim: 2.0 m x 0.7 m domain, 200x100 elements of 10x7 mm, bucket created with its bottom surface against the free surface and tooth already engaged in the material [p117]",
        "layer_creation": "sim: material created layer-by-layer, K0 = 1 - sin(phi), equilibrated with local damping 0.8 [p96]",
        "dem_initial_state": "NOT FOUND in thesis (in Coetzee 2000)",
        "container_length": "glass panels 2200 mm long, 820 mm high [p90]; usable bed height NOT STATED",
    },
    "mpm_bucket_sim": {
        "domain_m": [2.0, 0.7], "elements": [200, 100], "element_mm": [10, 7],
        "bc": "bottom fixed horizontal+vertical(+rotation polar); two sides fixed horizontal (+rotation polar); boundaries too far to matter [p117]",
        "interface": "no Cosserat rotation at bucket-material interface; bucket rigid [p117]",
        "friction_bucket_steel_deg": 14,
        "bucket_input_motion": "10 mm/s horizontal + constant vertical velocity from experiment",
    },
}

# Bucket profile, Fig 6.3 p93 (pdf114). The figure is an ISOMETRIC dimensioned drawing (dims in mm):
#   140  straight vertical back wall
#   R_i 100  inner radius of the quarter-circle that turns the wall into the floor
#   190  floor length (dimension drawn between arc end and start of tooth, far side)
#   40   tooth/lip run, drawn along the horizontal iso axis, 40 deg (angle drawn to horizontal at tooth)
#   200  bucket width (= glass panel spacing, plane-strain slice)
#   15   dimension arrows at floor/arc junction: meaning NOT labelled; candidates: plate thickness
#        (sim panels draw the plate ~15 mm thick in DEM/PICCUS figures) -> NOT CONFIRMED
#   Om   moment centre: 75 (oblique, on the mounting plate) and 93 (vertical) from the wall top mount
# Geometry was re-derived by measuring the PDF vector drawing in iso projection (checks passed:
# floor horizontal; wall + arc consistent with 140 + R100; tooth angle atan(33.6/40) = 40.0 deg
# measured vs 40 labelled). Floor length measured from drawing = ~200 mm vs 190 labelled (+-5 mm
# line-width/thickness ambiguity). Use the labelled 190.
# Coordinates: x forward (drag direction, toward the tooth), z up, ORIGIN = cutting edge (tooth tip),
# mm, polygon of the MATERIAL-FACING (inner) surface centreline; plate thickness NOT FOUND.
BUCKET_PROFILE_MM = {
    "source": "Fig 6.3 p93 labelled dims [THESIS-TEXT] + iso-drawing measurement [drawing-measured]",
    "origin": "cutting edge (tooth tip) = (0,0); +x = drag direction; +z up; bucket floor horizontal in drawing",
    "lip_angle_deg": "40 below horizontal [labelled 40 deg, tooth drawn on the horizontal iso axis; drawing-measured 40.0 +-1.5]",
    "polyline_inner_surface_mm": [(0.0, 0.0), (-40.0, 33.56), (-230.0, 33.56), (-238.72, 33.94), (-247.36, 35.08), (-255.88, 36.97), (-264.2, 39.59), (-272.26, 42.93), (-280.0, 46.96), (-287.36, 51.64), (-294.28, 56.96), (-300.71, 62.85), (-306.6, 69.28), (-311.92, 76.2), (-316.6, 83.56), (-320.63, 91.3), (-323.97, 99.36), (-326.59, 107.68), (-328.48, 116.2), (-329.62, 124.84), (-330.0, 133.56), (-330.0, 273.56)],
    "arc_centre_mm": (-230.0, 133.56), "arc_radius_mm": 100.0, "arc_from_to": "floor-end (-230,33.56) -> wall base (-330,133.56)",
    "wall_mm": "vertical from (-330,133.56) up to (-330,273.56) (140 mm)",
    "overall_mm": "length (tip to back wall) 330 + thickness; height 273.6 + thickness (paper-2007 says total length ~350 mm [coetzee 2007 TEXT], consistent with ~330 + plate thickness ~15-20)",
    "width_mm": 200,
    "thickness_mm": "NOT FOUND (15 on the drawing is unlabelled; sim figures show ~15-25 mm)",
    "uncertainty": "labelled dims exact to drawing precision (+-1 mm); floor 190 vs 200 measured ambiguity +-5-10 mm; unlabelled thickness",
    "initial_attitude": "NOT STATED: attack angle of profile relative to arm 2 'can be changed' [p92]; Figs 6.26-6.29 show the floor horizontal at start of drag and tilting is not mentioned -> floor horizontal [drawing-measured from photos/sims]",
}

# Digitized Fig 6.25 (p118, pdf139) is a VECTOR plot: coordinates below are the plotted path vertices
# calibrated to axes (x: 0..800 mm, 0 at frame left; y: 0..250 N). Calibration error ~ +-0.5 N / +-1 mm.
# Overall this REPLACES the raster digitization in MEASURED above (they agree to ~ +-5-10 N; the
# vector curve is the better source). The experimental trace is a single run (no repeats or error bars
# published; thesis does not state number of repeats). Within-bin sd below = trace noise, not scatter.
# NOTE ON LABELS: the legend colours are Experiment=red, Nonpolar=blue, Polar=magenta, DEM=black.
# The older raster-based dicts above (PUBLISHED_MODEL_CURVES_DIGITIZED) appear to have the DEM and
# polar_MPM labels SWAPPED (their 'polar' ~17->122 N @562 mm matches the black DEM trace here; their
# 'DEM' 33->161 N matches the magenta Polar trace).
THESIS_BUCKET_MEASURED = {
    "source": "Fig 6.25 p118 (pdf139), [VECTOR-FIG]; single experimental run (Coetzee 2000 rig)",
    "x_displacement_mm_bin_centres_25mm": [12.5, 37.5, 62.5, 87.5, 112.5, 137.5, 162.5, 187.5, 212.5, 237.5, 262.5, 287.5, 312.5, 337.5, 362.5, 387.5, 412.5, 437.5, 462.5, 487.5, 512.5, 537.5, 562.5, 587.5, 612.5, 637.5, 662.5, 687.5, 712.5, 737.5, 762.5, 787.5],
    "experiment_draft_force_N_bin_mean": [40.6, 61.3, 60.4, 58.5, 62.5, 64.9, 69.7, 72.9, 77.0, 76.0, 79.5, 88.5, 91.5, 89.1, 89.3, 97.7, 99.7, 105.8, 111.6, 118.2, 126.2, 134.6, 142.0, 152.3, 166.1, 176.9, 192.0, 202.7, 216.6, 226.3, 238.0, 249.9],
    "experiment_within_bin_sd_N": [15.3, 6.6, 8.0, 4.8, 7.9, 5.6, 8.1, 10.1, 7.8, 5.5, 6.2, 6.3, 5.9, 6.4, 8.3, 7.6, 6.8, 7.1, 7.0, 8.1, 8.5, 6.3, 5.1, 3.4, 6.4, 8.6, 10.9, 4.9, 4.4, 3.0, 2.4, 2.8],
    "experiment_note": "bin 0-25 mm includes the initial elastic/engagement rise from ~0 N (value 40.6 is the bin mean); end value ~250 N at 800 mm",
    "experiment_raw_polyline_mm_N": [[3.0, 15.0], [6.5, 37.7], [12.5, 50.8], [17.8, 60.4], [23.1, 39.1], [27.2, 70.7], [32.6, 56.7], [37.9, 63.2], [42.0, 64.2], [47.3, 51.5], [52.7, 73.5], [56.8, 55.3], [62.1, 57.4], [67.4, 64.9], [73.3, 50.8], [76.9, 57.4], [82.2, 57.4], [88.1, 56.0], [91.7, 67.7], [97.0, 53.9], [102.9, 60.4], [108.2, 59.0], [111.8, 56.0], [117.7, 78.0], [123.0, 59.0], [126.5, 61.8], [132.5, 64.2], [137.8, 57.4], [141.3, 74.3], [147.2, 67.0], [152.6, 70.7], [157.9, 70.7], [162.0, 58.3], [167.3, 83.2], [172.7, 65.6], [176.8, 72.1], [182.1, 76.6], [187.4, 58.3], [193.3, 89.0], [196.9, 68.4], [202.2, 78.0], [208.1, 83.9], [211.7, 64.9], [217.0, 86.2], [222.9, 72.1], [226.5, 75.0], [231.8, 83.2], [237.7, 67.0], [243.0, 80.1], [246.6, 75.0], [252.5, 75.9], [257.8, 90.7], [261.3, 72.1], [267.2, 80.1], [272.6, 78.7], [278.5, 82.5], [282.0, 99.3], [287.3, 82.5], [293.3, 87.6], [296.8, 90.7], [302.1, 81.8], [308.0, 99.3], [311.6, 92.8], [316.9, 89.0], [322.8, 94.9], [328.1, 81.8], [331.7, 97.9], [337.0, 81.8], [342.9, 92.1], [346.5, 92.1], [351.8, 76.6], [357.7, 96.5], [363.0, 83.2], [366.6, 91.4], [372.5, 98.6], [377.8, 87.6], [381.3, 109.6], [387.3, 93.5], [392.6, 95.6], [398.5, 102.4], [402.0, 89.0], [407.4, 97.9], [413.3, 97.9], [416.8, 104.5], [422.1, 108.9], [428.0, 96.5], [431.6, 105.9], [436.9, 103.8], [442.8, 104.5], [448.1, 118.3], [451.7, 105.2], [457.6, 109.6], [462.9, 111.8], [466.5, 106.6], [472.4, 124.9], [477.7, 113.2], [483.0, 120.0], [487.2, 120.7], [492.5, 106.6], [497.8, 130.7], [501.9, 115.5], [507.3, 127.2], [512.6, 127.9], [516.7, 120.0], [522.0, 140.3], [527.4, 128.6], [533.3, 136.6], [536.8, 140.3], [542.1, 125.8], [548.1, 141.8], [551.6, 134.5], [556.9, 141.8], [562.8, 149.3], [568.2, 138.9], [571.7, 145.5], [577.6, 147.6], [582.9, 150.7], [586.5, 155.1], [592.4, 150.7], [597.7, 157.2], [601.3, 157.2], [607.2, 161.0], [612.5, 170.6], [617.8, 166.8], [622.0, 175.0], [627.3, 170.6], [632.6, 164.7], [636.7, 183.0], [642.0, 177.1], [647.4, 188.9], [653.3, 185.8], [656.8, 177.1], [662.1, 200.6], [668.1, 188.9], [671.6, 207.8], [676.9, 198.5], [682.8, 197.5], [686.4, 207.8], [691.7, 200.6], [697.6, 209.3], [702.9, 211.6], [706.5, 213.0], [712.4, 216.0], [717.7, 218.2], [721.3, 224.0], [727.2, 224.7], [732.5, 223.3], [737.8, 226.1], [742.0, 225.4], [747.3, 232.0], [752.6, 238.8], [756.7, 234.3], [762.1, 238.1], [767.4, 237.1], [771.5, 241.6], [776.8, 246.8], [782.2, 247.5], [788.1, 249.1], [791.6, 254.3], [796.9, 251.9]],
    "curve_end_N_at_800mm": {"experiment": 250.0, "nonpolar": 209.4, "polar": 160.7, "dem": 191.8},
    "uncertainty": "+-1 N / +-1 mm calibration; real experimental noise +-5-10 N around the smoothed trace; repeat scatter NOT FOUND",
    "vertical_force": "NOT FOUND (not plotted)",
    "fill_volume_or_mass": "NOT FOUND (only photos Figs 6.26-6.29 and free-surface curves Fig 6.30 p123)",
    "number_of_repeats": "NOT FOUND (one trace shown)",
}

THESIS_MODEL_CURVES_VECTOR = {
    "source": "Fig 6.25 p118 [VECTOR-FIG]; same axes/bins as THESIS_BUCKET_MEASURED",
    "x_displacement_mm_bin_centres_25mm": [12.5, 37.5, 62.5, 87.5, 112.5, 137.5, 162.5, 187.5, 212.5, 237.5, 262.5, 287.5, 312.5, 337.5, 362.5, 387.5, 412.5, 437.5, 462.5, 487.5, 512.5, 537.5, 562.5, 587.5, 612.5, 637.5, 662.5, 687.5, 712.5, 737.5, 762.5, 787.5],
    "Nonpolar_MPM_N": [27.1, 41.1, 47.5, 52.9, 57.4, 63.2, 66.1, 69.7, 77.2, 81.0, 83.3, 88.2, 90.7, 95.5, 103.6, 105.0, 111.6, 117.1, 120.1, 126.9, 133.1, 138.3, 146.0, 152.0, 156.5, 164.5, 171.3, 175.7, 187.5, 193.1, 198.4, 209.4],
    "Polar_MPM_N": [27.2, 33.6, 37.4, 40.7, 44.7, 47.8, 51.8, 53.9, 57.5, 60.5, 62.4, 66.9, 68.2, 72.0, 75.3, 80.2, 83.5, 88.7, 91.3, 96.4, 100.7, 106.4, 111.7, 118.7, 123.0, 127.5, 133.3, 138.8, 145.2, 150.8, 155.4, 160.7],
    "DEM_N": [14.2, 22.8, 31.2, 30.5, 38.0, 37.1, 41.0, 40.1, 42.3, 46.4, 48.2, 55.9, 55.9, 60.3, 68.5, 73.9, 75.3, 85.1, 93.2, 98.6, 110.0, 114.3, 123.8, 125.8, 135.7, 143.3, 156.6, 149.0, 163.8, 167.2, 173.2, 191.8],
    "Nonpolar_within_bin_sd_N": [14.6, 3.5, 1.6, 3.1, 2.8, 1.9, 6.0, 2.8, 2.3, 4.0, 2.3, 3.6, 5.1, 2.7, 4.0, 4.6, 3.1, 5.8, 6.0, 2.5, 6.3, 5.2, 4.5, 7.6, 5.4, 3.9, 7.4, 4.2, 4.9, 7.7, 4.3, 7.3],
    "Polar_within_bin_sd_N": [9.7, 3.7, 2.1, 3.2, 2.1, 2.4, 2.4, 2.5, 2.3, 2.6, 2.9, 2.7, 3.1, 2.6, 3.2, 3.2, 3.0, 3.1, 3.2, 3.8, 3.8, 3.7, 3.3, 3.5, 3.3, 3.6, 3.7, 3.9, 3.7, 4.5, 3.8, 4.8],
    "DEM_within_bin_sd_N": [4.9, 4.7, 4.4, 3.6, 2.7, 3.9, 5.1, 4.7, 4.5, 4.9, 4.4, 5.0, 5.8, 4.9, 5.8, 7.5, 6.9, 4.8, 7.6, 8.8, 8.3, 6.9, 8.4, 9.8, 6.0, 9.7, 10.9, 7.3, 9.8, 8.0, 7.3, 8.0],
}

# Vertical bucket motion. The thesis states only that the vertical velocity was constant and used as
# input (p117). No number, no plot. Estimated here from the bucket position in the nonpolar and polar
# simulation panels (Figs 6.26-6.29, whose bucket motion is the prescribed experimental path):
# floor-bottom height above panel bottom measured in image pixels at 100..800 mm, panels are
# bottom-aligned (page rects) and share horizontal scale. Pixel scale 0.588 px/mm from the bucket's own
# wall-top-to-floor-bottom extent (170 px = ~289 mm incl. ~15 mm plate). Bucket sinks DEEPER into the bed
# as it is dragged (tip goes below initial surface), it does not rise.
VERTICAL_PATH = {
    "source": "[drawing-measured] raster panels of Figs 6.26-6.29 (p119-122); thesis text p117 for 'constant vertical velocity'",
    "sign": "negative = bucket moves DOWN relative to ground",
    "floor_bottom_height_above_panel_bottom_px_nonpolar": {100:113,200:100,300:88,400:71,500:65,600:56,700:42,800:25},
    "floor_bottom_height_above_panel_bottom_px_polar": {100:113,200:100,300:89,400:78.5,500:67,600:65,700:46,800:27},
    "vertical_displacement_relative_to_100mm_mm_nonpolar_est": {100:0,200:-22,300:-42,400:-71,500:-82,600:-97,700:-121,800:-150},
    "linear_fit_dz_per_dx": -0.20, "fit_range_dx_mm": (100, 800), "fit_resid_sd_mm": 5,
    "at_10mm_per_s_vertical_velocity_mm_s": -2.0,
    "uncertainty": "slope -0.20 +- 0.04 (scale 0.588 px/mm +-8%, bottom-alignment assumption, black-pixel thresholding); polar panels give -0.19",
    "state_at_start": "tooth engaged: floor ~at free surface at 100 mm panel (Fig 6.26); value for 0-100 mm NOT FOUND",
    "caveat": "path is the simulation input; assumed equal to the experiment's measured near-linear path per text p117",
}

BLADE_SETUP = {
    "source": "Sec 6.2.1/6.2.5 p90-96, Fig 6.12-6.14",
    "geometry": "flat vertical blade (theta = 0, attack angle 0 to vertical), height 350 mm [p91]; width = glass spacing ~200 mm [p90]; blade thickness/edge shape NOT FOUND",
    "depths_h_mm": [150, 250, 350],
    "blade_vertical_dof": "arm 2 locked (no vertical motion) [p92]",
    "velocity_mm_s": 10,
    "material": "same corn as bucket (Table 6.1: phi 26 deg, psi 2 deg, c 0, rho 778 kg/m3, E 2.76 MPa, nu 0.2 assumed, phi_glass 12 deg, phi_steel 14 deg)",
    "mpm_sim": "mesh 120x70 (10x10 mm) fine and 80x50 (15x14 mm) coarse [p108]; layer-by-layer creation with blade in place, K0 = 1 - sin(phi), damping 0.8; sides fixed in x1, bottom fixed; no Cosserat rotation at blade [p96]",
    "dilatancy": "psi = 2 deg chosen because it best matched the experimental free surface (tested 0, 2, 10, 26 deg) [p100]",
    "averaging": "average draft force taken from sim/experiment/DEM between 15 mm and 35 mm blade displacement [p104]",
    "bed_depth_total": "NOT FOUND (blade depth h = depth of blade tip below free surface)",
    "side_wall_friction_note": "thesis suggests glass friction (phi 12 deg) adds to measured force vs plane-strain sim, explaining sim under-prediction [p104]",
}

BLADE_MEASURED = {
    "source": "Fig 6.14 p105 (pdf126), [VECTOR-FIG] marker centres, +-1.5 N (triangle centroids +-3 N); only 3 depths, no force-displacement experimental curve published",
    "depth_h_mm": [150, 250, 350],
    "experiment_avg_draft_force_N": [96.6, 205.8, 386.1],
    "DEM_avg_draft_force_N": [105.6, 228.5, 367.0],
    "nonpolar_PICCUS_avg_draft_force_N": [71.7, 182.3, 351.5],
    "polar_PICCUS_avg_draft_force_N": [52.4, 154.0, 307.5],
    "Sokolovski_N_at_depths": [63.0, 171.4, 334.3],
    "Sokolovski_curve_h_mm_N": [(0.7,-0.3),(41.2,4.6),(80.8,19.0),(121.4,40.2),(161.0,71.9),(200.7,110.3),(241.2,160.0),(280.8,214.7),(321.4,282.0),(361.0,354.2)],
    "x1_force_vs_displacement": "experimental curves NOT FOUND; only simulated curves Figs 6.12/6.13 (0-40 mm)",
    "x2_force": "experimental NOT FOUND (sim x2 negative: material pushes blade up)",
    "repeats_scatter": "NOT FOUND",
    "comment": "Cleaner benchmark? Pros: full geometry (350 mm vertical flat blade, 200 mm wide, 10 mm/s, 3 depths, rigid, no vertical DOF), material table complete; Cons: only 3 averaged points over 15-35 mm displacement, no curve.",
}

THESIS_PUBLISHED_ERRORS = {
    "bucket_draft_force_Fig6_25": {
        "nonpolar_MPM": "[THESIS-TEXT p117] corresponds well to measured up to 600 mm, under-predicts thereafter, error 12% at 800 mm",
        "DEM": "[THESIS-TEXT p117] about 20 N lower than measured up to 600 mm",
        "polar_MPM": "[THESIS-TEXT p117] lower than both measured and nonpolar (no number)",
        "vector_cross_check": {
            "nonpolar_at_800_vs_exp": "209.4 vs 250.0 N = -16% on the curve end value (text says 12%; maybe measured on smoothed value near ~220 N, not stated)",
            "DEM_minus_exp_N_100_to_600": "about -16 to -22 N  (e.g. 100 mm: 41-62, 300 mm: 75-91, 500 mm: 110-126)",
            "polar_minus_exp_N": "about -15 to -25 N throughout to ~600 mm; 160.7 vs 250 N at 800 mm = -36%",
        },
    },
    "blade_avg_draft_force_Fig6_14": {
        "nonpolar_vs_experiment": "[THESIS-TEXT p104] underestimates by 25%, 11%, 9% for h = 150, 250, 350 mm (vector check: -25.8, -11.4, -9.0)",
        "polar_vs_experiment": "[THESIS-TEXT p105] 30%, 21%, 20% lower",
        "polar_vs_nonpolar": "[THESIS-TEXT p105] 28%, 14%, 10% lower",
        "nonpolar_vs_Sokolovski": "[THESIS-TEXT p104] overestimates by 12%, 6%, 5%",
        "DEM_vs_experiment": "[THESIS-TEXT p105] higher at 150 and 250 mm, lower at 350 mm (vector: +9%, +11%, -5%)",
    },
    "bucket_pressures_800mm": "[THESIS-TEXT Figs 6.35-6.36 p130] max normal stress on bucket inside: DEM 12.8 kPa, nonpolar 15.1, polar 9.8; max shear: 3.2, 3.5, 2.3 kPa; maxima at the tooth",
    "free_surface": "nonpolar best for blade free surface (p101); for bucket nonpolar best to 200 mm, polar closer 300-600 mm, DEM under-predicts upheave throughout (p117, Fig 6.30)",
}

THESIS_MATERIAL = {
    "source": "Table 6.1 p95 (Experiment column), Table 6.2 p102",
    "measured_tests": "oedometer (E_oed -> E via E = E_oed (1+nu)(1-2nu)/(1-nu)); shear box (phi, c); shear box half replaced with glass or steel sheet (phi_glass, phi_steel) [p93-94]",
    "measured": {"E_MPa": 2.76, "density_kg_m3": 778, "friction_angle_deg": 26, "cohesion_Pa": 0,
                "friction_with_glass_deg": 12, "friction_with_steel_deg": 14, "tension_cutoff_Pa": 0},
    "assumed_or_derived": {"poisson_nu": "0.2 estimated, NOT measured", "dilatancy_psi_deg": "2, NOT measured: picked as best fit of free surface among 0/2/10/26 deg (this is a fit of the experiment's free surface, p100)",
                          "tension_cutoff": "0 Pa assumed (listed with experiment column)"},
    "macro_within_range_of": "Reimbert & Reimbert (1976) [p95]",
    "cosserat_extra": {"a1_a2_a3": (0.375, 0.125, 1), "Gc_MPa": 2.3, "l_mm": 10, "J33_m2": 12.5e-6},
    "dem": {"E_MPa": 1.84, "phi_deg": 24, "particle": "2 discs rigidly joined (clump), kn=ks 420 kN/m, rho_p 855, mu 0.1, steel 14 deg"},
    "grain_size_mm": "corn grains ~ 5-9 x 4-6 x 3-6 (Fig 6.5a p94, ranges); mean ~10 used as characteristic length",
    "bulk_density_in_bin": "778 kg/m3 (single value in table; no separate in-bin vs loose measurement)",
    "angle_of_repose": "NOT FOUND (not mentioned in thesis)",
    "wall_friction_glass_deg": 12, "wall_friction_steel_deg": 14,
    "note_for_reproduction": "bucket surface friction 14 deg (steel); side glass/steel-panel friction 12 deg matters for the 2D rig: thesis argues glass friction raises measured forces above plane-strain sim (p104). A 3D MPM with 200 mm width and side walls at 12 deg can reproduce this without fitting.",
}

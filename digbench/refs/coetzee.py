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

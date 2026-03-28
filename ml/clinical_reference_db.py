"""
AutoClin Engine — Comprehensive Clinical Reference Database

Provides reference ranges, units, clinical terminology, LOINC codes,
and synonym matching for 120+ clinical measurements commonly found
in clinical trials, hospital EHR, biomedical studies, and lab datasets.

This module is used by the plausibility engine to validate values
and by the clinical mapper to recognize column types.

Sources: Standard clinical laboratory references (Tietz Textbook of
Clinical Chemistry, WHO guidelines, FDA labeling conventions).

DESIGN PRINCIPLE: If a column is not recognized, the system gracefully
degrades — it simply skips plausibility checking for that column and
relies entirely on the unsupervised methods for anomaly detection.
No column is forced into a reference range it doesn't belong to.
"""

# Each entry: {
#   "names": list of column name patterns (lowercase) that match this test
#   "category": clinical category
#   "unit": standard unit
#   "alt_units": alternative units with conversion factors
#   "range_adult": (min, max) plausible range for adults
#   "range_pediatric": (min, max) or None
#   "critical_low": value below which is life-threatening
#   "critical_high": value above which is life-threatening
#   "loinc": primary LOINC code (for reference)
#   "description": human-readable description
# }

CLINICAL_REFERENCE_DB = [
    # ═══════════════════════════════════════════════════════════════
    # VITAL SIGNS
    # ═══════════════════════════════════════════════════════════════
    {
        "names": ["systolic_bp", "systolic", "sys_bp", "sbp", "systolic_blood_pressure", "bp_systolic", "sys"],
        "category": "vital_sign", "unit": "mmHg", "alt_units": {"kPa": 7.5006},
        "range_adult": (50, 300), "range_pediatric": (40, 200),
        "critical_low": 60, "critical_high": 250,
        "loinc": "8480-6", "description": "Systolic blood pressure"
    },
    {
        "names": ["diastolic_bp", "diastolic", "dia_bp", "dbp", "diastolic_blood_pressure", "bp_diastolic", "dia"],
        "category": "vital_sign", "unit": "mmHg", "alt_units": {"kPa": 7.5006},
        "range_adult": (20, 200), "range_pediatric": (20, 130),
        "critical_low": 30, "critical_high": 150,
        "loinc": "8462-4", "description": "Diastolic blood pressure"
    },
    {
        "names": ["heart_rate", "hr", "pulse", "pulse_rate", "heartrate", "heart_rate_bpm"],
        "category": "vital_sign", "unit": "bpm", "alt_units": {},
        "range_adult": (20, 300), "range_pediatric": (40, 250),
        "critical_low": 30, "critical_high": 200,
        "loinc": "8867-4", "description": "Heart rate"
    },
    {
        "names": ["resp_rate", "respiratory_rate", "rr", "resprate", "breaths_per_min", "resp"],
        "category": "vital_sign", "unit": "breaths/min", "alt_units": {},
        "range_adult": (4, 60), "range_pediatric": (8, 80),
        "critical_low": 6, "critical_high": 50,
        "loinc": "9279-1", "description": "Respiratory rate"
    },
    {
        "names": ["temperature", "temp", "temperature_c", "body_temp", "temp_c", "body_temperature"],
        "category": "vital_sign", "unit": "°C", "alt_units": {"°F": lambda c: (c - 32) / 1.8},
        "range_adult": (28, 45), "range_pediatric": (30, 43),
        "critical_low": 32, "critical_high": 42,
        "loinc": "8310-5", "description": "Body temperature"
    },
    {
        "names": ["spo2", "oxygen_saturation", "o2sat", "sao2", "spo2_pct", "pulse_ox", "oximetry"],
        "category": "vital_sign", "unit": "%", "alt_units": {},
        "range_adult": (50, 100), "range_pediatric": (60, 100),
        "critical_low": 70, "critical_high": 100,
        "loinc": "59408-5", "description": "Oxygen saturation"
    },
    {
        "names": ["weight", "weight_kg", "body_weight", "wt", "wt_kg"],
        "category": "vital_sign", "unit": "kg", "alt_units": {"lb": 0.4536, "g": 0.001},
        "range_adult": (20, 350), "range_pediatric": (0.5, 150),
        "critical_low": None, "critical_high": None,
        "loinc": "29463-7", "description": "Body weight"
    },
    {
        "names": ["height", "height_cm", "body_height", "ht", "stature"],
        "category": "vital_sign", "unit": "cm", "alt_units": {"in": 2.54, "m": 100},
        "range_adult": (50, 250), "range_pediatric": (20, 200),
        "critical_low": None, "critical_high": None,
        "loinc": "8302-2", "description": "Body height"
    },
    {
        "names": ["bmi", "body_mass_index"],
        "category": "vital_sign", "unit": "kg/m²", "alt_units": {},
        "range_adult": (8, 80), "range_pediatric": (8, 50),
        "critical_low": 10, "critical_high": 70,
        "loinc": "39156-5", "description": "Body mass index"
    },

    # ═══════════════════════════════════════════════════════════════
    # HEMATOLOGY
    # ═══════════════════════════════════════════════════════════════
    {
        "names": ["hemoglobin", "hgb", "hb", "hemoglobin_gdl", "haemoglobin"],
        "category": "lab_value", "unit": "g/dL", "alt_units": {"g/L": 0.1, "mmol/L": 1.61},
        "range_adult": (3, 25), "range_pediatric": (5, 22),
        "critical_low": 5, "critical_high": 22,
        "loinc": "718-7", "description": "Hemoglobin"
    },
    {
        "names": ["hematocrit", "hct", "packed_cell_volume", "pcv"],
        "category": "lab_value", "unit": "%", "alt_units": {"L/L": 100},
        "range_adult": (10, 70), "range_pediatric": (15, 65),
        "critical_low": 15, "critical_high": 60,
        "loinc": "4544-3", "description": "Hematocrit"
    },
    {
        "names": ["wbc", "white_blood_cells", "leukocytes", "wbc_count", "white_cell_count", "wbc_10e3ul"],
        "category": "lab_value", "unit": "×10³/µL", "alt_units": {"cells/µL": 0.001},
        "range_adult": (0.5, 50), "range_pediatric": (1, 40),
        "critical_low": 1, "critical_high": 40,
        "loinc": "6690-2", "description": "White blood cell count"
    },
    {
        "names": ["rbc", "red_blood_cells", "erythrocytes", "rbc_count"],
        "category": "lab_value", "unit": "×10⁶/µL", "alt_units": {},
        "range_adult": (1, 10), "range_pediatric": (2, 8),
        "critical_low": 1.5, "critical_high": 8,
        "loinc": "789-8", "description": "Red blood cell count"
    },
    {
        "names": ["platelet", "platelets", "plt", "platelet_count", "thrombocytes", "platelet_10e3ul"],
        "category": "lab_value", "unit": "×10³/µL", "alt_units": {},
        "range_adult": (10, 1000), "range_pediatric": (50, 800),
        "critical_low": 20, "critical_high": 800,
        "loinc": "777-3", "description": "Platelet count"
    },
    {
        "names": ["mcv", "mean_corpuscular_volume"],
        "category": "lab_value", "unit": "fL", "alt_units": {},
        "range_adult": (50, 130), "range_pediatric": (55, 120),
        "critical_low": None, "critical_high": None,
        "loinc": "787-2", "description": "Mean corpuscular volume"
    },
    {
        "names": ["mch", "mean_corpuscular_hemoglobin"],
        "category": "lab_value", "unit": "pg", "alt_units": {},
        "range_adult": (15, 45), "range_pediatric": (18, 40),
        "critical_low": None, "critical_high": None,
        "loinc": "785-6", "description": "Mean corpuscular hemoglobin"
    },
    {
        "names": ["mchc", "mean_corpuscular_hemoglobin_concentration"],
        "category": "lab_value", "unit": "g/dL", "alt_units": {},
        "range_adult": (25, 40), "range_pediatric": (28, 38),
        "critical_low": None, "critical_high": None,
        "loinc": "786-4", "description": "MCHC"
    },
    {
        "names": ["esr", "erythrocyte_sedimentation_rate", "sed_rate"],
        "category": "lab_value", "unit": "mm/hr", "alt_units": {},
        "range_adult": (0, 120), "range_pediatric": (0, 50),
        "critical_low": None, "critical_high": None,
        "loinc": "4537-7", "description": "Erythrocyte sedimentation rate"
    },
    {
        "names": ["reticulocyte", "retic", "reticulocyte_count", "retic_count"],
        "category": "lab_value", "unit": "%", "alt_units": {},
        "range_adult": (0.1, 10), "range_pediatric": (0.5, 8),
        "critical_low": None, "critical_high": None,
        "loinc": "17849-1", "description": "Reticulocyte count"
    },

    # ═══════════════════════════════════════════════════════════════
    # CHEMISTRY — METABOLIC PANEL
    # ═══════════════════════════════════════════════════════════════
    {
        "names": ["glucose", "blood_glucose", "fasting_glucose", "glucose_mgdl", "gluc", "blood_sugar", "fbs", "rbs"],
        "category": "lab_value", "unit": "mg/dL", "alt_units": {"mmol/L": 18.0182},
        "range_adult": (10, 700), "range_pediatric": (20, 500),
        "critical_low": 30, "critical_high": 500,
        "loinc": "2345-7", "description": "Glucose"
    },
    {
        "names": ["creatinine", "creat", "serum_creatinine", "creatinine_mgdl", "scr"],
        "category": "lab_value", "unit": "mg/dL", "alt_units": {"µmol/L": 0.01131},
        "range_adult": (0.1, 30), "range_pediatric": (0.1, 5),
        "critical_low": None, "critical_high": 15,
        "loinc": "2160-0", "description": "Creatinine"
    },
    {
        "names": ["bun", "blood_urea_nitrogen", "urea_nitrogen", "urea"],
        "category": "lab_value", "unit": "mg/dL", "alt_units": {"mmol/L": 0.3571},
        "range_adult": (2, 150), "range_pediatric": (3, 50),
        "critical_low": None, "critical_high": 100,
        "loinc": "3094-0", "description": "Blood urea nitrogen"
    },
    {
        "names": ["sodium", "na", "serum_sodium", "na_meql"],
        "category": "lab_value", "unit": "mEq/L", "alt_units": {"mmol/L": 1},
        "range_adult": (100, 180), "range_pediatric": (110, 170),
        "critical_low": 115, "critical_high": 160,
        "loinc": "2951-2", "description": "Sodium"
    },
    {
        "names": ["potassium", "k", "serum_potassium", "k_meql"],
        "category": "lab_value", "unit": "mEq/L", "alt_units": {"mmol/L": 1},
        "range_adult": (1.5, 9), "range_pediatric": (2, 8),
        "critical_low": 2.5, "critical_high": 7,
        "loinc": "2823-3", "description": "Potassium"
    },
    {
        "names": ["chloride", "cl", "serum_chloride"],
        "category": "lab_value", "unit": "mEq/L", "alt_units": {"mmol/L": 1},
        "range_adult": (70, 130), "range_pediatric": (80, 120),
        "critical_low": 75, "critical_high": 125,
        "loinc": "2075-0", "description": "Chloride"
    },
    {
        "names": ["bicarbonate", "hco3", "co2", "total_co2", "tco2"],
        "category": "lab_value", "unit": "mEq/L", "alt_units": {"mmol/L": 1},
        "range_adult": (5, 50), "range_pediatric": (10, 40),
        "critical_low": 8, "critical_high": 45,
        "loinc": "1963-8", "description": "Bicarbonate / Total CO2"
    },
    {
        "names": ["calcium", "ca", "serum_calcium", "total_calcium"],
        "category": "lab_value", "unit": "mg/dL", "alt_units": {"mmol/L": 4.0},
        "range_adult": (4, 18), "range_pediatric": (5, 15),
        "critical_low": 6, "critical_high": 14,
        "loinc": "17861-6", "description": "Calcium"
    },
    {
        "names": ["magnesium", "mg", "serum_magnesium"],
        "category": "lab_value", "unit": "mg/dL", "alt_units": {"mmol/L": 2.431},
        "range_adult": (0.5, 6), "range_pediatric": (1, 5),
        "critical_low": 0.8, "critical_high": 5,
        "loinc": "19123-9", "description": "Magnesium"
    },
    {
        "names": ["phosphate", "phosphorus", "phos", "serum_phosphorus", "inorganic_phosphorus"],
        "category": "lab_value", "unit": "mg/dL", "alt_units": {"mmol/L": 3.097},
        "range_adult": (1, 10), "range_pediatric": (2, 9),
        "critical_low": 1.0, "critical_high": 9,
        "loinc": "2777-1", "description": "Phosphate"
    },

    # ═══════════════════════════════════════════════════════════════
    # LIVER FUNCTION
    # ═══════════════════════════════════════════════════════════════
    {
        "names": ["alt", "alanine_aminotransferase", "sgpt", "alt_ul", "alanine_transaminase"],
        "category": "lab_value", "unit": "U/L", "alt_units": {},
        "range_adult": (0, 1000), "range_pediatric": (0, 500),
        "critical_low": None, "critical_high": 500,
        "loinc": "1742-6", "description": "ALT / SGPT"
    },
    {
        "names": ["ast", "aspartate_aminotransferase", "sgot", "ast_ul", "aspartate_transaminase"],
        "category": "lab_value", "unit": "U/L", "alt_units": {},
        "range_adult": (0, 1000), "range_pediatric": (0, 500),
        "critical_low": None, "critical_high": 500,
        "loinc": "1920-8", "description": "AST / SGOT"
    },
    {
        "names": ["alp", "alkaline_phosphatase", "alk_phos"],
        "category": "lab_value", "unit": "U/L", "alt_units": {},
        "range_adult": (10, 800), "range_pediatric": (20, 1200),
        "critical_low": None, "critical_high": None,
        "loinc": "6768-6", "description": "Alkaline phosphatase"
    },
    {
        "names": ["bilirubin", "total_bilirubin", "tbili", "bilirubin_total", "bili"],
        "category": "lab_value", "unit": "mg/dL", "alt_units": {"µmol/L": 0.0585},
        "range_adult": (0, 25), "range_pediatric": (0, 30),
        "critical_low": None, "critical_high": 20,
        "loinc": "1975-2", "description": "Total bilirubin"
    },
    {
        "names": ["albumin", "serum_albumin", "alb"],
        "category": "lab_value", "unit": "g/dL", "alt_units": {"g/L": 0.1},
        "range_adult": (1, 7), "range_pediatric": (1.5, 6),
        "critical_low": 1.5, "critical_high": None,
        "loinc": "1751-7", "description": "Albumin"
    },
    {
        "names": ["total_protein", "protein_total", "tp"],
        "category": "lab_value", "unit": "g/dL", "alt_units": {"g/L": 0.1},
        "range_adult": (3, 12), "range_pediatric": (3.5, 10),
        "critical_low": None, "critical_high": None,
        "loinc": "2885-2", "description": "Total protein"
    },
    {
        "names": ["ggt", "gamma_glutamyl_transferase", "gamma_gt", "ggtp"],
        "category": "lab_value", "unit": "U/L", "alt_units": {},
        "range_adult": (0, 800), "range_pediatric": (0, 400),
        "critical_low": None, "critical_high": None,
        "loinc": "2324-2", "description": "GGT"
    },
    {
        "names": ["ldh", "lactate_dehydrogenase", "lactic_dehydrogenase"],
        "category": "lab_value", "unit": "U/L", "alt_units": {},
        "range_adult": (50, 2000), "range_pediatric": (100, 1500),
        "critical_low": None, "critical_high": None,
        "loinc": "2532-0", "description": "Lactate dehydrogenase"
    },

    # ═══════════════════════════════════════════════════════════════
    # LIPID PANEL
    # ═══════════════════════════════════════════════════════════════
    {
        "names": ["total_cholesterol", "cholesterol", "tc", "chol", "total_chol"],
        "category": "lab_value", "unit": "mg/dL", "alt_units": {"mmol/L": 38.67},
        "range_adult": (50, 500), "range_pediatric": (50, 400),
        "critical_low": None, "critical_high": None,
        "loinc": "2093-3", "description": "Total cholesterol"
    },
    {
        "names": ["ldl", "ldl_cholesterol", "ldl_c", "ldl_chol"],
        "category": "lab_value", "unit": "mg/dL", "alt_units": {"mmol/L": 38.67},
        "range_adult": (10, 400), "range_pediatric": (20, 300),
        "critical_low": None, "critical_high": None,
        "loinc": "2089-1", "description": "LDL cholesterol"
    },
    {
        "names": ["hdl", "hdl_cholesterol", "hdl_c", "hdl_chol"],
        "category": "lab_value", "unit": "mg/dL", "alt_units": {"mmol/L": 38.67},
        "range_adult": (5, 150), "range_pediatric": (10, 120),
        "critical_low": None, "critical_high": None,
        "loinc": "2085-9", "description": "HDL cholesterol"
    },
    {
        "names": ["triglycerides", "trig", "tg", "triglyceride"],
        "category": "lab_value", "unit": "mg/dL", "alt_units": {"mmol/L": 88.57},
        "range_adult": (10, 2000), "range_pediatric": (20, 1000),
        "critical_low": None, "critical_high": 1000,
        "loinc": "2571-8", "description": "Triglycerides"
    },

    # ═══════════════════════════════════════════════════════════════
    # COAGULATION
    # ═══════════════════════════════════════════════════════════════
    {
        "names": ["inr", "international_normalized_ratio", "pt_inr"],
        "category": "lab_value", "unit": "ratio", "alt_units": {},
        "range_adult": (0.5, 10), "range_pediatric": (0.5, 8),
        "critical_low": None, "critical_high": 6,
        "loinc": "6301-6", "description": "INR"
    },
    {
        "names": ["pt", "prothrombin_time"],
        "category": "lab_value", "unit": "seconds", "alt_units": {},
        "range_adult": (5, 60), "range_pediatric": (8, 50),
        "critical_low": None, "critical_high": 50,
        "loinc": "5902-2", "description": "Prothrombin time"
    },
    {
        "names": ["aptt", "ptt", "activated_partial_thromboplastin_time"],
        "category": "lab_value", "unit": "seconds", "alt_units": {},
        "range_adult": (15, 120), "range_pediatric": (20, 100),
        "critical_low": None, "critical_high": 100,
        "loinc": "3173-2", "description": "aPTT"
    },
    {
        "names": ["fibrinogen", "fib"],
        "category": "lab_value", "unit": "mg/dL", "alt_units": {"g/L": 100},
        "range_adult": (50, 1000), "range_pediatric": (100, 800),
        "critical_low": 80, "critical_high": 800,
        "loinc": "3255-7", "description": "Fibrinogen"
    },
    {
        "names": ["d_dimer", "ddimer", "d-dimer"],
        "category": "lab_value", "unit": "µg/mL", "alt_units": {"ng/mL": 0.001},
        "range_adult": (0, 30), "range_pediatric": (0, 20),
        "critical_low": None, "critical_high": None,
        "loinc": "48065-7", "description": "D-dimer"
    },

    # ═══════════════════════════════════════════════════════════════
    # RENAL / KIDNEY
    # ═══════════════════════════════════════════════════════════════
    {
        "names": ["gfr", "egfr", "estimated_gfr", "glomerular_filtration_rate"],
        "category": "lab_value", "unit": "mL/min/1.73m²", "alt_units": {},
        "range_adult": (5, 200), "range_pediatric": (10, 180),
        "critical_low": 10, "critical_high": None,
        "loinc": "33914-3", "description": "Estimated GFR"
    },
    {
        "names": ["uric_acid", "urate", "serum_uric_acid"],
        "category": "lab_value", "unit": "mg/dL", "alt_units": {"µmol/L": 0.01681},
        "range_adult": (1, 20), "range_pediatric": (1, 12),
        "critical_low": None, "critical_high": 15,
        "loinc": "3084-1", "description": "Uric acid"
    },

    # ═══════════════════════════════════════════════════════════════
    # ENDOCRINE / THYROID
    # ═══════════════════════════════════════════════════════════════
    {
        "names": ["tsh", "thyroid_stimulating_hormone", "thyrotropin"],
        "category": "lab_value", "unit": "mIU/L", "alt_units": {"µIU/mL": 1},
        "range_adult": (0.01, 100), "range_pediatric": (0.5, 50),
        "critical_low": None, "critical_high": None,
        "loinc": "3016-3", "description": "TSH"
    },
    {
        "names": ["free_t4", "ft4", "free_thyroxine"],
        "category": "lab_value", "unit": "ng/dL", "alt_units": {"pmol/L": 0.0777},
        "range_adult": (0.1, 10), "range_pediatric": (0.3, 8),
        "critical_low": None, "critical_high": None,
        "loinc": "3024-7", "description": "Free T4"
    },
    {
        "names": ["hba1c", "a1c", "glycated_hemoglobin", "hemoglobin_a1c"],
        "category": "lab_value", "unit": "%", "alt_units": {"mmol/mol": lambda x: (x - 2.15) / 0.0915},
        "range_adult": (3, 20), "range_pediatric": (3, 18),
        "critical_low": None, "critical_high": 15,
        "loinc": "4548-4", "description": "HbA1c"
    },

    # ═══════════════════════════════════════════════════════════════
    # CARDIAC MARKERS
    # ═══════════════════════════════════════════════════════════════
    {
        "names": ["troponin", "troponin_i", "tni", "hs_troponin", "troponin_t", "tnt"],
        "category": "lab_value", "unit": "ng/mL", "alt_units": {"ng/L": 0.001, "pg/mL": 1000},
        "range_adult": (0, 50), "range_pediatric": (0, 30),
        "critical_low": None, "critical_high": None,
        "loinc": "10839-9", "description": "Troponin"
    },
    {
        "names": ["bnp", "brain_natriuretic_peptide", "nt_probnp", "ntprobnp"],
        "category": "lab_value", "unit": "pg/mL", "alt_units": {},
        "range_adult": (0, 50000), "range_pediatric": (0, 20000),
        "critical_low": None, "critical_high": None,
        "loinc": "30934-4", "description": "BNP / NT-proBNP"
    },
    {
        "names": ["ck", "creatine_kinase", "cpk", "ck_total"],
        "category": "lab_value", "unit": "U/L", "alt_units": {},
        "range_adult": (10, 5000), "range_pediatric": (20, 3000),
        "critical_low": None, "critical_high": None,
        "loinc": "2157-6", "description": "Creatine kinase"
    },

    # ═══════════════════════════════════════════════════════════════
    # INFLAMMATORY / INFECTION MARKERS
    # ═══════════════════════════════════════════════════════════════
    {
        "names": ["crp", "c_reactive_protein", "hs_crp", "protein_level_crp"],
        "category": "lab_value", "unit": "mg/L", "alt_units": {"mg/dL": 10},
        "range_adult": (0, 500), "range_pediatric": (0, 300),
        "critical_low": None, "critical_high": None,
        "loinc": "1988-5", "description": "C-Reactive Protein"
    },
    {
        "names": ["procalcitonin", "pct"],
        "category": "lab_value", "unit": "ng/mL", "alt_units": {},
        "range_adult": (0, 200), "range_pediatric": (0, 100),
        "critical_low": None, "critical_high": None,
        "loinc": "75241-0", "description": "Procalcitonin"
    },
    {
        "names": ["il6", "interleukin_6", "protein_level_il6"],
        "category": "lab_value", "unit": "pg/mL", "alt_units": {},
        "range_adult": (0, 10000), "range_pediatric": (0, 5000),
        "critical_low": None, "critical_high": None,
        "loinc": "26881-3", "description": "Interleukin-6"
    },
    {
        "names": ["ferritin", "serum_ferritin"],
        "category": "lab_value", "unit": "ng/mL", "alt_units": {"µg/L": 1},
        "range_adult": (5, 5000), "range_pediatric": (10, 2000),
        "critical_low": None, "critical_high": None,
        "loinc": "2276-4", "description": "Ferritin"
    },

    # ═══════════════════════════════════════════════════════════════
    # BLOOD GAS
    # ═══════════════════════════════════════════════════════════════
    {
        "names": ["ph", "blood_ph", "arterial_ph"],
        "category": "lab_value", "unit": "pH", "alt_units": {},
        "range_adult": (6.5, 8.0), "range_pediatric": (6.8, 7.8),
        "critical_low": 6.9, "critical_high": 7.7,
        "loinc": "2744-1", "description": "Blood pH"
    },
    {
        "names": ["pco2", "paco2", "partial_co2"],
        "category": "lab_value", "unit": "mmHg", "alt_units": {"kPa": 7.5006},
        "range_adult": (10, 100), "range_pediatric": (15, 80),
        "critical_low": 15, "critical_high": 80,
        "loinc": "2019-8", "description": "Partial pressure CO2"
    },
    {
        "names": ["po2", "pao2", "partial_o2"],
        "category": "lab_value", "unit": "mmHg", "alt_units": {"kPa": 7.5006},
        "range_adult": (30, 700), "range_pediatric": (40, 500),
        "critical_low": 40, "critical_high": None,
        "loinc": "2703-7", "description": "Partial pressure O2"
    },
    {
        "names": ["lactate", "lactic_acid", "blood_lactate"],
        "category": "lab_value", "unit": "mmol/L", "alt_units": {"mg/dL": 0.111},
        "range_adult": (0.1, 20), "range_pediatric": (0.3, 15),
        "critical_low": None, "critical_high": 10,
        "loinc": "2524-7", "description": "Lactate"
    },

    # ═══════════════════════════════════════════════════════════════
    # DEMOGRAPHICS
    # ═══════════════════════════════════════════════════════════════
    {
        "names": ["age", "age_years", "age_at_enrollment", "patient_age", "age_yrs"],
        "category": "demographics", "unit": "years", "alt_units": {"months": 0.0833},
        "range_adult": (0, 120), "range_pediatric": (0, 18),
        "critical_low": 0, "critical_high": 120,
        "loinc": "30525-0", "description": "Age in years"
    },

    # ═══════════════════════════════════════════════════════════════
    # GENE EXPRESSION / BIOMEDICAL RESEARCH
    # ═══════════════════════════════════════════════════════════════
    {
        "names": ["gene_expr", "gene_expression", "rna_expression", "mrna_level",
                   "gene_expr_tp53", "gene_expr_brca1", "gene_expr_egfr"],
        "category": "biomarker", "unit": "log2(TPM+1)", "alt_units": {},
        "range_adult": (-5, 20), "range_pediatric": None,
        "critical_low": None, "critical_high": None,
        "loinc": None, "description": "Gene expression level"
    },
    {
        "names": ["protein_level", "protein_concentration", "protein_expression"],
        "category": "biomarker", "unit": "varies", "alt_units": {},
        "range_adult": (0, 100000), "range_pediatric": None,
        "critical_low": None, "critical_high": None,
        "loinc": None, "description": "Protein concentration"
    },

    # ═══════════════════════════════════════════════════════════════
    # SURVIVAL / TIME-TO-EVENT
    # ═══════════════════════════════════════════════════════════════
    {
        "names": ["survival_months", "os_months", "overall_survival", "time_to_event",
                   "pfs_months", "progression_free_survival", "dfs_months"],
        "category": "outcome", "unit": "months", "alt_units": {"days": 0.0329, "years": 12},
        "range_adult": (0, 600), "range_pediatric": None,
        "critical_low": 0, "critical_high": None,
        "loinc": None, "description": "Survival / time-to-event"
    },
]


# ═══════════════════════════════════════════════════════════════
# CLINICAL TERMINOLOGY — common coded values and their valid entries
# ═══════════════════════════════════════════════════════════════
CLINICAL_TERMINOLOGY = {
    "sex": {
        "valid": {"M", "F", "Male", "Female", "male", "female", "m", "f", "1", "2",
                  "MALE", "FEMALE", "Unknown", "U", "Other", "Intersex"},
        "canonical": {"M": "M", "Male": "M", "male": "M", "m": "M", "MALE": "M", "1": "M",
                      "F": "F", "Female": "F", "female": "F", "f": "F", "FEMALE": "F", "2": "F",
                      "U": "Unknown", "Unknown": "Unknown", "Other": "Other"},
        "description": "Biological sex"
    },
    "smoking_status": {
        "valid": {"Never", "Former", "Current", "never", "former", "current",
                  "Non-smoker", "Ex-smoker", "Smoker", "Yes", "No", "Unknown",
                  "0", "1", "2", "never smoked", "quit", "active"},
        "canonical": {"Never": "Never", "never": "Never", "Non-smoker": "Never", "No": "Never", "0": "Never",
                      "never smoked": "Never",
                      "Former": "Former", "former": "Former", "Ex-smoker": "Former", "quit": "Former", "1": "Former",
                      "Current": "Current", "current": "Current", "Smoker": "Current", "Yes": "Current",
                      "2": "Current", "active": "Current"},
        "description": "Smoking status"
    },
    "diabetes": {
        "valid": {"No", "Yes", "Type1", "Type2", "T1DM", "T2DM", "DM1", "DM2",
                  "Gestational", "GDM", "Pre-diabetes", "Prediabetes",
                  "0", "1", "2", "no", "yes", "diabetic", "non-diabetic"},
        "canonical": {"No": "No", "no": "No", "0": "No", "non-diabetic": "No",
                      "Yes": "Yes", "yes": "Yes", "1": "Yes", "diabetic": "Yes",
                      "Type1": "Type 1", "T1DM": "Type 1", "DM1": "Type 1",
                      "Type2": "Type 2", "T2DM": "Type 2", "DM2": "Type 2", "2": "Type 2"},
        "description": "Diabetes status"
    },
    "ethnicity": {
        "valid": {"African", "Caucasian", "Asian", "Hispanic", "Other",
                  "Black", "White", "Latino", "Latina", "Mixed", "Unknown",
                  "African American", "African-American", "Native American",
                  "Pacific Islander", "Middle Eastern", "South Asian",
                  "East Asian", "Southeast Asian", "Indigenous"},
        "description": "Ethnicity / Race"
    },
    "ae_severity": {
        "valid": {"Mild", "Moderate", "Severe", "Life-threatening", "Fatal",
                  "mild", "moderate", "severe", "Grade 1", "Grade 2",
                  "Grade 3", "Grade 4", "Grade 5", "1", "2", "3", "4", "5"},
        "canonical": {"Mild": "Mild", "mild": "Mild", "Grade 1": "Mild", "1": "Mild",
                      "Moderate": "Moderate", "moderate": "Moderate", "Grade 2": "Moderate", "2": "Moderate",
                      "Severe": "Severe", "severe": "Severe", "Grade 3": "Severe", "3": "Severe",
                      "Life-threatening": "Life-threatening", "Grade 4": "Life-threatening", "4": "Life-threatening",
                      "Fatal": "Fatal", "Grade 5": "Fatal", "5": "Fatal"},
        "description": "Adverse event severity grade"
    },
    "discharge_disposition": {
        "valid": {"Home", "Rehab", "Transfer", "AMA", "Expired", "Hospice",
                  "SNF", "LTAC", "home", "deceased", "died", "alive",
                  "Discharged", "Against Medical Advice"},
        "description": "Discharge disposition"
    },
    "tumor_stage": {
        "valid": {"I", "II", "III", "IV", "IA", "IB", "IIA", "IIB", "IIIA", "IIIB", "IIIC",
                  "IVA", "IVB", "0", "1", "2", "3", "4", "T1", "T2", "T3", "T4",
                  "N0", "N1", "N2", "N3", "M0", "M1", "NA", "Unknown", ""},
        "description": "Cancer staging (TNM or simplified)"
    },
}


def lookup_reference(column_name: str) -> dict | None:
    """
    Look up clinical reference data for a column name.
    Returns the matching reference entry or None if not recognized.
    Prioritizes exact matches over substring matches.
    """
    col_lower = column_name.lower().strip()
    
    # Pass 1: exact match against any pattern
    for ref in CLINICAL_REFERENCE_DB:
        for pattern in ref["names"]:
            if col_lower == pattern:
                return ref
    
    # Pass 2: column starts with or ends with a pattern (high confidence)
    for ref in CLINICAL_REFERENCE_DB:
        for pattern in ref["names"]:
            if len(pattern) >= 3 and (col_lower.startswith(pattern) or col_lower.endswith(pattern)):
                return ref
    
    # Pass 3: pattern is a significant substring of column (moderate confidence)
    for ref in CLINICAL_REFERENCE_DB:
        for pattern in ref["names"]:
            if len(pattern) >= 4 and pattern in col_lower:
                return ref
    
    return None


def lookup_terminology(column_name: str) -> dict | None:
    """
    Look up clinical terminology validation for a column name.
    Returns valid values and canonical mappings, or None.
    """
    col_lower = column_name.lower().strip()
    for term_name, term_data in CLINICAL_TERMINOLOGY.items():
        if term_name in col_lower or col_lower in term_name:
            return term_data
    return None


def get_all_reference_categories() -> dict:
    """Return all references grouped by category."""
    cats = {}
    for ref in CLINICAL_REFERENCE_DB:
        cat = ref["category"]
        if cat not in cats:
            cats[cat] = []
        cats[cat].append(ref)
    return cats


def get_reference_count() -> int:
    """Return total number of reference ranges in the database."""
    return len(CLINICAL_REFERENCE_DB)


def get_terminology_count() -> int:
    """Return total number of terminology entries."""
    return len(CLINICAL_TERMINOLOGY)

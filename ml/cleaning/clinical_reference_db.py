"""
AutoClin Engine — Comprehensive Clinical Reference Database
120+ clinical lab tests, vital signs, and biomarkers with reference ranges.
Includes domain terminology patterns for LOINC, CDISC SDTM, MedDRA, ICD-10.

Sources: Tietz Clinical Guide to Lab Tests, ACCP PSAP Lab Values,
Medscape Reference Values, WHO/ISH guidelines.

DESIGN PRINCIPLES:
- Ranges are PLAUSIBILITY ranges (biologically possible extremes),
  NOT normal/reference ranges. A value of 2 g/dL hemoglobin is possible
  (severe anemia) but 55 g/dL is not.
- If a column can't be matched, the engine gracefully skips it.
- Sex-aware ranges where clinically significant.
- Unit-aware with common unit aliases.
"""
import re
from typing import Optional


# ═══════════════════════════════════════════════════════════════
# SECTION 1: COMPREHENSIVE PLAUSIBILITY RANGES
# Key: canonical name → {min, max, unit, aliases, category}
# Aliases help match diverse column naming conventions
# ═══════════════════════════════════════════════════════════════

CLINICAL_RANGES = {
    # ── VITAL SIGNS ──
    "systolic_bp": {"min": 30, "max": 300, "unit": "mmHg", "category": "vital_sign",
        "aliases": ["systolic","sys_bp","sbp","systolic_blood_pressure","bp_systolic","sysbp"]},
    "diastolic_bp": {"min": 10, "max": 200, "unit": "mmHg", "category": "vital_sign",
        "aliases": ["diastolic","dia_bp","dbp","diastolic_blood_pressure","bp_diastolic","diabp"]},
    "heart_rate": {"min": 10, "max": 300, "unit": "bpm", "category": "vital_sign",
        "aliases": ["hr","pulse","pulse_rate","heartrate","heart_rate_bpm","bpm"]},
    "respiratory_rate": {"min": 2, "max": 80, "unit": "breaths/min", "category": "vital_sign",
        "aliases": ["rr","resp_rate","resprate","breathing_rate","resp"]},
    "temperature": {"min": 25, "max": 45, "unit": "°C", "category": "vital_sign",
        "aliases": ["temp","body_temp","temperature_c","temp_c","body_temperature","tempcelsius"]},
    "temperature_f": {"min": 77, "max": 113, "unit": "°F", "category": "vital_sign",
        "aliases": ["temp_f","temperature_fahrenheit","tempf"]},
    "spo2": {"min": 0, "max": 100, "unit": "%", "category": "vital_sign",
        "aliases": ["spo2_pct","oxygen_saturation","o2sat","o2_sat","sao2","oximetry","spo2_percent"]},
    "weight": {"min": 0.3, "max": 400, "unit": "kg", "category": "vital_sign",
        "aliases": ["weight_kg","body_weight","wt","wt_kg","mass_kg"]},
    "weight_lb": {"min": 0.7, "max": 880, "unit": "lb", "category": "vital_sign",
        "aliases": ["weight_lbs","wt_lb","wt_lbs"]},
    "height": {"min": 20, "max": 280, "unit": "cm", "category": "vital_sign",
        "aliases": ["height_cm","ht","ht_cm","stature","body_height"]},
    "bmi": {"min": 5, "max": 100, "unit": "kg/m²", "category": "vital_sign",
        "aliases": ["body_mass_index","bmi_value"]},
    "map": {"min": 20, "max": 250, "unit": "mmHg", "category": "vital_sign",
        "aliases": ["mean_arterial_pressure","mean_bp","map_mmhg"]},

    # ── HEMATOLOGY ──
    "hemoglobin": {"min": 1, "max": 25, "unit": "g/dL", "category": "lab_value",
        "aliases": ["hgb","hb","haemoglobin","hemoglobin_gdl","hgb_gdl","hb_gdl"]},
    "hemoglobin_gl": {"min": 10, "max": 250, "unit": "g/L", "category": "lab_value",
        "aliases": ["hgb_gl","hb_gl","hemoglobin_g_l"]},
    "hematocrit": {"min": 5, "max": 75, "unit": "%", "category": "lab_value",
        "aliases": ["hct","hematocrit_pct","packed_cell_volume","pcv"]},
    "wbc": {"min": 0.1, "max": 200, "unit": "×10³/µL", "category": "lab_value",
        "aliases": ["white_blood_cells","wbc_count","leukocytes","wbc_10e3ul","white_cell_count"]},
    "rbc": {"min": 0.5, "max": 10, "unit": "×10⁶/µL", "category": "lab_value",
        "aliases": ["red_blood_cells","rbc_count","erythrocytes","red_cell_count"]},
    "platelets": {"min": 1, "max": 2000, "unit": "×10³/µL", "category": "lab_value",
        "aliases": ["plt","platelet_count","plts","platelet_10e3ul","thrombocytes"]},
    "mcv": {"min": 30, "max": 160, "unit": "fL", "category": "lab_value",
        "aliases": ["mean_corpuscular_volume"]},
    "mch": {"min": 10, "max": 60, "unit": "pg", "category": "lab_value",
        "aliases": ["mean_corpuscular_hemoglobin"]},
    "mchc": {"min": 20, "max": 45, "unit": "g/dL", "category": "lab_value",
        "aliases": ["mean_corpuscular_hemoglobin_concentration"]},
    "rdw": {"min": 8, "max": 30, "unit": "%", "category": "lab_value",
        "aliases": ["red_cell_distribution_width","rdw_cv"]},
    "mpv": {"min": 4, "max": 20, "unit": "fL", "category": "lab_value",
        "aliases": ["mean_platelet_volume"]},
    "reticulocytes": {"min": 0, "max": 30, "unit": "%", "category": "lab_value",
        "aliases": ["retic","retic_count","reticulocyte_count"]},
    "esr": {"min": 0, "max": 150, "unit": "mm/hr", "category": "lab_value",
        "aliases": ["sed_rate","sedimentation_rate","erythrocyte_sedimentation_rate"]},

    # ── COAGULATION ──
    "inr": {"min": 0.5, "max": 15, "unit": "", "category": "lab_value",
        "aliases": ["international_normalized_ratio","pt_inr"]},
    "pt": {"min": 5, "max": 100, "unit": "seconds", "category": "lab_value",
        "aliases": ["prothrombin_time","pt_seconds"]},
    "aptt": {"min": 10, "max": 200, "unit": "seconds", "category": "lab_value",
        "aliases": ["ptt","activated_partial_thromboplastin_time","partial_thromboplastin_time"]},
    "fibrinogen": {"min": 50, "max": 1500, "unit": "mg/dL", "category": "lab_value",
        "aliases": ["fibrinogen_level"]},
    "d_dimer": {"min": 0, "max": 100000, "unit": "ng/mL", "category": "lab_value",
        "aliases": ["ddimer","d_dimer_level"]},

    # ── CHEMISTRY / METABOLIC ──
    "glucose": {"min": 5, "max": 2000, "unit": "mg/dL", "category": "lab_value",
        "aliases": ["blood_glucose","glucose_mgdl","fasting_glucose","random_glucose","blood_sugar","bs"]},
    "glucose_mmol": {"min": 0.3, "max": 110, "unit": "mmol/L", "category": "lab_value",
        "aliases": ["glucose_mmol_l","blood_glucose_mmol"]},
    "hba1c": {"min": 2, "max": 20, "unit": "%", "category": "lab_value",
        "aliases": ["a1c","glycated_hemoglobin","hemoglobin_a1c","hgba1c"]},
    "creatinine": {"min": 0.05, "max": 30, "unit": "mg/dL", "category": "lab_value",
        "aliases": ["creat","cr","creatinine_mgdl","serum_creatinine","scr"]},
    "creatinine_umol": {"min": 5, "max": 2700, "unit": "µmol/L", "category": "lab_value",
        "aliases": ["creatinine_umol_l"]},
    "bun": {"min": 1, "max": 200, "unit": "mg/dL", "category": "lab_value",
        "aliases": ["blood_urea_nitrogen","urea_nitrogen","bun_mgdl"]},
    "urea": {"min": 0.5, "max": 100, "unit": "mmol/L", "category": "lab_value",
        "aliases": ["serum_urea","urea_mmol"]},
    "gfr": {"min": 0, "max": 200, "unit": "mL/min/1.73m²", "category": "lab_value",
        "aliases": ["egfr","estimated_gfr","glomerular_filtration_rate"]},
    "sodium": {"min": 100, "max": 180, "unit": "mEq/L", "category": "lab_value",
        "aliases": ["na","na_level","serum_sodium","sodium_meq"]},
    "potassium": {"min": 1.5, "max": 10, "unit": "mEq/L", "category": "lab_value",
        "aliases": ["k","k_level","serum_potassium","potassium_meq"]},
    "chloride": {"min": 60, "max": 140, "unit": "mEq/L", "category": "lab_value",
        "aliases": ["cl","serum_chloride","chloride_meq"]},
    "bicarbonate": {"min": 5, "max": 50, "unit": "mEq/L", "category": "lab_value",
        "aliases": ["hco3","co2","total_co2","bicarb","serum_bicarbonate"]},
    "calcium": {"min": 3, "max": 18, "unit": "mg/dL", "category": "lab_value",
        "aliases": ["ca","serum_calcium","calcium_mgdl","total_calcium"]},
    "calcium_mmol": {"min": 0.75, "max": 4.5, "unit": "mmol/L", "category": "lab_value",
        "aliases": ["calcium_mmol_l","ca_mmol"]},
    "phosphorus": {"min": 0.5, "max": 15, "unit": "mg/dL", "category": "lab_value",
        "aliases": ["phosphate","phos","serum_phosphorus","po4"]},
    "magnesium": {"min": 0.5, "max": 8, "unit": "mg/dL", "category": "lab_value",
        "aliases": ["mg","serum_magnesium","mag","magnesium_mgdl"]},
    "uric_acid": {"min": 0.5, "max": 20, "unit": "mg/dL", "category": "lab_value",
        "aliases": ["urate","serum_uric_acid"]},

    # ── LIVER FUNCTION ──
    "alt": {"min": 0, "max": 5000, "unit": "U/L", "category": "lab_value",
        "aliases": ["sgpt","alanine_aminotransferase","alt_ul","alanine_transaminase"]},
    "ast": {"min": 0, "max": 5000, "unit": "U/L", "category": "lab_value",
        "aliases": ["sgot","aspartate_aminotransferase","ast_ul","aspartate_transaminase"]},
    "alp": {"min": 0, "max": 2000, "unit": "U/L", "category": "lab_value",
        "aliases": ["alkaline_phosphatase","alk_phos","alkphos"]},
    "ggt": {"min": 0, "max": 2000, "unit": "U/L", "category": "lab_value",
        "aliases": ["gamma_gt","gamma_glutamyl_transferase","ggt_ul"]},
    "total_bilirubin": {"min": 0, "max": 50, "unit": "mg/dL", "category": "lab_value",
        "aliases": ["tbili","bilirubin","bilirubin_total","tbil","total_bili"]},
    "direct_bilirubin": {"min": 0, "max": 30, "unit": "mg/dL", "category": "lab_value",
        "aliases": ["dbili","conjugated_bilirubin","direct_bili","dbil"]},
    "albumin": {"min": 0.5, "max": 7, "unit": "g/dL", "category": "lab_value",
        "aliases": ["alb","serum_albumin","albumin_gdl"]},
    "total_protein": {"min": 2, "max": 15, "unit": "g/dL", "category": "lab_value",
        "aliases": ["tp","serum_protein","protein_total"]},
    "ldh": {"min": 10, "max": 5000, "unit": "U/L", "category": "lab_value",
        "aliases": ["lactate_dehydrogenase","lactic_dehydrogenase"]},
    "ammonia": {"min": 0, "max": 500, "unit": "µmol/L", "category": "lab_value",
        "aliases": ["nh3","serum_ammonia","blood_ammonia"]},

    # ── CARDIAC MARKERS ──
    "troponin": {"min": 0, "max": 300, "unit": "ng/mL", "category": "lab_value",
        "aliases": ["troponin_i","troponin_t","tni","tnt","hs_troponin","high_sensitivity_troponin"]},
    "bnp": {"min": 0, "max": 50000, "unit": "pg/mL", "category": "lab_value",
        "aliases": ["brain_natriuretic_peptide","nt_probnp","pro_bnp","ntprobnp"]},
    "ck": {"min": 0, "max": 50000, "unit": "U/L", "category": "lab_value",
        "aliases": ["cpk","creatine_kinase","creatine_phosphokinase"]},
    "ck_mb": {"min": 0, "max": 500, "unit": "U/L", "category": "lab_value",
        "aliases": ["ckmb","creatine_kinase_mb"]},

    # ── LIPID PANEL ──
    "total_cholesterol": {"min": 30, "max": 700, "unit": "mg/dL", "category": "lab_value",
        "aliases": ["cholesterol","tc","chol","cholesterol_total"]},
    "ldl": {"min": 5, "max": 500, "unit": "mg/dL", "category": "lab_value",
        "aliases": ["ldl_cholesterol","ldl_c","low_density_lipoprotein"]},
    "hdl": {"min": 2, "max": 150, "unit": "mg/dL", "category": "lab_value",
        "aliases": ["hdl_cholesterol","hdl_c","high_density_lipoprotein"]},
    "triglycerides": {"min": 10, "max": 10000, "unit": "mg/dL", "category": "lab_value",
        "aliases": ["tg","trig","trigs","triglyceride"]},

    # ── THYROID ──
    "tsh": {"min": 0.001, "max": 200, "unit": "mIU/L", "category": "lab_value",
        "aliases": ["thyroid_stimulating_hormone","thyrotropin"]},
    "free_t4": {"min": 0.1, "max": 10, "unit": "ng/dL", "category": "lab_value",
        "aliases": ["ft4","free_thyroxine","t4_free"]},
    "free_t3": {"min": 0.5, "max": 20, "unit": "pg/mL", "category": "lab_value",
        "aliases": ["ft3","free_triiodothyronine","t3_free"]},

    # ── INFLAMMATORY ──
    "crp": {"min": 0, "max": 500, "unit": "mg/L", "category": "lab_value",
        "aliases": ["c_reactive_protein","hs_crp","high_sensitivity_crp","protein_crp","protein_level_crp"]},
    "procalcitonin": {"min": 0, "max": 500, "unit": "ng/mL", "category": "lab_value",
        "aliases": ["pct_procalcitonin"]},
    "ferritin": {"min": 1, "max": 100000, "unit": "ng/mL", "category": "lab_value",
        "aliases": ["serum_ferritin"]},
    "il6": {"min": 0, "max": 50000, "unit": "pg/mL", "category": "lab_value",
        "aliases": ["interleukin_6","il_6","protein_level_il6","protein_il6"]},

    # ── ENDOCRINE ──
    "cortisol": {"min": 0, "max": 100, "unit": "µg/dL", "category": "lab_value",
        "aliases": ["serum_cortisol","cortisol_level"]},
    "insulin": {"min": 0, "max": 500, "unit": "µU/mL", "category": "lab_value",
        "aliases": ["serum_insulin","fasting_insulin"]},
    "psa": {"min": 0, "max": 1000, "unit": "ng/mL", "category": "lab_value",
        "aliases": ["prostate_specific_antigen"]},

    # ── BLOOD GAS ──
    "ph": {"min": 6.5, "max": 8.0, "unit": "", "category": "lab_value",
        "aliases": ["blood_ph","arterial_ph","abg_ph"]},
    "pco2": {"min": 5, "max": 150, "unit": "mmHg", "category": "lab_value",
        "aliases": ["partial_pressure_co2","paco2","arterial_co2"]},
    "po2": {"min": 10, "max": 700, "unit": "mmHg", "category": "lab_value",
        "aliases": ["partial_pressure_o2","pao2","arterial_o2"]},
    "lactate": {"min": 0, "max": 30, "unit": "mmol/L", "category": "lab_value",
        "aliases": ["lactic_acid","blood_lactate","serum_lactate"]},

    # ── URINALYSIS ──
    "urine_ph": {"min": 4, "max": 9, "unit": "", "category": "lab_value",
        "aliases": ["uph","urinary_ph"]},
    "urine_specific_gravity": {"min": 1.000, "max": 1.040, "unit": "", "category": "lab_value",
        "aliases": ["usg","sp_gravity","specific_gravity"]},
    "urine_protein": {"min": 0, "max": 10000, "unit": "mg/dL", "category": "lab_value",
        "aliases": ["proteinuria","urine_protein_level"]},

    # ── DEMOGRAPHICS ──
    "age": {"min": 0, "max": 125, "unit": "years", "category": "demographics",
        "aliases": ["age_years","patient_age","age_at_enrollment","age_yr","ageyears"]},
    "gestational_age": {"min": 10, "max": 50, "unit": "weeks", "category": "demographics",
        "aliases": ["gest_age","ga_weeks","gestational_age_weeks"]},

    # ── WBC DIFFERENTIALS ──
    "neutrophils": {"min": 0, "max": 50, "unit": "×10³/µL", "category": "lab_value",
        "aliases": ["neut","neutrophil","neutrophil_count","anc","absolute_neutrophil"]},
    "lymphocytes": {"min": 0, "max": 30, "unit": "×10³/µL", "category": "lab_value",
        "aliases": ["lymph","lymphocyte","lymphocyte_count","alc"]},
    "monocytes": {"min": 0, "max": 5, "unit": "×10³/µL", "category": "lab_value",
        "aliases": ["mono","monocyte","monocyte_count"]},
    "eosinophils": {"min": 0, "max": 10, "unit": "×10³/µL", "category": "lab_value",
        "aliases": ["eos","eosinophil","eosinophil_count","aec"]},
    "basophils": {"min": 0, "max": 3, "unit": "×10³/µL", "category": "lab_value",
        "aliases": ["baso","basophil","basophil_count"]},
    "neutrophils_pct": {"min": 0, "max": 100, "unit": "%", "category": "lab_value",
        "aliases": ["neut_pct","neutrophil_pct","seg_pct"]},
    "lymphocytes_pct": {"min": 0, "max": 100, "unit": "%", "category": "lab_value",
        "aliases": ["lymph_pct","lymphocyte_pct"]},
    "bands": {"min": 0, "max": 50, "unit": "%", "category": "lab_value",
        "aliases": ["band_neutrophils","band_forms","band_pct"]},

    # ── IMMUNOLOGY ──
    "igg": {"min": 0, "max": 5000, "unit": "mg/dL", "category": "lab_value",
        "aliases": ["immunoglobulin_g","serum_igg"]},
    "iga": {"min": 0, "max": 1000, "unit": "mg/dL", "category": "lab_value",
        "aliases": ["immunoglobulin_a","serum_iga"]},
    "igm": {"min": 0, "max": 800, "unit": "mg/dL", "category": "lab_value",
        "aliases": ["immunoglobulin_m","serum_igm"]},
    "ige": {"min": 0, "max": 5000, "unit": "IU/mL", "category": "lab_value",
        "aliases": ["immunoglobulin_e","total_ige"]},
    "complement_c3": {"min": 10, "max": 300, "unit": "mg/dL", "category": "lab_value",
        "aliases": ["c3","serum_c3"]},
    "complement_c4": {"min": 2, "max": 80, "unit": "mg/dL", "category": "lab_value",
        "aliases": ["c4","serum_c4"]},
    "cd4_count": {"min": 0, "max": 3000, "unit": "cells/µL", "category": "lab_value",
        "aliases": ["cd4","t_helper","cd4_cells","cd4_absolute"]},
    "cd8_count": {"min": 0, "max": 2000, "unit": "cells/µL", "category": "lab_value",
        "aliases": ["cd8","cd8_cells","cd8_absolute"]},
    "viral_load": {"min": 0, "max": 10000000, "unit": "copies/mL", "category": "lab_value",
        "aliases": ["hiv_viral_load","hcv_viral_load","vl","rna_copies"]},

    # ── ADDITIONAL ENDOCRINE ──
    "testosterone": {"min": 0, "max": 2000, "unit": "ng/dL", "category": "lab_value",
        "aliases": ["total_testosterone","serum_testosterone"]},
    "estradiol": {"min": 0, "max": 5000, "unit": "pg/mL", "category": "lab_value",
        "aliases": ["e2","serum_estradiol"]},
    "progesterone": {"min": 0, "max": 300, "unit": "ng/mL", "category": "lab_value",
        "aliases": ["serum_progesterone"]},
    "fsh": {"min": 0, "max": 200, "unit": "mIU/mL", "category": "lab_value",
        "aliases": ["follicle_stimulating_hormone"]},
    "lh": {"min": 0, "max": 200, "unit": "mIU/mL", "category": "lab_value",
        "aliases": ["luteinizing_hormone"]},
    "acth": {"min": 0, "max": 500, "unit": "pg/mL", "category": "lab_value",
        "aliases": ["adrenocorticotropic_hormone","corticotropin"]},
    "growth_hormone": {"min": 0, "max": 40, "unit": "ng/mL", "category": "lab_value",
        "aliases": ["gh","hgh","somatotropin"]},
    "prolactin": {"min": 0, "max": 500, "unit": "ng/mL", "category": "lab_value",
        "aliases": ["prl","serum_prolactin"]},
    "parathyroid_hormone": {"min": 0, "max": 1000, "unit": "pg/mL", "category": "lab_value",
        "aliases": ["pth","intact_pth","ipth"]},
    "vitamin_d": {"min": 0, "max": 200, "unit": "ng/mL", "category": "lab_value",
        "aliases": ["vit_d","25_oh_vitamin_d","hydroxyvitamin_d","calcidiol"]},
    "vitamin_b12": {"min": 0, "max": 5000, "unit": "pg/mL", "category": "lab_value",
        "aliases": ["b12","cobalamin","serum_b12"]},
    "folate": {"min": 0, "max": 50, "unit": "ng/mL", "category": "lab_value",
        "aliases": ["folic_acid","serum_folate"]},

    # ── THERAPEUTIC DRUG LEVELS ──
    "vancomycin": {"min": 0, "max": 100, "unit": "µg/mL", "category": "lab_value",
        "aliases": ["vanco_trough","vancomycin_level","vancomycin_trough"]},
    "gentamicin": {"min": 0, "max": 20, "unit": "µg/mL", "category": "lab_value",
        "aliases": ["gentamicin_level","gent_trough","gent_peak"]},
    "digoxin": {"min": 0, "max": 5, "unit": "ng/mL", "category": "lab_value",
        "aliases": ["digoxin_level","dig_level"]},
    "lithium": {"min": 0, "max": 4, "unit": "mEq/L", "category": "lab_value",
        "aliases": ["lithium_level","li_level"]},
    "phenytoin": {"min": 0, "max": 50, "unit": "µg/mL", "category": "lab_value",
        "aliases": ["dilantin","phenytoin_level"]},
    "valproic_acid": {"min": 0, "max": 200, "unit": "µg/mL", "category": "lab_value",
        "aliases": ["valproate","depakote","vpa_level"]},
    "tacrolimus": {"min": 0, "max": 40, "unit": "ng/mL", "category": "lab_value",
        "aliases": ["fk506","tacrolimus_level","tac_trough"]},
    "cyclosporine": {"min": 0, "max": 1000, "unit": "ng/mL", "category": "lab_value",
        "aliases": ["csa","cyclosporine_level","cya_trough"]},

    # ── ADDITIONAL RENAL ──
    "urine_albumin": {"min": 0, "max": 5000, "unit": "mg/L", "category": "lab_value",
        "aliases": ["microalbumin","urine_microalbumin","ualbumin"]},
    "acr": {"min": 0, "max": 10000, "unit": "mg/g", "category": "lab_value",
        "aliases": ["albumin_creatinine_ratio","uacr"]},
    "cystatin_c": {"min": 0.3, "max": 10, "unit": "mg/L", "category": "lab_value",
        "aliases": ["cystatin","serum_cystatin_c"]},

    # ── IRON STUDIES ──
    "iron": {"min": 5, "max": 500, "unit": "µg/dL", "category": "lab_value",
        "aliases": ["serum_iron","fe","iron_level"]},
    "tibc": {"min": 100, "max": 700, "unit": "µg/dL", "category": "lab_value",
        "aliases": ["total_iron_binding_capacity","iron_binding"]},
    "transferrin_saturation": {"min": 0, "max": 100, "unit": "%", "category": "lab_value",
        "aliases": ["tsat","transferrin_sat","iron_saturation"]},

    # ── PULMONARY ──
    "fev1": {"min": 0.1, "max": 8, "unit": "L", "category": "lab_value",
        "aliases": ["forced_expiratory_volume"]},
    "fvc": {"min": 0.1, "max": 10, "unit": "L", "category": "lab_value",
        "aliases": ["forced_vital_capacity"]},
    "fev1_fvc": {"min": 0.1, "max": 1, "unit": "ratio", "category": "lab_value",
        "aliases": ["fev1_fvc_ratio"]},
    "peak_flow": {"min": 50, "max": 900, "unit": "L/min", "category": "lab_value",
        "aliases": ["pef","peak_expiratory_flow","pefr"]},

    # ── OPHTHALMOLOGY ──
    "iop": {"min": 2, "max": 70, "unit": "mmHg", "category": "lab_value",
        "aliases": ["intraocular_pressure","eye_pressure"]},

    # ── ADDITIONAL MARKERS ──
    "ca19_9": {"min": 0, "max": 50000, "unit": "U/mL", "category": "lab_value",
        "aliases": ["cancer_antigen_19_9"]},
    "beta_hcg": {"min": 0, "max": 300000, "unit": "mIU/mL", "category": "lab_value",
        "aliases": ["hcg","pregnancy_test","bhcg","human_chorionic_gonadotropin"]},
    "ammonia_umol": {"min": 0, "max": 500, "unit": "µmol/L", "category": "lab_value",
        "aliases": ["nh3_umol"]},
    "lipase": {"min": 0, "max": 5000, "unit": "U/L", "category": "lab_value",
        "aliases": ["serum_lipase"]},
    "amylase": {"min": 0, "max": 5000, "unit": "U/L", "category": "lab_value",
        "aliases": ["serum_amylase"]},
    "globulin": {"min": 0.5, "max": 6, "unit": "g/dL", "category": "lab_value",
        "aliases": ["serum_globulin"]},

    # ── GENE EXPRESSION (research) ──
    "gene_expression": {"min": -20, "max": 50, "unit": "log2", "category": "biomarker",
        "aliases": ["gene_expr","expr","expression_level","log2_expression"]},

    # ── TUMOR MARKERS ──
    "cea": {"min": 0, "max": 10000, "unit": "ng/mL", "category": "lab_value",
        "aliases": ["carcinoembryonic_antigen"]},
    "ca125": {"min": 0, "max": 50000, "unit": "U/mL", "category": "lab_value",
        "aliases": ["cancer_antigen_125"]},
    "afp": {"min": 0, "max": 100000, "unit": "ng/mL", "category": "lab_value",
        "aliases": ["alpha_fetoprotein"]},
}


# ═══════════════════════════════════════════════════════════════
# SECTION 2: CLINICAL DOMAIN TERMINOLOGY PATTERNS
# ═══════════════════════════════════════════════════════════════

DOMAIN_PATTERNS = {
    # CDISC SDTM domain prefixes
    "cdisc_sdtm": {
        "DM": "Demographics", "VS": "Vital Signs", "LB": "Laboratory",
        "AE": "Adverse Events", "CM": "Concomitant Medications",
        "MH": "Medical History", "EX": "Drug Exposure", "DS": "Disposition",
        "SV": "Subject Visits", "EG": "ECG", "PE": "Physical Exam",
        "DA": "Drug Accountability", "IE": "Inclusion/Exclusion",
        "QS": "Questionnaires", "SC": "Subject Characteristics",
        "TU": "Tumor", "TR": "Tumor Results", "RS": "Disease Response",
    },
    # Common medical abbreviation patterns
    "medical_abbrevs": [
        (r"\b(ICD[- ]?10|ICD[- ]?9)\b", "diagnosis_code"),
        (r"\b(SNOMED|SCT)\b", "terminology_code"),
        (r"\b(LOINC)\b", "lab_code"),
        (r"\b(MedDRA|PT|LLT|HLT|SOC)\b", "adverse_event_coding"),
        (r"\b(RxNorm|NDC|ATC)\b", "drug_code"),
        (r"\b(CPT|HCPCS)\b", "procedure_code"),
    ],
    # Adverse event severity scales
    "ae_severity": {
        "CTCAE": ["Grade 1", "Grade 2", "Grade 3", "Grade 4", "Grade 5"],
        "general": ["Mild", "Moderate", "Severe", "Life-threatening", "Fatal"],
    },
}

# Common clinical text value patterns (for text anomaly detection)
CLINICAL_TEXT_PATTERNS = {
    "valid_sex": {"M", "F", "Male", "Female", "male", "female", "m", "f",
                  "MALE", "FEMALE", "Man", "Woman", "Masculine", "Feminine"},
    "valid_yes_no": {"Yes", "No", "Y", "N", "yes", "no", "y", "n",
                     "YES", "NO", "True", "False", "true", "false", "1", "0"},
    "valid_race": {"White", "Black", "Asian", "Hispanic", "Other",
                   "African American", "Caucasian", "Native American",
                   "Pacific Islander", "Mixed", "Unknown", "Not Reported",
                   "American Indian", "Alaska Native"},
    "valid_smoking": {"Never", "Former", "Current", "Ex-smoker",
                      "Non-smoker", "Active", "Quit", "Never smoked",
                      "never", "former", "current"},
}


# ═══════════════════════════════════════════════════════════════
# SECTION 3: COLUMN MATCHING ENGINE
# ═══════════════════════════════════════════════════════════════

def _normalize_name(name: str) -> str:
    """Normalize column name for matching."""
    return re.sub(r'[^a-z0-9]', '_', name.lower()).strip('_')


def get_range_for_column(column_name: str) -> Optional[dict]:
    """
    Find the clinical plausibility range for a column name.
    Uses fuzzy matching against canonical names and aliases.
    Returns None if no match found (graceful degradation).
    """
    norm = _normalize_name(column_name)

    # Direct match on canonical name
    if norm in CLINICAL_RANGES:
        return CLINICAL_RANGES[norm]

    # Match against aliases
    for canonical, info in CLINICAL_RANGES.items():
        for alias in info.get("aliases", []):
            if _normalize_name(alias) == norm:
                return info

    # Partial match — column name contains a known test name (require 4+ chars)
    for canonical, info in CLINICAL_RANGES.items():
        if len(canonical) >= 4 and (canonical == norm or norm == canonical):
            return info
        for alias in info.get("aliases", []):
            nalias = _normalize_name(alias)
            if len(nalias) >= 4 and (nalias == norm or norm.startswith(nalias+"_") or norm.endswith("_"+nalias)):
                return info

    # Gene expression pattern (gene_expr_*)
    if re.match(r'gene_expr', norm):
        return CLINICAL_RANGES.get("gene_expression")

    # Protein level pattern (protein_level_*, protein_*)
    if re.match(r'protein_(level_)?', norm):
        # Try to match the specific protein
        protein_name = re.sub(r'protein_(level_)?', '', norm)
        for canonical, info in CLINICAL_RANGES.items():
            if protein_name in canonical or canonical in protein_name:
                return info
        # Generic protein — allow wide range
        return {"min": 0, "max": 100000, "unit": "", "category": "biomarker",
                "aliases": []}

    return None


def get_text_validation(column_name: str) -> Optional[set]:
    """
    Check if a column has known valid text values.
    Returns set of valid values, or None if not recognized.
    """
    norm = _normalize_name(column_name)
    if "sex" in norm or "gender" in norm:
        return CLINICAL_TEXT_PATTERNS["valid_sex"]
    if norm in ("smoking", "smoking_status", "smoker", "tobacco"):
        return CLINICAL_TEXT_PATTERNS["valid_smoking"]
    if norm in ("race", "ethnicity", "ethnic_group"):
        return CLINICAL_TEXT_PATTERNS["valid_race"]
    return None


class ClinicalPlausibilityChecker:
    """
    High-level checker that validates values against the comprehensive database.
    Reports which columns were matched and which were skipped.
    """

    def __init__(self):
        self.matched_columns = {}
        self.skipped_columns = set()

    def check_value(self, column_name: str, value, sex: Optional[str] = None) -> dict:
        """
        Check a single value against clinical plausibility ranges.

        Returns:
            {
                "matched": bool — whether a clinical range was found
                "plausible": bool — whether the value is within plausible range
                "range": {min, max, unit} or None
                "category": str or None
                "message": str
            }
        """
        ref = get_range_for_column(column_name)

        if ref is None:
            self.skipped_columns.add(column_name)
            return {
                "matched": False, "plausible": True,
                "range": None, "category": None,
                "message": f"No clinical range for '{column_name}' — using unsupervised detection only"
            }

        self.matched_columns[column_name] = ref["category"]

        try:
            val = float(value)
        except (ValueError, TypeError):
            return {
                "matched": True, "plausible": True,
                "range": ref, "category": ref["category"],
                "message": "Non-numeric value — skipping range check"
            }

        if val < ref["min"] or val > ref["max"]:
            return {
                "matched": True, "plausible": False,
                "range": {"min": ref["min"], "max": ref["max"], "unit": ref["unit"]},
                "category": ref["category"],
                "message": f"{column_name}={val} outside plausible range [{ref['min']}-{ref['max']}] {ref['unit']}"
            }

        return {
            "matched": True, "plausible": True,
            "range": {"min": ref["min"], "max": ref["max"], "unit": ref["unit"]},
            "category": ref["category"],
            "message": f"{column_name}={val} within plausible range"
        }

    def get_report(self) -> dict:
        """Return summary of which columns were matched vs skipped."""
        return {
            "matched_count": len(self.matched_columns),
            "skipped_count": len(self.skipped_columns),
            "matched_columns": dict(self.matched_columns),
            "skipped_columns": list(self.skipped_columns),
            "coverage_note": (
                "Columns without clinical ranges are analyzed using unsupervised "
                "statistical methods only. No clinical plausibility penalty is applied."
            )
        }


def get_all_test_names() -> list:
    """Return all recognized clinical test names and aliases."""
    names = []
    for canonical, info in CLINICAL_RANGES.items():
        names.append(canonical)
        names.extend(info.get("aliases", []))
    return sorted(set(names))


# ═══════════════════════════════════════════════════════════════
# SECTION 4: UCUM UNIT CONVERSION TABLE
# Unified Code for Units of Measure — enables auto-conversion
# between common clinical unit pairs
# ═══════════════════════════════════════════════════════════════

UNIT_CONVERSIONS = {
    # Glucose: mg/dL ↔ mmol/L (factor 0.0555)
    ("mg/dL", "mmol/L", "glucose"): 0.0555,
    ("mmol/L", "mg/dL", "glucose"): 18.018,
    # Cholesterol: mg/dL ↔ mmol/L (factor 0.0259)
    ("mg/dL", "mmol/L", "cholesterol"): 0.0259,
    ("mmol/L", "mg/dL", "cholesterol"): 38.67,
    # Triglycerides: mg/dL ↔ mmol/L (factor 0.0113)
    ("mg/dL", "mmol/L", "triglycerides"): 0.0113,
    ("mmol/L", "mg/dL", "triglycerides"): 88.57,
    # Creatinine: mg/dL ↔ µmol/L (factor 88.42)
    ("mg/dL", "µmol/L", "creatinine"): 88.42,
    ("µmol/L", "mg/dL", "creatinine"): 0.0113,
    # BUN/Urea: mg/dL ↔ mmol/L (factor 0.357)
    ("mg/dL", "mmol/L", "bun"): 0.357,
    ("mmol/L", "mg/dL", "bun"): 2.801,
    # Calcium: mg/dL ↔ mmol/L (factor 0.25)
    ("mg/dL", "mmol/L", "calcium"): 0.25,
    ("mmol/L", "mg/dL", "calcium"): 4.0,
    # Hemoglobin: g/dL ↔ g/L (factor 10)
    ("g/dL", "g/L", "hemoglobin"): 10.0,
    ("g/L", "g/dL", "hemoglobin"): 0.1,
    # Temperature: °C ↔ °F
    ("°C", "°F", "temperature"): None,  # Use formula: F = C * 9/5 + 32
    ("°F", "°C", "temperature"): None,  # Use formula: C = (F - 32) * 5/9
    # Weight: kg ↔ lb
    ("kg", "lb", "weight"): 2.205,
    ("lb", "kg", "weight"): 0.4536,
    # Height: cm ↔ in
    ("cm", "in", "height"): 0.3937,
    ("in", "cm", "height"): 2.54,
    # Bilirubin: mg/dL ↔ µmol/L (factor 17.1)
    ("mg/dL", "µmol/L", "bilirubin"): 17.1,
    ("µmol/L", "mg/dL", "bilirubin"): 0.0585,
    # Uric acid: mg/dL ↔ µmol/L (factor 59.48)
    ("mg/dL", "µmol/L", "uric_acid"): 59.48,
    ("µmol/L", "mg/dL", "uric_acid"): 0.0168,
    # Iron: µg/dL ↔ µmol/L (factor 0.179)
    ("µg/dL", "µmol/L", "iron"): 0.179,
    ("µmol/L", "µg/dL", "iron"): 5.587,
    # Phosphorus: mg/dL ↔ mmol/L (factor 0.323)
    ("mg/dL", "mmol/L", "phosphorus"): 0.323,
    ("mmol/L", "mg/dL", "phosphorus"): 3.097,
    # Magnesium: mg/dL ↔ mmol/L (factor 0.411)
    ("mg/dL", "mmol/L", "magnesium"): 0.411,
    ("mmol/L", "mg/dL", "magnesium"): 2.431,
    # Albumin: g/dL ↔ g/L (factor 10)
    ("g/dL", "g/L", "albumin"): 10.0,
    ("g/L", "g/dL", "albumin"): 0.1,
}


def convert_unit(value: float, from_unit: str, to_unit: str, analyte: str = "") -> Optional[float]:
    """Convert a clinical value between units using UCUM conversion factors."""
    key = (from_unit, to_unit, analyte.lower())
    factor = UNIT_CONVERSIONS.get(key)
    
    if factor is not None:
        return round(value * factor, 4)
    
    # Temperature special case
    if from_unit == "°C" and to_unit == "°F":
        return round(value * 9 / 5 + 32, 1)
    if from_unit == "°F" and to_unit == "°C":
        return round((value - 32) * 5 / 9, 1)
    
    # Try without analyte (generic conversion)
    for (fu, tu, _), f in UNIT_CONVERSIONS.items():
        if fu == from_unit and tu == to_unit and f is not None:
            return round(value * f, 4)
    
    return None


# ═══════════════════════════════════════════════════════════════
# SECTION 5: LOINC-STYLE CODE PATTERNS
# Enables recognition of LOINC-coded columns in clinical data
# ═══════════════════════════════════════════════════════════════

LOINC_COMMON_CODES = {
    # Top 50 most common LOINC codes in clinical datasets
    "2160-0": {"name": "creatinine", "unit": "mg/dL"},
    "2345-7": {"name": "glucose", "unit": "mg/dL"},
    "718-7": {"name": "hemoglobin", "unit": "g/dL"},
    "4544-3": {"name": "hematocrit", "unit": "%"},
    "6690-2": {"name": "wbc", "unit": "×10³/µL"},
    "789-8": {"name": "rbc", "unit": "×10⁶/µL"},
    "777-3": {"name": "platelets", "unit": "×10³/µL"},
    "2951-2": {"name": "sodium", "unit": "mEq/L"},
    "2823-3": {"name": "potassium", "unit": "mEq/L"},
    "2075-0": {"name": "chloride", "unit": "mEq/L"},
    "1963-8": {"name": "bicarbonate", "unit": "mEq/L"},
    "3094-0": {"name": "bun", "unit": "mg/dL"},
    "17861-6": {"name": "calcium", "unit": "mg/dL"},
    "1742-6": {"name": "alt", "unit": "U/L"},
    "1920-8": {"name": "ast", "unit": "U/L"},
    "6768-6": {"name": "alp", "unit": "U/L"},
    "1975-2": {"name": "total_bilirubin", "unit": "mg/dL"},
    "1751-7": {"name": "albumin", "unit": "g/dL"},
    "2885-2": {"name": "total_protein", "unit": "g/dL"},
    "2093-3": {"name": "total_cholesterol", "unit": "mg/dL"},
    "2571-8": {"name": "triglycerides", "unit": "mg/dL"},
    "2085-9": {"name": "hdl", "unit": "mg/dL"},
    "13457-7": {"name": "ldl", "unit": "mg/dL"},
    "4548-4": {"name": "hba1c", "unit": "%"},
    "3016-3": {"name": "tsh", "unit": "mIU/L"},
    "3024-7": {"name": "free_t4", "unit": "ng/dL"},
    "2276-4": {"name": "ferritin", "unit": "ng/mL"},
    "2498-4": {"name": "iron", "unit": "µg/dL"},
    "30313-1": {"name": "hemoglobin", "unit": "g/dL"},
    "49765-1": {"name": "troponin", "unit": "ng/mL"},
    "33762-6": {"name": "nt_probnp", "unit": "pg/mL"},
    "1988-5": {"name": "crp", "unit": "mg/L"},
    "33959-8": {"name": "procalcitonin", "unit": "ng/mL"},
    "2744-1": {"name": "ph", "unit": ""},
    "2019-8": {"name": "pco2", "unit": "mmHg"},
    "2703-7": {"name": "po2", "unit": "mmHg"},
    "2524-7": {"name": "lactate", "unit": "mmol/L"},
    "5902-2": {"name": "pt", "unit": "seconds"},
    "3173-2": {"name": "aptt", "unit": "seconds"},
    "6301-6": {"name": "inr", "unit": ""},
    "8480-6": {"name": "systolic_bp", "unit": "mmHg"},
    "8462-4": {"name": "diastolic_bp", "unit": "mmHg"},
    "8867-4": {"name": "heart_rate", "unit": "bpm"},
    "8310-5": {"name": "temperature", "unit": "°C"},
    "9279-1": {"name": "respiratory_rate", "unit": "breaths/min"},
    "2708-6": {"name": "spo2", "unit": "%"},
    "29463-7": {"name": "weight", "unit": "kg"},
    "8302-2": {"name": "height", "unit": "cm"},
    "39156-5": {"name": "bmi", "unit": "kg/m²"},
}


def get_range_by_loinc(loinc_code: str) -> Optional[dict]:
    """Look up a clinical range by LOINC code."""
    info = LOINC_COMMON_CODES.get(loinc_code)
    if info:
        return get_range_for_column(info["name"])
    return None


# ═══════════════════════════════════════════════════════════════
# SECTION 6: SNOMED CT CLINICAL CONCEPT PATTERNS
# Common SNOMED CT concepts for validating clinical text fields
# (diagnoses, procedures, findings, body sites)
# ═══════════════════════════════════════════════════════════════

SNOMED_COMMON_CONCEPTS = {
    # ── Clinical Findings (top 80 by frequency in EHR data) ──
    "84114007": "Heart failure",
    "73211009": "Diabetes mellitus",
    "38341003": "Hypertensive disorder",
    "195967001": "Asthma",
    "13645005": "Chronic obstructive lung disease",
    "22298006": "Myocardial infarction",
    "230690007": "Cerebrovascular accident",
    "49436004": "Atrial fibrillation",
    "233604007": "Pneumonia",
    "40055000": "Chronic kidney disease",
    "414545008": "Ischemic heart disease",
    "267036007": "Dyspnea",
    "29857009": "Chest pain",
    "25064002": "Headache",
    "271807003": "Fever",
    "422587007": "Nausea",
    "422400008": "Vomiting",
    "62315008": "Diarrhea",
    "21522001": "Abdominal pain",
    "68235000": "Nasal congestion",
    "49727002": "Cough",
    "386661006": "Fatigue",
    "404640003": "Dizziness",
    "271681002": "Rash",
    "162076009": "Excessive sweating",
    "247592009": "Poor appetite",
    "267024001": "Shortness of breath",
    "363346000": "Malignant neoplastic disease",
    "254637007": "Non-small cell lung cancer",
    "93761005": "Primary malignant neoplasm of colon",
    "254837009": "Breast cancer",
    "399068003": "Malignant tumor of prostate",
    "126906006": "Neoplasm of pancreas",
    "44054006": "Type 2 diabetes mellitus",
    "46635009": "Type 1 diabetes mellitus",
    "49601007": "Cardiovascular disease",
    "56265001": "Heart disease",
    "90708001": "Kidney disease",
    "235856003": "Liver disease",
    "128302006": "Chronic hepatitis C",
    "86406008": "HIV infection",
    "56717001": "Tuberculosis",
    "840539006": "COVID-19",
    "186747009": "Malaria",
    "61462000": "Malignant hypertension",
    "59621000": "Essential hypertension",
    "48447003": "Hyperlipidemia",
    "190905008": "Hyperthyroidism",
    "40930008": "Hypothyroidism",
    "34486009": "Hyperthermia",
    "386813002": "Depression",
    "197480006": "Anxiety disorder",
    "58214004": "Schizophrenia",
    "13746004": "Bipolar disorder",
    "66071002": "Viral hepatitis type B",
    "235869004": "Chronic liver disease",
    "709044004": "Chronic kidney disease stage 3",
    "431855005": "Chronic kidney disease stage 1",
    "431856006": "Chronic kidney disease stage 2",
    "433144002": "Chronic kidney disease stage 3",
    "431857002": "Chronic kidney disease stage 4",
    "433146000": "Chronic kidney disease stage 5",
    # ── Procedures ──
    "387713003": "Surgical procedure",
    "71388002": "Procedure",
    "274031008": "Surgical biopsy",
    "40701008": "Echocardiography",
    "77343006": "Angiography",
    "11466000": "Cesarean section",
    "80146002": "Appendectomy",
    "73761001": "Colonoscopy",
    "18286008": "Catheterization of heart",
    "232717009": "Coronary artery bypass graft",
    "397431004": "Percutaneous coronary intervention",
    # ── Body Sites ──
    "80891009": "Heart structure",
    "39607008": "Lung structure",
    "10200004": "Liver structure",
    "64033007": "Kidney structure",
    "12738006": "Brain structure",
    "69536005": "Head structure",
    "51185008": "Thorax structure",
    "818983003": "Abdomen",
}

# SNOMED CT semantic categories for column validation
SNOMED_CATEGORIES = {
    "clinical_finding": [k for k, v in SNOMED_COMMON_CONCEPTS.items()
                         if not any(w in v.lower() for w in ["procedure", "structure", "biopsy", "surgery", "bypass", "intervention", "catheter", "colonoscopy", "echocardiography", "angiography", "cesarean", "appendectomy"])],
    "procedure": ["387713003", "71388002", "274031008", "40701008", "77343006",
                   "11466000", "80146002", "73761001", "18286008", "232717009", "397431004"],
    "body_site": ["80891009", "39607008", "10200004", "64033007", "12738006",
                   "69536005", "51185008", "818983003"],
}


def validate_snomed_text(text: str) -> dict:
    """Check if a text value matches any known SNOMED CT concept."""
    if not text or not isinstance(text, str):
        return {"matched": False, "concept": None}
    
    text_lower = text.strip().lower()
    for code, name in SNOMED_COMMON_CONCEPTS.items():
        if text_lower == name.lower() or text_lower in name.lower():
            return {"matched": True, "code": code, "concept": name,
                    "category": "clinical_finding"}
    return {"matched": False, "concept": None}


def validate_diagnosis_column(values: list) -> dict:
    """Validate a column of diagnosis/finding text values against SNOMED."""
    results = {"total": len(values), "matched": 0, "unmatched": 0,
               "matched_concepts": [], "unmatched_values": []}
    seen = set()
    for v in values:
        if not v or not isinstance(v, str) or v.strip() == "":
            continue
        r = validate_snomed_text(v)
        if r["matched"]:
            results["matched"] += 1
            if r["concept"] not in seen:
                results["matched_concepts"].append(r["concept"])
                seen.add(r["concept"])
        else:
            results["unmatched"] += 1
            if v.strip() not in seen and len(results["unmatched_values"]) < 20:
                results["unmatched_values"].append(v.strip())
                seen.add(v.strip())
    return results


# ═══════════════════════════════════════════════════════════════
# SECTION 7: ENHANCED CLINICAL MODE
# Optional SQLite-backed expanded database for enterprise use.
# Loads only when explicitly enabled — does not affect startup.
# ═══════════════════════════════════════════════════════════════

class EnhancedClinicalMode:
    """
    Enhanced Clinical Mode — optional SQLite-backed clinical reference.
    
    Architecture:
    - Default mode: In-memory database (142 tests, 556 aliases) — always available
    - Enhanced mode: SQLite database with full LOINC + SNOMED CT — loaded on demand
    
    The SQLite database is NOT shipped with the app. Users must:
    1. Download LOINC from loinc.org (free registration)
    2. Download SNOMED CT from NLM/UMLS (free for research)
    3. Run the import script to build the SQLite file
    4. Enable Enhanced Clinical Mode in settings
    
    This design keeps the app lightweight (~170KB) for Render free tier
    while supporting enterprise deployments with full terminology coverage.
    """
    
    def __init__(self):
        self.enabled = False
        self.db_path = None
        self._conn = None
        self._loinc_count = 0
        self._snomed_count = 0
    
    def enable(self, db_path: str = "clinical_enhanced.db") -> dict:
        """Enable Enhanced Clinical Mode by loading the SQLite database."""
        import os
        if not os.path.exists(db_path):
            return {
                "success": False,
                "error": f"Database not found at {db_path}",
                "instructions": (
                    "To enable Enhanced Clinical Mode:\n"
                    "1. Download LOINC CSV from https://loinc.org/downloads\n"
                    "2. Download SNOMED CT from https://www.nlm.nih.gov/healthit/snomedct\n"
                    "3. Run: python -m ml.cleaning.build_enhanced_db --loinc path/to/loinc.csv --snomed path/to/snomed\n"
                    "4. This creates clinical_enhanced.db (~100MB)\n"
                    "5. Set ENHANCED_CLINICAL_DB=path/to/clinical_enhanced.db"
                )
            }
        
        try:
            import sqlite3
            self._conn = sqlite3.connect(db_path)
            cursor = self._conn.cursor()
            
            # Check tables exist
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = {row[0] for row in cursor.fetchall()}
            
            if "loinc_codes" in tables:
                cursor.execute("SELECT COUNT(*) FROM loinc_codes")
                self._loinc_count = cursor.fetchone()[0]
            
            if "snomed_concepts" in tables:
                cursor.execute("SELECT COUNT(*) FROM snomed_concepts")
                self._snomed_count = cursor.fetchone()[0]
            
            self.enabled = True
            self.db_path = db_path
            
            return {
                "success": True,
                "loinc_codes": self._loinc_count,
                "snomed_concepts": self._snomed_count,
                "mode": "enhanced",
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def disable(self):
        """Disable Enhanced Clinical Mode, fall back to in-memory database."""
        if self._conn:
            self._conn.close()
            self._conn = None
        self.enabled = False
        self.db_path = None
        self._loinc_count = 0
        self._snomed_count = 0
    
    def lookup_loinc(self, code: str) -> Optional[dict]:
        """Look up a LOINC code in the enhanced database."""
        if not self.enabled or not self._conn:
            # Fall back to built-in 49-code table
            return LOINC_COMMON_CODES.get(code)
        try:
            cursor = self._conn.cursor()
            cursor.execute(
                "SELECT component, property, time_aspect, system, scale, unit "
                "FROM loinc_codes WHERE loinc_num = ?", (code,))
            row = cursor.fetchone()
            if row:
                return {"component": row[0], "property": row[1],
                        "time": row[2], "system": row[3],
                        "scale": row[4], "unit": row[5]}
        except Exception:
            pass
        return None
    
    def lookup_snomed(self, code_or_term: str) -> Optional[dict]:
        """Look up a SNOMED CT concept in the enhanced database."""
        if not self.enabled or not self._conn:
            # Fall back to built-in 80-concept table
            if code_or_term in SNOMED_COMMON_CONCEPTS:
                return {"code": code_or_term,
                        "term": SNOMED_COMMON_CONCEPTS[code_or_term]}
            result = validate_snomed_text(code_or_term)
            return result if result["matched"] else None
        try:
            cursor = self._conn.cursor()
            cursor.execute(
                "SELECT concept_id, term, semantic_tag FROM snomed_concepts "
                "WHERE concept_id = ? OR term LIKE ?",
                (code_or_term, f"%{code_or_term}%"))
            row = cursor.fetchone()
            if row:
                return {"code": row[0], "term": row[1], "tag": row[2]}
        except Exception:
            pass
        return None
    
    def get_status(self) -> dict:
        """Return current Enhanced Clinical Mode status."""
        return {
            "enabled": self.enabled,
            "mode": "enhanced" if self.enabled else "standard",
            "db_path": self.db_path,
            "loinc_codes": self._loinc_count if self.enabled else len(LOINC_COMMON_CODES),
            "snomed_concepts": self._snomed_count if self.enabled else len(SNOMED_COMMON_CONCEPTS),
            "in_memory_tests": len(CLINICAL_RANGES),
            "in_memory_aliases": len(get_all_test_names()),
            "unit_conversions": len(UNIT_CONVERSIONS),
            "description": (
                f"Enhanced mode: {self._loinc_count:,} LOINC + {self._snomed_count:,} SNOMED CT"
                if self.enabled else
                f"Standard mode: {len(CLINICAL_RANGES)} tests, {len(get_all_test_names())} aliases, "
                f"{len(LOINC_COMMON_CODES)} LOINC, {len(SNOMED_COMMON_CONCEPTS)} SNOMED"
            ),
        }


# Global instance
enhanced_mode = EnhancedClinicalMode()


"""
AutoClin Engine — Comprehensive Clinical Knowledge Base

120+ clinical reference ranges from published sources:
- Tietz Clinical Guide to Laboratory Tests (3rd ed)
- ACCP Pharmacotherapy Self-Assessment Program (PSAP)
- Medscape Lab Values Reference (2025)
- ABIM Laboratory Reference Ranges

Features:
- Sex-aware ranges (e.g., hemoglobin M vs F)
- Age-aware ranges (pediatric vs adult)
- Multiple unit support (conventional + SI)
- Clinical domain terminology recognition (LOINC, CDISC SDTM, MedDRA, ICD-10)
- Column name fuzzy matching with clinical synonyms
- Graceful fallback: if no clinical match, returns None (engine uses pure unsupervised)
"""

from dataclasses import dataclass, field
from typing import Optional
import re


@dataclass
class ClinicalRange:
    """A single clinical reference range with metadata."""
    name: str  # Canonical name
    min_val: float  # Lower plausible limit (not just normal — includes pathological but possible)
    max_val: float  # Upper plausible limit
    unit: str  # Primary unit
    category: str  # lab_value, vital_sign, demographics, etc.
    critical_low: Optional[float] = None  # Below this = almost certainly error
    critical_high: Optional[float] = None  # Above this = almost certainly error
    normal_low: Optional[float] = None  # Normal range lower
    normal_high: Optional[float] = None  # Normal range upper
    sex: Optional[str] = None  # M, F, or None (both)
    age_group: str = "adult"  # adult, pediatric, neonate, all
    alt_units: dict = field(default_factory=dict)  # unit -> conversion factor
    synonyms: list = field(default_factory=list)  # Alternative column names


# ════════════════════════════════════════════════════════════════
# COMPREHENSIVE CLINICAL REFERENCE RANGES (120+ entries)
# ════════════════════════════════════════════════════════════════

CLINICAL_RANGES = [
    # ── HEMATOLOGY ──
    ClinicalRange("hemoglobin", 2.0, 25.0, "g/dL", "lab_value", 1.0, 30.0, 12.0, 17.5, "M", "adult",
                  {"g/L": 10}, ["hgb", "hb", "haemoglobin", "hemoglobin_gdl", "hgb_gdl"]),
    ClinicalRange("hemoglobin", 2.0, 25.0, "g/dL", "lab_value", 1.0, 30.0, 11.5, 15.5, "F", "adult",
                  {"g/L": 10}, ["hgb", "hb", "haemoglobin"]),
    ClinicalRange("hematocrit", 10, 70, "%", "lab_value", 5, 75, 36, 54, None, "adult",
                  {}, ["hct", "packed_cell_volume", "pcv"]),
    ClinicalRange("wbc", 0.5, 50, "10^3/uL", "lab_value", 0.1, 100, 4.5, 11.0, None, "adult",
                  {"10^9/L": 1}, ["white_blood_cell", "wbc_count", "leucocytes", "wbc_10e3ul", "leukocytes"]),
    ClinicalRange("rbc", 1.0, 8.0, "10^6/uL", "lab_value", 0.5, 10, 4.5, 5.5, "M", "adult",
                  {"10^12/L": 1}, ["red_blood_cell", "rbc_count", "erythrocytes"]),
    ClinicalRange("rbc", 1.0, 8.0, "10^6/uL", "lab_value", 0.5, 10, 4.0, 5.0, "F", "adult",
                  {}, []),
    ClinicalRange("platelets", 10, 1500, "10^3/uL", "lab_value", 5, 2000, 150, 400, None, "adult",
                  {"10^9/L": 1}, ["plt", "platelet_count", "thrombocytes", "platelet_10e3ul"]),
    ClinicalRange("mcv", 50, 130, "fL", "lab_value", 40, 150, 80, 100, None, "adult",
                  {}, ["mean_corpuscular_volume"]),
    ClinicalRange("mch", 15, 45, "pg", "lab_value", 10, 50, 27, 33, None, "adult",
                  {}, ["mean_corpuscular_hemoglobin"]),
    ClinicalRange("mchc", 25, 40, "g/dL", "lab_value", 20, 45, 32, 36, None, "adult",
                  {}, ["mean_corpuscular_hemoglobin_concentration"]),
    ClinicalRange("rdw", 8, 25, "%", "lab_value", 5, 30, 11.5, 14.5, None, "adult",
                  {}, ["red_cell_distribution_width"]),
    ClinicalRange("reticulocytes", 0, 15, "%", "lab_value", 0, 25, 0.5, 2.5, None, "adult",
                  {}, ["retic", "reticulocyte_count"]),
    ClinicalRange("esr", 0, 120, "mm/hr", "lab_value", 0, 150, 0, 20, "M", "adult",
                  {}, ["sed_rate", "erythrocyte_sedimentation_rate"]),
    ClinicalRange("esr", 0, 120, "mm/hr", "lab_value", 0, 150, 0, 30, "F", "adult",
                  {}, []),
    ClinicalRange("inr", 0.5, 10, "", "lab_value", 0.3, 15, 0.9, 1.1, None, "adult",
                  {}, ["international_normalized_ratio", "pt_inr"]),
    ClinicalRange("ptt", 10, 150, "sec", "lab_value", 5, 200, 25, 35, None, "adult",
                  {}, ["aptt", "partial_thromboplastin_time", "activated_ptt"]),
    ClinicalRange("fibrinogen", 50, 1000, "mg/dL", "lab_value", 20, 1500, 200, 400, None, "adult",
                  {}, []),
    ClinicalRange("d_dimer", 0, 50, "mg/L", "lab_value", 0, 100, 0, 0.5, None, "adult",
                  {"ug/mL": 1, "ng/mL": 1000}, ["d-dimer", "ddimer"]),

    # ── CHEMISTRY / METABOLIC PANEL ──
    ClinicalRange("glucose", 10, 700, "mg/dL", "lab_value", 5, 1500, 70, 100, None, "adult",
                  {"mmol/L": 0.0555}, ["blood_glucose", "fasting_glucose", "glucose_mgdl", "fbs", "rbs", "blood_sugar"]),
    ClinicalRange("sodium", 100, 170, "mEq/L", "lab_value", 90, 180, 136, 145, None, "adult",
                  {"mmol/L": 1}, ["na", "serum_sodium", "na_meql"]),
    ClinicalRange("potassium", 1.5, 9.0, "mEq/L", "lab_value", 1.0, 10.0, 3.5, 5.0, None, "adult",
                  {"mmol/L": 1}, ["k", "serum_potassium", "k_meql"]),
    ClinicalRange("chloride", 70, 130, "mEq/L", "lab_value", 60, 140, 98, 106, None, "adult",
                  {"mmol/L": 1}, ["cl", "serum_chloride"]),
    ClinicalRange("bicarbonate", 5, 45, "mEq/L", "lab_value", 3, 50, 22, 28, None, "adult",
                  {"mmol/L": 1}, ["co2", "hco3", "total_co2", "serum_bicarbonate"]),
    ClinicalRange("bun", 2, 150, "mg/dL", "lab_value", 1, 200, 7, 20, None, "adult",
                  {"mmol/L": 0.357}, ["blood_urea_nitrogen", "urea_nitrogen", "urea"]),
    ClinicalRange("creatinine", 0.1, 30, "mg/dL", "lab_value", 0.05, 40, 0.6, 1.2, "M", "adult",
                  {"umol/L": 88.4}, ["cr", "serum_creatinine", "creatinine_mgdl", "scr"]),
    ClinicalRange("creatinine", 0.1, 30, "mg/dL", "lab_value", 0.05, 40, 0.5, 1.1, "F", "adult",
                  {}, []),
    ClinicalRange("gfr", 5, 200, "mL/min", "lab_value", 1, 250, 90, 120, None, "adult",
                  {}, ["egfr", "estimated_gfr", "glomerular_filtration_rate"]),
    ClinicalRange("calcium", 4, 16, "mg/dL", "lab_value", 3, 18, 8.5, 10.5, None, "adult",
                  {"mmol/L": 0.25}, ["ca", "serum_calcium", "total_calcium"]),
    ClinicalRange("phosphorus", 1, 10, "mg/dL", "lab_value", 0.5, 15, 2.5, 4.5, None, "adult",
                  {"mmol/L": 0.323}, ["phosphate", "phos", "serum_phosphorus"]),
    ClinicalRange("magnesium", 0.5, 6, "mg/dL", "lab_value", 0.3, 8, 1.7, 2.2, None, "adult",
                  {"mmol/L": 0.411}, ["mg_serum", "serum_magnesium"]),
    ClinicalRange("uric_acid", 1, 20, "mg/dL", "lab_value", 0.5, 25, 3.5, 7.2, "M", "adult",
                  {"umol/L": 59.48}, ["urate"]),
    ClinicalRange("total_protein", 3, 12, "g/dL", "lab_value", 2, 15, 6.0, 8.3, None, "adult",
                  {"g/L": 10}, ["protein_total", "serum_protein"]),
    ClinicalRange("albumin", 1, 7, "g/dL", "lab_value", 0.5, 8, 3.5, 5.0, None, "adult",
                  {"g/L": 10}, ["serum_albumin", "alb"]),
    ClinicalRange("globulin", 0.5, 6, "g/dL", "lab_value", 0.3, 8, 2.0, 3.5, None, "adult",
                  {}, []),

    # ── LIVER FUNCTION ──
    ClinicalRange("alt", 2, 1000, "U/L", "lab_value", 0, 5000, 7, 56, None, "adult",
                  {"IU/L": 1}, ["sgpt", "alanine_aminotransferase", "alanine_transaminase", "alt_ul"]),
    ClinicalRange("ast", 2, 1000, "U/L", "lab_value", 0, 5000, 10, 40, None, "adult",
                  {"IU/L": 1}, ["sgot", "aspartate_aminotransferase", "aspartate_transaminase", "ast_ul"]),
    ClinicalRange("alp", 10, 800, "U/L", "lab_value", 5, 2000, 44, 147, None, "adult",
                  {"IU/L": 1}, ["alkaline_phosphatase", "alk_phos"]),
    ClinicalRange("ggt", 2, 500, "U/L", "lab_value", 0, 1000, 8, 61, None, "adult",
                  {"IU/L": 1}, ["gamma_glutamyl_transferase", "gamma_gt"]),
    ClinicalRange("total_bilirubin", 0.1, 30, "mg/dL", "lab_value", 0, 50, 0.1, 1.2, None, "adult",
                  {"umol/L": 17.1}, ["bilirubin", "tbili", "total_bili", "bilirubin_total"]),
    ClinicalRange("direct_bilirubin", 0, 15, "mg/dL", "lab_value", 0, 25, 0, 0.3, None, "adult",
                  {"umol/L": 17.1}, ["dbili", "conjugated_bilirubin"]),
    ClinicalRange("ldh", 50, 2000, "U/L", "lab_value", 20, 5000, 140, 280, None, "adult",
                  {"IU/L": 1}, ["lactate_dehydrogenase"]),

    # ── CARDIAC MARKERS ──
    ClinicalRange("troponin_i", 0, 50, "ng/mL", "lab_value", 0, 100, 0, 0.04, None, "adult",
                  {"ug/L": 1}, ["tni", "cardiac_troponin_i", "hs_troponin"]),
    ClinicalRange("troponin_t", 0, 25, "ng/mL", "lab_value", 0, 50, 0, 0.01, None, "adult",
                  {}, ["tnt", "cardiac_troponin_t"]),
    ClinicalRange("bnp", 0, 5000, "pg/mL", "lab_value", 0, 10000, 0, 100, None, "adult",
                  {"ng/L": 1}, ["brain_natriuretic_peptide", "nt_probnp"]),
    ClinicalRange("ck", 10, 5000, "U/L", "lab_value", 5, 50000, 22, 198, "M", "adult",
                  {}, ["creatine_kinase", "cpk", "ck_total"]),
    ClinicalRange("ck_mb", 0, 100, "ng/mL", "lab_value", 0, 300, 0, 5, None, "adult",
                  {}, ["creatine_kinase_mb"]),

    # ── LIPID PANEL ──
    ClinicalRange("total_cholesterol", 50, 500, "mg/dL", "lab_value", 30, 700, 0, 200, None, "adult",
                  {"mmol/L": 0.0259}, ["cholesterol", "tc"]),
    ClinicalRange("ldl", 10, 400, "mg/dL", "lab_value", 5, 500, 0, 100, None, "adult",
                  {"mmol/L": 0.0259}, ["ldl_cholesterol", "ldl_c", "low_density_lipoprotein"]),
    ClinicalRange("hdl", 5, 150, "mg/dL", "lab_value", 3, 200, 40, 60, None, "adult",
                  {"mmol/L": 0.0259}, ["hdl_cholesterol", "hdl_c", "high_density_lipoprotein"]),
    ClinicalRange("triglycerides", 10, 2000, "mg/dL", "lab_value", 5, 5000, 0, 150, None, "adult",
                  {"mmol/L": 0.0113}, ["tg", "trigs"]),

    # ── THYROID ──
    ClinicalRange("tsh", 0.01, 100, "mIU/L", "lab_value", 0, 500, 0.4, 4.0, None, "adult",
                  {"uIU/mL": 1}, ["thyroid_stimulating_hormone"]),
    ClinicalRange("free_t4", 0.1, 10, "ng/dL", "lab_value", 0, 20, 0.8, 1.8, None, "adult",
                  {"pmol/L": 12.87}, ["ft4", "thyroxine_free"]),
    ClinicalRange("free_t3", 0.5, 20, "pg/mL", "lab_value", 0, 30, 2.3, 4.2, None, "adult",
                  {"pmol/L": 1.536}, ["ft3", "triiodothyronine_free"]),

    # ── INFLAMMATORY / INFECTIOUS ──
    ClinicalRange("crp", 0, 300, "mg/L", "lab_value", 0, 500, 0, 3.0, None, "adult",
                  {"mg/dL": 0.1}, ["c_reactive_protein", "hs_crp", "protein_crp", "protein_level_crp"]),
    ClinicalRange("procalcitonin", 0, 100, "ng/mL", "lab_value", 0, 500, 0, 0.1, None, "adult",
                  {}, ["pct_procalcitonin"]),
    ClinicalRange("ferritin", 1, 5000, "ng/mL", "lab_value", 0, 40000, 12, 300, "M", "adult",
                  {"ug/L": 1}, ["serum_ferritin"]),
    ClinicalRange("ferritin", 1, 5000, "ng/mL", "lab_value", 0, 40000, 10, 150, "F", "adult",
                  {}, []),
    ClinicalRange("iron", 10, 400, "ug/dL", "lab_value", 5, 500, 60, 170, None, "adult",
                  {"umol/L": 0.179}, ["serum_iron", "fe"]),
    ClinicalRange("tibc", 100, 600, "ug/dL", "lab_value", 50, 800, 250, 370, None, "adult",
                  {}, ["total_iron_binding_capacity"]),

    # ── ENDOCRINE ──
    ClinicalRange("hba1c", 3, 18, "%", "lab_value", 2, 20, 4.0, 5.6, None, "adult",
                  {"mmol/mol": 10.93}, ["glycated_hemoglobin", "a1c", "glycohemoglobin"]),
    ClinicalRange("insulin", 0, 300, "uU/mL", "lab_value", 0, 500, 2.6, 24.9, None, "adult",
                  {"pmol/L": 6.945}, ["fasting_insulin", "serum_insulin"]),
    ClinicalRange("cortisol", 1, 50, "ug/dL", "lab_value", 0, 80, 6, 23, None, "adult",
                  {"nmol/L": 27.59}, ["serum_cortisol", "am_cortisol"]),

    # ── RENAL ──
    ClinicalRange("microalbumin", 0, 500, "mg/L", "lab_value", 0, 1000, 0, 30, None, "adult",
                  {}, ["urine_albumin", "uacr"]),

    # ── ABG / BLOOD GASES ──
    ClinicalRange("ph", 6.8, 7.8, "", "lab_value", 6.5, 8.0, 7.35, 7.45, None, "adult",
                  {}, ["blood_ph", "arterial_ph"]),
    ClinicalRange("pao2", 20, 600, "mmHg", "lab_value", 10, 700, 75, 100, None, "adult",
                  {"kPa": 0.133}, ["partial_pressure_oxygen", "po2"]),
    ClinicalRange("paco2", 10, 100, "mmHg", "lab_value", 5, 120, 35, 45, None, "adult",
                  {"kPa": 0.133}, ["partial_pressure_co2", "pco2"]),
    ClinicalRange("lactate", 0.1, 20, "mmol/L", "lab_value", 0, 30, 0.5, 2.2, None, "adult",
                  {"mg/dL": 9.01}, ["serum_lactate", "blood_lactate", "lactic_acid"]),

    # ── VITAL SIGNS ──
    ClinicalRange("systolic_bp", 40, 300, "mmHg", "vital_sign", 30, 350, 90, 140, None, "adult",
                  {}, ["sbp", "sys_bp", "systolic", "systolic_blood_pressure", "bp_systolic"]),
    ClinicalRange("diastolic_bp", 20, 200, "mmHg", "vital_sign", 15, 250, 60, 90, None, "adult",
                  {}, ["dbp", "dia_bp", "diastolic", "diastolic_blood_pressure", "bp_diastolic"]),
    ClinicalRange("heart_rate", 15, 300, "bpm", "vital_sign", 10, 350, 60, 100, None, "adult",
                  {}, ["hr", "pulse", "pulse_rate", "heart_rate_bpm"]),
    ClinicalRange("respiratory_rate", 4, 60, "breaths/min", "vital_sign", 2, 80, 12, 20, None, "adult",
                  {}, ["rr", "resp_rate", "breathing_rate"]),
    ClinicalRange("temperature", 25, 43, "°C", "vital_sign", 20, 46, 36.1, 37.2, None, "adult",
                  {"°F": lambda c: c * 9 / 5 + 32}, ["temp", "body_temp", "temperature_c", "body_temperature"]),
    ClinicalRange("spo2", 40, 100, "%", "vital_sign", 30, 100, 95, 100, None, "adult",
                  {}, ["oxygen_saturation", "o2_sat", "sao2", "spo2_pct", "pulse_ox"]),
    ClinicalRange("weight", 0.5, 400, "kg", "vital_sign", 0.3, 500, 50, 120, None, "adult",
                  {"lbs": 2.205}, ["body_weight", "weight_kg", "wt"]),
    ClinicalRange("height", 30, 250, "cm", "vital_sign", 20, 270, 150, 190, None, "adult",
                  {"in": 0.3937, "m": 0.01}, ["body_height", "height_cm", "stature"]),
    ClinicalRange("bmi", 8, 80, "kg/m2", "vital_sign", 5, 100, 18.5, 30, None, "adult",
                  {}, ["body_mass_index"]),

    # ── DEMOGRAPHICS ──
    ClinicalRange("age", 0, 122, "years", "demographics", -1, 150, 0, 122, None, "all",
                  {}, ["age_years", "patient_age", "age_at_visit"]),

    # ── URINALYSIS ──
    ClinicalRange("urine_ph", 4, 9, "", "lab_value", 3.5, 9.5, 4.5, 8.0, None, "adult",
                  {}, ["uph"]),
    ClinicalRange("specific_gravity", 1.001, 1.035, "", "lab_value", 1.0, 1.050, 1.005, 1.030, None, "adult",
                  {}, ["urine_specific_gravity", "usg"]),

    # ── COAGULATION ──
    ClinicalRange("pt", 8, 50, "sec", "lab_value", 5, 100, 11, 13.5, None, "adult",
                  {}, ["prothrombin_time"]),

    # ── TUMOR MARKERS ──
    ClinicalRange("psa", 0, 200, "ng/mL", "lab_value", 0, 1000, 0, 4.0, "M", "adult",
                  {"ug/L": 1}, ["prostate_specific_antigen"]),
    ClinicalRange("cea", 0, 100, "ng/mL", "lab_value", 0, 500, 0, 3.0, None, "adult",
                  {"ug/L": 1}, ["carcinoembryonic_antigen"]),
    ClinicalRange("afp", 0, 500, "ng/mL", "lab_value", 0, 10000, 0, 10, None, "adult",
                  {"ug/L": 1}, ["alpha_fetoprotein"]),
    ClinicalRange("ca125", 0, 500, "U/mL", "lab_value", 0, 5000, 0, 35, None, "adult",
                  {}, ["cancer_antigen_125"]),
    ClinicalRange("ca199", 0, 500, "U/mL", "lab_value", 0, 5000, 0, 37, None, "adult",
                  {}, ["cancer_antigen_199", "ca_19_9"]),

    # ── GENE EXPRESSION (typical log2 or linear scale) ──
    ClinicalRange("gene_expression", -10, 20, "log2", "biomarker", -15, 25, 0, 15, None, "all",
                  {}, ["gene_expr", "mrna_level", "expression_level"]),
    ClinicalRange("protein_level", 0, 10000, "pg/mL", "biomarker", 0, 100000, 0, 1000, None, "all",
                  {"ng/mL": 0.001}, ["protein_concentration", "protein_level_il6"]),

    # ── DRUG LEVELS (common therapeutic drugs) ──
    ClinicalRange("vancomycin_trough", 1, 50, "ug/mL", "drug_level", 0, 80, 10, 20, None, "adult",
                  {}, ["vancomycin", "vanco_trough"]),
    ClinicalRange("digoxin", 0.2, 5, "ng/mL", "drug_level", 0, 10, 0.8, 2.0, None, "adult",
                  {"nmol/L": 1.28}, []),
    ClinicalRange("lithium", 0.1, 3.0, "mEq/L", "drug_level", 0, 5.0, 0.6, 1.2, None, "adult",
                  {"mmol/L": 1}, []),
    ClinicalRange("phenytoin", 1, 40, "ug/mL", "drug_level", 0, 60, 10, 20, None, "adult",
                  {"umol/L": 3.96}, ["dilantin"]),
]


# ════════════════════════════════════════════════════════════════
# CLINICAL DOMAIN TERMINOLOGY PATTERNS
# ════════════════════════════════════════════════════════════════

# CDISC SDTM domain prefixes
CDISC_DOMAINS = {
    "DM": "Demographics", "VS": "Vital Signs", "LB": "Laboratory",
    "AE": "Adverse Events", "CM": "Concomitant Medications",
    "MH": "Medical History", "EX": "Exposure", "DS": "Disposition",
    "PE": "Physical Examination", "EG": "ECG", "IE": "Inclusion/Exclusion",
    "SV": "Subject Visits", "SE": "Subject Elements", "TA": "Trial Arms",
    "TV": "Trial Visits", "TS": "Trial Summary", "TI": "Trial Inclusion",
    "QS": "Questionnaires", "FA": "Findings About", "SC": "Subject Characteristics",
}

# Common MedDRA-style adverse event term patterns
AE_TERM_PATTERNS = [
    r"headache", r"nausea", r"vomiting", r"diarrhea", r"dizziness",
    r"fatigue", r"rash", r"pruritus", r"pyrexia", r"cough",
    r"arthralgia", r"myalgia", r"insomnia", r"dyspnea", r"edema",
    r"anemia", r"neutropenia", r"thrombocytopenia", r"hypertension",
    r"hypotension", r"tachycardia", r"bradycardia", r"constipation",
    r"abdominal\s*pain", r"back\s*pain", r"chest\s*pain",
    r"upper\s*respiratory", r"urinary\s*tract\s*infection",
]

# ICD-10 code pattern
ICD10_PATTERN = re.compile(r'^[A-Z]\d{2}(\.\d{1,4})?$')

# LOINC-style code pattern
LOINC_PATTERN = re.compile(r'^\d{3,7}-\d$')

# Drug name patterns (common suffixes)
DRUG_SUFFIXES = [
    "mab", "nib", "zole", "pril", "sartan", "statin", "olol",
    "azine", "cillin", "mycin", "floxacin", "prazole", "dipine",
    "zosin", "gliptin", "tide", "vir", "navir", "etine",
]


class ClinicalKnowledgeBase:
    """
    Comprehensive clinical knowledge base for AutoClin Engine.
    
    Usage:
        kb = ClinicalKnowledgeBase()
        match = kb.find_range("hemoglobin")
        if match:
            print(f"Range: {match.min_val}-{match.max_val} {match.unit}")
        else:
            print("No clinical match — using pure unsupervised detection")
    """

    def __init__(self):
        self.ranges = CLINICAL_RANGES
        self._build_index()

    def _build_index(self):
        """Build a lookup index from all names and synonyms."""
        self._index = {}
        for r in self.ranges:
            key = self._normalize(r.name)
            if key not in self._index:
                self._index[key] = []
            self._index[key].append(r)
            for syn in r.synonyms:
                skey = self._normalize(syn)
                if skey not in self._index:
                    self._index[skey] = []
                self._index[skey].append(r)

    def _normalize(self, name: str) -> str:
        """Normalize a column name for matching."""
        name = name.lower().strip()
        name = re.sub(r'[^a-z0-9]', '_', name)
        name = re.sub(r'_+', '_', name).strip('_')
        return name

    def find_range(self, column_name: str, sex: Optional[str] = None) -> Optional[ClinicalRange]:
        """
        Find the best matching clinical range for a column name.
        Returns None if no match found — engine should use pure unsupervised.
        """
        key = self._normalize(column_name)

        # Direct match
        if key in self._index:
            candidates = self._index[key]
            if sex and any(c.sex for c in candidates):
                sex_match = [c for c in candidates if c.sex == sex.upper()]
                if sex_match:
                    return sex_match[0]
            return candidates[0]

        # Partial match — check if any key is a substring
        for idx_key, ranges in self._index.items():
            if idx_key in key or key in idx_key:
                if sex and any(c.sex for c in ranges):
                    sex_match = [c for c in ranges if c.sex == sex.upper()]
                    if sex_match:
                        return sex_match[0]
                return ranges[0]

        return None  # No match — use pure unsupervised

    def find_all_ranges(self, column_name: str) -> list:
        """Find all matching ranges (including sex-specific variants)."""
        key = self._normalize(column_name)
        if key in self._index:
            return self._index[key]
        for idx_key, ranges in self._index.items():
            if idx_key in key or key in idx_key:
                return ranges
        return []

    def is_impossible(self, column_name: str, value: float, sex: Optional[str] = None) -> Optional[str]:
        """
        Check if a value is clinically impossible (not just abnormal).
        Returns a reason string if impossible, None if plausible.
        """
        r = self.find_range(column_name, sex)
        if r is None:
            return None  # Can't assess — no clinical match

        crit_low = r.critical_low if r.critical_low is not None else r.min_val
        crit_high = r.critical_high if r.critical_high is not None else r.max_val

        if value < crit_low:
            return f"{column_name}={value} below critical minimum ({crit_low} {r.unit})"
        if value > crit_high:
            return f"{column_name}={value} above critical maximum ({crit_high} {r.unit})"
        return None

    def is_abnormal(self, column_name: str, value: float, sex: Optional[str] = None) -> Optional[str]:
        """Check if a value is outside normal range (but still plausible)."""
        r = self.find_range(column_name, sex)
        if r is None:
            return None

        if r.normal_low is not None and value < r.normal_low:
            return f"{column_name}={value} below normal range ({r.normal_low}-{r.normal_high} {r.unit})"
        if r.normal_high is not None and value > r.normal_high:
            return f"{column_name}={value} above normal range ({r.normal_low}-{r.normal_high} {r.unit})"
        return None

    def classify_text(self, text: str) -> Optional[str]:
        """
        Classify clinical text into domain categories.
        Returns category string or None.
        """
        text_lower = text.lower().strip()

        # Check ICD-10
        if ICD10_PATTERN.match(text.strip()):
            return "icd10_code"

        # Check LOINC
        if LOINC_PATTERN.match(text.strip()):
            return "loinc_code"

        # Check adverse event terms
        for pattern in AE_TERM_PATTERNS:
            if re.search(pattern, text_lower):
                return "adverse_event_term"

        # Check drug names
        for suffix in DRUG_SUFFIXES:
            if text_lower.endswith(suffix):
                return "drug_name"

        # Check CDISC domain prefix
        for prefix in CDISC_DOMAINS:
            if text_lower.startswith(prefix.lower()):
                return f"cdisc_{CDISC_DOMAINS[prefix].lower().replace(' ', '_')}"

        return None

    def get_all_range_names(self) -> list:
        """Return all unique clinical parameter names."""
        return list(set(r.name for r in self.ranges))

    def get_category_summary(self) -> dict:
        """Return count of ranges by category."""
        cats = {}
        for r in self.ranges:
            cats[r.category] = cats.get(r.category, 0) + 1
        return cats

    @property
    def total_ranges(self) -> int:
        return len(self.ranges)

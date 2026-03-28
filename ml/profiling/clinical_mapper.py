"""
AutoClin Engine Clinical Field Mapper
Auto-detects clinical variable types using column-name regex patterns,
value-range heuristics, and cardinality analysis.
"""
import re
from dataclasses import dataclass
from typing import Optional
import numpy as np
import pandas as pd


@dataclass
class ClinicalMapping:
    column_name: str
    clinical_type: Optional[str]
    confidence: float
    reference_range: Optional[dict] = None
    user_confirmed: bool = False


CLINICAL_PATTERNS = {
    "patient_id": [
        r"(?i)(subject|patient|subj|pt|participant)[\s_\-]*(id|num|no|number|code|key)",
        r"(?i)^(subjid|ptid|pid|patid|usubjid|mrn|record_id)$",
    ],
    "visit_date": [
        r"(?i)(visit|encounter|assess|collection)[\s_\-]*(date|dt|datetime)",
        r"(?i)^(visitdt|vdate|asmtdt|rfstdtc|dmdtc)$",
    ],
    "lab_value": [
        r"(?i)(lab|analyte|test|result|biomarker)[\s_\-]*(value|result|orres|stresn)",
        r"(?i)^(lborres|lbstresn|aval)$",
        r"(?i)(hemoglobin|hgb|wbc|rbc|platelet|creatinine|glucose|alt|ast|bun|albumin|sodium|potassium|calcium|cholesterol|hba1c|troponin|crp|tsh)",
    ],
    "vital_sign": [
        r"(?i)(systolic|diastolic|sbp|dbp|blood[\s_]?pressure|bp)",
        r"(?i)(heart[\s_]?rate|hr|pulse)",
        r"(?i)(temperature|temp|body[\s_]?temp)",
        r"(?i)(respiratory[\s_]?rate|resp[\s_]?rate|rr)",
        r"(?i)(oxygen[\s_]?sat|spo2|o2[\s_]?sat)",
        r"(?i)(weight|wt|height|ht|bmi)",
    ],
    "adverse_event": [
        r"(?i)(adverse|ae|event|safety|toxicity)",
        r"(?i)(aeterm|aedecod|aesev|aeser|aerel)",
    ],
    "demographics": [
        r"(?i)^(age|sex|gender|race|ethnic|ethnicity|dob|arm|armcd|treatment[\s_]?group)$",
    ],
    "site_id": [
        r"(?i)(site|center|centre|clinic|facility)[\s_\-]*(id|num|code|name)",
        r"(?i)^(siteid|siteno|country|invid)$",
    ],
    "visit_id": [
        r"(?i)(visit|epoch|period|timepoint)[\s_\-]*(id|num|code|name|label)",
        r"(?i)^(visitnum|visitname|visit|avisitn|avisit)$",
    ],
    "drug_exposure": [
        r"(?i)(drug|medication|dose|dosage|treatment|therapy|conmed)",
        r"(?i)(exdose|extrt|cmdecod|cmtrt)",
    ],
    "eligibility": [
        r"(?i)(eligible|inclusion|exclusion|criteria|ietest)",
    ],
}

CLINICAL_REFERENCE_RANGES = {}
# Build from comprehensive ClinicalKnowledgeBase
try:
    from ml.clinical_knowledge import ClinicalKnowledgeBase
    _kb = ClinicalKnowledgeBase()
    for r in _kb.ranges:
        key = r.name
        CLINICAL_REFERENCE_RANGES[key] = {
            "min": r.min_val, "max": r.max_val, "unit": r.unit
        }
        for syn in r.synonyms:
            CLINICAL_REFERENCE_RANGES[syn] = {
                "min": r.min_val, "max": r.max_val, "unit": r.unit
            }
    _HAS_COMPREHENSIVE_DB = True
except ImportError:
    _HAS_COMPREHENSIVE_DB = False
    CLINICAL_REFERENCE_RANGES = {
        "systolic_bp": {"min": 50.0, "max": 300.0, "unit": "mmHg"},
        "diastolic_bp": {"min": 20.0, "max": 200.0, "unit": "mmHg"},
        "heart_rate": {"min": 20.0, "max": 300.0, "unit": "bpm"},
        "hemoglobin": {"min": 1.0, "max": 25.0, "unit": "g/dL"},
        "glucose": {"min": 5.0, "max": 1000.0, "unit": "mg/dL"},
        "creatinine": {"min": 0.1, "max": 30.0, "unit": "mg/dL"},
        "age": {"min": 0.0, "max": 125.0, "unit": "years"},
    }





class ClinicalFieldMapper:
    def __init__(self):
        self.patterns = CLINICAL_PATTERNS
        self.reference_ranges = CLINICAL_REFERENCE_RANGES

    def map_fields(self, df: pd.DataFrame) -> list[ClinicalMapping]:
        return [self._map_column(df[col], col) for col in df.columns]

    def _map_column(self, series: pd.Series, name: str) -> ClinicalMapping:
        best_type, best_conf, best_ref = None, 0.0, None

        for clinical_type, patterns in self.patterns.items():
            for pattern in patterns:
                if re.search(pattern, name):
                    conf = 0.8
                    if conf > best_conf:
                        best_type, best_conf = clinical_type, conf
                    break

        numeric = pd.to_numeric(series.dropna(), errors="coerce").dropna()
        name_lower = name.lower()
        if len(numeric) > 10:
            for ref_name, ref_range in self.reference_ranges.items():
                ref_kws = ref_name.replace("_", " ").split()
                if any(kw in name_lower for kw in ref_kws):
                    in_range = ((numeric >= ref_range["min"] * 0.5) &
                                (numeric <= ref_range["max"] * 1.5)).mean()
                    if in_range > 0.7 and 0.85 > best_conf:
                        vital_refs = {"systolic_bp", "diastolic_bp", "heart_rate",
                                      "temperature", "spo2", "bmi", "weight", "height"}
                        best_type = "vital_sign" if ref_name in vital_refs else "lab_value"
                        best_conf = 0.85
                        best_ref = ref_range

        return ClinicalMapping(column_name=name, clinical_type=best_type,
                               confidence=round(best_conf, 2), reference_range=best_ref)

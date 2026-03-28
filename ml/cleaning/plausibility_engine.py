"""
AutoClin Engine Clinical Plausibility Constraint Engine
Uses ClinicalKnowledgeBase (120+ ranges) for validation.
Gracefully falls back when no clinical match exists.
"""
import numpy as np
from typing import Optional

try:
    from ml.clinical_knowledge import ClinicalKnowledgeBase
except ImportError:
    ClinicalKnowledgeBase = None

LEGACY_RANGES = {
    "systolic_bp":(50,300),"diastolic_bp":(20,200),"heart_rate":(15,300),
    "temperature":(25,43),"spo2":(40,100),"hemoglobin":(3,25),
    "glucose":(10,700),"creatinine":(0.1,30),"sodium":(100,170),
    "potassium":(1.5,9.0),"age":(0,122),"bmi":(8,80),"weight":(0.5,400),
}

class PlausibilityEngine:
    def __init__(self):
        if ClinicalKnowledgeBase:
            self.kb = ClinicalKnowledgeBase()
            self.mode = "comprehensive"
        else:
            self.kb = None
            self.mode = "legacy"

    def validate_correction(self, column_name, proposed_value, sex=None):
        if self.kb:
            r = self.kb.find_range(column_name, sex)
            if r is None:
                return {"valid":True,"reason":"No clinical match — unsupervised only.","matched":False,"range":"none","mode":"no_match"}
            if np.isnan(proposed_value) or np.isinf(proposed_value):
                return {"valid":False,"reason":"NaN/Inf rejected.","matched":True,"range":f"{r.min_val}-{r.max_val}","mode":"comprehensive"}
            cl = r.critical_low if r.critical_low is not None else r.min_val
            ch = r.critical_high if r.critical_high is not None else r.max_val
            if proposed_value < cl or proposed_value > ch:
                return {"valid":False,"reason":f"{column_name}={proposed_value} outside [{cl}-{ch}] {r.unit}","matched":True,"range":f"{cl}-{ch} {r.unit}","mode":"comprehensive"}
            return {"valid":True,"reason":f"Within [{cl}-{ch}] {r.unit}","matched":True,"range":f"{cl}-{ch} {r.unit}","mode":"comprehensive"}
        # Legacy
        col = column_name.lower()
        for name,(lo,hi) in LEGACY_RANGES.items():
            if name in col:
                if proposed_value < lo or proposed_value > hi:
                    return {"valid":False,"reason":f"Outside [{lo}-{hi}]","matched":True,"range":f"{lo}-{hi}","mode":"legacy"}
                return {"valid":True,"reason":f"Within [{lo}-{hi}]","matched":True,"range":f"{lo}-{hi}","mode":"legacy"}
        return {"valid":True,"reason":"No match.","matched":False,"range":"none","mode":"no_match"}

    def check_column(self, column_name, values, sex=None):
        if self.kb:
            r = self.kb.find_range(column_name, sex)
            if not r:
                return {"matched":False,"column":column_name,"impossible_count":0,"impossible_indices":[],"mode":"no_match"}
            cl = r.critical_low if r.critical_low is not None else r.min_val
            ch = r.critical_high if r.critical_high is not None else r.max_val
            impossible = []
            for i,v in enumerate(values):
                try:
                    fv = float(v)
                    if np.isnan(fv): continue
                    if fv < cl or fv > ch: impossible.append(i)
                except (ValueError,TypeError): continue
            return {"matched":True,"column":column_name,"range":f"{cl}-{ch} {r.unit}",
                    "normal_range":f"{r.normal_low}-{r.normal_high} {r.unit}" if r.normal_low else None,
                    "impossible_count":len(impossible),"impossible_indices":impossible,"mode":"comprehensive"}
        return {"matched":False,"column":column_name,"impossible_count":0,"impossible_indices":[],"mode":"no_match"}

    @property
    def total_ranges(self):
        return self.kb.total_ranges if self.kb else len(LEGACY_RANGES)

    def compute_plausibility_flags(self, df, clinical_mappings=None):
        """
        Check all numeric columns against clinical ranges.
        Returns a boolean array: True = implausible, False = plausible.
        Used by the benchmarking scorer to compute Clinical Plausibility (CP).
        """
        import pandas as pd
        flags = np.zeros(len(df), dtype=bool)
        for col in df.columns:
            if not pd.api.types.is_numeric_dtype(df[col]):
                continue
            result = self.check_column(col, df[col].values)
            if result["matched"] and result["impossible_count"] > 0:
                for idx in result["impossible_indices"]:
                    if idx < len(flags):
                        flags[idx] = True
        return flags

"""
Generate 3 sample noisy clinical datasets for AutoClin Engine testing.
Each has realistic structure with injected anomalies of various types.
"""
import numpy as np
import pandas as pd
import os

OUT = "/mnt/user-data/outputs/clinengine_output/sample_datasets"
os.makedirs(OUT, exist_ok=True)
rng = np.random.RandomState(42)


def gen_clinical_trial(n=800):
    """Multi-site randomized clinical trial with labs, vitals, AEs."""
    sites = ["SITE-US-001","SITE-US-002","SITE-UK-003","SITE-DE-004","SITE-GH-005"]
    arms = ["Placebo","Drug_Low","Drug_High"]
    records = []
    for pid in range(1, n//4+1):
        site = rng.choice(sites)
        arm = rng.choice(arms)
        age = rng.randint(22,78)
        sex = rng.choice(["M","F"])
        bmi = round(rng.normal(26,4),1)
        base_sbp = rng.normal(128,16)
        base_dbp = rng.normal(78,10)
        base_hgb = rng.normal(13.5 if sex=="M" else 12.0,1.5)
        for visit in range(1,5):
            ae_term = rng.choice(["None","Headache","Nausea","Fatigue","Dizziness","Rash","None","None"],p=[.35,.12,.1,.1,.08,.05,.1,.1])
            ae_sev = rng.choice(["Mild","Moderate","Severe"],p=[.7,.25,.05]) if ae_term!="None" else ""
            records.append({
                "subject_id":f"SUBJ-{pid:04d}","site_id":site,"treatment_arm":arm,
                "visit_number":visit,
                "visit_date":f"2025-{rng.randint(1,13):02d}-{rng.randint(1,29):02d}",
                "age_years":age,"sex":sex,"bmi":round(bmi+rng.normal(0,.3),1),
                "systolic_bp":round(base_sbp+rng.normal(0,5)+visit*rng.normal(0,2),1),
                "diastolic_bp":round(base_dbp+rng.normal(0,4),1),
                "heart_rate":round(rng.normal(72,10),1),
                "temperature_c":round(rng.normal(36.7,.4),1),
                "hemoglobin_gdl":round(base_hgb+rng.normal(0,.5),1),
                "wbc_10e3ul":round(rng.normal(7.5,2),1),
                "platelet_10e3ul":round(rng.normal(250,60),0),
                "creatinine_mgdl":round(rng.normal(1.0,.3),2),
                "alt_ul":round(rng.normal(25,12),0),
                "glucose_mgdl":round(rng.normal(95,20),0),
                "ae_term":ae_term,"ae_severity":ae_sev,
                "conmed_name":rng.choice(["None","Paracetamol","Ibuprofen","Omeprazole","None"],p=[.5,.2,.1,.1,.1]),
                "notes":rng.choice(["","Patient compliant","Missed morning dose","Rash observed on arm","Slight dizziness reported",""],p=[.4,.2,.1,.1,.1,.1]),
            })
    df = pd.DataFrame(records)
    # Inject anomalies
    idx = rng.choice(len(df),12,replace=False)
    for i in idx[:3]: df.at[i,"systolic_bp"]=rng.choice([320,-15,999,450])
    for i in idx[3:5]: df.at[i,"hemoglobin_gdl"]=rng.choice([55,-3,0.1])
    for i in idx[5:7]: df.at[i,"age_years"]=rng.choice([200,-5,999])
    for i in idx[7:8]: df.at[i,"glucose_mgdl"]=round(df.at[i,"glucose_mgdl"]/18,1) # unit mix
    for i in idx[8:10]: df.at[i,"sex"]=rng.choice(["Male","female","MALE"])
    for i in idx[10:11]: df.at[i,"visit_date"]="2030-15-45" # bad date
    for i in idx[11:12]: df.at[i,"temperature_c"]=round(rng.choice([42.5,28.0]),1) # extreme temp
    # Duplicates
    df = pd.concat([df, df.iloc[rng.choice(len(df),5)]], ignore_index=True)
    # Missing
    for i in rng.choice(len(df),20,replace=False):
        df.at[i,"hemoglobin_gdl"]=np.nan; df.at[i,"wbc_10e3ul"]=np.nan
    # Site drift: SITE-GH has higher BP
    gh = df["site_id"]=="SITE-GH-005"
    df.loc[gh,"systolic_bp"] = df.loc[gh,"systolic_bp"]+15
    return df


def gen_biomedical(n=500):
    """Biomedical research dataset — gene expression + clinical phenotypes."""
    records = []
    for pid in range(1,n+1):
        group = rng.choice(["Control","Case"],p=[.4,.6])
        age = rng.randint(30,75)
        sex = rng.choice(["M","F"])
        records.append({
            "sample_id":f"BIO-{pid:05d}",
            "study_group":group,"age":age,"sex":sex,
            "tissue_type":rng.choice(["Blood","Tissue","CSF","Urine"],p=[.4,.3,.15,.15]),
            "collection_date":f"2024-{rng.randint(1,13):02d}-{rng.randint(1,29):02d}",
            "gene_expr_TP53":round(rng.normal(5.2,1.5),2),
            "gene_expr_BRCA1":round(rng.normal(3.8,1.2),2),
            "gene_expr_EGFR":round(rng.normal(6.1,2.0),2),
            "protein_level_CRP":round(rng.lognormal(1,0.8),2),
            "protein_level_IL6":round(rng.lognormal(0.5,1),2),
            "bmi":round(rng.normal(27,5),1),
            "smoking_status":rng.choice(["Never","Former","Current"],p=[.5,.3,.2]),
            "diabetes":rng.choice(["No","Yes","Type1","Type2"],p=[.7,.1,.05,.15]),
            "tumor_stage":rng.choice(["","I","II","III","IV","NA"],p=[.4,.15,.15,.12,.08,.1]) if group=="Case" else "",
            "survival_months":round(rng.exponential(24),1) if group=="Case" else np.nan,
            "lab_notes":rng.choice(["","Sample hemolyzed","Repeat requested","QC pass","Lipemic sample","Low volume"],p=[.5,.1,.08,.2,.07,.05]),
        })
    df = pd.DataFrame(records)
    # Anomalies
    idx = rng.choice(len(df),15,replace=False)
    for i in idx[:3]: df.at[i,"gene_expr_TP53"]=rng.choice([-5,50,999])
    for i in idx[3:5]: df.at[i,"protein_level_CRP"]=rng.choice([5000,-10])
    for i in idx[5:7]: df.at[i,"bmi"]=rng.choice([0.5,95,-3])
    for i in idx[7:9]: df.at[i,"age"]=rng.choice([5,150])
    for i in idx[9:11]: df.at[i,"smoking_status"]=rng.choice(["smoker","YES","unknown"])
    for i in idx[11:13]: df.at[i,"diabetes"]=rng.choice(["DM2","diabetic","Y"])
    for i in idx[13:15]: df.at[i,"sex"]=rng.choice(["Male","female","m"])
    df = pd.concat([df, df.iloc[rng.choice(len(df),3)]], ignore_index=True)
    for i in rng.choice(len(df),15,replace=False):
        df.at[i,"gene_expr_BRCA1"]=np.nan; df.at[i,"protein_level_IL6"]=np.nan
    return df


def gen_patient_care(n=1000):
    """Hospital patient care EHR-like dataset."""
    depts = ["Emergency","Cardiology","Internal Medicine","Surgery","Pediatrics","Oncology"]
    records = []
    for pid in range(1,n+1):
        dept = rng.choice(depts)
        age = rng.randint(1,95)
        sex = rng.choice(["M","F"])
        records.append({
            "mrn":f"MRN-{rng.randint(100000,999999)}",
            "encounter_id":f"ENC-{pid:06d}",
            "department":dept,
            "admission_date":f"2025-{rng.randint(1,13):02d}-{rng.randint(1,29):02d}",
            "discharge_date":f"2025-{rng.randint(1,13):02d}-{rng.randint(1,29):02d}",
            "age":age,"sex":sex,
            "ethnicity":rng.choice(["African","Caucasian","Asian","Hispanic","Other"],p=[.3,.3,.15,.15,.1]),
            "weight_kg":round(rng.normal(70 if age>18 else age*2.5, 15 if age>18 else 5),1),
            "height_cm":round(rng.normal(170 if age>18 else 50+age*3, 10),0),
            "systolic_bp":round(rng.normal(125 if dept!="Pediatrics" else 100, 18),0),
            "diastolic_bp":round(rng.normal(78 if dept!="Pediatrics" else 65, 12),0),
            "heart_rate":round(rng.normal(80,15),0),
            "resp_rate":round(rng.normal(16,3),0),
            "spo2_pct":round(min(100,rng.normal(97,2)),1),
            "temperature_c":round(rng.normal(36.8,.5),1),
            "hemoglobin":round(rng.normal(13,2),1),
            "wbc":round(rng.normal(8,3),1),
            "creatinine":round(rng.normal(1.0,.4),2),
            "sodium":round(rng.normal(140,4),0),
            "potassium":round(rng.normal(4.2,.5),1),
            "primary_dx":rng.choice(["Chest Pain","Pneumonia","Heart Failure","Fracture","Appendicitis","Anemia","Hypertension","DKA","Sepsis","Stroke"],p=[.12,.1,.09,.08,.06,.08,.12,.06,.08,.21]) + rng.choice(["", " - acute", " - chronic"],p=[.6,.25,.15]),
            "discharge_disposition":rng.choice(["Home","Rehab","Transfer","AMA","Expired"],p=[.65,.15,.1,.05,.05]),
            "clinical_notes":rng.choice([
                "","Patient stable, improving","Oxygen supplementation required",
                "Labs trending down","Awaiting imaging results","Family at bedside",
                "Consult requested for cardiology","Pain managed with analgesics",
                "Follow-up in 2 weeks","Discharge planning initiated",
            ]),
        })
    df = pd.DataFrame(records)
    # Anomalies
    idx = rng.choice(len(df),20,replace=False)
    for i in idx[:4]: df.at[i,"systolic_bp"]=rng.choice([350,-20,0,500])
    for i in idx[4:6]: df.at[i,"hemoglobin"]=rng.choice([50,-1,0])
    for i in idx[6:8]: df.at[i,"potassium"]=rng.choice([12,0.5,15])
    for i in idx[8:10]: df.at[i,"sodium"]=rng.choice([200,80,0])
    for i in idx[10:12]: df.at[i,"age"]=rng.choice([200,-3,0])
    for i in idx[12:14]: df.at[i,"spo2_pct"]=rng.choice([150,-5,0])
    for i in idx[14:16]: df.at[i,"temperature_c"]=rng.choice([45,20,0])
    for i in idx[16:18]: df.at[i,"weight_kg"]=rng.choice([500,-10,0])
    for i in idx[18:20]: df.at[i,"sex"]=rng.choice(["Male","female","MALE","m"])
    df = pd.concat([df, df.iloc[rng.choice(len(df),8)]], ignore_index=True)
    for i in rng.choice(len(df),30,replace=False):
        cols = rng.choice(["hemoglobin","wbc","creatinine","sodium","potassium"],2,replace=False)
        for c in cols: df.at[i,c]=np.nan
    return df


print("Generating sample datasets...")

df1 = gen_clinical_trial()
df1.to_csv(f"{OUT}/sample_clinical_trial.csv", index=False)
print(f"  Clinical Trial: {df1.shape} — saved")

df2 = gen_biomedical()
df2.to_csv(f"{OUT}/sample_biomedical_study.csv", index=False)
print(f"  Biomedical Study: {df2.shape} — saved")

df3 = gen_patient_care()
df3.to_csv(f"{OUT}/sample_patient_care.csv", index=False)
print(f"  Patient Care EHR: {df3.shape} — saved")

# Run benchmark across all 3
print("\nRunning benchmark across all 3 datasets...")
import sys
sys.path.insert(0, "/home/claude/clinengine")
from ml.orchestrator import PipelineOrchestrator, PipelineConfig

config = PipelineConfig(mode="suggestion", random_seed=42)
results = {}
for name, df in [("Clinical Trial", df1), ("Biomedical", df2), ("Patient Care", df3)]:
    print(f"\n  Running pipeline on {name}...")
    orch = PipelineOrchestrator(config)
    r = orch.run(df, run_id=f"bench-{name.lower().replace(' ','_')}")
    results[name] = r
    print(f"    Anomalies: {r.total_anomalies}, Noise: {r.noise_percentage:.2f}%, Trust: {r.trust_score:.1f}")

# Benchmark comparison table
print("\n" + "="*80)
print("BENCHMARK COMPARISON ACROSS 3 DATASETS")
print("="*80)
print(f"{'Dataset':<20} {'Rows':>6} {'Anomalies':>10} {'Noise%':>8} {'Trust':>7} {'Method':>18} {'Mode':>10}")
print("-"*80)
for name, r in results.items():
    print(f"{name:<20} {r.total_rows:>6} {r.total_anomalies:>10} {r.noise_percentage:>7.2f}% {r.trust_score:>6.1f} {','.join(r.selected_methods):>18} {r.selection_mode:>10}")

print("\nMethod Performance Across Datasets:")
print(f"{'Method':<20}", end="")
for name in results: print(f" {name:>18}", end="")
print()
print("-"*80)
all_methods = set()
for r in results.values():
    for mr in r.method_rankings: all_methods.add(mr["method"])
for method in sorted(all_methods):
    print(f"{method:<20}", end="")
    for name, r in results.items():
        mr = next((m for m in r.method_rankings if m["method"]==method), None)
        if mr: print(f" {mr['composite']:>18.4f}", end="")
        else: print(f" {'N/A':>18}", end="")
    print()

# Save benchmark CSV
import csv
with open(f"{OUT}/benchmark_comparison.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["Dataset","Rows","Anomalies","Noise%","Trust","Selected Method","Mode"])
    for name, r in results.items():
        w.writerow([name,r.total_rows,r.total_anomalies,f"{r.noise_percentage:.2f}",
                     f"{r.trust_score:.1f}",",".join(r.selected_methods),r.selection_mode])

print(f"\nBenchmark saved to {OUT}/benchmark_comparison.csv")
print("All sample datasets generated!")

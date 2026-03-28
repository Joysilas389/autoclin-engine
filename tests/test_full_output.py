"""
AutoClin Engine Full Output Test
Runs pipeline + generates PDF report, Excel workbook, and CSV exports.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd

from ml.orchestrator import PipelineOrchestrator, PipelineConfig
from reporting.pdf_generator import PDFReportGenerator
from reporting.excel_generator import ExcelGenerator
from reporting.csv_exporter import CSVExporter
from reporting.chart_builder import ChartBuilder


def generate_data(n=200, seed=42):
    rng = np.random.RandomState(seed)
    records = []
    sites = ['SITE-A', 'SITE-B', 'SITE-C']
    for pid in range(1, n+1):
        site = rng.choice(sites)
        age = rng.randint(25, 80)
        sex = rng.choice(['M', 'F'])
        for v in range(1, 4):
            records.append({
                'patient_id': f'PT-{pid:03d}', 'site_id': site, 'visit_num': v,
                'visit_date': f'2025-{rng.randint(1,13):02d}-{rng.randint(1,29):02d}',
                'age': age, 'sex': sex,
                'systolic_bp': round(rng.normal(125, 15), 1),
                'heart_rate': round(rng.normal(72, 10), 1),
                'hemoglobin': round(rng.normal(13, 1.5), 1),
                'glucose': round(rng.normal(100, 20), 1),
                'creatinine': round(rng.normal(1.0, 0.3), 2),
            })
    df = pd.DataFrame(records)
    # Inject anomalies
    for i in rng.choice(len(df), 5, replace=False):
        df.at[i, 'systolic_bp'] = rng.choice([350, -10, 999])
    for i in rng.choice(len(df), 3, replace=False):
        df.at[i, 'hemoglobin'] = rng.choice([0.1, 55, -2])
    for i in rng.choice(len(df), 3, replace=False):
        df.at[i, 'age'] = rng.choice([200, -5, 999])
    return df


def main():
    print("Generating synthetic clinical data...")
    df = generate_data()
    print(f"Dataset: {df.shape}")

    print("\nRunning AutoClin Engine pipeline...")
    config = PipelineConfig(mode="suggestion", random_seed=42)
    orch = PipelineOrchestrator(config)
    orch.set_progress_callback(lambda p, pct, msg: print(f"  Phase {p}: {msg}"))
    result = orch.run(df, run_id="output-test-001")

    out_dir = "/mnt/user-data/outputs/clinengine_output"
    os.makedirs(out_dir, exist_ok=True)

    # Generate PDF
    print("\nGenerating PDF report...")
    pdf_gen = PDFReportGenerator()
    pdf_path = os.path.join(out_dir, "clinengine_report.pdf")
    pdf_gen.generate(result, pdf_path, dataset_name="Synthetic Clinical Trial")
    print(f"  PDF: {pdf_path}")

    # Generate Excel
    print("Generating Excel workbook...")
    xlsx_gen = ExcelGenerator()
    xlsx_path = os.path.join(out_dir, "clinengine_workbook.xlsx")
    xlsx_gen.generate(result, xlsx_path)
    print(f"  Excel: {xlsx_path}")

    # Generate CSVs
    print("Generating CSV exports...")
    csv_exp = CSVExporter()
    csv_exp.export_anomaly_log(result.anomaly_classifications,
                               os.path.join(out_dir, "anomaly_log.csv"))
    csv_exp.export_method_comparison(result.method_rankings,
                                     os.path.join(out_dir, "method_comparison.csv"))
    if result.audit_entries:
        csv_exp.export_cleaning_audit(result.audit_entries,
                                      os.path.join(out_dir, "audit_trail.csv"))
    print(f"  CSVs written to {out_dir}")

    # Generate standalone charts
    print("Generating charts...")
    cb = ChartBuilder()
    from collections import Counter
    sev = Counter(c.severity for c in result.anomaly_classifications)
    types = Counter(c.anomaly_type for c in result.anomaly_classifications)

    with open(os.path.join(out_dir, "trust_gauge.png"), 'wb') as f:
        f.write(cb.trust_score_gauge(result.trust_score))
    with open(os.path.join(out_dir, "severity_pie.png"), 'wb') as f:
        f.write(cb.severity_distribution_pie(dict(sev)))
    with open(os.path.join(out_dir, "type_bar.png"), 'wb') as f:
        f.write(cb.anomaly_type_bar(dict(types)))
    with open(os.path.join(out_dir, "method_radar.png"), 'wb') as f:
        f.write(cb.method_comparison_radar(result.method_rankings))
    print("  Charts written")

    print(f"\n{'='*60}")
    print(f"ALL OUTPUTS GENERATED IN: {out_dir}")
    print(f"{'='*60}")
    print(f"  - clinengine_report.pdf (or .html)")
    print(f"  - clinengine_workbook.xlsx")
    print(f"  - anomaly_log.csv")
    print(f"  - method_comparison.csv")
    print(f"  - trust_gauge.png")
    print(f"  - severity_pie.png")
    print(f"  - type_bar.png")
    print(f"  - method_radar.png")

    # Print summary
    print(f"\n{'='*60}")
    print("PIPELINE SUMMARY")
    print(f"{'='*60}")
    print(f"  Anomalies:  {result.total_anomalies}")
    print(f"  Noise:      {result.noise_percentage:.2f}%")
    print(f"  Trust:      {result.trust_score:.1f}/100")
    print(f"  Methods:    {result.selected_methods} ({result.selection_mode})")
    print(f"  Duration:   {result.duration_ms/1000:.1f}s")


if __name__ == "__main__":
    main()

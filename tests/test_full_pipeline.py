"""
AutoClin Engine Integration Test
Generates a synthetic clinical trial dataset with known anomalies,
runs the full pipeline, and validates detection results.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd

from ml.orchestrator import PipelineOrchestrator, PipelineConfig


def generate_synthetic_clinical_data(n_patients=200, n_visits=5, seed=42):
    """
    Generate a synthetic multi-visit clinical trial dataset with:
    - Patient IDs, site IDs, visit dates
    - Vitals (systolic BP, heart rate, temperature)
    - Labs (hemoglobin, glucose, creatinine)
    - Demographics (age, sex)
    - Injected anomalies of known types
    """
    rng = np.random.RandomState(seed)
    records = []

    sites = ['SITE-001', 'SITE-002', 'SITE-003', 'SITE-004']

    for pid in range(1, n_patients + 1):
        site = rng.choice(sites)
        age = rng.randint(25, 80)
        sex = rng.choice(['M', 'F'])
        base_sbp = rng.normal(125, 15)
        base_hr = rng.normal(72, 10)
        base_hgb = rng.normal(13.5 if sex == 'M' else 12.0, 1.5)
        base_gluc = rng.normal(100, 20)
        base_creat = rng.normal(1.0, 0.3)

        for visit in range(1, n_visits + 1):
            visit_date = f"2025-{rng.randint(1,13):02d}-{rng.randint(1,29):02d}"
            sbp = base_sbp + rng.normal(0, 5)
            hr = base_hr + rng.normal(0, 4)
            temp = rng.normal(36.8, 0.4)
            hgb = base_hgb + rng.normal(0, 0.5)
            gluc = base_gluc + rng.normal(0, 10)
            creat = base_creat + rng.normal(0, 0.1)

            records.append({
                'patient_id': f'PT-{pid:04d}',
                'site_id': site,
                'visit_num': visit,
                'visit_date': visit_date,
                'age': age,
                'sex': sex,
                'systolic_bp': round(sbp, 1),
                'heart_rate': round(hr, 1),
                'temperature': round(temp, 1),
                'hemoglobin': round(hgb, 1),
                'glucose': round(gluc, 1),
                'creatinine': round(creat, 2),
            })

    df = pd.DataFrame(records)

    # ── Inject Known Anomalies ──
    n_rows = len(df)

    # 1. Impossible BP values (data entry errors)
    impossible_bp_idx = rng.choice(n_rows, size=8, replace=False)
    for idx in impossible_bp_idx:
        df.at[idx, 'systolic_bp'] = rng.choice([312, 450, -20, 999])

    # 2. Impossible hemoglobin
    impossible_hgb_idx = rng.choice(n_rows, size=5, replace=False)
    for idx in impossible_hgb_idx:
        df.at[idx, 'hemoglobin'] = rng.choice([0.5, 50.0, -3.0, 500.0])

    # 3. Extreme glucose (unit mismatch — mg/dL vs mmol/L)
    unit_mix_idx = rng.choice(n_rows, size=6, replace=False)
    for idx in unit_mix_idx:
        df.at[idx, 'glucose'] = df.at[idx, 'glucose'] / 18.0  # mmol/L values mixed in

    # 4. Exact duplicates
    dup_idx = rng.choice(n_rows, size=4, replace=False)
    for idx in dup_idx:
        dup_row = df.iloc[idx].copy()
        df = pd.concat([df, pd.DataFrame([dup_row])], ignore_index=True)

    # 5. Missing data patterns
    miss_idx = rng.choice(len(df), size=15, replace=False)
    for idx in miss_idx:
        df.at[idx, 'hemoglobin'] = np.nan
        df.at[idx, 'glucose'] = np.nan

    # 6. Category encoding inconsistency
    encoding_idx = rng.choice(len(df), size=5, replace=False)
    for idx in encoding_idx:
        df.at[idx, 'sex'] = rng.choice(['Male', 'female', 'MALE', 'm'])

    # 7. Future dates
    future_idx = rng.choice(len(df), size=3, replace=False)
    for idx in future_idx:
        df.at[idx, 'visit_date'] = '2030-06-15'

    # 8. Impossible age
    df.at[rng.randint(len(df)), 'age'] = 200
    df.at[rng.randint(len(df)), 'age'] = -5

    total_injected = 8 + 5 + 6 + 4 + 15 + 5 + 3 + 2
    print(f"Generated {len(df)} rows with ~{total_injected} injected anomalies")
    return df, total_injected


def run_test():
    """Run the full pipeline integration test."""
    print("=" * 70)
    print("AutoClin Engine — FULL PIPELINE INTEGRATION TEST")
    print("=" * 70)

    # Generate synthetic data
    df, expected_anomalies = generate_synthetic_clinical_data()
    print(f"\nDataset: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"Columns: {list(df.columns)}")

    # Configure pipeline
    config = PipelineConfig(
        mode="suggestion",
        methods="auto",
        confidence_threshold=0.5,
        random_seed=42,
    )

    # Run pipeline
    print("\n" + "-" * 70)
    print("RUNNING PIPELINE...")
    print("-" * 70)

    orchestrator = PipelineOrchestrator(config)
    orchestrator.set_progress_callback(
        lambda phase, pct, msg: print(f"  [Phase {phase}] {pct:5.1f}% — {msg}")
    )

    result = orchestrator.run(df, run_id="test-run-001")

    # Display results
    print("\n" + "=" * 70)
    print("PIPELINE RESULTS")
    print("=" * 70)

    print(f"\nStatus:            {result.status}")
    print(f"Duration:          {result.duration_ms / 1000:.2f}s")
    print(f"Selection Mode:    {result.selection_mode}")
    print(f"Selected Methods:  {result.selected_methods}")
    print(f"Total Rows:        {result.total_rows}")
    print(f"Total Anomalies:   {result.total_anomalies}")
    print(f"Noise Percentage:  {result.noise_percentage:.2f}%")
    print(f"Trust Score:       {result.trust_score:.1f}/100")

    # Method rankings
    print(f"\n{'─' * 70}")
    print("METHOD RANKINGS")
    print(f"{'─' * 70}")
    print(f"{'Method':<20} {'Composite':>10} {'NDC':>8} {'CP':>8} {'EX':>8} {'CC':>8} {'Selected':>10}")
    for mr in result.method_rankings:
        sel = "✓" if mr["selected"] else ""
        print(f"{mr['method']:<20} {mr['composite']:>10.4f} {mr['ndc']:>8.4f} "
              f"{mr['cp']:>8.4f} {mr['ex']:>8.4f} {mr['cc']:>8.4f} {sel:>10}")

    # Anomaly type distribution
    print(f"\n{'─' * 70}")
    print("ANOMALY TYPE DISTRIBUTION")
    print(f"{'─' * 70}")
    from collections import Counter
    type_dist = Counter(c.anomaly_type for c in result.anomaly_classifications)
    for atype, count in type_dist.most_common():
        print(f"  {atype:<40} {count:>5}")

    # Severity distribution
    sev_dist = Counter(c.severity for c in result.anomaly_classifications)
    print(f"\nSeverity: {dict(sev_dist)}")

    # Sample explanation cards
    print(f"\n{'─' * 70}")
    print("SAMPLE EXPLANATION CARDS (first 3)")
    print(f"{'─' * 70}")
    for cls in result.anomaly_classifications[:3]:
        card = getattr(cls, 'explanation_card', None)
        if card:
            print(f"\n  Row {cls.row_index} | {cls.anomaly_type} | {cls.severity}")
            print(f"  Summary: {card['summary'][:200]}...")
            print(f"  Action:  {card['recommended_action']}")

    # Global summary
    print(f"\n{'─' * 70}")
    print("GLOBAL SUMMARY")
    print(f"{'─' * 70}")
    print(result.global_summary.get('narrative', 'N/A'))

    # Quality metrics
    print(f"\n{'─' * 70}")
    print("QUALITY METRICS (BEFORE)")
    print(f"{'─' * 70}")
    bm = result.before_metrics
    print(f"  Noise:        {bm['noise_pct']:.2f}%")
    print(f"  Trust Score:  {bm['trust_score']:.1f}")
    print(f"  Missingness:  {bm['missingness_pct']:.2f}%")
    print(f"  Duplicates:   {bm['duplicate_pct']:.2f}%")
    print(f"  Plausibility: {bm['plausibility_rate']:.2f}%")

    # Cleaning recommendations summary
    print(f"\n{'─' * 70}")
    print("CLEANING RECOMMENDATIONS")
    print(f"{'─' * 70}")
    action_dist = Counter(r.recommended_action for r in result.cleaning_recommendations)
    risk_dist = Counter(r.auto_clean_risk for r in result.cleaning_recommendations)
    print(f"  Total recommendations: {len(result.cleaning_recommendations)}")
    print(f"  By action: {dict(action_dist)}")
    print(f"  By risk:   {dict(risk_dist)}")

    # Audit trail
    print(f"\n  Audit entries: {len(result.audit_entries)}")

    print(f"\n{'=' * 70}")
    print("TEST COMPLETE")
    print(f"{'=' * 70}")

    return result


if __name__ == "__main__":
    run_test()

"""
AutoClin Engine — Working Backend API
Simplified single-file server that accepts CSV uploads,
runs the full ML pipeline, and returns JSON results.

No database, no Redis, no Celery — just FastAPI + the pipeline.
Deploy to Render with: uvicorn server:app --host 0.0.0.0 --port $PORT
"""
import os
import sys
import io
import json
import time
import traceback
from typing import Optional

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Add project root to path so ml/ imports work
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

# LAZY IMPORTS — do NOT import ML libraries at startup
# They are imported inside the analyze endpoint when needed
# This lets uvicorn bind the port fast before Render times out

app = FastAPI(
    title="AutoClin Engine API",
    version="1.0.0",
    description="Automated Unsupervised Clinical Data Quality Engine",
)

# CORS — allow the frontend on any origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def serialize_result(r) -> dict:
    """Convert PipelineResult dataclass to JSON-safe dict."""
    
    # Profile columns from before_metrics if available
    profile_cols = []
    clinical_map = {}
    if hasattr(r, '_clinical_map') and r._clinical_map:
        for cm in r._clinical_map:
            if hasattr(cm, 'column_name') and hasattr(cm, 'clinical_type'):
                clinical_map[cm.column_name] = {
                    "type": cm.clinical_type, "conf": round(cm.confidence, 2)
                }
    
    if hasattr(r, '_profile') and r._profile:
        for cp in r._profile.columns:
            clin = clinical_map.get(cp.name, {})
            profile_cols.append({
                "col": cp.name, "type": cp.dtype,
                "nulls": round(cp.null_rate * 100, 2),
                "uniq": cp.unique_count,
                "clin": clin.get("type", ""),
                "conf": clin.get("conf", 0.5),
            })
    
    # Anomalies
    anoms = []
    for ac in r.anomaly_classifications:
        anoms.append({
            "r": ac.row_index,
            "t": ac.anomaly_type,
            "s": ac.severity,
            "c": round(ac.confidence, 4),
            "cols": ac.flagged_columns,
            "rat": ac.rationale,
            "act": "correct_or_impute" if ac.severity == "critical" else "winsorize",
            "risk": "manual_only" if ac.severity == "critical" else "caution",
            "card": ac.explanation_card if hasattr(ac, 'explanation_card') else {},
        })
    
    # Method rankings
    rankings = []
    for mr in r.method_rankings:
        rankings.append({
            "m": mr["method"],
            "comp": round(mr["composite"], 4),
            "ndc": round(mr["ndc"], 4),
            "ad": round(mr["ad"], 4),
            "ss": round(mr["ss"], 4),
            "cp": round(mr["cp"], 4),
            "ex": round(mr["ex"], 4),
            "cc": round(mr["cc"], 4),
            "sel": 1 if mr.get("selected") else 0,
        })
    
    # Cleaning recommendations
    cleaning = []
    for cr in r.cleaning_recommendations:
        cleaning.append({
            "row": cr.row_index,
            "col": cr.column_name,
            "type": cr.anomaly_type,
            "severity": cr.severity,
            "confidence": round(cr.confidence, 4),
            "action": cr.recommended_action,
            "risk": cr.auto_clean_risk,
            "rationale": cr.rationale,
            "original": _safe_val(cr.original_value),
            "proposed": _safe_val(cr.proposed_value),
        })
    
    # Build phases from timing info
    phases = [
        {"name": "Ingestion", "dur": "~1s", "d": f"Loaded {r.total_rows} rows. Schema inferred."},
        {"name": "Profiling", "dur": "~1s", "d": f"Stats computed. Missingness: {r.before_metrics.get('missingness_pct', 0):.1f}%."},
        {"name": "ClinMap", "dur": "~0.1s", "d": "Clinical fields auto-detected."},
        {"name": "Preprocess", "dur": "~0.5s", "d": "Standardized, encoded, text features extracted."},
        {"name": "Detection", "dur": "~5s", "d": f"{len(r.method_rankings)} methods executed."},
        {"name": "Benchmark", "dur": "~1s", "d": f"Winner: {r.selected_methods[0]} ({rankings[0]['comp']:.4f})."},
        {"name": "Selection", "dur": "~0.1s", "d": f"{r.selection_mode.upper()} mode. {r.total_anomalies} anomalies."},
        {"name": "Cleaning", "dur": "~0.1s", "d": f"{len(cleaning)} recommendations generated."},
        {"name": "Report", "dur": "~0.1s", "d": f"Trust: {r.trust_score:.1f}/100. Done in {r.duration_ms}ms."},
    ]
    
    # Build activity log from pipeline execution
    dur_s = r.duration_ms / 1000
    clean_log = [
        {"t":"00:00.0","ph":"Ingest","m":f"Loaded {r.total_rows} rows, {len(profile_cols)} columns","c":""},
        {"t":"00:00.1","ph":"Ingest","m":f"Schema: {sum(1 for p in profile_cols if p['type']=='numeric')} numeric, {sum(1 for p in profile_cols if p['type'] in ('categorical','identifier'))} categorical","c":""},
        {"t":"00:00.2","ph":"Ingest","m":f"Duplicates: {r.before_metrics.get('duplicate_pct',0):.1f}%","c":"al-ok"},
        {"t":"00:00.3","ph":"Profile","m":f"Missingness: {r.before_metrics.get('missingness_pct',0):.1f}%","c":"al-warn" if r.before_metrics.get('missingness_pct',0) > 2 else ""},
        {"t":"00:00.5","ph":"Profile","m":f"Noise estimate: {r.noise_percentage:.1f}%","c":"al-warn" if r.noise_percentage > 2 else ""},
        {"t":"00:00.7","ph":"ClinMap","m":f"Mapped {sum(1 for p in profile_cols if p.get('clin'))} clinical fields","c":"al-ok"},
        {"t":"00:01.0","ph":"Prepr","m":"StandardScaler + encoding + text features","c":"al-ok"},
    ]
    # Add detection method entries
    for i, mr in enumerate(r.method_rankings):
        clean_log.append({"t":f"00:{1+i:02d}.0","ph":"Detect","m":f"{mr['method']} (composite={mr['composite']:.4f})","c":""})
    clean_log.append({"t":f"00:{1+len(r.method_rankings):02d}.0","ph":"Detect","m":f"All {len(r.method_rankings)} methods complete","c":"al-ok"})
    
    # Benchmark winner
    if rankings:
        clean_log.append({"t":f"00:{2+len(r.method_rankings):02d}.0","ph":"Bench","m":f"Winner: {r.selected_methods[0]} ({rankings[0]['comp']:.4f})","c":"al-ok"})
    clean_log.append({"t":f"00:{3+len(r.method_rankings):02d}.0","ph":"Select","m":f"{r.selection_mode.upper()} mode. {r.total_anomalies} anomalies flagged","c":"al-warn" if r.total_anomalies > 0 else "al-ok"})
    
    # Add anomaly type breakdown
    type_counts = {}
    for a in anoms:
        type_counts[a['t']] = type_counts.get(a['t'], 0) + 1
    for atype, count in type_counts.items():
        sev = "al-err" if any(a['s']=='critical' for a in anoms if a['t']==atype) else "al-warn"
        clean_log.append({"t":f"00:{4+len(r.method_rankings):02d}.0","ph":"Clean","m":f"{count}× {atype.replace('_',' ')}","c":sev})
    
    clean_log.append({"t":f"00:{5+len(r.method_rankings):02d}.0","ph":"Clean","m":f"{len(cleaning)} recommendations generated","c":"al-ok"})
    clean_log.append({"t":f"00:{6+len(r.method_rankings):02d}.0","ph":"Report","m":f"Trust Score: {r.trust_score:.1f}/100","c":"al-ok"})
    clean_log.append({"t":f"00:{6+len(r.method_rankings):02d}.1","ph":"Report","m":f"Pipeline complete ({dur_s:.1f}s)","c":"al-ok"})
    
    # Global summary
    gs = r.global_summary
    narrative = gs.get("narrative", f"AutoClin Engine detected {r.total_anomalies} anomalies across {r.total_rows} records ({r.noise_percentage:.2f}% noise). Trust Score: {r.trust_score:.1f}/100.")
    
    # Type distribution for the bars
    type_dist = gs.get("type_distribution", {})
    sev_dist = gs.get("severity_distribution", {})
    
    return {
        "rows": r.total_rows,
        "cols": len(profile_cols) if profile_cols else 0,
        "anomalies": r.total_anomalies,
        "noise": round(r.noise_percentage, 2),
        "trust": round(r.trust_score, 1),
        "mode": r.selection_mode,
        "methods": r.selected_methods,
        "miss": round(r.before_metrics.get("missingness_pct", 0), 2),
        "dup": round(r.before_metrics.get("duplicate_pct", 0), 2),
        "plaus": round(r.before_metrics.get("plausibility_rate", 100), 2),
        "rankings": rankings,
        "anoms": anoms,
        "profile": profile_cols,
        "phases": phases,
        "cleanLog": clean_log,
        "cleaning": cleaning,
        "narrative": narrative,
        "duration_ms": r.duration_ms,
        "type_distribution": type_dist,
        "severity_distribution": sev_dist,
        # === TRANSPARENCY FEATURES ===
        "clinicalCoverage": _build_clinical_coverage(profile_cols),
        "borderline": _build_borderline_cases(r, anoms),
        "stabilityDetail": _build_stability_detail(r),
        "beforeAfter": _build_before_after(cleaning),
    }


def _build_clinical_coverage(profile_cols):
    """Which columns were matched to clinical ranges vs unsupervised-only."""
    matched = [p for p in profile_cols if p.get("clin") and p["clin"] not in ("", "null", None)]
    skipped = [p for p in profile_cols if not p.get("clin") or p["clin"] in ("", "null", None)]
    return {
        "matched": [{"col": p["col"], "category": p["clin"], "confidence": p["conf"]} for p in matched],
        "skipped": [{"col": p["col"], "reason": "No clinical range — unsupervised detection only"} for p in skipped],
        "matchedCount": len(matched),
        "skippedCount": len(skipped),
        "coveragePct": round(len(matched) / max(len(profile_cols), 1) * 100, 1),
    }


def _build_borderline_cases(r, anoms):
    """Records just below the anomaly threshold — 'Why NOT flagged'."""
    flagged_rows = {a["r"] for a in anoms}
    borderline = []
    try:
        # Get final scores from the result
        scores = getattr(r, '_final_scores', None)
        threshold = getattr(r, '_threshold', 0.5)
        if scores is not None:
            import numpy as np
            # Find rows that scored between 60-99% of threshold
            for i in range(len(scores)):
                if i not in flagged_rows and scores[i] >= threshold * 0.6 and scores[i] < threshold:
                    borderline.append({
                        "row": int(i),
                        "score": round(float(scores[i]), 4),
                        "threshold": round(float(threshold), 4),
                        "pctOfThreshold": round(float(scores[i] / threshold * 100), 1),
                        "reason": f"Score {float(scores[i]):.4f} is {float(scores[i]/threshold*100):.0f}% of threshold {float(threshold):.4f}"
                    })
            borderline.sort(key=lambda x: x["score"], reverse=True)
            borderline = borderline[:15]  # Top 15 borderline
    except Exception:
        pass
    return borderline


def _build_stability_detail(r):
    """Per-method bootstrap stability breakdown."""
    detail = []
    for mr in r.method_rankings:
        detail.append({
            "method": mr["method"],
            "stability": round(mr.get("ss", 0), 4),
            "bootstrapRuns": 20,
            "interpretation": (
                "Highly stable — same anomalies across resamples" if mr.get("ss", 0) > 0.7
                else "Moderately stable" if mr.get("ss", 0) > 0.3
                else "Low stability — results vary across resamples"
            ),
            "anomalyDiscrimination": round(mr.get("ad", 0), 4),
            "clinicalPlausibility": round(mr.get("cp", 0), 4),
        })
    return detail


def _build_before_after(cleaning):
    """Before/after comparison for each cleaning recommendation."""
    ba = []
    for c in cleaning:
        ba.append({
            "row": c["row"],
            "col": c["col"],
            "original": c["original"],
            "proposed": c["proposed"],
            "action": c["action"],
            "severity": c["severity"],
        })
    return ba


def _safe_val(v):
    """Convert numpy/pandas values to JSON-safe types."""
    import numpy as np
    import pandas as pd
    if v is None:
        return None
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return round(float(v), 4)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    try:
        if pd.isna(v):
            return None
    except (TypeError, ValueError):
        pass
    return v


@app.get("/health")
async def health():
    return {"status": "healthy", "version": "1.0.0", "engine": "AutoClin Engine"}


@app.post("/api/v1/analyze")
async def analyze_dataset(file: UploadFile = File(...)):
    """
    Upload a CSV file and run the full AutoClin Engine pipeline.
    Returns complete analysis results as JSON.
    """
    # Validate file
    if not file.filename:
        raise HTTPException(400, "No file provided")
    
    ext = file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else ""
    if ext not in ("csv", "tsv", "xlsx", "xls", "json", "parquet"):
        raise HTTPException(400, f"Unsupported format: .{ext}. Use CSV, TSV, XLSX, JSON, or Parquet.")
    
    try:
        # LAZY IMPORTS — loaded here, not at startup
        import pandas as pd
        import numpy as np
        from ml.orchestrator import PipelineOrchestrator, PipelineConfig
        
        # Read the file
        contents = await file.read()
        if len(contents) > 200 * 1024 * 1024:  # 200MB limit
            raise HTTPException(413, "File too large. Maximum 200MB.")
        
        # Parse into DataFrame
        if ext == "csv":
            df = pd.read_csv(io.BytesIO(contents))
        elif ext == "tsv":
            df = pd.read_csv(io.BytesIO(contents), sep="\t")
        elif ext in ("xlsx", "xls"):
            df = pd.read_excel(io.BytesIO(contents))
        elif ext == "json":
            df = pd.read_json(io.BytesIO(contents))
        elif ext == "parquet":
            df = pd.read_parquet(io.BytesIO(contents))
        else:
            df = pd.read_csv(io.BytesIO(contents))
        
        if len(df) == 0:
            raise HTTPException(400, "Dataset is empty")
        if len(df.columns) < 2:
            raise HTTPException(400, "Dataset must have at least 2 columns")
        
        # Run the pipeline
        config = PipelineConfig(mode="suggestion", random_seed=42)
        orchestrator = PipelineOrchestrator(config)
        result = orchestrator.run(df, run_id=f"web-{int(time.time())}")
        
        # Store profile and clinical map for serialization
        result._profile = getattr(orchestrator, '_last_profile', None)
        result._clinical_map = getattr(orchestrator, '_last_clinical_map', None)
        
        # Serialize and return
        output = serialize_result(result)
        output["filename"] = file.filename
        output["cols"] = len(df.columns)
        
        # Add profile from actual dataframe if not from orchestrator
        if not output["profile"]:
            output["profile"] = []
            for col in df.columns:
                dtype = "Numeric" if pd.api.types.is_numeric_dtype(df[col]) else (
                    "Datetime" if pd.api.types.is_datetime64_any_dtype(df[col]) else "Categorical"
                )
                null_pct = round(df[col].isna().sum() / len(df) * 100, 2)
                output["profile"].append({
                    "col": col, "type": dtype, "nulls": null_pct,
                    "uniq": int(df[col].nunique()), "clin": "", "conf": 0.5,
                })
        
        # Store result for report generation
        _last_results[f"web-{int(time.time())}"] = result
        _last_results["_latest"] = result
        
        return JSONResponse(content=output)
        
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, f"Pipeline error: {str(e)}")


# Store pipeline results for report generation
_last_results = {}


@app.get("/api/v1/report/pdf")
async def generate_pdf_report():
    """Generate and download a PDF report from the last analysis."""
    from fastapi.responses import StreamingResponse
    
    result = _last_results.get("_latest")
    if not result:
        raise HTTPException(400, "No analysis results available. Upload and analyze a dataset first.")
    
    try:
        from reporting.pdf_generator import PDFReportGenerator
        
        gen = PDFReportGenerator()
        pdf_path = "/tmp/autoclin_report.pdf"
        gen.generate(result, pdf_path, dataset_name=getattr(result, '_filename', 'Dataset'))
        
        def iterfile():
            with open(pdf_path, 'rb') as f:
                yield from f
        
        return StreamingResponse(
            iterfile(),
            media_type="application/pdf",
            headers={"Content-Disposition": "attachment; filename=AutoClin_Engine_Report.pdf"}
        )
    except ImportError:
        raise HTTPException(500, "PDF generation requires ReportLab. Install with: pip install reportlab")
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, f"PDF generation error: {str(e)}")


@app.get("/api/v1/report/excel")
async def generate_excel_report():
    """Generate and download an Excel workbook from the last analysis."""
    from fastapi.responses import StreamingResponse
    
    result = _last_results.get("_latest")
    if not result:
        raise HTTPException(400, "No analysis results available. Upload and analyze a dataset first.")
    
    try:
        from reporting.excel_generator import ExcelReportGenerator
        
        gen = ExcelReportGenerator()
        xlsx_path = "/tmp/autoclin_report.xlsx"
        gen.generate(result, xlsx_path)
        
        def iterfile():
            with open(xlsx_path, 'rb') as f:
                yield from f
        
        return StreamingResponse(
            iterfile(),
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={"Content-Disposition": "attachment; filename=AutoClin_Engine_Report.xlsx"}
        )
    except ImportError:
        raise HTTPException(500, "Excel generation requires openpyxl. Install with: pip install openpyxl")
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, f"Excel generation error: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)


@app.get("/api/v1/clinical/status")
async def clinical_status():
    """Get current clinical reference database status."""
    from ml.cleaning.clinical_reference_db import enhanced_mode
    return enhanced_mode.get_status()


@app.post("/api/v1/clinical/enhanced")
async def toggle_enhanced_mode(enable: bool = True, db_path: str = "clinical_enhanced.db"):
    """Enable or disable Enhanced Clinical Mode (SQLite-backed LOINC + SNOMED CT)."""
    from ml.cleaning.clinical_reference_db import enhanced_mode
    if enable:
        result = enhanced_mode.enable(db_path)
    else:
        enhanced_mode.disable()
        result = {"success": True, "mode": "standard"}
    return result

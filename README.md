# AutoClin Engine

**Automated Unsupervised Clinical Data Quality Engine**

> Upload any clinical dataset. Get anomalies detected, explained, and cleaned — automatically.

**Live Demo:** [https://autoclinengine.netlify.app](https://autoclinengine.netlify.app)

---

## What It Does

AutoClin Engine is a cloud-native Progressive Web Application that automates data quality assurance for clinical research, clinical trials, hospital patient care, biomedical studies, and mixed healthcare datasets from anywhere in the world.

**Upload a CSV. Get results in seconds:**
- 5 unsupervised ML methods run and scored automatically
- Best method selected across 6 evaluation dimensions
- Every anomaly explained with clinical context
- 19-category anomaly taxonomy
- Before/after quality metrics with transparency features
- Professional PDF and Excel reports
- Full audit trail for regulatory compliance

## Clinical Reference Database

AutoClin Engine ships with a comprehensive built-in clinical knowledge base:

### Standard Mode (always available, in-memory)
| Resource | Count | Coverage |
|----------|-------|----------|
| Clinical reference tests | 142 | Labs, vitals, demographics, biomarkers |
| Name/alias matches | 556 | Handles sbp, hgb, K, ALT, nt_probnp etc. |
| LOINC codes | 49 | Most common lab and vital LOINC codes |
| SNOMED CT concepts | 80+ | Clinical findings, procedures, body sites |
| UCUM unit conversions | 32 | mg/dL to mmol/L, g/dL to g/L, °C to °F etc. |

Categories covered: hematology, chemistry/metabolic, liver function, cardiac markers, lipid panel, thyroid, inflammatory, coagulation, blood gas, urinalysis, endocrine, immunology, WBC differentials, therapeutic drug levels, iron studies, tumor markers, pulmonary, ophthalmology, demographics, biomarkers.

### Enhanced Clinical Mode (enterprise, SQLite-backed)
For institutions needing complete terminology coverage:

| Resource | Count | Source |
|----------|-------|--------|
| LOINC codes | 100,000+ | loinc.org (free registration) |
| SNOMED CT concepts | 350,000+ | NLM/UMLS (free for research) |

Enhanced mode loads from a SQLite database file only when explicitly enabled. It does not affect startup time or memory usage in standard mode.

**To enable Enhanced Clinical Mode:**
1. Download LOINC CSV from https://loinc.org/downloads (free registration)
2. Download SNOMED CT from https://www.nlm.nih.gov/healthit/snomedct (free for research via UMLS)
3. Run: `python -m ml.cleaning.build_enhanced_db --loinc path/to/loinc.csv --snomed path/to/snomed`
4. This creates `clinical_enhanced.db` (~100MB)
5. Call `POST /api/v1/clinical/enhanced?enable=true&db_path=clinical_enhanced.db`

The standard in-memory database is the fast layer. The SQLite database is the comprehensive layer. Unrecognized columns always fall back to unsupervised statistical methods — no penalty applied.

## Detection Methods

| Method | What It Detects | Strength |
|--------|----------------|----------|
| Isolation Forest | Global outliers in high dimensions | Fast, works on any data |
| HDBSCAN | Variable-density cluster anomalies | Handles subpopulations |
| Local Outlier Factor | Contextual anomalies | Finds local density issues |
| Autoencoder | Complex nonlinear patterns | Reconstruction error scoring |
| Robust PCA | Sparse corruption in records | Finds scattered entry errors |

Each method is scored on 6 dimensions: Anomaly Discrimination, Stability (20 bootstrap resamples), Clinical Plausibility, Explainability, Computational Cost, and Noise Detection Confidence. The best method or weighted ensemble is selected automatically.

## Transparency Features

- **Feature contribution** per anomaly — which columns drove each flag
- **Clinical reference coverage report** — matched vs skipped columns with reasons
- **Bootstrap stability analysis** — per-method consistency across 20 resamples
- **Borderline cases** — "Why NOT flagged" records near the detection threshold
- **Before/after values** — original flagged value vs proposed correction
- **Anomaly score distribution** — bell curve visualization with severity coloring
- **SNOMED CT validation** — diagnosis columns checked against clinical ontology

## Try It Now

1. Open [autoclinengine.netlify.app](https://autoclinengine.netlify.app)

**Settings:** Click Settings in the sidebar to view Clinical Reference Mode, toggle Enhanced Clinical Mode, and see database statistics.

2. Click **Upload** in the sidebar
3. Download a **Sample Dataset** (Clinical Trial, Biomedical, or Patient EHR)
4. Upload the CSV file
5. Wait 15-30 seconds for the ML pipeline
6. Explore: Dashboard, Analysis, Anomalies, Cleaning, Methods, Activity Log, Reports

**Note:** Backend runs on Render free tier — first request after inactivity takes 30-60s to wake.

## Architecture

```
Frontend (Netlify)              Backend (Render)
+-------------------+          +---------------------------+
| Single HTML PWA   | -------> | FastAPI + ML Pipeline     |
| 12 Views          |          | 5 Detection Methods       |
| Dark/Light mode   |          | 142 Clinical Tests        |
| HTML/CSS/SVG      |          | SNOMED CT + LOINC + UCUM  |
| charts            |          | PDF/Excel Reports         |
+-------------------+          +---------------------------+
                                         |
                               +---------------------------+
                               | Enhanced Mode (optional)  |
                               | SQLite: 100K+ LOINC       |
                               |         350K+ SNOMED CT   |
                               +---------------------------+
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | /health | Health check |
| POST | /api/v1/analyze | Upload CSV, run full 9-phase pipeline |
| GET | /api/v1/report/pdf | Download PDF report |
| GET | /api/v1/report/excel | Download Excel workbook |
| GET | /api/v1/clinical/status | Get clinical reference DB status |
| POST | /api/v1/clinical/enhanced | Enable/disable Enhanced Clinical Mode |

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Frontend | HTML5 PWA, CSS Grid, SVG, Canvas API |
| Backend | Python 3.11, FastAPI, Uvicorn |
| ML | scikit-learn, HDBSCAN, NumPy, SciPy |
| Clinical DB | 142 tests, LOINC, SNOMED CT, UCUM |
| Enhanced DB | SQLite (optional, for enterprise) |
| Reporting | ReportLab (PDF), openpyxl (Excel) |
| Deployment | Netlify (frontend), Render (backend) |

## Anomaly Taxonomy (19 Categories)

Extreme numeric outlier, Contextual outlier, Site-specific anomaly, Temporal inconsistency, Duplicate/near-duplicate, Impossible biological value, Suspicious missingness, Unit mismatch, Data entry typo, Category encoding inconsistency, Drift-related anomaly, Cross-field contradiction, Visit-sequence inconsistency, Latent cluster-isolated observation, Graph-isolated patient record, High reconstruction error, Distributional contamination, Rare but plausible case, Rare and implausible case

## Developer

**Dr. Silas Joy Agbesi**
Data Scientist Intern | Medical Officer | Deputy Clinical Coordinator

A medical professional combining hands-on clinical expertise with data science to build tools that improve healthcare data quality worldwide.

## License

This project is for research and educational purposes. See the Disclaimer page in the app for regulatory and clinical use notices.

---

**Version 1.3.0** — March 2026

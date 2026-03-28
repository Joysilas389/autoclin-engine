"""
AutoClin Engine Professional PDF Report Generator
Clean, properly spaced multi-page PDF using ReportLab.
No text overlaps, no container collisions.
"""
import io
from datetime import datetime, timezone
from collections import Counter

from reportlab.lib.pagesizes import A4
from reportlab.lib.colors import HexColor, white, black
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    PageBreak, Image, HRFlowable, KeepTogether,
)
from reporting.chart_builder import ChartBuilder

W, H = A4
NAVY = HexColor("#0f172a")
BLUE = HexColor("#1e40af")
SKY = HexColor("#3b82f6")
LIGHT_BLUE = HexColor("#dbeafe")
LIGHT_BG = HexColor("#f8fafc")
TXT = HexColor("#1e293b")
TXT2 = HexColor("#475569")
MUTED = HexColor("#94a3b8")
BORDER = HexColor("#e2e8f0")
GREEN = HexColor("#16a34a")
YELLOW = HexColor("#d97706")
RED = HexColor("#dc2626")
SEV_COLORS = {"critical": RED, "high": HexColor("#ef4444"), "medium": YELLOW, "low": GREEN}


def _make_styles():
    return {
        "title": ParagraphStyle("T", fontName="Helvetica-Bold", fontSize=26,
                                textColor=NAVY, leading=32, spaceAfter=8),
        "subtitle": ParagraphStyle("Sub", fontName="Helvetica", fontSize=13,
                                   textColor=TXT2, leading=18, spaceAfter=4),
        "h1": ParagraphStyle("H1", fontName="Helvetica-Bold", fontSize=17,
                             textColor=BLUE, leading=22, spaceBefore=0, spaceAfter=10),
        "h2": ParagraphStyle("H2", fontName="Helvetica-Bold", fontSize=12,
                             textColor=NAVY, leading=16, spaceBefore=14, spaceAfter=6),
        "body": ParagraphStyle("Body", fontName="Helvetica", fontSize=9.5,
                               textColor=TXT, leading=14, spaceAfter=6, alignment=TA_JUSTIFY),
        "small": ParagraphStyle("Small", fontName="Helvetica", fontSize=8,
                                textColor=TXT2, leading=11, spaceAfter=4),
        "center": ParagraphStyle("Ctr", fontName="Helvetica", fontSize=9,
                                 textColor=MUTED, alignment=TA_CENTER, spaceAfter=4),
        "callout": ParagraphStyle("Call", fontName="Helvetica", fontSize=9.5,
                                  textColor=BLUE, leading=14, spaceAfter=10,
                                  leftIndent=10, rightIndent=10,
                                  backColor=LIGHT_BLUE, borderPadding=10),
        "warning": ParagraphStyle("Warn", fontName="Helvetica", fontSize=9,
                                  textColor=HexColor("#92400e"), leading=13,
                                  leftIndent=10, rightIndent=10, spaceAfter=10,
                                  backColor=HexColor("#fef3c7"), borderPadding=10),
    }


def _header_footer(canvas, doc):
    canvas.saveState()
    w, h = A4
    canvas.setStrokeColor(SKY)
    canvas.setLineWidth(1.5)
    canvas.line(35, h - 38, w - 35, h - 38)
    canvas.setFont("Helvetica-Bold", 7.5)
    canvas.setFillColor(BLUE)
    canvas.drawString(35, h - 30, "AutoClin Engine")
    canvas.setFont("Helvetica", 7)
    canvas.setFillColor(MUTED)
    canvas.drawRightString(w - 35, h - 30, "Clinical Data Quality Report")
    canvas.setStrokeColor(BORDER)
    canvas.setLineWidth(0.5)
    canvas.line(35, 35, w - 35, 35)
    canvas.setFont("Helvetica", 7)
    canvas.setFillColor(MUTED)
    canvas.drawString(35, 22, "CONFIDENTIAL")
    canvas.drawCentredString(w / 2, 22, f"Page {doc.page}")
    canvas.drawRightString(w - 35, 22, "Dr. Silas Joy Agbesi")
    canvas.restoreState()


def _table_style(font_size=8.5):
    return TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("TEXTCOLOR", (0, 0), (-1, 0), white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, 0), font_size),
        ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 1), (-1, -1), font_size),
        ("TEXTCOLOR", (0, 1), (-1, -1), TXT),
        ("VALIGN", (0, 0), (-1, 0), "MIDDLE"),
        ("VALIGN", (0, 1), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [white, LIGHT_BG]),
        ("LINEBELOW", (0, 0), (-1, 0), 1, SKY),
        ("GRID", (0, 0), (-1, -1), 0.25, BORDER),
    ])


class PDFReportGenerator:
    def __init__(self):
        self.cb = ChartBuilder(dpi=180)
        self.s = _make_styles()

    def _chart_image(self, chart_bytes, width, height):
        """Return Image flowable or empty spacer if no chart data."""
        if not chart_bytes:
            return Spacer(1, 6)
        return Image(io.BytesIO(chart_bytes), width=width, height=height)

    def generate(self, result, output_path, dataset_name="Dataset"):
        doc = SimpleDocTemplate(
            output_path, pagesize=A4,
            leftMargin=35, rightMargin=35, topMargin=50, bottomMargin=50,
        )
        uw = W - 70
        story = []

        story += self._cover(result, dataset_name, uw)
        story.append(PageBreak())
        story += self._toc(uw)
        story.append(PageBreak())
        story += self._executive_summary(result, uw)
        story.append(PageBreak())
        story += self._dataset_profile(result, uw)
        story.append(PageBreak())
        story += self._anomaly_overview(result, uw)
        story.append(PageBreak())
        story += self._method_comparison(result, uw)
        story.append(PageBreak())
        story += self._flagged_records(result, uw)
        story.append(PageBreak())
        story += self._cleaning_recs(result, uw)
        story.append(PageBreak())
        story += self._quality_metrics(result, uw)
        story.append(PageBreak())
        story += self._disclaimer_and_about(result, uw)

        doc.build(story, onFirstPage=_header_footer, onLaterPages=_header_footer)
        return output_path

    def _cover(self, r, name, uw):
        s = []
        s.append(Spacer(1, 100))
        s.append(Paragraph("AutoClin Engine", self.s["title"]))
        s.append(Paragraph("Automated Unsupervised Clinical Data Quality Engine", self.s["subtitle"]))
        s.append(Spacer(1, 8))
        s.append(HRFlowable(width=80, thickness=3, color=SKY, spaceAfter=30))

        # Trust gauge chart
        gauge_bytes = self.cb.trust_score_gauge(r.trust_score)
        s.append(self._chart_image(gauge_bytes, 220, 165))
        s.append(Spacer(1, 25))

        # Key metrics table
        data = [
            ["Total Rows", f"{r.total_rows:,}",
             "Total Anomalies", str(r.total_anomalies)],
            ["Noise Rate", f"{r.noise_percentage:.2f}%",
             "Trust Score", f"{r.trust_score:.1f} / 100"],
            ["Selection Mode", r.selection_mode.title(),
             "Selected Method(s)", ", ".join(r.selected_methods)],
        ]
        t = Table(data, colWidths=[uw * 0.2, uw * 0.3, uw * 0.2, uw * 0.3])
        t.setStyle(TableStyle([
            ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
            ("FONTNAME", (2, 0), (2, -1), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("TEXTCOLOR", (0, 0), (0, -1), TXT2),
            ("TEXTCOLOR", (2, 0), (2, -1), TXT2),
            ("TEXTCOLOR", (1, 0), (1, -1), TXT),
            ("TEXTCOLOR", (3, 0), (3, -1), TXT),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("LINEBELOW", (0, 0), (-1, -1), 0.3, BORDER),
        ]))
        s.append(t)
        s.append(Spacer(1, 30))

        now = datetime.now(timezone.utc).strftime("%B %d, %Y at %H:%M UTC")
        s.append(Paragraph(f"Dataset: {name}", self.s["center"]))
        s.append(Paragraph(f"Report generated: {now}", self.s["center"]))
        s.append(Paragraph("AutoClin Engine v1.0.0", self.s["center"]))
        s.append(Spacer(1, 15))
        s.append(HRFlowable(width=uw, thickness=2, color=SKY, spaceAfter=8))
        s.append(Paragraph("CONFIDENTIAL — FOR INTERNAL USE ONLY", self.s["center"]))
        return s

    def _toc(self, uw):
        s = []
        s.append(Paragraph("Table of Contents", self.s["h1"]))
        s.append(HRFlowable(width=uw, thickness=1, color=SKY, spaceAfter=14))
        items = [
            "1.  Executive Summary",
            "2.  Dataset Profile",
            "3.  Anomaly & Noise Overview",
            "4.  Method Comparison & Selection",
            "5.  Flagged Records Detail",
            "6.  Cleaning Recommendations",
            "7.  Quality Metrics",
            "8.  Disclaimer & About the Developer",
        ]
        for item in items:
            s.append(Paragraph(item, ParagraphStyle("toc", fontName="Helvetica",
                               fontSize=11, leading=20, leftIndent=24, textColor=TXT)))
        return s

    def _executive_summary(self, r, uw):
        s = []
        s.append(Paragraph("1.  Executive Summary", self.s["h1"]))
        s.append(HRFlowable(width=uw, thickness=1, color=SKY, spaceAfter=12))

        narrative = r.global_summary.get("narrative", "No summary available.")
        s.append(Paragraph(narrative, self.s["callout"]))
        s.append(Spacer(1, 10))

        drivers = r.global_summary.get("top_drivers", [])
        if drivers:
            s.append(Paragraph("Top Anomaly Drivers", self.s["h2"]))
            data = [["#", "Feature", "Contribution", "Clinical Label"]]
            for i, d in enumerate(drivers[:8], 1):
                data.append([str(i), d["column"],
                             f"{d['contribution_pct']:.1f}%",
                             d.get("clinical_label") or "—"])
            t = Table(data, colWidths=[25, uw * 0.35, 80, uw * 0.30])
            t.setStyle(_table_style())
            s.append(t)

        sev = r.global_summary.get("severity_distribution", {})
        if sev:
            s.append(Spacer(1, 12))
            s.append(Paragraph("Severity Breakdown", self.s["h2"]))
            data = [["Severity", "Count"]]
            for sv in ["critical", "high", "medium", "low"]:
                if sev.get(sv, 0) > 0:
                    data.append([sv.upper(), str(sev[sv])])
            t = Table(data, colWidths=[100, 80])
            ts = _table_style()
            for i, sv in enumerate(["critical", "high", "medium", "low"]):
                row = next((j for j, r in enumerate(data[1:], 1) if r[0] == sv.upper()), None)
                if row:
                    ts.add("TEXTCOLOR", (0, row), (0, row), SEV_COLORS.get(sv, TXT))
                    ts.add("FONTNAME", (0, row), (0, row), "Helvetica-Bold")
            t.setStyle(ts)
            s.append(t)
        return s

    def _dataset_profile(self, r, uw):
        s = []
        s.append(Paragraph("2.  Dataset Profile", self.s["h1"]))
        s.append(HRFlowable(width=uw, thickness=1, color=SKY, spaceAfter=12))

        bm = r.before_metrics
        data = [
            ["Metric", "Value"],
            ["Total Rows", f"{bm.get('total_rows', r.total_rows):,}"],
            ["Total Anomalies Detected", str(r.total_anomalies)],
            ["Noise Rate", f"{bm.get('noise_pct', 0):.2f}%"],
            ["Missingness Rate", f"{bm.get('missingness_pct', 0):.2f}%"],
            ["Duplicate Rate", f"{bm.get('duplicate_pct', 0):.2f}%"],
            ["Clinical Plausibility", f"{bm.get('plausibility_rate', 0):.1f}%"],
            ["Data Trust Score", f"{bm.get('trust_score', 0):.1f} / 100"],
        ]
        t = Table(data, colWidths=[uw * 0.55, uw * 0.45])
        t.setStyle(_table_style())
        s.append(t)
        return s

    def _anomaly_overview(self, r, uw):
        s = []
        s.append(Paragraph("3.  Anomaly & Noise Overview", self.s["h1"]))
        s.append(HRFlowable(width=uw, thickness=1, color=SKY, spaceAfter=12))

        s.append(Paragraph(
            f"AutoClin Engine detected <b>{r.total_anomalies}</b> anomalies across "
            f"<b>{r.total_rows:,}</b> records, yielding a noise rate of "
            f"<b>{r.noise_percentage:.2f}%</b>.", self.s["body"]))
        s.append(Spacer(1, 8))

        td = Counter(c.anomaly_type for c in r.anomaly_classifications)
        s.append(Paragraph("Anomaly Type Distribution", self.s["h2"]))
        data = [["Anomaly Type", "Count", "Proportion"]]
        for atype, cnt in td.most_common():
            pct = cnt / r.total_anomalies * 100 if r.total_anomalies else 0
            data.append([atype.replace("_", " ").title(), str(cnt), f"{pct:.1f}%"])
        t = Table(data, colWidths=[uw * 0.50, 60, 80])
        t.setStyle(_table_style())
        s.append(t)
        s.append(Spacer(1, 14))

        # Type bar chart
        chart_bytes = self.cb.anomaly_type_bar(dict(td))
        s.append(self._chart_image(chart_bytes, uw * 0.9, 180))
        s.append(Spacer(1, 12))

        # Severity pie
        sd = Counter(c.severity for c in r.anomaly_classifications)
        if sd:
            sev_bytes = self.cb.severity_distribution_pie(dict(sd))
            s.append(self._chart_image(sev_bytes, 260, 200))
        return s

    def _method_comparison(self, r, uw):
        s = []
        s.append(Paragraph("4.  Method Comparison & Selection", self.s["h1"]))
        s.append(HRFlowable(width=uw, thickness=1, color=SKY, spaceAfter=12))

        s.append(Paragraph(
            f"Evaluated <b>{len(r.method_rankings)}</b> unsupervised methods. "
            f"Selection mode: <b>{r.selection_mode}</b>. "
            f"Selected: <b>{', '.join(r.selected_methods)}</b>.", self.s["body"]))
        s.append(Spacer(1, 8))

        s.append(Paragraph("Method Scoring Matrix", self.s["h2"]))
        cw = (uw - 100) / 7
        data = [["Method", "Comp.", "NDC", "AD", "SS", "CP", "EX", "CC"]]
        for mr in r.method_rankings:
            sel = " *" if mr.get("selected") else ""
            data.append([
                mr["method"] + sel,
                f"{mr['composite']:.3f}", f"{mr['ndc']:.3f}", f"{mr['ad']:.3f}",
                f"{mr['ss']:.3f}", f"{mr['cp']:.3f}", f"{mr['ex']:.3f}", f"{mr['cc']:.3f}",
            ])
        t = Table(data, colWidths=[100] + [cw] * 7)
        ts = _table_style(8)
        for i, mr in enumerate(r.method_rankings, 1):
            if mr.get("selected"):
                ts.add("BACKGROUND", (0, i), (-1, i), LIGHT_BLUE)
                ts.add("FONTNAME", (0, i), (0, i), "Helvetica-Bold")
        t.setStyle(ts)
        s.append(t)
        s.append(Spacer(1, 14))

        # Radar chart
        radar_bytes = self.cb.method_comparison_radar(r.method_rankings)
        s.append(self._chart_image(radar_bytes, 300, 280))
        s.append(Spacer(1, 8))
        s.append(Paragraph(
            "<b>AD</b> = Anomaly Discrimination  |  <b>SS</b> = Stability  |  "
            "<b>CP</b> = Clinical Plausibility  |  <b>EX</b> = Explainability  |  "
            "<b>CC</b> = Cost  |  <b>NDC</b> = Noise Detection Confidence  |  "
            "* = selected method", self.s["small"]))
        return s

    def _flagged_records(self, r, uw):
        s = []
        s.append(Paragraph("5.  Flagged Records Detail", self.s["h1"]))
        s.append(HRFlowable(width=uw, thickness=1, color=SKY, spaceAfter=12))

        mx = min(40, len(r.anomaly_classifications))
        sorted_cls = sorted(r.anomaly_classifications, key=lambda c: c.confidence, reverse=True)

        s.append(Paragraph(
            f"Top {mx} of {len(r.anomaly_classifications)} flagged records, "
            f"sorted by confidence.", self.s["body"]))
        s.append(Spacer(1, 6))

        # Use Paragraph objects so text wraps inside cells
        cell_style = ParagraphStyle("cell", fontName="Helvetica", fontSize=7.5,
                                    leading=10, textColor=TXT)
        bold_cell = ParagraphStyle("bcell", fontName="Helvetica-Bold", fontSize=7.5,
                                   leading=10, textColor=TXT)

        data = [["Row", "Type", "Sev.", "Conf.", "Columns", "Rationale"]]
        for c in sorted_cls[:mx]:
            cols = ", ".join(c.flagged_columns[:2])
            sev_color = SEV_COLORS.get(c.severity, TXT)
            data.append([
                Paragraph(str(c.row_index), cell_style),
                Paragraph(c.anomaly_type.replace("_", " "), cell_style),
                Paragraph(c.severity.upper(), ParagraphStyle("sev", fontName="Helvetica-Bold",
                          fontSize=7.5, leading=10, textColor=sev_color)),
                Paragraph(f"{c.confidence:.2f}", cell_style),
                Paragraph(cols, cell_style),
                Paragraph(c.rationale, cell_style),
            ])
        t = Table(data, colWidths=[30, uw * 0.18, 38, 35, uw * 0.14, uw * 0.34])
        t.setStyle(_table_style(7.5))
        s.append(t)

        # Explanation cards
        s.append(Spacer(1, 16))
        s.append(Paragraph("Sample Explanation Cards", self.s["h2"]))
        for c in sorted_cls[:3]:
            card = getattr(c, "explanation_card", None)
            if card:
                s.append(Spacer(1, 6))
                s.append(Paragraph(
                    f"<b>Row #{c.row_index}</b> — "
                    f"{c.anomaly_type.replace('_', ' ').title()} "
                    f"({c.severity.upper()})", self.s["body"]))
                s.append(Paragraph(card.get("summary", ""), self.s["callout"]))
        return s

    def _cleaning_recs(self, r, uw):
        s = []
        s.append(Paragraph("6.  Cleaning Recommendations", self.s["h1"]))
        s.append(HRFlowable(width=uw, thickness=1, color=SKY, spaceAfter=12))

        ad = Counter(x.recommended_action for x in r.cleaning_recommendations)
        rd = Counter(x.auto_clean_risk for x in r.cleaning_recommendations)

        s.append(Paragraph(
            f"<b>{len(r.cleaning_recommendations)}</b> cleaning actions recommended: "
            f"<b>{rd.get('safe', 0)}</b> auto-safe, "
            f"<b>{rd.get('caution', 0)}</b> caution, "
            f"<b>{rd.get('manual_only', 0)}</b> manual review.", self.s["body"]))
        s.append(Spacer(1, 8))

        data = [["Row", "Column", "Action", "Severity", "Risk", "Rationale"]]
        cell_style = ParagraphStyle("cell2", fontName="Helvetica", fontSize=7.5,
                                    leading=10, textColor=TXT)
        for x in r.cleaning_recommendations[:30]:
            data.append([
                Paragraph(str(x.row_index), cell_style),
                Paragraph(x.column_name, cell_style),
                Paragraph(x.recommended_action.replace("_", " "), cell_style),
                Paragraph(x.severity.upper(), ParagraphStyle("sev2", fontName="Helvetica-Bold",
                          fontSize=7.5, leading=10,
                          textColor=SEV_COLORS.get(x.severity, TXT))),
                Paragraph(x.auto_clean_risk.replace("_", " ").upper(), cell_style),
                Paragraph(x.rationale, cell_style),
            ])
        t = Table(data, colWidths=[30, uw * 0.13, uw * 0.14, 48, 56, uw * 0.32])
        t.setStyle(_table_style(7.5))
        s.append(t)
        return s

    def _quality_metrics(self, r, uw):
        s = []
        s.append(Paragraph("7.  Quality Metrics", self.s["h1"]))
        s.append(HRFlowable(width=uw, thickness=1, color=SKY, spaceAfter=12))

        bm = r.before_metrics
        am = r.after_metrics

        if am:
            data = [["Metric", "Before", "After", "Change"]]
        else:
            data = [["Metric", "Value"]]

        for label, key, unit in [
            ("Noise Rate", "noise_pct", "%"),
            ("Trust Score", "trust_score", " / 100"),
            ("Missingness", "missingness_pct", "%"),
            ("Duplicate Rate", "duplicate_pct", "%"),
            ("Plausibility", "plausibility_rate", "%"),
        ]:
            bv = bm.get(key, 0)
            row = [label, f"{bv:.2f}{unit}"]
            if am:
                av = am.get(key, 0)
                ch = av - bv
                arrow = "improved" if (key != "trust_score" and ch < 0) or (key == "trust_score" and ch > 0) else "same" if ch == 0 else "worse"
                row.append(f"{av:.2f}{unit}")
                row.append(f"{abs(ch):.2f} ({arrow})")
            data.append(row)

        if am:
            t = Table(data, colWidths=[uw * 0.25, uw * 0.25, uw * 0.25, uw * 0.25])
        else:
            t = Table(data, colWidths=[uw * 0.5, uw * 0.5])
        t.setStyle(_table_style())
        s.append(t)

        if am:
            s.append(Spacer(1, 14))
            ba_bytes = self.cb.before_after_bars(bm, am)
            s.append(self._chart_image(ba_bytes, uw * 0.8, 200))
        return s

    def _disclaimer_and_about(self, r, uw):
        s = []
        s.append(Paragraph("8.  Disclaimer", self.s["h1"]))
        s.append(HRFlowable(width=uw, thickness=1, color=SKY, spaceAfter=12))
        s.append(Paragraph(
            "IMPORTANT: This report is generated by AutoClin Engine, an automated "
            "unsupervised machine learning data quality assessment tool. Findings "
            "and recommendations are produced algorithmically and should NOT be "
            "interpreted as definitive clinical judgments.", self.s["warning"]))
        s.append(Spacer(1, 6))

        disclaimers = [
            "All flagged anomalies must be reviewed by qualified clinical data managers or clinicians before any action is taken on clinical datasets.",
            "Unsupervised methods may flag statistically unusual but clinically valid observations. False positives are possible.",
            "Auto-applied corrections are limited to deterministic, reversible operations. All transformations are logged in the immutable audit ledger.",
            "This tool supports but does not replace regulatory-compliant data management (ICH-GCP, FDA 21 CFR Part 11, HIPAA, GDPR).",
            "AutoClin Engine is provided as-is with no guarantees regarding completeness or accuracy. Users should validate results against source documents.",
            "AutoClin Engine does not retain patient data beyond the session. Users must ensure appropriate de-identification.",
        ]
        for d in disclaimers:
            s.append(Paragraph(f"   •   {d}", self.s["body"]))

        s.append(Spacer(1, 24))
        s.append(Paragraph("About the Developer", self.s["h1"]))
        s.append(HRFlowable(width=uw, thickness=1, color=SKY, spaceAfter=12))
        s.append(Paragraph("<b>Dr. Silas Joy Agbesi</b>",
                           ParagraphStyle("name", fontName="Helvetica-Bold",
                                          fontSize=14, textColor=NAVY, spaceAfter=4)))
        s.append(Paragraph(
            "Data Scientist Intern  |  Medical Officer  |  Deputy Clinical Coordinator",
            ParagraphStyle("role", fontName="Helvetica", fontSize=10,
                           textColor=BLUE, spaceAfter=10)))
        s.append(Paragraph(
            "Dr. Silas Joy Agbesi is a medical professional with a passion for "
            "applying data science and machine learning to improve healthcare data "
            "quality. As a Medical Officer and currently serving as Deputy Clinical "
            "Coordinator, Dr. Agbesi brings a unique blend of clinical expertise and "
            "technical skill to AutoClin Engine. His work bridges clinical practice and "
            "data-driven innovation, ensuring that automated quality tools remain "
            "clinically grounded, explainable, and safe for real-world healthcare.",
            self.s["body"]))

        s.append(Spacer(1, 24))
        s.append(HRFlowable(width=uw, thickness=2, color=SKY, spaceAfter=10))
        s.append(Paragraph("AutoClin Engine v1.0.0 — Automated Unsupervised Clinical Data Quality Engine", self.s["center"]))
        now = datetime.now(timezone.utc).strftime("%B %d, %Y")
        s.append(Paragraph(f"Report generated {now}", self.s["center"]))
        return s

"""
AutoClin Engine Chart Builder
Generates chart images (PNG/SVG) for embedding in PDF/DOCX reports.
Uses matplotlib when available, falls back to no-chart mode.
"""
import io
import base64
from typing import Optional
import numpy as np

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


COLORS = {
    "primary": "#3b82f6",
    "success": "#22c55e",
    "warning": "#f59e0b",
    "danger": "#ef4444",
    "critical": "#dc2626",
    "neutral": "#94a3b8",
    "bg": "#f8fafc",
    "severity": {
        "critical": "#dc2626",
        "high": "#ef4444",
        "medium": "#f59e0b",
        "low": "#22c55e",
    },
}


class ChartBuilder:
    """Builds charts for AutoClin Engine reports."""

    def __init__(self, dpi: int = 150, figsize: tuple = (8, 4)):
        self.dpi = dpi
        self.figsize = figsize

    def _fig_to_png_bytes(self, fig) -> bytes:
        if not HAS_MPL:
            return b''
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=self.dpi, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        plt.close(fig)
        buf.seek(0)
        return buf.read()

    def _fig_to_base64(self, fig) -> str:
        data = self._fig_to_png_bytes(fig)
        return base64.b64encode(data).decode('utf-8')

    def trust_score_gauge(self, score: float) -> bytes:
        if not HAS_MPL: return b''
        """Semicircular gauge for Data Trust Score."""
        fig, ax = plt.subplots(figsize=(4, 3))
        theta = np.linspace(np.pi, 0, 100)

        # Background arc
        ax.plot(np.cos(theta), np.sin(theta), color='#e2e8f0', linewidth=20, solid_capstyle='round')

        # Score arc
        filled = int(score)
        theta_filled = np.linspace(np.pi, np.pi - (np.pi * score / 100), max(2, filled))
        color = '#22c55e' if score >= 80 else '#f59e0b' if score >= 60 else '#ef4444'
        ax.plot(np.cos(theta_filled), np.sin(theta_filled), color=color, linewidth=20, solid_capstyle='round')

        ax.text(0, -0.1, f'{score:.0f}', ha='center', va='center', fontsize=36, fontweight='bold', color=color)
        ax.text(0, -0.35, 'Data Trust Score', ha='center', va='center', fontsize=11, color='#64748b')

        ax.set_xlim(-1.3, 1.3)
        ax.set_ylim(-0.5, 1.3)
        ax.set_aspect('equal')
        ax.axis('off')
        return self._fig_to_png_bytes(fig)

    def severity_distribution_pie(self, severity_dist: dict) -> bytes:
        if not HAS_MPL: return b''
        """Pie chart of anomaly severity distribution."""
        fig, ax = plt.subplots(figsize=(5, 4))
        labels = list(severity_dist.keys())
        sizes = list(severity_dist.values())
        colors_list = [COLORS["severity"].get(s, COLORS["neutral"]) for s in labels]

        wedges, texts, autotexts = ax.pie(
            sizes, labels=labels, colors=colors_list, autopct='%1.0f%%',
            startangle=90, textprops={'fontsize': 10}
        )
        ax.set_title('Anomaly Severity Distribution', fontsize=13, fontweight='bold', pad=15)
        return self._fig_to_png_bytes(fig)

    def anomaly_type_bar(self, type_dist: dict, max_types: int = 10) -> bytes:
        if not HAS_MPL: return b''
        """Horizontal bar chart of anomaly type distribution."""
        sorted_types = sorted(type_dist.items(), key=lambda x: x[1], reverse=True)[:max_types]
        labels = [t[0].replace('_', ' ').title() for t in sorted_types]
        values = [t[1] for t in sorted_types]

        fig, ax = plt.subplots(figsize=(8, max(3, len(labels) * 0.5)))
        bars = ax.barh(labels[::-1], values[::-1], color=COLORS["primary"], height=0.6)
        ax.set_xlabel('Count', fontsize=11)
        ax.set_title('Anomaly Types', fontsize=13, fontweight='bold')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        for bar, val in zip(bars, values[::-1]):
            ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2, str(val),
                    va='center', fontsize=9)
        plt.tight_layout()
        return self._fig_to_png_bytes(fig)

    def method_comparison_radar(self, method_rankings: list) -> bytes:
        if not HAS_MPL: return b''
        """Radar chart comparing method scores."""
        dimensions = ['AD', 'SS', 'CP', 'EX', 'CC']
        fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
        angles = np.linspace(0, 2 * np.pi, len(dimensions), endpoint=False).tolist()
        angles += angles[:1]

        colors_cycle = ['#3b82f6', '#ef4444', '#22c55e', '#f59e0b', '#8b5cf6']
        for i, mr in enumerate(method_rankings[:5]):
            values = [mr.get('ad', 0), mr.get('ss', 0), mr.get('cp', 0),
                      mr.get('ex', 0), mr.get('cc', 0)]
            values += values[:1]
            color = colors_cycle[i % len(colors_cycle)]
            ax.plot(angles, values, 'o-', linewidth=1.5, label=mr['method'], color=color)
            ax.fill(angles, values, alpha=0.1, color=color)

        ax.set_thetagrids([a * 180/np.pi for a in angles[:-1]], dimensions)
        ax.set_ylim(0, 1)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=9)
        ax.set_title('Method Comparison', fontsize=13, fontweight='bold', pad=20)
        return self._fig_to_png_bytes(fig)

    def before_after_bars(self, before: dict, after: dict) -> bytes:
        if not HAS_MPL: return b''
        """Grouped bar chart comparing before vs after metrics."""
        metrics = ['Noise %', 'Missingness %', 'Duplicates %']
        before_vals = [before.get('noise_pct', 0), before.get('missingness_pct', 0),
                       before.get('duplicate_pct', 0)]
        after_vals = [after.get('noise_pct', 0), after.get('missingness_pct', 0),
                      after.get('duplicate_pct', 0)]

        x = np.arange(len(metrics))
        width = 0.35

        fig, ax = plt.subplots(figsize=(7, 4))
        bars1 = ax.bar(x - width/2, before_vals, width, label='Before', color='#ef4444', alpha=0.8)
        bars2 = ax.bar(x + width/2, after_vals, width, label='After', color='#22c55e', alpha=0.8)

        ax.set_ylabel('Percentage', fontsize=11)
        ax.set_title('Before vs After Cleaning', fontsize=13, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(metrics, fontsize=10)
        ax.legend()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        for bar in bars1:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    f'{bar.get_height():.1f}', ha='center', va='bottom', fontsize=9)
        for bar in bars2:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    f'{bar.get_height():.1f}', ha='center', va='bottom', fontsize=9)

        plt.tight_layout()
        return self._fig_to_png_bytes(fig)

    def noise_reduction_waterfall(self, type_reductions: dict) -> bytes:
        if not HAS_MPL: return b''
        """Waterfall chart showing noise reduction by category."""
        categories = list(type_reductions.keys())[:8]
        values = [type_reductions[c] for c in categories]
        labels = [c.replace('_', ' ').title()[:20] for c in categories]

        fig, ax = plt.subplots(figsize=(8, 4))
        cumulative = 0
        for i, (label, val) in enumerate(zip(labels, values)):
            ax.bar(i, val, bottom=cumulative, color=COLORS["primary"], alpha=0.8, width=0.6)
            cumulative += val

        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
        ax.set_ylabel('Anomalies Resolved', fontsize=11)
        ax.set_title('Noise Reduction by Category', fontsize=13, fontweight='bold')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tight_layout()
        return self._fig_to_png_bytes(fig)

    def feature_contribution_bar(self, contributions: dict, top_n: int = 10) -> bytes:
        if not HAS_MPL: return b''
        """Horizontal bar chart of feature contributions for a single anomaly."""
        sorted_c = sorted(contributions.items(), key=lambda x: abs(x[1]), reverse=True)[:top_n]
        labels = [c[0] for c in sorted_c][::-1]
        values = [abs(c[1]) * 100 for c in sorted_c][::-1]

        fig, ax = plt.subplots(figsize=(6, max(2, len(labels) * 0.4)))
        ax.barh(labels, values, color=COLORS["primary"], height=0.5)
        ax.set_xlabel('Contribution %', fontsize=10)
        ax.set_title('Feature Contribution to Anomaly', fontsize=12, fontweight='bold')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tight_layout()
        return self._fig_to_png_bytes(fig)

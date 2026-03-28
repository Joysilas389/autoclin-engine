"""
AutoClin Engine CSV Exporter
Exports anomaly logs and cleaning audit as CSV.
"""
import csv
import io
from typing import Optional


class CSVExporter:
    """Exports pipeline results as CSV files."""

    def export_anomaly_log(self, classifications: list, output_path: str):
        """Export anomaly classifications as a CSV file."""
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                'row_index', 'anomaly_type', 'severity', 'confidence',
                'flagged_columns', 'rationale',
            ])
            for cls in classifications:
                writer.writerow([
                    cls.row_index, cls.anomaly_type, cls.severity,
                    f"{cls.confidence:.3f}",
                    '; '.join(cls.flagged_columns),
                    cls.rationale,
                ])

    def export_cleaning_audit(self, audit_entries: list, output_path: str):
        """Export cleaning audit trail as CSV."""
        if not audit_entries:
            return

        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                'transaction_id', 'timestamp', 'row_index', 'column_name',
                'original_value', 'new_value', 'action', 'method_trigger',
                'risk_level', 'rationale',
            ])
            for entry in audit_entries:
                writer.writerow([
                    entry.get('transaction_id', ''),
                    entry.get('timestamp', ''),
                    entry.get('row_index', ''),
                    entry.get('column_name', ''),
                    entry.get('original_value', ''),
                    entry.get('new_value', ''),
                    entry.get('action', ''),
                    entry.get('method_trigger', ''),
                    entry.get('risk_level', ''),
                    entry.get('rationale', ''),
                ])

    def export_method_comparison(self, method_rankings: list, output_path: str):
        """Export method comparison table as CSV."""
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                'method', 'composite', 'ndc', 'ad', 'ss', 'cp', 'ex', 'cc',
                'anomaly_count', 'duration_ms', 'selected',
            ])
            for mr in method_rankings:
                writer.writerow([
                    mr['method'], f"{mr['composite']:.4f}", f"{mr['ndc']:.4f}",
                    f"{mr['ad']:.4f}", f"{mr['ss']:.4f}", f"{mr['cp']:.4f}",
                    f"{mr['ex']:.4f}", f"{mr['cc']:.4f}",
                    mr.get('anomaly_count', 0), mr.get('duration_ms', 0),
                    mr.get('selected', False),
                ])

"""
AutoClin Engine Cleaning Audit Ledger
Immutable, append-only log of every transformation with reproducibility metadata.
"""
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional
import uuid
import json


@dataclass
class AuditEntry:
    transaction_id: str
    timestamp: str
    run_id: str
    anomaly_id: Optional[str]
    row_index: int
    column_name: str
    original_value: object
    new_value: object
    action: str
    method_trigger: str
    risk_level: str
    approved_by: Optional[str]
    rationale: str
    reproducibility: dict


class AuditLedger:
    """Append-only audit ledger for all cleaning transformations."""

    def __init__(self, run_id: str, clinengine_version: str = "1.0.0", random_seed: int = 42):
        self.run_id = run_id
        self.version = clinengine_version
        self.random_seed = random_seed
        self.entries: list[AuditEntry] = []
        self._counter = 0

    def log(
        self,
        row_index: int,
        column_name: str,
        original_value: object,
        new_value: object,
        action: str,
        method_trigger: str,
        risk_level: str,
        rationale: str,
        anomaly_id: Optional[str] = None,
        approved_by: Optional[str] = None,
        method_params: Optional[dict] = None,
    ) -> AuditEntry:
        """Log a single transformation."""
        self._counter += 1
        now = datetime.now(timezone.utc).isoformat()
        date_str = now[:10].replace("-", "")
        txn_id = f"TXN-{date_str}-{self._counter:04d}"

        entry = AuditEntry(
            transaction_id=txn_id,
            timestamp=now,
            run_id=self.run_id,
            anomaly_id=anomaly_id,
            row_index=row_index,
            column_name=column_name,
            original_value=original_value,
            new_value=new_value,
            action=action,
            method_trigger=method_trigger,
            risk_level=risk_level,
            approved_by=approved_by,
            rationale=rationale,
            reproducibility={
                "clinengine_version": self.version,
                "random_seed": self.random_seed,
                "method_params": method_params or {},
            },
        )
        self.entries.append(entry)
        return entry

    def to_records(self) -> list[dict]:
        """Export all entries as serializable dicts."""
        return [
            {
                "transaction_id": e.transaction_id,
                "timestamp": e.timestamp,
                "run_id": e.run_id,
                "anomaly_id": e.anomaly_id,
                "row_index": e.row_index,
                "column_name": e.column_name,
                "original_value": str(e.original_value) if e.original_value is not None else None,
                "new_value": str(e.new_value) if e.new_value is not None else None,
                "action": e.action,
                "method_trigger": e.method_trigger,
                "risk_level": e.risk_level,
                "approved_by": e.approved_by,
                "rationale": e.rationale,
                "reproducibility": e.reproducibility,
            }
            for e in self.entries
        ]

    def get_replay(self) -> list[dict]:
        """Get chronological replay of all transformations."""
        return sorted(self.to_records(), key=lambda e: e["timestamp"])

    @property
    def count(self) -> int:
        return len(self.entries)

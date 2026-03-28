"""
AutoClin Engine Duplicate Detector
Exact duplicates via row hashing, near-duplicates via blocking + Jaccard similarity.
"""
import hashlib
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
import xxhash


@dataclass
class DuplicateCluster:
    """A group of duplicate or near-duplicate records."""
    cluster_id: int
    row_indices: list[int]
    similarity: float  # 1.0 for exact, <1.0 for near-duplicates
    duplicate_type: str  # "exact" or "near"
    most_complete_index: int  # row with fewest nulls


@dataclass
class DuplicateReport:
    """Summary of duplicate detection results."""
    exact_duplicate_count: int
    near_duplicate_count: int
    exact_clusters: list[DuplicateCluster]
    near_clusters: list[DuplicateCluster]
    total_rows: int

    @property
    def exact_rate(self) -> float:
        return self.exact_duplicate_count / self.total_rows if self.total_rows > 0 else 0.0

    @property
    def total_duplicate_rate(self) -> float:
        total = self.exact_duplicate_count + self.near_duplicate_count
        return total / self.total_rows if self.total_rows > 0 else 0.0


class DuplicateDetector:
    """
    Detects exact and near-duplicate records in a DataFrame.
    
    Exact: row-level xxhash comparison.
    Near: blocking on identifier columns + Jaccard similarity on remaining fields.
    """

    def __init__(
        self,
        near_threshold: float = 0.85,
        blocking_columns: Optional[list[str]] = None,
        max_comparisons: int = 1_000_000,
    ):
        self.near_threshold = near_threshold
        self.blocking_columns = blocking_columns
        self.max_comparisons = max_comparisons

    def detect(self, df: pd.DataFrame) -> DuplicateReport:
        exact_clusters = self._find_exact_duplicates(df)
        exact_row_set = set()
        for cluster in exact_clusters:
            exact_row_set.update(cluster.row_indices[1:])  # keep first, rest are dupes

        near_clusters = self._find_near_duplicates(df, exclude_rows=exact_row_set)

        exact_count = sum(len(c.row_indices) - 1 for c in exact_clusters)
        near_count = sum(len(c.row_indices) - 1 for c in near_clusters)

        return DuplicateReport(
            exact_duplicate_count=exact_count,
            near_duplicate_count=near_count,
            exact_clusters=exact_clusters,
            near_clusters=near_clusters,
            total_rows=len(df),
        )

    def _find_exact_duplicates(self, df: pd.DataFrame) -> list[DuplicateCluster]:
        """Find exact duplicates via row hashing."""
        hashes: dict[str, list[int]] = {}
        for idx in range(len(df)):
            row_str = "|".join(str(v) for v in df.iloc[idx].values)
            h = xxhash.xxh64(row_str).hexdigest()
            if h not in hashes:
                hashes[h] = []
            hashes[h].append(idx)

        clusters = []
        cluster_id = 0
        for h, indices in hashes.items():
            if len(indices) > 1:
                # Find row with fewest nulls
                null_counts = [df.iloc[i].isna().sum() for i in indices]
                most_complete = indices[int(np.argmin(null_counts))]

                clusters.append(DuplicateCluster(
                    cluster_id=cluster_id,
                    row_indices=indices,
                    similarity=1.0,
                    duplicate_type="exact",
                    most_complete_index=most_complete,
                ))
                cluster_id += 1

        return clusters

    def _find_near_duplicates(
        self, df: pd.DataFrame, exclude_rows: set[int]
    ) -> list[DuplicateCluster]:
        """
        Find near-duplicates using blocking + Jaccard similarity.
        """
        if len(df) < 2:
            return []

        # Determine blocking columns
        block_cols = self.blocking_columns
        if block_cols is None:
            block_cols = self._auto_detect_blocking_columns(df)

        if not block_cols:
            # No blocking columns; skip near-duplicate detection for large datasets
            if len(df) > 5000:
                return []
            return self._brute_force_near_duplicates(df, exclude_rows)

        return self._blocked_near_duplicates(df, block_cols, exclude_rows)

    def _auto_detect_blocking_columns(self, df: pd.DataFrame) -> list[str]:
        """Auto-detect high-cardinality identifier columns for blocking."""
        candidates = []
        for col in df.columns:
            name_lower = col.lower()
            id_keywords = [
                "patient", "subject", "subj", "ptid", "record", "id",
                "identifier", "mrn", "ssn", "enrollment",
            ]
            if any(kw in name_lower for kw in id_keywords):
                if df[col].nunique() > 1:
                    candidates.append(col)
        return candidates[:2]  # Use at most 2 blocking columns

    def _blocked_near_duplicates(
        self, df: pd.DataFrame, block_cols: list[str], exclude_rows: set[int]
    ) -> list[DuplicateCluster]:
        """Near-duplicate detection with blocking to reduce comparison space."""
        clusters = []
        cluster_id = 1000  # offset from exact cluster IDs

        # Group by blocking columns
        grouped = df.groupby(block_cols, dropna=False)
        compare_cols = [c for c in df.columns if c not in block_cols]

        for _, group in grouped:
            if len(group) < 2:
                continue

            indices = group.index.tolist()
            visited = set()

            for i in range(len(indices)):
                if indices[i] in exclude_rows or indices[i] in visited:
                    continue
                cluster_members = [indices[i]]

                for j in range(i + 1, len(indices)):
                    if indices[j] in exclude_rows or indices[j] in visited:
                        continue
                    sim = self._jaccard_similarity(
                        df.iloc[indices[i]][compare_cols],
                        df.iloc[indices[j]][compare_cols],
                    )
                    if sim >= self.near_threshold:
                        cluster_members.append(indices[j])
                        visited.add(indices[j])

                if len(cluster_members) > 1:
                    null_counts = [df.iloc[idx].isna().sum() for idx in cluster_members]
                    most_complete = cluster_members[int(np.argmin(null_counts))]
                    avg_sim = self.near_threshold  # approximate

                    clusters.append(DuplicateCluster(
                        cluster_id=cluster_id,
                        row_indices=cluster_members,
                        similarity=avg_sim,
                        duplicate_type="near",
                        most_complete_index=most_complete,
                    ))
                    cluster_id += 1

        return clusters

    def _brute_force_near_duplicates(
        self, df: pd.DataFrame, exclude_rows: set[int]
    ) -> list[DuplicateCluster]:
        """Brute force for small datasets without blocking columns."""
        clusters = []
        cluster_id = 1000
        visited = set()
        n = len(df)

        for i in range(n):
            if i in exclude_rows or i in visited:
                continue
            cluster_members = [i]
            for j in range(i + 1, n):
                if j in exclude_rows or j in visited:
                    continue
                sim = self._jaccard_similarity(df.iloc[i], df.iloc[j])
                if sim >= self.near_threshold:
                    cluster_members.append(j)
                    visited.add(j)

            if len(cluster_members) > 1:
                null_counts = [df.iloc[idx].isna().sum() for idx in cluster_members]
                most_complete = cluster_members[int(np.argmin(null_counts))]
                clusters.append(DuplicateCluster(
                    cluster_id=cluster_id,
                    row_indices=cluster_members,
                    similarity=self.near_threshold,
                    duplicate_type="near",
                    most_complete_index=most_complete,
                ))
                cluster_id += 1

        return clusters

    @staticmethod
    def _jaccard_similarity(row_a: pd.Series, row_b: pd.Series) -> float:
        """Compute Jaccard similarity between two rows (treating non-null matching values)."""
        total_fields = len(row_a)
        if total_fields == 0:
            return 0.0

        matches = 0
        comparable = 0
        for val_a, val_b in zip(row_a.values, row_b.values):
            a_null = pd.isna(val_a)
            b_null = pd.isna(val_b)
            if a_null and b_null:
                matches += 1
                comparable += 1
            elif a_null or b_null:
                comparable += 1
            else:
                comparable += 1
                if str(val_a).strip() == str(val_b).strip():
                    matches += 1

        return matches / comparable if comparable > 0 else 0.0

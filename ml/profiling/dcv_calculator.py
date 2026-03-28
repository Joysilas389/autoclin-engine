"""
AutoClin Engine Dataset Characterization Vector (DCV) Calculator
Computes a 12-feature vector that drives method selection and filtering.
"""
from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist


@dataclass
class DatasetCharacterizationVector:
    n: int                    # Row count
    d: int                    # Column count (after preprocessing)
    d_over_n: float           # Dimensionality ratio
    missingness_burden: float # Proportion of all cells that are null
    numeric_proportion: float # Fraction of numeric columns
    categorical_cardinality: float  # Mean unique/n for categoricals
    temporal_flag: int        # 1 if datetime + patient IDs detected
    site_count: int           # Number of distinct sites
    estimated_noise_rate: float  # From profiling
    cluster_tendency: float   # Hopkins statistic
    skewness_spread: float    # Mean absolute skewness
    correlation_density: float  # Mean absolute pairwise correlation

    def to_dict(self) -> dict:
        return {
            "n": self.n, "d": self.d, "d_over_n": round(self.d_over_n, 4),
            "missingness_burden": round(self.missingness_burden, 4),
            "numeric_proportion": round(self.numeric_proportion, 4),
            "categorical_cardinality": round(self.categorical_cardinality, 4),
            "temporal_flag": self.temporal_flag,
            "site_count": self.site_count,
            "estimated_noise_rate": round(self.estimated_noise_rate, 4),
            "cluster_tendency": round(self.cluster_tendency, 4),
            "skewness_spread": round(self.skewness_spread, 4),
            "correlation_density": round(self.correlation_density, 4),
        }


class DCVCalculator:
    """Computes the Dataset Characterization Vector from profile results."""

    def compute(
        self,
        df: pd.DataFrame,
        profile_result,
        clinical_mappings: list,
    ) -> DatasetCharacterizationVector:

        n = profile_result.row_count
        d = profile_result.col_count
        d_over_n = d / n if n > 0 else 0.0

        # Numeric proportion
        numeric_cols = [c for c in profile_result.columns if c.dtype == "numeric"]
        numeric_prop = len(numeric_cols) / d if d > 0 else 0.0

        # Categorical cardinality
        cat_cols = [c for c in profile_result.columns if c.dtype == "categorical"]
        cat_card = 0.0
        if cat_cols:
            cat_card = np.mean([c.cardinality_ratio for c in cat_cols])

        # Temporal flag
        has_datetime = any(c.dtype == "datetime" for c in profile_result.columns) or \
                       any(m.clinical_type == "visit_date" for m in clinical_mappings if m.clinical_type)
        has_patient_id = any(m.clinical_type == "patient_id" for m in clinical_mappings if m.clinical_type)
        temporal_flag = 1 if (has_datetime and has_patient_id) else 0

        # Site count
        site_count = 0
        site_mappings = [m for m in clinical_mappings if m.clinical_type == "site_id"]
        if site_mappings:
            site_col = site_mappings[0].column_name
            if site_col in df.columns:
                site_count = df[site_col].nunique()

        # Skewness spread
        skewness_vals = [c.skewness for c in profile_result.columns
                         if c.skewness is not None]
        skewness_spread = np.mean(np.abs(skewness_vals)) if skewness_vals else 0.0

        # Correlation density
        corr_density = 0.0
        if profile_result.correlation_matrix and "values" in profile_result.correlation_matrix:
            corr_mat = np.array(profile_result.correlation_matrix["values"])
            np.fill_diagonal(corr_mat, 0)
            corr_density = np.mean(np.abs(corr_mat))

        # Hopkins statistic (cluster tendency)
        cluster_tendency = self._compute_hopkins(df, numeric_cols)

        return DatasetCharacterizationVector(
            n=n, d=d, d_over_n=d_over_n,
            missingness_burden=profile_result.overall_missingness,
            numeric_proportion=numeric_prop,
            categorical_cardinality=cat_card,
            temporal_flag=temporal_flag,
            site_count=site_count,
            estimated_noise_rate=profile_result.noise_estimate,
            cluster_tendency=cluster_tendency,
            skewness_spread=skewness_spread,
            correlation_density=corr_density,
        )

    def _compute_hopkins(self, df: pd.DataFrame, numeric_col_profiles: list) -> float:
        """
        Compute Hopkins statistic to assess cluster tendency.
        Values near 0.5 indicate uniform (no clusters); >0.7 indicates clusterable.
        """
        numeric_col_names = [c.name for c in numeric_col_profiles]
        if len(numeric_col_names) < 2:
            return 0.5  # default: no cluster tendency assessment

        numeric_df = df[numeric_col_names].apply(pd.to_numeric, errors="coerce").dropna()
        if len(numeric_df) < 30:
            return 0.5

        # Sample size for Hopkins
        m = min(int(len(numeric_df) * 0.1), 200)
        if m < 5:
            return 0.5

        try:
            data = numeric_df.values
            n = len(data)

            # Random sample of m points from the dataset
            rng = np.random.RandomState(42)
            sample_idx = rng.choice(n, size=m, replace=False)
            sample_points = data[sample_idx]

            # Generate m random points in the data space
            mins = data.min(axis=0)
            maxs = data.max(axis=0)
            random_points = rng.uniform(mins, maxs, size=(m, data.shape[1]))

            # Compute nearest-neighbor distances
            from scipy.spatial import cKDTree
            tree = cKDTree(data)

            u_distances = np.array([tree.query(rp, k=1)[0] for rp in random_points])
            w_distances = np.array([tree.query(sp, k=2)[0][1] for sp in sample_points])

            u_sum = np.sum(u_distances ** data.shape[1])
            w_sum = np.sum(w_distances ** data.shape[1])

            H = u_sum / (u_sum + w_sum) if (u_sum + w_sum) > 0 else 0.5
            return float(np.clip(H, 0, 1))
        except Exception:
            return 0.5

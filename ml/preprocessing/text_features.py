"""
AutoClin Engine Text Feature Extractor
Converts free-text and high-cardinality string columns into numeric features
for unsupervised anomaly detection.

Strategies:
1. String length features (length, word count, digit ratio, special char ratio)
2. TF-IDF sparse encoding (top-k terms, truncated SVD for dimensionality reduction)
3. Character-level statistics (entropy, case ratio, whitespace ratio)
4. Pattern flags (email-like, date-like, numeric-like, empty/null patterns)

These features let unsupervised methods detect:
- Encoding inconsistencies (mixed languages, garbled text)
- Data entry anomalies (unusually long/short entries, wrong field content)
- Format violations (free text in structured fields)
- Suspicious patterns (identical repeated text, placeholder values)
"""
import numpy as np
import pandas as pd
from typing import Optional
import math
from collections import Counter
import re


class TextFeatureExtractor:
    """
    Extracts numeric features from text columns for anomaly detection.
    Does NOT require any pretrained models — pure statistical features.
    """

    def __init__(self, max_tfidf_features: int = 20, min_doc_freq: int = 2):
        self.max_tfidf_features = max_tfidf_features
        self.min_doc_freq = min_doc_freq

    def extract(self, series: pd.Series, col_name: str) -> tuple[np.ndarray, list[str]]:
        """
        Extract numeric features from a text column.
        Returns (feature_matrix, feature_names).
        """
        n = len(series)
        str_series = series.fillna("").astype(str)

        features = {}

        # ── 1. String length features ──
        lengths = str_series.str.len().values.astype(float)
        features[f"{col_name}__len"] = lengths

        word_counts = str_series.str.split().str.len().fillna(0).values.astype(float)
        features[f"{col_name}__words"] = word_counts

        # ── 2. Character composition ratios ──
        digit_ratio = str_series.apply(
            lambda x: sum(c.isdigit() for c in x) / max(len(x), 1)
        ).values.astype(float)
        features[f"{col_name}__digit_ratio"] = digit_ratio

        alpha_ratio = str_series.apply(
            lambda x: sum(c.isalpha() for c in x) / max(len(x), 1)
        ).values.astype(float)
        features[f"{col_name}__alpha_ratio"] = alpha_ratio

        upper_ratio = str_series.apply(
            lambda x: sum(c.isupper() for c in x) / max(len(x), 1)
        ).values.astype(float)
        features[f"{col_name}__upper_ratio"] = upper_ratio

        space_ratio = str_series.apply(
            lambda x: sum(c.isspace() for c in x) / max(len(x), 1)
        ).values.astype(float)
        features[f"{col_name}__space_ratio"] = space_ratio

        special_ratio = str_series.apply(
            lambda x: sum(not c.isalnum() and not c.isspace() for c in x) / max(len(x), 1)
        ).values.astype(float)
        features[f"{col_name}__special_ratio"] = special_ratio

        # ── 3. Shannon entropy of characters ──
        features[f"{col_name}__entropy"] = str_series.apply(self._char_entropy).values.astype(float)

        # ── 4. Pattern flags ──
        features[f"{col_name}__is_empty"] = (lengths == 0).astype(float)

        features[f"{col_name}__looks_numeric"] = str_series.apply(
            lambda x: 1.0 if re.match(r'^[\d\.\-\+eE,]+$', x.strip()) else 0.0
        ).values.astype(float)

        features[f"{col_name}__has_date_pattern"] = str_series.apply(
            lambda x: 1.0 if re.search(r'\d{2,4}[-/]\d{1,2}[-/]\d{1,2}', x) else 0.0
        ).values.astype(float)

        # ── 5. Uniqueness signal ──
        value_counts = str_series.value_counts()
        freq = str_series.map(value_counts).values.astype(float)
        features[f"{col_name}__value_freq"] = freq

        # Rarity: values appearing only once
        features[f"{col_name}__is_rare"] = (freq == 1).astype(float)

        # ── 6. TF-IDF features (if enough unique tokens) ──
        tfidf_features, tfidf_names = self._tfidf_features(str_series, col_name)
        if tfidf_features is not None:
            for i, name in enumerate(tfidf_names):
                features[name] = tfidf_features[:, i]

        # Stack into matrix
        names = list(features.keys())
        X = np.column_stack([features[n] for n in names])

        return X, names

    def _char_entropy(self, text: str) -> float:
        """Shannon entropy of character distribution in a string."""
        if not text:
            return 0.0
        counts = Counter(text)
        length = len(text)
        return -sum(
            (c / length) * math.log2(c / length)
            for c in counts.values()
            if c > 0
        )

    def _tfidf_features(self, series: pd.Series, col_name: str
                        ) -> tuple[Optional[np.ndarray], list[str]]:
        """
        Simple TF-IDF with SVD dimensionality reduction.
        Falls back gracefully if sklearn is available.
        """
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.decomposition import TruncatedSVD

            non_empty = series[series.str.len() > 0]
            if len(non_empty) < 10:
                return None, []

            # Only do TF-IDF if there are enough distinct words
            all_words = set()
            for text in non_empty.head(500):
                all_words.update(text.lower().split())
            if len(all_words) < 5:
                return None, []

            n_features = min(self.max_tfidf_features, len(all_words))
            n_components = min(5, n_features - 1)
            if n_components < 1:
                return None, []

            vectorizer = TfidfVectorizer(
                max_features=n_features,
                min_df=self.min_doc_freq,
                stop_words="english",
                token_pattern=r'(?u)\b\w+\b',
            )

            tfidf_matrix = vectorizer.fit_transform(series)

            if tfidf_matrix.shape[1] < 2:
                return None, []

            svd = TruncatedSVD(n_components=n_components, random_state=42)
            reduced = svd.fit_transform(tfidf_matrix)

            names = [f"{col_name}__tfidf_svd_{i}" for i in range(n_components)]
            return reduced, names

        except Exception:
            return None, []


def extract_text_features_for_dataset(
    df: pd.DataFrame, profile, max_text_cols: int = 10
) -> tuple[Optional[np.ndarray], list[str]]:
    """
    Convenience function: extract text features for all text/high-cardinality columns.
    Returns combined feature matrix and names, or (None, []) if no text columns.
    """
    extractor = TextFeatureExtractor()
    all_parts = []
    all_names = []

    text_cols = []
    for cp in profile.columns:
        if cp.dtype in ("free_text", "identifier") and cp.name in df.columns:
            text_cols.append(cp.name)
        elif cp.dtype == "categorical" and cp.unique_count > 50 and cp.name in df.columns:
            text_cols.append(cp.name)

    for col in text_cols[:max_text_cols]:
        X_text, names = extractor.extract(df[col], col)
        if X_text.shape[1] > 0:
            all_parts.append(X_text)
            all_names.extend(names)

    if not all_parts:
        return None, []

    return np.hstack(all_parts), all_names

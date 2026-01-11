"""
Feature Collection

Container for extracted features with querying and manipulation methods.
"""

from typing import List, Optional, Dict, Tuple, Any
import pandas as pd
import numpy as np
import warnings


class FeatureCollection:
    """
    Container for extracted features from multiple windows/recordings.

    Manages features in DataFrame format with subject IDs and metadata.
    Provides methods for:
    - Feature selection
    - Normalization
    - Merging with labels
    - Splitting train/test
    - Export

    Attributes:
        features_df (pd.DataFrame): DataFrame with features (rows=windows, cols=features)
        subject_ids (List[str]): Subject ID for each row
        window_indices (List[int]): Window index for each row (if applicable)
        feature_names (List[str]): List of feature column names
        metadata (dict): Additional metadata
    """

    def __init__(
        self,
        features_df: pd.DataFrame,
        subject_ids: Optional[List[str]] = None,
        window_indices: Optional[List[int]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize feature collection.

        Args:
            features_df: DataFrame with features
            subject_ids: Subject IDs for each row
            window_indices: Window indices for each row
            metadata: Additional metadata
        """
        if not isinstance(features_df, pd.DataFrame):
            raise TypeError("features_df must be a pandas DataFrame")

        self.features_df = features_df.copy()
        self.subject_ids = subject_ids or [None] * len(features_df)
        self.window_indices = window_indices or list(range(len(features_df)))
        self.metadata = metadata or {}

        if len(self.subject_ids) != len(features_df):
            raise ValueError("subject_ids length must match DataFrame length")

        if len(self.window_indices) != len(features_df):
            raise ValueError("window_indices length must match DataFrame length")

    @classmethod
    def from_dict(
        cls,
        features_dict: Dict[str, Dict[str, float]],
        subject_ids: Optional[List[str]] = None
    ) -> 'FeatureCollection':
        """
        Create from dictionary of features.

        Args:
            features_dict: Dict mapping row_id to feature dict
            subject_ids: Optional subject IDs

        Returns:
            FeatureCollection instance
        """
        df = pd.DataFrame.from_dict(features_dict, orient='index')

        if subject_ids is None:
            subject_ids = [None] * len(df)

        return cls(df, subject_ids=subject_ids)

    @classmethod
    def concatenate(
        cls,
        collections: List['FeatureCollection']
    ) -> 'FeatureCollection':
        """
        Concatenate multiple feature collections.

        Args:
            collections: List of FeatureCollection objects

        Returns:
            Combined FeatureCollection
        """
        if not collections:
            raise ValueError("No collections to concatenate")

        # Concatenate DataFrames
        dfs = [c.features_df for c in collections]
        combined_df = pd.concat(dfs, ignore_index=True)

        # Combine subject IDs and window indices
        combined_subject_ids = []
        combined_window_indices = []

        for collection in collections:
            combined_subject_ids.extend(collection.subject_ids)
            combined_window_indices.extend(collection.window_indices)

        return cls(
            combined_df,
            subject_ids=combined_subject_ids,
            window_indices=combined_window_indices
        )

    @property
    def feature_names(self) -> List[str]:
        """
        Get list of feature names.

        Returns:
            List of feature column names
        """
        return self.features_df.columns.tolist()

    @property
    def n_features(self) -> int:
        """
        Number of features.

        Returns:
            Number of feature columns
        """
        return self.features_df.shape[1]

    @property
    def n_samples(self) -> int:
        """
        Number of samples (rows).

        Returns:
            Number of rows
        """
        return self.features_df.shape[0]

    @property
    def shape(self) -> Tuple[int, int]:
        """
        Shape of feature matrix.

        Returns:
            (n_samples, n_features)
        """
        return self.features_df.shape

    def get_features_array(self) -> np.ndarray:
        """
        Get features as NumPy array.

        Returns:
            Feature array (n_samples, n_features)
        """
        return self.features_df.values

    def get_subject_features(
        self,
        subject_id: str
    ) -> pd.DataFrame:
        """
        Get all features for a specific subject.

        Args:
            subject_id: Subject identifier

        Returns:
            DataFrame with features for this subject
        """
        mask = [s == subject_id for s in self.subject_ids]
        return self.features_df[mask].reset_index(drop=True)

    def select_features(
        self,
        feature_names: List[str],
        inplace: bool = False
    ) -> 'FeatureCollection':
        """
        Select subset of features.

        Args:
            feature_names: List of feature names to keep
            inplace: Modify in place or return new instance

        Returns:
            FeatureCollection with selected features
        """
        # Validate feature names
        missing = set(feature_names) - set(self.feature_names)
        if missing:
            raise ValueError(f"Features not found: {missing}")

        selected_df = self.features_df[feature_names].copy()

        if inplace:
            self.features_df = selected_df
            return self
        else:
            return FeatureCollection(
                selected_df,
                subject_ids=self.subject_ids.copy(),
                window_indices=self.window_indices.copy(),
                metadata=self.metadata.copy()
            )

    def drop_features(
        self,
        feature_names: List[str],
        inplace: bool = False
    ) -> 'FeatureCollection':
        """
        Drop features.

        Args:
            feature_names: Features to drop
            inplace: Modify in place

        Returns:
            FeatureCollection without dropped features
        """
        remaining = [f for f in self.feature_names if f not in feature_names]
        return self.select_features(remaining, inplace=inplace)

    def normalize(
        self,
        method: str = 'zscore',
        inplace: bool = False
    ) -> 'FeatureCollection':
        """
        Normalize features.

        Args:
            method: 'zscore', 'minmax', or 'robust'
            inplace: Modify in place

        Returns:
            Normalized FeatureCollection
        """
        normalized_df = self.features_df.copy()

        if method == 'zscore':
            # Z-score normalization: (x - mean) / std
            normalized_df = (normalized_df - normalized_df.mean()) / normalized_df.std()

        elif method == 'minmax':
            # Min-Max normalization: (x - min) / (max - min)
            normalized_df = (normalized_df - normalized_df.min()) / (normalized_df.max() - normalized_df.min())

        elif method == 'robust':
            # Robust scaling: (x - median) / IQR
            median = normalized_df.median()
            q1 = normalized_df.quantile(0.25)
            q3 = normalized_df.quantile(0.75)
            iqr = q3 - q1
            normalized_df = (normalized_df - median) / (iqr + 1e-8)

        else:
            raise ValueError(f"Unknown normalization method: {method}")

        if inplace:
            self.features_df = normalized_df
            return self
        else:
            return FeatureCollection(
                normalized_df,
                subject_ids=self.subject_ids.copy(),
                window_indices=self.window_indices.copy(),
                metadata=self.metadata.copy()
            )

    def handle_missing_values(
        self,
        strategy: str = 'mean',
        inplace: bool = False
    ) -> 'FeatureCollection':
        """
        Handle missing values in features.

        Args:
            strategy: 'mean', 'median', 'drop', or 'zero'
            inplace: Modify in place

        Returns:
            FeatureCollection with handled missing values
        """
        df = self.features_df.copy()

        if strategy == 'drop':
            # Drop rows with any NaN
            mask = ~df.isna().any(axis=1)
            df = df[mask]
            subject_ids = [s for s, m in zip(self.subject_ids, mask) if m]
            window_indices = [w for w, m in zip(self.window_indices, mask) if m]

        elif strategy == 'mean':
            # Fill with column mean
            df = df.fillna(df.mean())

        elif strategy == 'median':
            # Fill with column median
            df = df.fillna(df.median())

        elif strategy == 'zero':
            # Fill with zero
            df = df.fillna(0)

        else:
            raise ValueError(f"Unknown missing value strategy: {strategy}")

        if inplace:
            self.features_df = df
            if strategy == 'drop':
                self.subject_ids = subject_ids
                self.window_indices = window_indices
            return self
        else:
            return FeatureCollection(
                df,
                subject_ids=subject_ids if strategy == 'drop' else self.subject_ids.copy(),
                window_indices=window_indices if strategy == 'drop' else self.window_indices.copy(),
                metadata=self.metadata.copy()
            )

    def merge_with_labels(
        self,
        labels_df: pd.DataFrame,
        on: str = 'SubjectID',
        outcome: Optional[str] = None,
        also_on: Optional[str] = None
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Merge features with labels.

        Args:
            labels_df: DataFrame with labels
            on: Column to merge on (e.g., 'SubjectID')
            outcome: If specified, return only this outcome's labels
            also_on: Additional column to merge on (e.g., 'RecordingDate') to prevent duplicates

        Returns:
            Tuple of (merged_features_df, labels_array)
        """
        # Add subject IDs to features for merging
        features_with_ids = self.features_df.copy()
        features_with_ids[on] = self.subject_ids

        # DEBUG: Identify which subject IDs are missing labels
        feature_subject_ids = set(features_with_ids[on].unique())
        label_subject_ids = set(labels_df[on].unique()) if on in labels_df.columns else set()

        missing_in_labels = feature_subject_ids - label_subject_ids
        if missing_in_labels:
            print(f"\n⚠️  WARNING: {len(missing_in_labels)} subject ID(s) in features have NO labels:")
            for sid in sorted(missing_in_labels):
                count = len(features_with_ids[features_with_ids[on] == sid])
                print(f"    - {repr(sid)}: {count} recording(s)")
            print(f"    These {sum(len(features_with_ids[features_with_ids[on] == sid]) for sid in missing_in_labels)} recordings will be EXCLUDED from analysis!\n")

        # Determine merge columns
        if also_on and also_on in features_with_ids.columns and also_on in labels_df.columns:
            merge_on = [on, also_on]
        else:
            merge_on = on

        # Merge
        merged = features_with_ids.merge(labels_df, on=merge_on, how='inner')

        # Extract labels
        if outcome is None:
            # Try to find outcome column
            possible_outcomes = [c for c in merged.columns if c not in self.feature_names and c != on]
            if len(possible_outcomes) != 1:
                raise ValueError(f"Cannot determine outcome column. Found: {possible_outcomes}")
            outcome = possible_outcomes[0]

        labels = merged[outcome].values
        features_only = merged.drop(columns=[outcome] + [c for c in merged.columns if c not in self.feature_names and c != on])

        n_original = len(self.features_df)
        n_merged = len(merged)

        if n_merged < n_original:
            warnings.warn(
                f"Merge reduced dataset from {n_original} to {n_merged} samples. "
                f"Some features may not have corresponding labels."
            )

        return features_only, labels

    def split_by_subjects(
        self,
        test_subjects: List[str]
    ) -> Tuple['FeatureCollection', 'FeatureCollection']:
        """
        Split into train/test by subjects.

        Args:
            test_subjects: List of subject IDs for test set

        Returns:
            Tuple of (train_collection, test_collection)
        """
        test_mask = [s in test_subjects for s in self.subject_ids]
        train_mask = [not m for m in test_mask]

        train_df = self.features_df[train_mask].reset_index(drop=True)
        test_df = self.features_df[test_mask].reset_index(drop=True)

        train_subjects = [s for s, m in zip(self.subject_ids, train_mask) if m]
        test_subjects_actual = [s for s, m in zip(self.subject_ids, test_mask) if m]

        train_windows = [w for w, m in zip(self.window_indices, train_mask) if m]
        test_windows = [w for w, m in zip(self.window_indices, test_mask) if m]

        train_collection = FeatureCollection(
            train_df,
            subject_ids=train_subjects,
            window_indices=train_windows,
            metadata=self.metadata.copy()
        )

        test_collection = FeatureCollection(
            test_df,
            subject_ids=test_subjects_actual,
            window_indices=test_windows,
            metadata=self.metadata.copy()
        )

        return train_collection, test_collection

    def get_feature_statistics(self) -> pd.DataFrame:
        """
        Get statistics for each feature.

        Returns:
            DataFrame with mean, std, min, max, etc. for each feature
        """
        # Select only numeric columns (exclude SubjectID, RecordingDate, etc.)
        numeric_df = self.features_df.select_dtypes(include=[np.number])

        stats = pd.DataFrame({
            'mean': numeric_df.mean(),
            'std': numeric_df.std(),
            'min': numeric_df.min(),
            'max': numeric_df.max(),
            'median': numeric_df.median(),
            'q25': numeric_df.quantile(0.25),
            'q75': numeric_df.quantile(0.75),
        })
        return stats

    def get_correlation_matrix(
        self,
        method: str = 'pearson'
    ) -> pd.DataFrame:
        """
        Get feature correlation matrix.

        Args:
            method: 'pearson', 'spearman', or 'kendall'

        Returns:
            Correlation matrix DataFrame
        """
        # Select only numeric columns (exclude SubjectID, RecordingDate, etc.)
        numeric_df = self.features_df.select_dtypes(include=[np.number])
        return numeric_df.corr(method=method)

    def remove_correlated_features(
        self,
        threshold: float = 0.95,
        inplace: bool = False
    ) -> 'FeatureCollection':
        """
        Remove highly correlated features.

        Args:
            threshold: Correlation threshold
            inplace: Modify in place

        Returns:
            FeatureCollection with reduced features
        """
        corr_matrix = self.get_correlation_matrix()

        # Find pairs of highly correlated features
        to_drop = set()
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                if abs(corr_matrix.iloc[i, j]) > threshold:
                    # Drop the second feature
                    to_drop.add(corr_matrix.columns[j])

        if not to_drop:
            if inplace:
                return self
            else:
                return FeatureCollection(
                    self.features_df.copy(),
                    subject_ids=self.subject_ids.copy(),
                    window_indices=self.window_indices.copy(),
                    metadata=self.metadata.copy()
                )

        return self.drop_features(list(to_drop), inplace=inplace)

    def to_excel(
        self,
        filepath: str,
        include_subject_ids: bool = True
    ) -> None:
        """
        Export features to Excel.

        Args:
            filepath: Output file path
            include_subject_ids: Include subject ID column (ignored if SubjectID already exists)
        """
        df = self.features_df.copy()

        # Only insert SubjectID if requested AND it doesn't already exist
        # Handles case where one subject has multiple recordings, causing SubjectID to already be in df
        if include_subject_ids and self.subject_ids is not None and 'SubjectID' not in df.columns:
            df.insert(0, 'SubjectID', self.subject_ids)

        try:
            df.to_excel(filepath, index=False)
            print(f"Features exported to: {filepath}")
        except Exception as e:
            raise ValueError(f"Failed to export to Excel: {e}")

    def to_csv(
        self,
        filepath: str,
        include_subject_ids: bool = True
    ) -> None:
        """
        Export features to CSV.

        Args:
            filepath: Output file path
            include_subject_ids: Include subject ID column (ignored if SubjectID already exists)
        """
        df = self.features_df.copy()

        # Only insert SubjectID if requested AND it doesn't already exist
        # Handles case where one subject has multiple recordings, causing SubjectID to already be in df
        if include_subject_ids and self.subject_ids is not None and 'SubjectID' not in df.columns:
            df.insert(0, 'SubjectID', self.subject_ids)

        try:
            df.to_csv(filepath, index=False)
            print(f"Features exported to: {filepath}")
        except Exception as e:
            raise ValueError(f"Failed to export to CSV: {e}")

    def copy(self) -> 'FeatureCollection':
        """
        Create deep copy.

        Returns:
            New FeatureCollection instance
        """
        return FeatureCollection(
            self.features_df.copy(),
            subject_ids=self.subject_ids.copy(),
            window_indices=self.window_indices.copy(),
            metadata={k: v.copy() if isinstance(v, dict) else v
                     for k, v in self.metadata.items()}
        )

    def __len__(self) -> int:
        """Number of samples."""
        return len(self.features_df)

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"FeatureCollection(n_samples={self.n_samples}, "
            f"n_features={self.n_features})"
        )
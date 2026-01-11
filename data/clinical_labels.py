"""
Clinical Labels Management

This module manages clinical outcome labels for subjects.
Handles loading from Excel, label mapping, and querying.
"""

from typing import Optional, Dict, Any, List, Tuple, Union
import pandas as pd
import numpy as np
import warnings
from pathlib import Path


class ClinicalLabels:
    """
    Manages clinical outcome labels for respiratory recordings.

    Structure: One row per recording
    Columns: SubjectID (or recording identifier) + outcome columns

    Supported operations:
    - Load from Excel file
    - Get labels for specific subjects/outcomes
    - Remap label values (e.g., MCS → conscious)
    - Filter subjects by outcome availability
    - Export modified labels

    Attributes:
        labels_df (pd.DataFrame): DataFrame with labels
        subject_id_column (str): Name of subject ID column
        available_outcomes (List[str]): List of outcome column names
    """

    def __init__(
        self,
        labels_df: pd.DataFrame,
        subject_id_column: str = 'SubjectID'
    ):
        """
        Initialize clinical labels.

        Args:
            labels_df: DataFrame with subject IDs and outcome labels
            subject_id_column: Name of the column containing subject IDs
        """
        if subject_id_column not in labels_df.columns:
            raise ValueError(f"Subject ID column '{subject_id_column}' not found in DataFrame")

        self.labels_df = labels_df.copy()
        self.subject_id_column = subject_id_column

        # Set subject ID as index for faster lookup
        if self.labels_df.index.name != subject_id_column:
            self.labels_df = self.labels_df.set_index(subject_id_column)

    @classmethod
    def from_excel(
        cls,
        filepath: str,
        sheet_name: Union[str, int] = 0,
        subject_id_column: str = 'SubjectID'
    ) -> 'ClinicalLabels':
        """
        Load clinical labels from Excel file.

        Args:
            filepath: Path to Excel file
            sheet_name: Sheet name or index (default: first sheet)
            subject_id_column: Name of subject ID column

        Returns:
            ClinicalLabels instance
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")

        try:
            labels_df = pd.read_excel(filepath, sheet_name=sheet_name)
        except Exception as e:
            raise ValueError(f"Failed to load Excel file {filepath}: {e}")

        return cls(labels_df, subject_id_column)

    @classmethod
    def from_csv(
        cls,
        filepath: str,
        subject_id_column: str = 'SubjectID'
    ) -> 'ClinicalLabels':
        """
        Load clinical labels from CSV file.

        Args:
            filepath: Path to CSV file
            subject_id_column: Name of subject ID column

        Returns:
            ClinicalLabels instance
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")

        try:
            labels_df = pd.read_csv(filepath)
        except Exception as e:
            raise ValueError(f"Failed to load CSV file {filepath}: {e}")

        return cls(labels_df, subject_id_column)

    @property
    def available_outcomes(self) -> List[str]:
        """
        Get list of available outcome column names.

        Returns:
            List of outcome names (excluding subject ID column)
        """
        return self.labels_df.columns.tolist()

    @property
    def n_subjects(self) -> int:
        """
        Get number of subjects/recordings.

        Returns:
            Number of rows in labels DataFrame
        """
        return len(self.labels_df)

    def get_label(
        self,
        subject_id: str,
        outcome: str
    ) -> Any:
        """
        Get label for a specific subject and outcome.

        Args:
            subject_id: Subject identifier
            outcome: Outcome name (column name)

        Returns:
            Label value (can be numeric, string, or None if missing)

        Raises:
            KeyError: If subject or outcome not found
        """
        if subject_id not in self.labels_df.index:
            raise KeyError(f"Subject '{subject_id}' not found in labels")

        if outcome not in self.labels_df.columns:
            raise KeyError(f"Outcome '{outcome}' not found. Available: {self.available_outcomes}")

        value = self.labels_df.loc[subject_id, outcome]

        # Return None for NaN values
        if pd.isna(value):
            return None

        return value

    def get_labels_for_outcome(
        self,
        outcome: str,
        include_subject_ids: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Get all labels for a specific outcome.

        Args:
            outcome: Outcome name
            include_subject_ids: If True, return (subject_ids, labels)

        Returns:
            Array of labels, or tuple (subject_ids, labels) if include_subject_ids=True
        """
        if outcome not in self.labels_df.columns:
            raise KeyError(f"Outcome '{outcome}' not found. Available: {self.available_outcomes}")

        labels = self.labels_df[outcome].values

        if include_subject_ids:
            subject_ids = self.labels_df.index.values
            return subject_ids, labels

        return labels

    def get_subjects_with_outcome(
        self,
        outcome: str,
        exclude_missing: bool = True
    ) -> List[str]:
        """
        Get list of subjects that have a specific outcome label.

        Args:
            outcome: Outcome name
            exclude_missing: If True, exclude subjects with NaN/None labels

        Returns:
            List of subject IDs
        """
        if outcome not in self.labels_df.columns:
            raise KeyError(f"Outcome '{outcome}' not found. Available: {self.available_outcomes}")

        if exclude_missing:
            # Exclude NaN values
            mask = self.labels_df[outcome].notna()
            subjects = self.labels_df[mask].index.tolist()
        else:
            subjects = self.labels_df.index.tolist()

        return subjects

    def filter_by_subjects(
        self,
        subject_ids: List[str]
    ) -> 'ClinicalLabels':
        """
        Create new ClinicalLabels with only specified subjects.

        Args:
            subject_ids: List of subject IDs to keep

        Returns:
            New ClinicalLabels instance
        """
        # Find which subjects exist
        existing_subjects = [sid for sid in subject_ids if sid in self.labels_df.index]
        missing_subjects = [sid for sid in subject_ids if sid not in self.labels_df.index]

        if missing_subjects:
            warnings.warn(
                f"Subjects not found in labels: {missing_subjects}. "
                f"Keeping {len(existing_subjects)}/{len(subject_ids)} subjects."
            )

        filtered_df = self.labels_df.loc[existing_subjects].copy()
        filtered_df = filtered_df.reset_index()

        return ClinicalLabels(filtered_df, self.subject_id_column)

    def remap_labels(
        self,
        outcome: str,
        mapping: Dict[Any, Any],
        inplace: bool = False
    ) -> 'ClinicalLabels':
        """
        Remap label values for an outcome.

        Example: Remap MCS (value=2) to conscious (value=1)
            mapping = {2: 1}

        Args:
            outcome: Outcome to remap
            mapping: Dictionary mapping old values to new values
            inplace: If True, modify this instance; else return new instance

        Returns:
            ClinicalLabels instance (self if inplace=True, else new instance)
        """
        if outcome not in self.labels_df.columns:
            raise KeyError(f"Outcome '{outcome}' not found. Available: {self.available_outcomes}")

        if inplace:
            self.labels_df[outcome] = self.labels_df[outcome].replace(mapping)
            return self
        else:
            new_df = self.labels_df.copy()
            new_df[outcome] = new_df[outcome].replace(mapping)
            new_df = new_df.reset_index()
            return ClinicalLabels(new_df, self.subject_id_column)

    def exclude_labels(
        self,
        outcome: str,
        values_to_exclude: List[Any],
        inplace: bool = False
    ) -> 'ClinicalLabels':
        """
        Exclude certain label values (set to NaN).

        Args:
            outcome: Outcome column
            values_to_exclude: List of values to exclude
            inplace: If True, modify this instance

        Returns:
            ClinicalLabels instance
        """
        if outcome not in self.labels_df.columns:
            raise KeyError(f"Outcome '{outcome}' not found. Available: {self.available_outcomes}")

        if inplace:
            mask = self.labels_df[outcome].isin(values_to_exclude)
            self.labels_df.loc[mask, outcome] = np.nan
            return self
        else:
            new_df = self.labels_df.copy()
            mask = new_df[outcome].isin(values_to_exclude)
            new_df.loc[mask, outcome] = np.nan
            new_df = new_df.reset_index()
            return ClinicalLabels(new_df, self.subject_id_column)

    def add_outcome(
        self,
        outcome_name: str,
        labels: np.ndarray
    ) -> None:
        """
        Add a new outcome column.

        Args:
            outcome_name: Name of new outcome
            labels: Array of labels (must match number of subjects)
        """
        if len(labels) != len(self.labels_df):
            raise ValueError(
                f"Labels array length ({len(labels)}) must match number of subjects ({len(self.labels_df)})"
            )

        if outcome_name in self.labels_df.columns:
            warnings.warn(f"Outcome '{outcome_name}' already exists. Overwriting.")

        self.labels_df[outcome_name] = labels

    def get_label_distribution(
        self,
        outcome: str
    ) -> pd.Series:
        """
        Get distribution of labels for an outcome.

        Args:
            outcome: Outcome name

        Returns:
            Series with value counts
        """
        if outcome not in self.labels_df.columns:
            raise KeyError(f"Outcome '{outcome}' not found. Available: {self.available_outcomes}")

        return self.labels_df[outcome].value_counts(dropna=False)

    def merge_with_features(
        self,
        features_df: pd.DataFrame,
        outcome: str,
        on: str = 'SubjectID'
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Merge features with labels for an outcome.

        Useful for preparing data for classification.

        Args:
            features_df: DataFrame with features (must have subject ID column)
            outcome: Which outcome to merge
            on: Column name to merge on

        Returns:
            Tuple of (merged_features_df, labels_array)
        """
        if outcome not in self.labels_df.columns:
            raise KeyError(f"Outcome '{outcome}' not found. Available: {self.available_outcomes}")

        # Reset index to make subject ID a column
        labels_temp = self.labels_df.reset_index()

        # Select only subject ID and outcome columns
        labels_subset = labels_temp[[self.subject_id_column, outcome]]

        # DEBUG: Identify which subject IDs are missing labels
        feature_subject_ids = set(features_df[on].unique()) if on in features_df.columns else set()
        label_subject_ids = set(labels_subset[on].unique()) if on in labels_subset.columns else set()

        missing_in_labels = feature_subject_ids - label_subject_ids
        if missing_in_labels:
            print(f"\n⚠️  WARNING: {len(missing_in_labels)} subject ID(s) in features have NO labels for outcome '{outcome}':")
            for sid in sorted(missing_in_labels):
                count = len(features_df[features_df[on] == sid])
                print(f"    - {repr(sid)}: {count} recording(s)")
            print(f"    These {sum(len(features_df[features_df[on] == sid]) for sid in missing_in_labels)} recordings will be EXCLUDED from analysis!\n")

        # Merge
        merged = features_df.merge(labels_subset, on=on, how='inner')

        # Extract labels
        labels = merged[outcome].values

        # Remove outcome column from features
        features_merged = merged.drop(columns=[outcome])

        n_original = len(features_df)
        n_merged = len(merged)

        if n_merged < n_original:
            warnings.warn(
                f"Merge reduced dataset from {n_original} to {n_merged} samples. "
                f"Some features may not have corresponding labels."
            )

        return features_merged, labels

    def to_excel(
        self,
        filepath: str,
        sheet_name: str = 'Labels'
    ) -> None:
        """
        Export labels to Excel file.

        Args:
            filepath: Output file path
            sheet_name: Sheet name
        """
        output_df = self.labels_df.reset_index()

        try:
            output_df.to_excel(filepath, sheet_name=sheet_name, index=False)
            print(f"Labels exported to: {filepath}")
        except Exception as e:
            raise ValueError(f"Failed to export to Excel: {e}")

    def to_csv(self, filepath: str) -> None:
        """
        Export labels to CSV file.

        Args:
            filepath: Output file path
        """
        output_df = self.labels_df.reset_index()

        try:
            output_df.to_csv(filepath, index=False)
            print(f"Labels exported to: {filepath}")
        except Exception as e:
            raise ValueError(f"Failed to export to CSV: {e}")

    def summary(self) -> str:
        """
        Get summary string of labels.

        Returns:
            String describing available outcomes and their distributions
        """
        lines = []
        lines.append(f"Clinical Labels Summary")
        lines.append(f"=" * 50)
        lines.append(f"Number of subjects: {self.n_subjects}")
        lines.append(f"Number of outcomes: {len(self.available_outcomes)}")
        lines.append(f"\nAvailable outcomes:")

        for outcome in self.available_outcomes:
            lines.append(f"\n  {outcome}:")
            distribution = self.get_label_distribution(outcome)

            # Count missing values
            n_missing = self.labels_df[outcome].isna().sum()
            n_valid = len(self.labels_df) - n_missing

            lines.append(f"    Valid: {n_valid}, Missing: {n_missing}")
            lines.append(f"    Distribution:")

            for value, count in distribution.items():
                if pd.notna(value):
                    percentage = (count / len(self.labels_df)) * 100
                    lines.append(f"      {value}: {count} ({percentage:.1f}%)")

        return '\n'.join(lines)

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"ClinicalLabels(n_subjects={self.n_subjects}, "
            f"n_outcomes={len(self.available_outcomes)})"
        )

    def __len__(self) -> int:
        """Number of subjects."""
        return self.n_subjects
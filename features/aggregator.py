"""
Feature Aggregation

Aggregate window-level features to recording-level features.
Takes multiple feature dictionaries (one per window) and combines them
using statistical aggregation (mean, std, min, max, etc.).
"""

from typing import List, Dict, Optional, Callable
import numpy as np
import warnings


class FeatureAggregator:
    """
    Aggregate window-level features to recording-level features.

    Takes a list of feature dictionaries (one per window) and applies
    aggregation functions (mean, std, min, max, etc.) to create a single
    feature dictionary representing the entire recording.

    Usage:
        aggregator = FeatureAggregator(config)
        window_features = [extractor.extract(w.data, w.sampling_rate) for w in windows]
        recording_features = aggregator.aggregate(window_features)
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize feature aggregator.

        Args:
            config: Configuration dictionary
        """
        self.config = config or {}

        # Get aggregation configuration
        agg_config = self.config.get('features', {}).get('aggregation', {})

        # Which aggregation functions to apply
        self.aggregation_functions = agg_config.get(
            'functions',
            ['mean', 'std', 'min', 'max', 'median']
        )

        # Minimum number of windows required for aggregation
        self.min_windows = agg_config.get('min_windows', 1)

        # How to handle NaN values
        self.nan_policy = agg_config.get('nan_policy', 'omit')  # 'omit', 'propagate', or 'raise'

        # Minimum valid windows (non-NaN) required
        self.min_valid_windows = agg_config.get('min_valid_windows', 1)

        # Aggregation function mapping
        self._agg_func_map = {
            'mean': np.nanmean,
            'std': np.nanstd,
            'min': np.nanmin,
            'max': np.nanmax,
            'median': np.nanmedian,
            'sum': np.nansum,
            'q25': lambda x: np.nanpercentile(x, 25),
            'q75': lambda x: np.nanpercentile(x, 75),
            'iqr': lambda x: np.nanpercentile(x, 75) - np.nanpercentile(x, 25),
            'range': lambda x: np.nanmax(x) - np.nanmin(x),
            'cv': lambda x: np.nanstd(x) / np.nanmean(x) if np.nanmean(x) != 0 else np.nan,
        }

    def aggregate(
        self,
        window_features: List[Dict[str, float]],
        subject_id: Optional[str] = None,
        recording_date: Optional[str] = None
    ) -> Dict[str, float]:
        """
        Aggregate window-level features to recording level.

        Args:
            window_features: List of feature dictionaries (one per window)
            subject_id: Optional subject ID to include in output
            recording_date: Optional recording date to include in output

        Returns:
            Dictionary with aggregated features

        Raises:
            ValueError: If insufficient windows or invalid data
        """
        if len(window_features) < self.min_windows:
            raise ValueError(
                f"Insufficient windows for aggregation. "
                f"Got {len(window_features)}, need at least {self.min_windows}"
            )

        # Get all feature names from first window
        if len(window_features) == 0:
            return {}

        feature_names = list(window_features[0].keys())

        # Remove metadata columns if present
        metadata_cols = ['SubjectID', 'RecordingDate', 'WindowIndex', 'StartTime', 'EndTime', 'Duration']
        feature_names = [name for name in feature_names if name not in metadata_cols]

        # Initialize output
        aggregated = {}

        # Add metadata if provided
        if subject_id is not None:
            aggregated['SubjectID'] = subject_id
        if recording_date is not None:
            aggregated['RecordingDate'] = recording_date

        # Add number of windows
        aggregated['N_Windows'] = len(window_features)

        # Aggregate each feature
        for feature_name in feature_names:
            # Extract values for this feature across all windows
            values = []
            for window_feat in window_features:
                if feature_name in window_feat:
                    values.append(window_feat[feature_name])
                else:
                    values.append(np.nan)

            values = np.array(values)

            # Check how many valid (non-NaN) values
            n_valid = np.sum(~np.isnan(values))

            if n_valid < self.min_valid_windows:
                # Not enough valid windows
                for func_name in self.aggregation_functions:
                    agg_feature_name = f"{feature_name}_{func_name}"
                    aggregated[agg_feature_name] = np.nan
                continue

            # Apply aggregation functions
            for func_name in self.aggregation_functions:
                try:
                    agg_func = self._agg_func_map.get(func_name)
                    if agg_func is None:
                        warnings.warn(f"Unknown aggregation function: {func_name}")
                        continue

                    # Apply function
                    if self.nan_policy == 'propagate':
                        # If any NaN, result is NaN
                        if np.any(np.isnan(values)):
                            agg_value = np.nan
                        else:
                            agg_value = agg_func(values)
                    elif self.nan_policy == 'raise':
                        # Raise error if any NaN
                        if np.any(np.isnan(values)):
                            raise ValueError(f"NaN values found in {feature_name}")
                        agg_value = agg_func(values)
                    else:  # 'omit' (default)
                        # Ignore NaN values
                        agg_value = agg_func(values)

                    # Store aggregated feature
                    agg_feature_name = f"{feature_name}_{func_name}"
                    aggregated[agg_feature_name] = agg_value

                except Exception as e:
                    warnings.warn(f"Error aggregating {feature_name} with {func_name}: {e}")
                    agg_feature_name = f"{feature_name}_{func_name}"
                    aggregated[agg_feature_name] = np.nan

        # Add count of valid windows per feature (useful for quality assessment)
        aggregated['N_Valid_Windows'] = n_valid

        return aggregated

    def aggregate_by_recording(
        self,
        all_windows: List,  # List of SignalWindow objects
        feature_dicts: List[Dict[str, float]]
    ) -> Dict[str, Dict[str, float]]:
        """
        Aggregate windows grouped by recording.

        Useful when you have windows from multiple recordings and want to
        aggregate each recording separately.

        Args:
            all_windows: List of SignalWindow objects
            feature_dicts: List of feature dictionaries (same length as all_windows)

        Returns:
            Dictionary mapping recording_id -> aggregated_features
        """
        if len(all_windows) != len(feature_dicts):
            raise ValueError("Number of windows and feature dictionaries must match")

        # Group by recording
        recordings = {}
        for window, features in zip(all_windows, feature_dicts):
            recording_id = f"{window.subject_id}_{window.recording_date}"

            if recording_id not in recordings:
                recordings[recording_id] = {
                    'subject_id': window.subject_id,
                    'recording_date': window.recording_date,
                    'features': []
                }

            recordings[recording_id]['features'].append(features)

        # Aggregate each recording
        aggregated_by_recording = {}

        for recording_id, rec_data in recordings.items():
            try:
                aggregated = self.aggregate(
                    rec_data['features'],
                    subject_id=rec_data['subject_id'],
                    recording_date=rec_data['recording_date']
                )
                aggregated_by_recording[recording_id] = aggregated
            except Exception as e:
                warnings.warn(f"Failed to aggregate {recording_id}: {e}")
                aggregated_by_recording[recording_id] = None

        return aggregated_by_recording

    def add_custom_aggregation(self, name: str, func: Callable) -> None:
        """
        Add a custom aggregation function.

        Args:
            name: Name for the aggregation function
            func: Callable that takes an array and returns a scalar

        Example:
            aggregator.add_custom_aggregation('rms', lambda x: np.sqrt(np.nanmean(x**2)))
        """
        self._agg_func_map[name] = func

        if name not in self.aggregation_functions:
            self.aggregation_functions.append(name)

    def get_aggregated_feature_names(self, base_feature_names: List[str]) -> List[str]:
        """
        Get list of all aggregated feature names that will be created.

        Args:
            base_feature_names: List of base feature names (from windows)

        Returns:
            List of aggregated feature names
        """
        aggregated_names = ['N_Windows', 'N_Valid_Windows']

        for base_name in base_feature_names:
            for func_name in self.aggregation_functions:
                aggregated_names.append(f"{base_name}_{func_name}")

        return aggregated_names


class SimpleAggregator:
    """
    Simplified aggregator that just computes mean across windows.

    Useful for quick analysis or when you don't need multiple aggregation functions.
    """

    @staticmethod
    def aggregate_mean(window_features: List[Dict[str, float]]) -> Dict[str, float]:
        """
        Compute mean of each feature across windows.

        Args:
            window_features: List of feature dictionaries

        Returns:
            Dictionary with mean values
        """
        if len(window_features) == 0:
            return {}

        # Get feature names
        feature_names = list(window_features[0].keys())
        metadata_cols = ['SubjectID', 'RecordingDate', 'WindowIndex', 'StartTime', 'EndTime', 'Duration']
        feature_names = [name for name in feature_names if name not in metadata_cols]

        # Compute means
        aggregated = {}
        for feature_name in feature_names:
            values = [w[feature_name] for w in window_features if feature_name in w]
            values = np.array(values)
            aggregated[feature_name] = np.nanmean(values)

        aggregated['N_Windows'] = len(window_features)

        return aggregated

    @staticmethod
    def aggregate_median(window_features: List[Dict[str, float]]) -> Dict[str, float]:
        """
        Compute median of each feature across windows.

        Args:
            window_features: List of feature dictionaries

        Returns:
            Dictionary with median values
        """
        if len(window_features) == 0:
            return {}

        # Get feature names
        feature_names = list(window_features[0].keys())
        metadata_cols = ['SubjectID', 'RecordingDate', 'WindowIndex', 'StartTime', 'EndTime', 'Duration']
        feature_names = [name for name in feature_names if name not in metadata_cols]

        # Compute medians
        aggregated = {}
        for feature_name in feature_names:
            values = [w[feature_name] for w in window_features if feature_name in w]
            values = np.array(values)
            aggregated[feature_name] = np.nanmedian(values)

        aggregated['N_Windows'] = len(window_features)

        return aggregated


def aggregate_to_recording_level(
    windows: List,  # List of SignalWindow objects
    extractor,  # BreathingParameterExtractor instance
    aggregator: Optional[FeatureAggregator] = None
) -> Dict[str, float]:
    """
    Convenience function to extract and aggregate features in one call.

    Args:
        windows: List of SignalWindow objects from same recording
        extractor: BreathingParameterExtractor instance
        aggregator: FeatureAggregator instance (creates default if None)

    Returns:
        Dictionary with aggregated features

    Example:
        windows = generator.generate_windows(recording)
        extractor = BreathingParameterExtractor(config)
        aggregator = FeatureAggregator(config)
        recording_features = aggregate_to_recording_level(windows, extractor, aggregator)
    """
    if aggregator is None:
        aggregator = FeatureAggregator()

    # Extract features from each window
    window_features = []
    for window in windows:
        features = extractor.extract(window.data, window.sampling_rate)
        window_features.append(features)

    # Aggregate
    if len(windows) > 0:
        subject_id = windows[0].subject_id
        recording_date = windows[0].recording_date
    else:
        subject_id = None
        recording_date = None

    return aggregator.aggregate(window_features, subject_id, recording_date)
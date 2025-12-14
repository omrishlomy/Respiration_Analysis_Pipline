"""
Signal Cleaning and Quality Assessment

This module provides:
1. Data quality checking (gap detection, corruption, flat lines)
2. Signal cleaning (filtering, detrending, artifact removal)
3. Baseline correction
"""

from typing import Tuple, List, Dict, Optional
import numpy as np
from scipy import signal
from scipy.interpolate import interp1d
import warnings
from dataclasses import dataclass
import sys
import os

# =============================================================================
# CRITICAL IMPORT FIX: Ensure RespiratoryRecording is imported
# =============================================================================
try:
    # 1. Try Absolute Import (Standard)
    from data.recording import RespiratoryRecording
except ImportError:
    try:
        # 2. Try Relative Import (If running as module)
        from ..data.recording import RespiratoryRecording
    except ImportError:
        # 3. Fallback: Add project root to sys.path
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(current_dir))
        if project_root not in sys.path:
            sys.path.append(project_root)
        try:
            from respiratory_analysis.data.recording import RespiratoryRecording
        except ImportError:
             raise ImportError("❌ CRITICAL: Could not import RespiratoryRecording. Check project structure.")

# =============================================================================

@dataclass
class QualityIssue:
    """Container for a detected quality issue."""
    issue_type: str  # 'gap', 'corruption', 'flat', 'outlier'
    start_idx: int
    end_idx: int
    severity: str  # 'low', 'medium', 'high'
    description: str


@dataclass
class QualityReport:
    """Container for quality assessment results."""
    is_valid: bool
    total_samples: int
    valid_samples: int
    percent_valid: float
    issues: List[QualityIssue]
    recommendation: str  # 'accept', 'clean', 'reject'


class SignalCleaner:
    """
    Clean respiratory signals and assess data quality.
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize signal cleaner with configuration.
        """
        self.config = config or {}

        # Extract configuration
        cleaning_config = self.config.get('preprocessing', {}).get('cleaning', {})
        quality_config = self.config.get('data_quality', {})

        # Cleaning parameters
        self.remove_outliers = cleaning_config.get('remove_outliers', True)
        self.apply_filter = cleaning_config.get('apply_filter', False)
        self.filter_type = cleaning_config.get('filter_type', 'bandpass')
        self.lowcut = cleaning_config.get('lowcut', 0.05)
        self.highcut = cleaning_config.get('highcut', 0.5)
        self.filter_order = cleaning_config.get('filter_order', 4)
        self.detrend_method = cleaning_config.get('detrend', 'linear')

        # Quality checking parameters
        self.check_quality = quality_config.get('enabled', True)
        self.gap_threshold = quality_config.get('gap_threshold', 0.0)
        self.min_gap_duration = quality_config.get('min_gap_duration_sec', 1.0)
        self.flat_threshold = quality_config.get('flat_threshold', 0.001)
        self.min_flat_duration = quality_config.get('min_flat_duration_sec', 5.0)
        self.outlier_threshold_std = quality_config.get('outlier_threshold_std', 10.0)
        self.physiological_min = quality_config.get('physiological_min', -100.0)
        self.physiological_max = quality_config.get('physiological_max', 100.0)
        self.min_valid_percent = quality_config.get('min_valid_percent', 80.0)
        self.reject_threshold = quality_config.get('reject_threshold', 50.0)
        self.auto_clean = quality_config.get('auto_clean', True)

    def clean(self, recording):
        """
        Clean a respiratory recording.

        Args:
            recording: RespiratoryRecording instance

        Returns:
            New RespiratoryRecording with cleaned signal
        """
        # 1. Quality assessment
        if self.check_quality:
            quality_report = self.assess_quality(recording.data, recording.sampling_rate)

            if quality_report.recommendation == 'reject':
                raise ValueError(
                    f"Recording quality too poor: {quality_report.percent_valid:.1f}% valid. "
                    f"Detected {len(quality_report.issues)} issues."
                )

            # Auto-clean if needed
            if quality_report.recommendation == 'clean' and self.auto_clean:
                cleaned_signal = self._clean_quality_issues(
                    recording.data,
                    quality_report,
                    recording.sampling_rate
                )
            else:
                cleaned_signal = recording.data.copy()
        else:
            cleaned_signal = recording.data.copy()
            quality_report = None

        # 2. Remove outliers
        if self.remove_outliers:
            cleaned_signal = self._remove_outliers(cleaned_signal)

        # 3. Detrending
        if self.detrend_method:
            if np.any(np.isnan(cleaned_signal)):
                nan_mask = np.isnan(cleaned_signal)
                if not np.all(nan_mask):
                    x = np.arange(len(cleaned_signal))
                    x_valid = x[~nan_mask]
                    y_valid = cleaned_signal[~nan_mask]
                    if len(x_valid) > 1:
                        f = interp1d(x_valid, y_valid, kind='linear',
                                     fill_value='extrapolate', bounds_error=False)
                        cleaned_signal[nan_mask] = f(x[nan_mask])

            cleaned_signal = self._detrend(cleaned_signal, self.detrend_method)

        # 4. Filtering
        if self.apply_filter:
            cleaned_signal = self._filter_signal(
                cleaned_signal,
                recording.sampling_rate,
                self.filter_type,
                self.lowcut,
                self.highcut,
                self.filter_order
            )

        # Create new recording with cleaned data
        clean_recording = RespiratoryRecording(
            data=cleaned_signal,
            sampling_rate=recording.sampling_rate,
            subject_id=recording.subject_id,
            recording_date=recording.recording_date,
            metadata=recording.metadata.copy()
        )

        # Add cleaning metadata
        clean_recording.add_metadata('cleaned', True)
        clean_recording.add_metadata('cleaning_steps', {
            'outlier_removal': self.remove_outliers,
            'filtering': self.apply_filter,
            'detrending': self.detrend_method is not None
        })

        if quality_report:
            clean_recording.add_metadata('quality_report', {
                'percent_valid': quality_report.percent_valid,
                'recommendation': quality_report.recommendation,
                'n_issues': len(quality_report.issues)
            })

        return clean_recording

    def assess_quality(self, signal_data: np.ndarray, sampling_rate: float) -> QualityReport:
        """
        Assess signal quality and detect issues.
        """
        issues = []

        # 1. Detect gaps
        gap_issues = self._detect_gaps(signal_data, sampling_rate)
        issues.extend(gap_issues)

        # 2. Detect flat lines
        flat_issues = self._detect_flat_lines(signal_data, sampling_rate)
        issues.extend(flat_issues)

        # 3. Detect outliers
        outlier_issues = self._detect_outliers(signal_data)
        issues.extend(outlier_issues)

        # 4. Detect non-physiological values
        physio_issues = self._detect_non_physiological(signal_data)
        issues.extend(physio_issues)

        # Calculate valid data percentage
        total_samples = len(signal_data)
        invalid_mask = self._create_invalid_mask(signal_data, issues)
        valid_samples = np.sum(~invalid_mask)
        percent_valid = (valid_samples / total_samples) * 100.0 if total_samples > 0 else 0.0

        # Determine recommendation
        if percent_valid < self.reject_threshold:
            recommendation = 'reject'
            is_valid = False
        elif percent_valid < self.min_valid_percent:
            recommendation = 'clean'
            is_valid = True
        else:
            recommendation = 'accept'
            is_valid = True

        return QualityReport(
            is_valid=is_valid,
            total_samples=total_samples,
            valid_samples=valid_samples,
            percent_valid=percent_valid,
            issues=issues,
            recommendation=recommendation
        )

    def _detect_gaps(self, signal: np.ndarray, sampling_rate: float) -> List[QualityIssue]:
        issues = []
        nan_mask = np.isnan(signal)
        if np.any(nan_mask):
            gap_regions = self._find_continuous_regions(nan_mask)
            for start, end in gap_regions:
                duration = (end - start) / sampling_rate
                if duration >= self.min_gap_duration:
                    issues.append(QualityIssue('gap', start, end, 'high', f'NaN gap of {duration:.1f}s'))
        if abs(self.gap_threshold) < 1e-10:
            zero_mask = np.abs(signal) < 1e-10
            gap_regions = self._find_continuous_regions(zero_mask)
            for start, end in gap_regions:
                duration = (end - start) / sampling_rate
                if duration >= self.min_gap_duration:
                    issues.append(QualityIssue('gap', start, end, 'high', f'Zero gap of {duration:.1f}s'))
        return issues

    def _detect_flat_lines(self, signal: np.ndarray, sampling_rate: float) -> List[QualityIssue]:
        issues = []
        window_size = int(self.min_flat_duration * sampling_rate)
        if window_size < 2: return issues
        i = 0
        while i < len(signal) - window_size:
            window = signal[i:i+window_size]
            if np.std(window) < self.flat_threshold:
                start = i
                end = i + window_size
                while end < len(signal) and np.std(signal[start:end+1]) < self.flat_threshold: end += 1
                duration = (end - start) / sampling_rate
                issues.append(QualityIssue('flat', start, end, 'medium', f'Flat line {duration:.1f}s'))
                i = end
            else: i += 1
        return issues

    def _detect_outliers(self, signal: np.ndarray) -> List[QualityIssue]:
        issues = []
        valid_signal = signal[~np.isnan(signal)]
        if len(valid_signal) == 0: return issues
        median = np.median(valid_signal)
        mad = np.median(np.abs(valid_signal - median))
        if mad < 1e-10:
            mean = np.mean(valid_signal)
            std = np.std(valid_signal)
            threshold = self.outlier_threshold_std * std
            outlier_mask = np.abs(signal - mean) > threshold
        else:
            threshold = self.outlier_threshold_std * mad
            outlier_mask = np.abs(signal - median) > threshold
        outlier_regions = self._find_continuous_regions(outlier_mask)
        for start, end in outlier_regions:
            issues.append(QualityIssue('outlier', start, end, 'medium', f'Outlier region: {end-start} samples'))
        return issues

    def _detect_non_physiological(self, signal: np.ndarray) -> List[QualityIssue]:
        issues = []
        out_of_range_mask = (signal < self.physiological_min) | (signal > self.physiological_max)
        if np.any(out_of_range_mask):
            bad_regions = self._find_continuous_regions(out_of_range_mask)
            for start, end in bad_regions:
                issues.append(QualityIssue('corruption', start, end, 'high', f'Non-physio: {end-start} samples'))
        return issues

    def _find_continuous_regions(self, mask: np.ndarray) -> List[Tuple[int, int]]:
        if not np.any(mask): return []
        padded = np.concatenate([[False], mask, [False]])
        diff = np.diff(padded.astype(int))
        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0]
        return list(zip(starts, ends))

    def _create_invalid_mask(self, signal: np.ndarray, issues: List[QualityIssue]) -> np.ndarray:
        mask = np.zeros(len(signal), dtype=bool)
        for issue in issues: mask[issue.start_idx:issue.end_idx] = True
        return mask

    def _clean_quality_issues(self, signal: np.ndarray, quality_report: QualityReport, sampling_rate: float) -> np.ndarray:
        cleaned = signal.copy()
        invalid_mask = self._create_invalid_mask(signal, quality_report.issues)
        invalid_regions = self._find_continuous_regions(invalid_mask)
        max_interpolate_samples = int(10 * sampling_rate)
        for start, end in invalid_regions:
            gap_size = end - start
            if gap_size <= max_interpolate_samples:
                if start > 0 and end < len(signal):
                    cleaned[start:end] = np.linspace(cleaned[start-1], cleaned[end], gap_size + 2)[1:-1]
            else: cleaned[start:end] = np.nan
        return cleaned

    def _remove_outliers(self, signal: np.ndarray, threshold_std: float = 5.0) -> np.ndarray:
        cleaned = signal.copy()
        mean = np.nanmean(cleaned)
        std = np.nanstd(cleaned)
        outlier_mask = np.abs(cleaned - mean) > threshold_std * std
        if np.sum(outlier_mask) > 0:
            x = np.arange(len(cleaned))
            x_clean = x[~outlier_mask]
            y_clean = cleaned[~outlier_mask]
            if len(x_clean) > 1:
                f = interp1d(x_clean, y_clean, kind='linear', fill_value='extrapolate', bounds_error=False)
                cleaned[outlier_mask] = f(x[outlier_mask])
        return cleaned

    def _detrend(self, signal_data: np.ndarray, method: str = 'linear') -> np.ndarray:
        if method == 'linear': return signal.detrend(signal_data, type='linear')
        elif method == 'constant': return signal.detrend(signal_data, type='constant')
        return signal_data

    def _filter_signal(self, signal_data: np.ndarray, sampling_rate: float, filter_type: str, lowcut: float, highcut: float, order: int) -> np.ndarray:
        nyquist = sampling_rate / 2.0
        try:
            if filter_type == 'bandpass':
                if highcut >= nyquist: return signal_data
                b, a = signal.butter(order, [lowcut/nyquist, highcut/nyquist], btype='band')
            elif filter_type == 'lowpass':
                if highcut >= nyquist: return signal_data
                b, a = signal.butter(order, highcut/nyquist, btype='low')
            elif filter_type == 'highpass':
                if lowcut >= nyquist: return signal_data
                b, a = signal.butter(order, lowcut/nyquist, btype='high')
            else: return signal_data
            return signal.filtfilt(b, a, signal_data)
        except Exception as e:
            warnings.warn(f"Filtering failed: {e}. Returning original signal.")
            return signal_data
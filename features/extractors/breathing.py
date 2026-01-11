"""
Breathing Parameter Extraction (Enhanced with Multi-Peak Detection)

Python port of MATLAB breathing parameter algorithms:
- peaks_from_ts.m -> Peak detection (inhales/exhales)
- nested_FindPeaksProperties.m -> Connected components for breath boundaries
- calculate_zz.m -> 25 breathing metrics calculation

ENHANCED: Now properly detects bi-phasic and multi-phasic breaths
(e.g., sniffing patterns, interrupted breathing).

This is the core feature extraction module that extracts comprehensive
breathing parameters from respiratory signals.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
from scipy.signal import find_peaks
from scipy.ndimage import label
from scipy.interpolate import interp1d
from dataclasses import dataclass, field
import bisect  # For optimized pause calculation


@dataclass
class SubPeak:
    """Properties of a single sub-peak within a breath."""
    index: int                    # Index in original signal
    value: float                  # Amplitude at peak
    time: float                   # Time in seconds
    relative_time: float          # Time relative to breath start
    relative_amplitude: float     # Amplitude relative to main peak (0-1)
    is_main_peak: bool           # Whether this is the highest peak


@dataclass
class BreathPeak:
    """
    Structure to hold properties of a single breath peak.

    Enhanced to support multiple sub-peaks within a breath.
    """
    PeakLocation: int       # Index of MAIN peak in signal
    PeakValue: float        # Amplitude at MAIN peak
    Volume: float           # Area under curve (integral)
    Duration: float         # Duration of breath in seconds
    StartTime: float        # Start time in seconds
    Latency: float          # Time from start to MAIN peak in seconds
    NumberOfPeaks: int      # Number of peaks within this breath

    # NEW: Multi-peak support
    SubPeaks: List[SubPeak] = field(default_factory=list)
    InterPeakIntervals: List[float] = field(default_factory=list)
    PeakAmplitudeRatios: List[float] = field(default_factory=list)
    IsMultiPhasic: bool = False
    StartIndex: int = 0
    EndIndex: int = 0


class BreathingParameterExtractor:
    """
    Extract comprehensive breathing parameters from respiratory signals.

    Python port of MATLAB breathing parameter extraction:
    1. Peak detection (peaks_from_ts.m)
    2. Connected components analysis (nested_FindPeaksProperties.m)
    3. Metrics calculation (calculate_zz.m) - 25+ parameters

    ENHANCED: Now includes multi-peak detection for bi-phasic breaths.

    Usage:
        extractor = BreathingParameterExtractor(config)
        features = extractor.extract(window.data, window.sampling_rate)
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize extractor with configuration.

        Args:
            config: Configuration dictionary
        """
        self.config = config or {}

        # Peak detection parameters (from config or defaults)
        peak_config = self.config.get('peak_detection', {})
        self.min_peak_distance_sec = peak_config.get('min_peak_distance_seconds', 2.0)
        self.min_peak_width_sec = peak_config.get('min_peak_width_seconds', 0.4)
        self.min_peak_prominence = peak_config.get('min_peak_prominence', 0.01)
        self.percentile_peak_value = peak_config.get('percentile_peak_value', 90)
        self.peak_value_multiplier = peak_config.get('peak_value_multiplier', 0.01)
        self.min_duration = peak_config.get('min_duration', 0.25)
        self.volume_threshold = peak_config.get('volume_threshold', 0.0)

        # Metrics parameters
        metrics_config = self.config.get('breath_metrics', {})
        self.outlier_threshold_std = metrics_config.get('outlier_threshold_std', 3.0)
        self.min_pause_duration = metrics_config.get('min_pause_duration', 0.05)

        # Multi-peak detection parameters (NEW)
        multipeak_config = self.config.get('multi_peak', {})
        self.upsampling_factor = multipeak_config.get('upsampling_factor', 4)
        self.sub_peak_min_prominence = multipeak_config.get('sub_peak_min_prominence', 0.1)
        self.sub_peak_min_amplitude_ratio = multipeak_config.get('sub_peak_min_amplitude_ratio', 0.2)
        self.sub_peak_min_distance_sec = multipeak_config.get('sub_peak_min_distance_sec', 0.3)

    def extract(self, signal: np.ndarray, sampling_rate: float) -> Dict[str, float]:
        """
        Extract breathing parameters from signal.

        This is the main entry point.

        Args:
            signal: Respiratory signal (1D array)
            sampling_rate: Sampling rate in Hz

        Returns:
            Dictionary with 25+ breathing metrics (including multi-peak metrics)
        """
        # 1. Detect peaks with multi-peak support
        peaks = self._detect_peaks(signal, sampling_rate)

        # 2. Calculate metrics (original 25 + new multi-peak metrics)
        metrics = self._calculate_metrics(peaks)

        return metrics

    def extract_with_details(
        self,
        signal: np.ndarray,
        sampling_rate: float
    ) -> Tuple[Dict[str, float], List[BreathPeak]]:
        """
        Extract breathing parameters AND return detailed breath information.

        Useful for visualization or detailed analysis.

        Args:
            signal: Respiratory signal (1D array)
            sampling_rate: Sampling rate in Hz

        Returns:
            Tuple of (metrics dict, list of BreathPeak objects)
        """
        peaks = self._detect_peaks(signal, sampling_rate)
        metrics = self._calculate_metrics(peaks)
        return metrics, peaks

    def _detect_peaks(self, signal: np.ndarray, sampling_rate: float) -> List[BreathPeak]:
        """
        Detect breathing peaks (inhales and exhales) with multi-peak support.

        Enhanced port of MATLAB peaks_from_ts.m + nested_FindPeaksProperties.m

        Args:
            signal: Respiratory signal
            sampling_rate: Sampling rate in Hz

        Returns:
            List of BreathPeak objects
        """
        if len(signal) == 0:
            return []

        signal = np.asarray(signal).flatten()
        sample_length = 1.0 / sampling_rate

        # Step 1: Find candidate peaks
        inhale_indices, exhale_indices = self._find_candidate_peaks(signal, sampling_rate)

        if len(inhale_indices) == 0 and len(exhale_indices) == 0:
            return []

        # Step 2: Apply amplitude threshold
        inhale_indices, exhale_indices = self._apply_amplitude_threshold(
            signal, inhale_indices, exhale_indices
        )

        all_peak_indices = np.concatenate([inhale_indices, exhale_indices])

        if len(all_peak_indices) == 0:
            return []

        # Step 3: Find breath properties using connected components
        peaks = self._find_breath_properties_connected_components(
            signal, sampling_rate, all_peak_indices
        )

        # Step 4: Filter by duration and volume
        peaks = self._filter_peaks(peaks)

        return peaks

    def _find_candidate_peaks(
        self,
        signal: np.ndarray,
        sampling_rate: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Find candidate inhale and exhale peaks."""
        min_distance = int(self.min_peak_distance_sec * sampling_rate - 1)
        min_width = self.min_peak_width_sec * sampling_rate

        # Find inhale peaks (positive)
        inhale_indices, _ = find_peaks(
            signal,
            distance=max(1, min_distance),
            width=min_width,
            prominence=self.min_peak_prominence
        )

        # Find exhale peaks (negative)
        exhale_indices, _ = find_peaks(
            -signal,
            distance=max(1, min_distance),
            width=min_width,
            prominence=self.min_peak_prominence
        )

        return inhale_indices, exhale_indices

    def _apply_amplitude_threshold(
        self,
        signal: np.ndarray,
        inhale_indices: np.ndarray,
        exhale_indices: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply amplitude-based threshold."""
        if len(inhale_indices) == 0:
            return inhale_indices, exhale_indices

        inhale_values = signal[inhale_indices]

        # Calculate threshold
        high_percentile = np.percentile(inhale_values, self.percentile_peak_value)
        peaks_above = inhale_values[inhale_values >= high_percentile]

        if len(peaks_above) > 0:
            threshold = np.median(peaks_above)
        else:
            threshold = np.median(inhale_values)

        amplitude_thresh = self.peak_value_multiplier * threshold

        # Apply threshold
        inhale_mask = inhale_values > amplitude_thresh
        inhale_indices = inhale_indices[inhale_mask]

        if len(exhale_indices) > 0:
            exhale_values = signal[exhale_indices]
            exhale_mask = exhale_values < -amplitude_thresh
            exhale_indices = exhale_indices[exhale_mask]

        return inhale_indices, exhale_indices

    def _find_breath_properties_connected_components(
        self,
        signal: np.ndarray,
        sampling_rate: float,
        peak_indices: np.ndarray
    ) -> List[BreathPeak]:
        """
        Find breath properties using connected components analysis.

        Port of nested_FindPeaksProperties.m - this is key for multi-peak detection.
        """
        sample_length = 1.0 / sampling_rate

        # Upsample signal for better precision
        upsampled_length = len(signal) * self.upsampling_factor
        upsampled_signal = self._upsample_signal(signal, upsampled_length)

        # Create binary masks
        positive_mask = upsampled_signal > self.volume_threshold
        negative_mask = upsampled_signal < -self.volume_threshold

        # Find connected components
        positive_labels, n_positive = label(positive_mask)
        negative_labels, n_negative = label(negative_mask)

        # Combine labels
        combined_labels = positive_labels.copy()
        combined_labels[negative_mask] = negative_labels[negative_mask] + n_positive

        # Map peaks to components
        peak_to_component = {}
        for peak_idx in np.sort(peak_indices):
            upsampled_idx = min(int(peak_idx * self.upsampling_factor), len(combined_labels) - 1)
            component_id = combined_labels[upsampled_idx]
            if component_id > 0:
                if component_id not in peak_to_component:
                    peak_to_component[component_id] = []
                peak_to_component[component_id].append(peak_idx)

        # Process each breath
        peaks = []
        for component_id, component_peaks in peak_to_component.items():
            breath = self._process_breath_component(
                signal, upsampled_signal, combined_labels,
                component_id, component_peaks, sample_length, sampling_rate
            )
            if breath is not None:
                peaks.append(breath)

        peaks.sort(key=lambda b: b.StartTime)
        return peaks

    def _upsample_signal(self, signal: np.ndarray, target_length: int) -> np.ndarray:
        """Upsample signal using interpolation."""
        if len(signal) < 2:
            return signal

        x_original = np.arange(len(signal))
        x_upsampled = np.linspace(0, len(signal) - 1, target_length)

        interpolator = interp1d(x_original, signal, kind='linear', fill_value='extrapolate')
        return interpolator(x_upsampled)

    def _process_breath_component(
        self,
        signal: np.ndarray,
        upsampled_signal: np.ndarray,
        labels: np.ndarray,
        component_id: int,
        peak_indices: List[int],
        sample_length: float,
        sampling_rate: float
    ) -> Optional[BreathPeak]:
        """Process a single breath with all its sub-peaks."""
        if len(peak_indices) == 0:
            return None

        # Find component boundaries
        component_mask = labels == component_id
        component_indices = np.where(component_mask)[0]

        if len(component_indices) == 0:
            return None

        start_idx_upsampled = component_indices[0]
        end_idx_upsampled = component_indices[-1]

        start_idx = int(start_idx_upsampled / self.upsampling_factor)
        end_idx = min(int(end_idx_upsampled / self.upsampling_factor), len(signal) - 1)

        # Find main peak
        peak_values = signal[peak_indices]
        main_peak_local_idx = np.argmax(np.abs(peak_values))
        main_peak_idx = peak_indices[main_peak_local_idx]
        main_peak_value = signal[main_peak_idx]

        # Calculate breath properties
        breath_signal = signal[start_idx:end_idx + 1]

        # Use trapezoid (numpy >= 2.0) or trapz (older versions)
        try:
            volume = np.trapezoid(breath_signal) * sample_length
        except AttributeError:
            volume = np.trapz(breath_signal) * sample_length

        duration = (end_idx - start_idx) * sample_length
        start_time = start_idx * sample_length
        latency = (main_peak_idx - start_idx) * sample_length

        # Detect sub-peaks
        sub_peaks = self._detect_sub_peaks(
            signal, start_idx, end_idx, peak_indices,
            main_peak_idx, main_peak_value, sample_length
        )

        # Calculate multi-peak metrics
        inter_peak_intervals = []
        amplitude_ratios = []

        if len(sub_peaks) > 1:
            sorted_peaks = sorted(sub_peaks, key=lambda p: p.time)
            for i in range(1, len(sorted_peaks)):
                interval = sorted_peaks[i].time - sorted_peaks[i-1].time
                inter_peak_intervals.append(interval)

            for sp in sub_peaks:
                if not sp.is_main_peak:
                    amplitude_ratios.append(sp.relative_amplitude)

        return BreathPeak(
            PeakLocation=main_peak_idx,
            PeakValue=main_peak_value,
            Volume=volume,
            Duration=duration,
            StartTime=start_time,
            Latency=latency,
            NumberOfPeaks=len(sub_peaks),
            SubPeaks=sub_peaks,
            InterPeakIntervals=inter_peak_intervals,
            PeakAmplitudeRatios=amplitude_ratios,
            IsMultiPhasic=len(sub_peaks) > 1,
            StartIndex=start_idx,
            EndIndex=end_idx
        )

    def _detect_sub_peaks(
        self,
        signal: np.ndarray,
        start_idx: int,
        end_idx: int,
        known_peaks: List[int],
        main_peak_idx: int,
        main_peak_value: float,
        sample_length: float
    ) -> List[SubPeak]:
        """Detect all sub-peaks within a breath segment."""
        sub_peaks = []
        breath_signal = signal[start_idx:end_idx + 1]

        if len(breath_signal) < 3:
            return sub_peaks

        is_inhale = main_peak_value > 0

        # Find local peaks
        if is_inhale:
            local_peaks, _ = find_peaks(
                breath_signal,
                prominence=self.sub_peak_min_prominence * abs(main_peak_value),
                distance=max(1, int(self.sub_peak_min_distance_sec / sample_length))
            )
        else:
            local_peaks, _ = find_peaks(
                -breath_signal,
                prominence=self.sub_peak_min_prominence * abs(main_peak_value),
                distance=max(1, int(self.sub_peak_min_distance_sec / sample_length))
            )

        # Convert to global indices
        global_peaks = set((local_peaks + start_idx).tolist())

        # Include known peaks
        for kp in known_peaks:
            if start_idx <= kp <= end_idx:
                global_peaks.add(kp)

        # Create SubPeak objects
        for peak_idx in sorted(global_peaks):
            peak_value = signal[peak_idx]
            peak_time = peak_idx * sample_length
            relative_time = (peak_idx - start_idx) * sample_length

            if main_peak_value != 0:
                relative_amplitude = abs(peak_value) / abs(main_peak_value)
            else:
                relative_amplitude = 0.0

            if relative_amplitude >= self.sub_peak_min_amplitude_ratio or peak_idx == main_peak_idx:
                sub_peaks.append(SubPeak(
                    index=peak_idx,
                    value=peak_value,
                    time=peak_time,
                    relative_time=relative_time,
                    relative_amplitude=relative_amplitude,
                    is_main_peak=(peak_idx == main_peak_idx)
                ))

        return sub_peaks

    def _filter_peaks(self, peaks: List[BreathPeak]) -> List[BreathPeak]:
        """Filter peaks by duration and volume thresholds."""
        if len(peaks) == 0:
            return []

        peaks = [p for p in peaks if p.Duration >= self.min_duration]

        if len(peaks) == 0:
            return []

        volumes = np.array([abs(p.Volume) for p in peaks])
        median_volume = np.median(volumes)

        peaks = [p for p in peaks if abs(p.Volume) > self.volume_threshold * median_volume]

        return peaks

    def _calculate_metrics(self, peaks: List[BreathPeak]) -> Dict[str, float]:
        """
        Calculate 25+ breathing metrics from detected peaks.

        Port of MATLAB calculate_zz.m + NEW multi-peak metrics.

        Returns:
            Dictionary with all metrics
        """
        if len(peaks) == 0:
            return self._empty_metrics()

        # Separate inhales and exhales
        inhales = [p for p in peaks if p.PeakValue > 0]
        exhales = [p for p in peaks if p.PeakValue < 0]

        if len(inhales) == 0 or len(exhales) == 0:
            return self._empty_metrics()

        # Clean outliers
        clean_inhale = self._clean_outliers([p.__dict__ for p in inhales])
        clean_exhale = self._clean_outliers_exhale([p.__dict__ for p in exhales])

        metrics = {}

        # === ORIGINAL 25 METRICS ===

        # Basic volume and duration metrics
        metrics['Inhale_Volume'] = self._safe_mean(clean_inhale.get('Volume', []))
        metrics['Exhale_Volume'] = self._safe_mean(clean_exhale.get('Volume', []))
        metrics['Inhale_Duration'] = self._safe_mean(clean_inhale.get('Duration', []))
        metrics['Exhale_Duration'] = self._safe_mean(clean_exhale.get('Duration', []))
        metrics['Inhale_value'] = self._safe_mean(clean_inhale.get('PeakValue', []))
        metrics['Exhale_value'] = self._safe_mean(clean_exhale.get('PeakValue', []))

        # Inter-breath interval and rate
        inter_intervals = self._calculate_inter_breath_intervals(inhales)
        inter_clean = self._remove_outliers_1d(inter_intervals, self.outlier_threshold_std)
        metrics['Inter_breath_interval'] = self._safe_mean(inter_clean)

        if metrics['Inter_breath_interval'] > 0:
            metrics['Rate'] = 1.0 / metrics['Inter_breath_interval']
        else:
            metrics['Rate'] = np.nan

        # Tidal volume and minute ventilation
        metrics['Tidal_volume'] = metrics['Inhale_Volume'] + metrics['Exhale_Volume']
        metrics['Minute_Ventilation'] = metrics['Rate'] * metrics['Tidal_volume']

        # Duty cycles
        if metrics['Inter_breath_interval'] > 0:
            inhale_durations = clean_inhale.get('Duration', [])
            exhale_durations = clean_exhale.get('Duration', [])

            if len(inhale_durations) > 0:
                duty_cycles_inhale = np.array(inhale_durations) / metrics['Inter_breath_interval']
                metrics['Duty_Cycle_inhale'] = self._safe_mean(duty_cycles_inhale)
            else:
                metrics['Duty_Cycle_inhale'] = np.nan

            if len(exhale_durations) > 0:
                duty_cycles_exhale = np.array(exhale_durations) / metrics['Inter_breath_interval']
                metrics['Duty_Cycle_exhale'] = self._safe_mean(duty_cycles_exhale)
            else:
                metrics['Duty_Cycle_exhale'] = np.nan
        else:
            metrics['Duty_Cycle_inhale'] = np.nan
            metrics['Duty_Cycle_exhale'] = np.nan

        # Coefficients of variation (COV)
        metrics['COV_InhaleDutyCycle'] = self._calculate_cov(clean_inhale.get('Duration', []))
        metrics['COV_ExhaleDutyCycle'] = self._calculate_cov(clean_exhale.get('Duration', []))

        if len(inter_clean) > 0:
            inter_mean = np.mean(inter_clean)
            if inter_mean > 0:
                normalized_inter = inter_clean / inter_mean
                metrics['COV_BreathingRate'] = np.nanstd(normalized_inter)
            else:
                metrics['COV_BreathingRate'] = np.nan
        else:
            metrics['COV_BreathingRate'] = np.nan

        metrics['COV_InhaleVolume'] = self._calculate_cov(clean_inhale.get('Volume', []))
        metrics['COV_ExhaleVolume'] = self._calculate_cov(clean_exhale.get('Volume', []))

        # Pause metrics
        pause_metrics = self._calculate_pause_metrics(peaks, metrics['Inter_breath_interval'])
        metrics.update(pause_metrics)

        # === NEW MULTI-PEAK METRICS ===

        # Count of multi-phasic breaths
        multi_phasic = [p for p in peaks if p.IsMultiPhasic]
        metrics['MultiPhasic_Count'] = len(multi_phasic)
        metrics['MultiPhasic_Ratio'] = len(multi_phasic) / len(peaks) if len(peaks) > 0 else 0

        # Multi-phasic by type
        multi_phasic_inhales = [p for p in inhales if p.IsMultiPhasic]
        multi_phasic_exhales = [p for p in exhales if p.IsMultiPhasic]
        metrics['MultiPhasic_Inhale_Count'] = len(multi_phasic_inhales)
        metrics['MultiPhasic_Exhale_Count'] = len(multi_phasic_exhales)
        metrics['MultiPhasic_Inhale_Ratio'] = len(multi_phasic_inhales) / len(inhales) if len(inhales) > 0 else 0
        metrics['MultiPhasic_Exhale_Ratio'] = len(multi_phasic_exhales) / len(exhales) if len(exhales) > 0 else 0

        # Peaks per breath
        peaks_per_breath = [p.NumberOfPeaks for p in peaks]
        metrics['PeaksPerBreath_Mean'] = np.mean(peaks_per_breath)
        metrics['PeaksPerBreath_Max'] = max(peaks_per_breath)

        # Inter-peak intervals (within multi-phasic breaths)
        all_intervals = []
        for p in multi_phasic:
            all_intervals.extend(p.InterPeakIntervals)

        if len(all_intervals) > 0:
            metrics['InterPeakInterval_Mean'] = np.mean(all_intervals)
            metrics['InterPeakInterval_Std'] = np.std(all_intervals)
        else:
            metrics['InterPeakInterval_Mean'] = 0
            metrics['InterPeakInterval_Std'] = 0

        # Secondary peak amplitude ratios
        all_ratios = []
        for p in multi_phasic:
            all_ratios.extend(p.PeakAmplitudeRatios)

        if len(all_ratios) > 0:
            metrics['SecondaryPeakRatio_Mean'] = np.mean(all_ratios)
            metrics['SecondaryPeakRatio_Std'] = np.std(all_ratios)
        else:
            metrics['SecondaryPeakRatio_Mean'] = 0
            metrics['SecondaryPeakRatio_Std'] = 0

        return metrics

    def _clean_outliers(self, peaks_dicts: List[Dict]) -> Dict[str, np.ndarray]:
        """Remove outliers from peak properties (for inhales)."""
        if len(peaks_dicts) == 0:
            return {}

        properties = {
            'PeakValue': np.array([p['PeakValue'] for p in peaks_dicts]),
            'Volume': np.array([p['Volume'] for p in peaks_dicts]),
            'Duration': np.array([p['Duration'] for p in peaks_dicts]),
        }

        cleaned = {}
        for prop_name, values in properties.items():
            cleaned[prop_name] = self._remove_outliers_1d(values, self.outlier_threshold_std)

        return cleaned

    def _clean_outliers_exhale(self, peaks_dicts: List[Dict]) -> Dict[str, np.ndarray]:
        """Remove outliers from peak properties (for exhales)."""
        if len(peaks_dicts) == 0:
            return {}

        properties = {
            'PeakValue': np.abs([p['PeakValue'] for p in peaks_dicts]),
            'Volume': np.abs([p['Volume'] for p in peaks_dicts]),
            'Duration': np.array([p['Duration'] for p in peaks_dicts]),
        }

        cleaned = {}
        for prop_name, values in properties.items():
            cleaned[prop_name] = self._remove_outliers_1d(values, self.outlier_threshold_std)

        return cleaned

    def _remove_outliers_1d(self, data: np.ndarray, threshold_std: float) -> np.ndarray:
        """Remove outliers using standard deviation threshold."""
        if len(data) == 0:
            return data

        data = np.asarray(data)
        mean_val = np.nanmean(data)
        std_val = np.nanstd(data)

        if std_val == 0:
            return data

        mask = np.abs(data - mean_val) <= threshold_std * std_val
        return data[mask]

    def _calculate_inter_breath_intervals(self, inhales: List[BreathPeak]) -> np.ndarray:
        """Calculate intervals between consecutive inhales."""
        if len(inhales) < 2:
            return np.array([])

        sorted_inhales = sorted(inhales, key=lambda p: p.StartTime)
        intervals = []

        for i in range(1, len(sorted_inhales)):
            interval = sorted_inhales[i].StartTime - sorted_inhales[i-1].StartTime
            intervals.append(interval)

        return np.array(intervals)

    def _calculate_cov(self, data) -> float:
        """Calculate coefficient of variation."""
        if len(data) == 0:
            return np.nan

        data = np.asarray(data)
        mean_val = np.nanmean(data)

        if mean_val == 0:
            return np.nan

        return np.nanstd(data) / mean_val

    def _safe_mean(self, data) -> float:
        """Safely calculate mean, returning nan for empty data."""
        if len(data) == 0:
            return np.nan
        return np.nanmean(data)

    def _calculate_pause_metrics(
        self,
        peaks: List[BreathPeak],
        inter_breath_interval: float
    ) -> Dict[str, float]:
        """Calculate respiratory pause metrics."""
        metrics = {}

        inhales = [p for p in peaks if p.PeakValue > 0]
        exhales = [p for p in peaks if p.PeakValue < 0]

        # Post-inhale pause
        post_inhale_pauses = self._calculate_pauses(inhales, exhales, 'post_inhale')
        clean_post_inhale = self._remove_outliers_1d(
            np.array(post_inhale_pauses), self.outlier_threshold_std
        )

        metrics['Post_inhale_pause_mean'] = self._safe_mean(clean_post_inhale)
        metrics['Post_inhale_pause_std'] = np.nanstd(clean_post_inhale) if len(clean_post_inhale) > 0 else np.nan

        # Post-exhale pause
        post_exhale_pauses = self._calculate_pauses(inhales, exhales, 'post_exhale')
        clean_post_exhale = self._remove_outliers_1d(
            np.array(post_exhale_pauses), self.outlier_threshold_std
        )

        metrics['Post_exhale_pause_mean'] = self._safe_mean(clean_post_exhale)
        metrics['Post_exhale_pause_std'] = np.nanstd(clean_post_exhale) if len(clean_post_exhale) > 0 else np.nan

        # Pause duty cycles
        if inter_breath_interval > 0:
            if len(clean_post_inhale) > 0:
                metrics['Pause_DutyCycle_inhale'] = self._safe_mean(clean_post_inhale) / inter_breath_interval
            else:
                metrics['Pause_DutyCycle_inhale'] = np.nan

            if len(clean_post_exhale) > 0:
                metrics['Pause_DutyCycle_exhale'] = self._safe_mean(clean_post_exhale) / inter_breath_interval
            else:
                metrics['Pause_DutyCycle_exhale'] = np.nan
        else:
            metrics['Pause_DutyCycle_inhale'] = np.nan
            metrics['Pause_DutyCycle_exhale'] = np.nan

        return metrics

    def _calculate_pauses(
        self,
        inhales: List[BreathPeak],
        exhales: List[BreathPeak],
        pause_type: str
    ) -> List[float]:
        """Calculate pauses between breathing phases. OPTIMIZED: O(n) instead of O(n²)"""
        pauses = []

        if pause_type == 'post_inhale':
            # Pause after inhale = start of next exhale - end of inhale
            # OPTIMIZATION: Sort exhales once by StartTime, then use binary search
            sorted_exhales = sorted(exhales, key=lambda e: e.StartTime)

            for inhale in inhales:
                inhale_end = inhale.StartTime + inhale.Duration
                # Binary search for first exhale after inhale_end
                # Use bisect for O(log n) instead of O(n) linear search
                idx = bisect.bisect_left([e.StartTime for e in sorted_exhales], inhale_end)

                if idx < len(sorted_exhales):
                    next_exhale = sorted_exhales[idx]
                    pause = next_exhale.StartTime - inhale_end
                    if pause >= self.min_pause_duration:
                        pauses.append(pause)
        else:
            # Pause after exhale = start of next inhale - end of exhale
            # OPTIMIZATION: Sort inhales once by StartTime
            sorted_inhales = sorted(inhales, key=lambda i: i.StartTime)

            for exhale in exhales:
                exhale_end = exhale.StartTime + exhale.Duration
                # Binary search for first inhale after exhale_end
                idx = bisect.bisect_left([i.StartTime for i in sorted_inhales], exhale_end)

                if idx < len(sorted_inhales):
                    next_inhale = sorted_inhales[idx]
                    pause = next_inhale.StartTime - exhale_end
                    if pause >= self.min_pause_duration:
                        pauses.append(pause)

        return pauses

    def _empty_metrics(self) -> Dict[str, float]:
        """Return empty metrics dictionary when no breaths detected."""
        return {
            # Original 25 metrics
            'Inhale_Volume': np.nan,
            'Exhale_Volume': np.nan,
            'Inhale_Duration': np.nan,
            'Exhale_Duration': np.nan,
            'Inhale_value': np.nan,
            'Exhale_value': np.nan,
            'Inter_breath_interval': np.nan,
            'Rate': np.nan,
            'Tidal_volume': np.nan,
            'Minute_Ventilation': np.nan,
            'Duty_Cycle_inhale': np.nan,
            'Duty_Cycle_exhale': np.nan,
            'COV_InhaleDutyCycle': np.nan,
            'COV_ExhaleDutyCycle': np.nan,
            'COV_BreathingRate': np.nan,
            'COV_InhaleVolume': np.nan,
            'COV_ExhaleVolume': np.nan,
            'Post_inhale_pause_mean': np.nan,
            'Post_inhale_pause_std': np.nan,
            'Post_exhale_pause_mean': np.nan,
            'Post_exhale_pause_std': np.nan,
            'Pause_DutyCycle_inhale': np.nan,
            'Pause_DutyCycle_exhale': np.nan,
            # New multi-peak metrics
            'MultiPhasic_Count': 0,
            'MultiPhasic_Ratio': 0,
            'MultiPhasic_Inhale_Count': 0,
            'MultiPhasic_Exhale_Count': 0,
            'MultiPhasic_Inhale_Ratio': 0,
            'MultiPhasic_Exhale_Ratio': 0,
            'PeaksPerBreath_Mean': 0,
            'PeaksPerBreath_Max': 0,
            'InterPeakInterval_Mean': 0,
            'InterPeakInterval_Std': 0,
            'SecondaryPeakRatio_Mean': 0,
            'SecondaryPeakRatio_Std': 0,
        }

    def get_feature_names(self) -> List[str]:
        """Get list of all feature names produced by this extractor."""
        return list(self._empty_metrics().keys())


# =============================================================================
# BACKWARD COMPATIBILITY - Original BreathPeak without multi-peak fields
# =============================================================================

def convert_to_simple_peaks(peaks: List[BreathPeak]) -> List[Dict]:
    """Convert enhanced BreathPeak to simple dict format for backward compatibility."""
    return [{
        'PeakLocation': p.PeakLocation,
        'PeakValue': p.PeakValue,
        'Volume': p.Volume,
        'Duration': p.Duration,
        'StartTime': p.StartTime,
        'Latency': p.Latency,
        'NumberOfPeaks': p.NumberOfPeaks
    } for p in peaks]
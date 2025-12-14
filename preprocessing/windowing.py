"""
Signal Windowing

Split respiratory recordings into time blocks (windows) with optional overlap.

KEY FIX: Step size calculation
- WRONG: step_size = overlap (creates way too many windows)
- CORRECT: step_size = window_size - overlap (proper windowing)

Example: 40-minute recording with 5-min windows and 1-min overlap
- window_size = 300s
- overlap = 60s
- step_size = 300 - 60 = 240s (CORRECT)
- Expected windows = (2400 - 300) / 240 + 1 ≈ 9 windows (not 35!)
"""

from typing import List, Optional
import numpy as np
from dataclasses import dataclass


@dataclass
class WindowConfig:
    """
    Configuration for window generation.

    Attributes:
        window_size: Window size in seconds
        overlap: Overlap between consecutive windows in seconds
    """
    window_size: float  # seconds
    overlap: float  # seconds

    def __post_init__(self):
        """Validate configuration."""
        if self.window_size <= 0:
            raise ValueError(f"window_size must be positive, got {self.window_size}")
        if self.overlap < 0:
            raise ValueError(f"overlap must be non-negative, got {self.overlap}")
        if self.overlap >= self.window_size:
            raise ValueError(f"overlap ({self.overlap}) must be < window_size ({self.window_size})")


@dataclass
class SignalWindow:
    """
    Represents a single time window from a recording.

    Attributes:
        data: Signal data for this window
        sampling_rate: Sampling rate in Hz
        start_time: Start time in seconds (relative to recording start)
        end_time: End time in seconds
        window_index: Index of this window (0-based)
        subject_id: Subject identifier
        recording_date: Recording date
        metadata: Additional metadata
    """
    data: np.ndarray
    sampling_rate: float
    start_time: float
    end_time: float
    window_index: int
    subject_id: str
    recording_date: str
    metadata: dict = None

    def __post_init__(self):
        """Initialize metadata if None."""
        if self.metadata is None:
            self.metadata = {}

    @property
    def duration(self) -> float:
        """Window duration in seconds."""
        return self.end_time - self.start_time

    @property
    def n_samples(self) -> int:
        """Number of samples in window."""
        return len(self.data)

    @property
    def time_axis(self) -> np.ndarray:
        """Time axis for this window."""
        return np.arange(len(self.data)) / self.sampling_rate + self.start_time

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"SignalWindow(subject='{self.subject_id}', "
            f"window={self.window_index}, "
            f"time={self.start_time:.1f}-{self.end_time:.1f}s, "
            f"n_samples={self.n_samples})"
        )


class WindowGenerator:
    """
    Generate overlapping windows from respiratory recordings.

    Implements proper windowing with correct step size calculation:
    - step_size = window_size - overlap

    Usage:
        config = WindowConfig(window_size=300, overlap=60)  # 5 min, 1 min overlap
        generator = WindowGenerator(config)
        windows = generator.generate_windows(recording)

    Mathematical example:
        - Recording: 40 minutes = 2400 seconds
        - Window size: 300 seconds (5 minutes)
        - Overlap: 60 seconds (1 minute)
        - Step size: 300 - 60 = 240 seconds (CORRECT!)
        - Expected windows: floor((2400 - 300) / 240) + 1 = 10 windows
    """

    def __init__(self, config: WindowConfig):
        """
        Initialize window generator.

        Args:
            config: Window configuration
        """
        self.config = config

    def generate_windows(self, recording, min_windows: int = 1) -> List[SignalWindow]:
        """
        Generate windows from a respiratory recording.

        Args:
            recording: RespiratoryRecording instance
            min_windows: Minimum number of windows required

        Returns:
            List of SignalWindow objects

        Raises:
            ValueError: If recording too short for even one window
        """
        # Convert to samples
        window_size_samples = int(self.config.window_size * recording.sampling_rate)
        overlap_samples = int(self.config.overlap * recording.sampling_rate)

        # KEY FIX: step_size = window_size - overlap (not just overlap)
        step_size_samples = window_size_samples - overlap_samples

        if step_size_samples <= 0:
            raise ValueError(
                f"Invalid window configuration: "
                f"window_size ({self.config.window_size}s) <= overlap ({self.config.overlap}s)"
            )

        # Calculate number of windows
        n_samples = len(recording.data)

        # Formula: number of windows = floor((total_length - window_length) / step) + 1
        if n_samples >= window_size_samples:
            n_windows = int(np.floor((n_samples - window_size_samples) / step_size_samples)) + 1
        else:
            n_windows = 0

        if n_windows < min_windows:
            raise ValueError(
                f"Recording too short for {min_windows} window(s). "
                f"Duration: {recording.duration:.1f}s, "
                f"Window size: {self.config.window_size}s, "
                f"Got {n_windows} windows. "
                f"Need at least {self.config.window_size}s for one window."
            )

        # Generate windows
        windows = []
        for i in range(n_windows):
            start_idx = i * step_size_samples
            end_idx = start_idx + window_size_samples

            # Ensure we don't exceed signal length
            if end_idx <= n_samples:
                window_data = recording.data[start_idx:end_idx]
                start_time = start_idx / recording.sampling_rate
                end_time = end_idx / recording.sampling_rate

                window = SignalWindow(
                    data=window_data,
                    sampling_rate=recording.sampling_rate,
                    start_time=start_time,
                    end_time=end_time,
                    window_index=i,
                    subject_id=recording.subject_id,
                    recording_date=recording.recording_date,
                    metadata={
                        'source_recording': f"{recording.subject_id}_{recording.recording_date}",
                        'window_config': {
                            'window_size': self.config.window_size,
                            'overlap': self.config.overlap,
                            'step_size': self.config.window_size - self.config.overlap
                        }
                    }
                )

                windows.append(window)

        return windows

    def calculate_n_windows(self, recording) -> int:
        """
        Calculate number of windows without generating them.

        Args:
            recording: RespiratoryRecording instance

        Returns:
            Number of windows that would be generated
        """
        window_size_samples = int(self.config.window_size * recording.sampling_rate)
        step_size_samples = int((self.config.window_size - self.config.overlap) * recording.sampling_rate)
        n_samples = len(recording.data)

        if n_samples >= window_size_samples and step_size_samples > 0:
            return int(np.floor((n_samples - window_size_samples) / step_size_samples)) + 1
        else:
            return 0


class BatchWindowProcessor:
    """
    Process multiple recordings and generate windows.

    Useful for batch processing datasets.
    """

    def __init__(self, config: WindowConfig):
        """
        Initialize batch processor.

        Args:
            config: Window configuration
        """
        self.generator = WindowGenerator(config)

    def process_batch(
        self,
        recordings: List,
        skip_short: bool = True
    ) -> List[SignalWindow]:
        """
        Generate windows from multiple recordings.

        Args:
            recordings: List of RespiratoryRecording instances
            skip_short: If True, skip recordings too short for windowing

        Returns:
            Flat list of all windows from all recordings
        """
        all_windows = []
        skipped = []

        for recording in recordings:
            try:
                windows = self.generator.generate_windows(recording)
                all_windows.extend(windows)
            except ValueError as e:
                if skip_short:
                    skipped.append(recording.subject_id)
                else:
                    raise

        if skipped:
            print(f"Skipped {len(skipped)} short recordings: {skipped}")

        return all_windows

    def process_batch_by_recording(
        self,
        recordings: List,
        skip_short: bool = True
    ) -> List[List[SignalWindow]]:
        """
        Generate windows keeping them grouped by recording.

        Args:
            recordings: List of RespiratoryRecording instances
            skip_short: If True, skip recordings too short for windowing

        Returns:
            List of lists, where each inner list contains windows from one recording
        """
        windows_by_recording = []
        skipped = []

        for recording in recordings:
            try:
                windows = self.generator.generate_windows(recording)
                windows_by_recording.append(windows)
            except ValueError as e:
                if skip_short:
                    skipped.append(recording.subject_id)
                else:
                    raise

        if skipped:
            print(f"Skipped {len(skipped)} short recordings: {skipped}")

        return windows_by_recording
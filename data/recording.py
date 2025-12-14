"""
Respiratory Recording Data Structure

This module defines the core data structure for representing a single
respiratory recording with its metadata.
"""

from typing import Optional, Dict, Any, Tuple
import numpy as np
from datetime import datetime
import copy


class RespiratoryRecording:
    """
    Represents a single respiratory recording with metadata.

    This is the core data structure that holds:
    - Raw respiratory signal data
    - Sampling rate
    - Subject identification
    - Recording metadata (date, duration, quality metrics, etc.)

    Attributes:
        data (np.ndarray): Raw respiratory signal, shape (n_samples,)
        sampling_rate (float): Sampling rate in Hz
        subject_id (str): Subject identifier (4-letter code)
        recording_date (str): Date of recording (format: YYYYMMDD or other)
        metadata (dict): Additional metadata (session info, quality flags, etc.)
    """

    def __init__(
        self,
        data: np.ndarray,
        sampling_rate: float,
        subject_id: str,
        recording_date: str,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize a respiratory recording.

        Args:
            data: Raw respiratory signal array
            sampling_rate: Sampling rate in Hz
            subject_id: Subject identifier
            recording_date: Date of recording
            metadata: Optional additional metadata
        """
        # Ensure data is 1D array
        self.data = np.asarray(data).flatten()
        self.sampling_rate = float(sampling_rate)
        self.subject_id = str(subject_id)
        self.recording_date = str(recording_date)
        self.metadata = metadata if metadata is not None else {}

        # Validation
        if self.sampling_rate <= 0:
            raise ValueError(f"Sampling rate must be positive, got {sampling_rate}")
        if len(self.data) == 0:
            raise ValueError("Data array cannot be empty")

    @property
    def duration(self) -> float:
        """
        Get recording duration in seconds.

        Returns:
            Duration in seconds
        """
        return len(self.data) / self.sampling_rate

    @property
    def n_samples(self) -> int:
        """
        Get number of samples in recording.

        Returns:
            Number of samples
        """
        return len(self.data)

    @property
    def time_axis(self) -> np.ndarray:
        """
        Get time axis for the recording.

        Returns:
            Time array in seconds, shape (n_samples,)
        """
        return np.arange(len(self.data)) / self.sampling_rate

    def get_segment(
        self,
        start_time: float,
        end_time: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract a time segment from the recording.

        Args:
            start_time: Start time in seconds
            end_time: End time in seconds

        Returns:
            Tuple of (signal_segment, time_segment)
        """
        start_idx = int(start_time * self.sampling_rate)
        end_idx = int(end_time * self.sampling_rate)

        # Clip to valid range
        start_idx = max(0, start_idx)
        end_idx = min(len(self.data), end_idx)

        if start_idx >= end_idx:
            raise ValueError(f"Invalid time range: {start_time} to {end_time} seconds")

        signal_segment = self.data[start_idx:end_idx]
        time_segment = np.arange(start_idx, end_idx) / self.sampling_rate

        return signal_segment, time_segment

    def get_samples_range(
        self,
        start_idx: int,
        end_idx: int
    ) -> np.ndarray:
        """
        Extract samples by index range.

        Args:
            start_idx: Start sample index
            end_idx: End sample index

        Returns:
            Signal segment
        """
        if start_idx < 0 or end_idx > len(self.data) or start_idx >= end_idx:
            raise ValueError(f"Invalid sample range: {start_idx} to {end_idx}")

        return self.data[start_idx:end_idx]

    def copy(self) -> 'RespiratoryRecording':
        """
        Create a deep copy of the recording.

        Returns:
            New RespiratoryRecording instance
        """
        return RespiratoryRecording(
            data=self.data.copy(),
            sampling_rate=self.sampling_rate,
            subject_id=self.subject_id,
            recording_date=self.recording_date,
            metadata=copy.deepcopy(self.metadata)
        )

    def add_metadata(self, key: str, value: Any) -> None:
        """
        Add metadata to the recording.

        Args:
            key: Metadata key
            value: Metadata value
        """
        self.metadata[key] = value

    def get_metadata(self, key: str, default: Any = None) -> Any:
        """
        Get metadata value.

        Args:
            key: Metadata key
            default: Default value if key not found

        Returns:
            Metadata value or default
        """
        return self.metadata.get(key, default)

    def __repr__(self) -> str:
        """String representation of the recording."""
        return (
            f"RespiratoryRecording(subject_id='{self.subject_id}', "
            f"date='{self.recording_date}', "
            f"duration={self.duration:.2f}s, "
            f"fs={self.sampling_rate}Hz, "
            f"n_samples={self.n_samples})"
        )

    def __len__(self) -> int:
        """Length of recording (number of samples)."""
        return len(self.data)
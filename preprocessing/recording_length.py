"""
Recording Length Management

Utilities for truncating recordings to specific durations for
hyperparameter experiments.
"""

import numpy as np
from typing import Optional
from data.loaders import Recording


class RecordingLengthManager:
    """
    Manages truncation of recordings to specific durations.

    Useful for testing how model performance varies with recording length.
    """

    @staticmethod
    def truncate_recording(
        recording: Recording,
        duration_minutes: Optional[float] = None
    ) -> Recording:
        """
        Truncate a recording to a specific duration.

        Args:
            recording: The original recording
            duration_minutes: Duration in minutes. If None, return original recording.

        Returns:
            New Recording object with truncated data
        """
        if duration_minutes is None:
            # Return original recording unchanged
            return recording

        # Calculate number of samples for the desired duration
        duration_seconds = duration_minutes * 60
        n_samples_desired = int(duration_seconds * recording.sampling_rate)

        # If recording is shorter than desired length, return original
        if n_samples_desired >= len(recording.data):
            return recording

        # Truncate data
        truncated_data = recording.data[:n_samples_desired]

        # Create new recording with truncated data
        truncated_recording = Recording(
            data=truncated_data,
            sampling_rate=recording.sampling_rate,
            subject_id=recording.subject_id,
            recording_date=recording.recording_date,
            metadata={**recording.metadata, 'truncated_to_minutes': duration_minutes}
        )

        return truncated_recording

    @staticmethod
    def get_recording_duration_minutes(recording: Recording) -> float:
        """
        Get the duration of a recording in minutes.

        Args:
            recording: The recording

        Returns:
            Duration in minutes
        """
        duration_seconds = len(recording.data) / recording.sampling_rate
        return duration_seconds / 60

    @staticmethod
    def format_length_name(duration_minutes: Optional[float]) -> str:
        """
        Format recording length for display/filenames.

        Args:
            duration_minutes: Duration in minutes, or None for full recording

        Returns:
            Formatted string (e.g., "5min", "10min", "full")
        """
        if duration_minutes is None:
            return "full"
        return f"{int(duration_minutes)}min"

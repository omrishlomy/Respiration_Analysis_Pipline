"""
Data Loaders

This module provides data loaders for different file formats.
All loaders implement the DataLoader interface.
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.io
import warnings
import re
from datetime import datetime

from .recording import RespiratoryRecording


class DataLoader(ABC):
    """
    Abstract base class for data loaders.

    All data loaders must implement the load() method to read
    a file and return a RespiratoryRecording object.
    """

    @abstractmethod
    def load(self, filepath: str) -> RespiratoryRecording:
        """
        Load a respiratory recording from file.

        Args:
            filepath: Path to the file

        Returns:
            RespiratoryRecording object

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file format is invalid
        """
        pass

    @abstractmethod
    def load_batch(self, directory: str, pattern: str = "*.mat") -> List[RespiratoryRecording]:
        """
        Load multiple recordings from a directory.

        Args:
            directory: Directory containing recordings
            pattern: Glob pattern for file matching (e.g., "*.mat")

        Returns:
            List of RespiratoryRecording objects
        """
        pass

    def _extract_subject_id(self, filename: str) -> str:
        """
        Extract subject ID from filename.

        Override this method if your naming convention is different.
        Default: extracts first 4 characters, normalized

        Args:
            filename: Filename (without path)

        Returns:
            Subject ID (4-letter code, uppercase, no separators)

        Examples:
            "ABOU - 22.8.16.mat" → "ABOU"
            "AB_OU - 22.8.16.mat" → "ABOU"
            "ab-ou - 22.8.16.mat" → "ABOU"
        """
        # Remove extension and get base name
        base_name = Path(filename).stem

        # Extract subject ID (everything before first space or dash)
        # This handles "ABOU - 22.8.16" format
        subject_id = base_name.split(' ')[0].split('-')[0]

        # Remove any remaining separators
        subject_id = subject_id.replace('_', '').replace('-', '').replace(' ', '')

        # Convert to uppercase
        subject_id = subject_id.upper()

        # Take first 4 characters
        if len(subject_id) >= 4:
            return subject_id[:4]
        else:
            warnings.warn(f"Filename '{filename}' too short to extract 4-letter subject ID")
            return subject_id


class MATDataLoader(DataLoader):
    """
    Loader for MATLAB .mat files.

    Assumes the .mat file contains:
    - Respiratory signal data
    - Sampling rate
    - Optional: additional metadata fields

    Attributes:
        data_key (str): Key for respiratory data in .mat file
        fs_key (str): Key for sampling rate in .mat file
        auto_detect_keys (bool): Try to auto-detect keys if not found
    """

    def __init__(
        self,
        data_key: str = 'data',
        fs_key: str = 'fs',
        auto_detect_keys: bool = True,
        default_sampling_rate: float = 6.0
    ):
        """
        Initialize MAT file loader.

        Args:
            data_key: Key for respiratory data in .mat structure
            fs_key: Key for sampling rate in .mat structure
            auto_detect_keys: If True, try to find keys automatically
            default_sampling_rate: Default sampling rate if not found in file
        """
        self.data_key = data_key
        self.fs_key = fs_key
        self.auto_detect_keys = auto_detect_keys
        self.default_sampling_rate = default_sampling_rate

    def _find_data_key(self, mat_data: Dict[str, Any]) -> Optional[str]:
        """
        Auto-detect the data key in MAT file.

        Args:
            mat_data: Loaded MAT file dictionary

        Returns:
            Key name or None if not found
        """
        # Common key names for respiratory data
        common_keys = ['ds_data', 'data', 'signal', 'resp', 'respiratory',
                      'y', 'x', 'values', 'samples']

        # Get non-private keys
        available_keys = [k for k in mat_data.keys() if not k.startswith('__')]

        # Try exact matches first
        for key in common_keys:
            if key in available_keys:
                return key

        # If only one key, use it
        if len(available_keys) == 1:
            return available_keys[0]

        # Try case-insensitive match
        for key in common_keys:
            for avail_key in available_keys:
                if avail_key.lower() == key.lower():
                    return avail_key

        return None

    def _find_fs_key(self, mat_data: Dict[str, Any]) -> Optional[str]:
        """
        Auto-detect the sampling rate key in MAT file.

        Args:
            mat_data: Loaded MAT file dictionary

        Returns:
            Key name or None if not found
        """
        common_keys = ['fs', 'Fs', 'sampling_rate', 'sample_rate', 'sr', 'freq', 'frequency']
        available_keys = [k for k in mat_data.keys() if not k.startswith('__')]

        for key in common_keys:
            if key in available_keys:
                return key

        return None

    def load(self, filepath: str) -> RespiratoryRecording:
        """
        Load recording from .mat file.

        Extracts subject ID and date from filename.

        Args:
            filepath: Path to .mat file

        Returns:
            RespiratoryRecording object
        """
        filepath = Path(filepath)

        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")

        # Load MAT file
        try:
            mat_data = scipy.io.loadmat(str(filepath))
        except Exception as e:
            raise ValueError(f"Failed to load MAT file {filepath}: {e}")

        # Find data key
        data_key = self.data_key
        if data_key not in mat_data:
            if self.auto_detect_keys:
                detected_key = self._find_data_key(mat_data)
                if detected_key:
                    data_key = detected_key
                    warnings.warn(f"Auto-detected data key '{data_key}' in {filepath.name}")
                else:
                    available = [k for k in mat_data.keys() if not k.startswith('__')]
                    raise ValueError(
                        f"Could not find data key '{self.data_key}' in {filepath.name}. "
                        f"Available keys: {available}"
                    )
            else:
                raise ValueError(f"Data key '{self.data_key}' not found in {filepath.name}")

        # Extract data
        data = mat_data[data_key]
        if isinstance(data, np.ndarray):
            data = data.flatten()
        else:
            raise ValueError(f"Data must be numpy array, got {type(data)}")

        # Find sampling rate
        fs_key = self.fs_key
        sampling_rate = None

        if fs_key in mat_data:
            sampling_rate = float(mat_data[fs_key].flatten()[0])
        elif self.auto_detect_keys:
            detected_fs_key = self._find_fs_key(mat_data)
            if detected_fs_key:
                sampling_rate = float(mat_data[detected_fs_key].flatten()[0])
                warnings.warn(f"Auto-detected sampling rate key '{detected_fs_key}' in {filepath.name}")

        if sampling_rate is None:
            sampling_rate = self.default_sampling_rate
            warnings.warn(
                f"Sampling rate not found in {filepath.name}, using default: {sampling_rate} Hz"
            )

        # Extract subject ID
        subject_id = self._extract_subject_id(filepath.name)

        # Extract date from filename
        recording_date = self._extract_date_from_filename(filepath.name)

        # Create recording
        recording = RespiratoryRecording(
            data=data,
            sampling_rate=sampling_rate,
            subject_id=subject_id,
            recording_date=recording_date,
            metadata={'source_file': filepath.name}
        )

        return recording

    def load_batch(self, directory: str, pattern: str = "*.mat") -> List[RespiratoryRecording]:
        """
        Load all .mat files from directory.

        Args:
            directory: Directory path
            pattern: Filename pattern

        Returns:
            List of recordings
        """
        directory = Path(directory)
        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")

        files = sorted(directory.glob(pattern))
        recordings = []
        failed_files = []

        print(f"Loading {len(files)} files from {directory}...")

        for filepath in files:
            try:
                recording = self.load(filepath)
                recordings.append(recording)
            except Exception as e:
                failed_files.append((filepath.name, str(e)))
                warnings.warn(f"Failed to load {filepath.name}: {e}")

        # Print summary
        print(f"\nBatch loading summary:")
        print(f"  Successfully loaded: {len(recordings)} files")
        print(f"  Failed: {len(failed_files)} files")

        if failed_files:
            print(f"\nFailed files:")
            for filename, error in failed_files:
                print(f"  - {filename}: {error}")

        return recordings

    def _extract_date_from_filename(self, filename: str) -> str:
        """
        Extract recording date from filename.

        Expected format: ABCD_-_4_9_16.mat (DD_M_YY or DD_MM_YY)
        Returns: YYYYMMDD format

        Args:
            filename: Filename

        Returns:
            Date string in YYYYMMDD format
        """
        # Remove extension
        base_name = Path(filename).stem

        # Try to extract date pattern: DD_M_YY or DD_MM_YY or DD_MM_YYYY
        # Pattern after subject ID (first 4 chars)
        # Example: ABOU_-_4_9_16

        # Extract numbers from filename
        numbers = re.findall(r'\d+', base_name)

        if len(numbers) >= 3:
            # Assume last 3 numbers are day, month, year
            day = numbers[-3]
            month = numbers[-2]
            year = numbers[-1]

            # Pad day and month
            day = day.zfill(2)
            month = month.zfill(2)

            # Handle 2-digit year
            if len(year) == 2:
                year_int = int(year)
                # Assume 00-30 is 2000s, 31-99 is 1900s
                if year_int <= 30:
                    year = '20' + year
                else:
                    year = '19' + year

            return f"{year}{month}{day}"
        else:
            warnings.warn(f"Could not extract date from filename: {filename}")
            return "UNKNOWN"


class CSVDataLoader(DataLoader):
    """
    Loader for CSV files with time-series respiratory data.

    CSV format expected:
    - Column 1: Time (seconds) or sample index
    - Column 2: Respiratory signal
    - Header row with 'time' and 'signal' (or configurable)

    Attributes:
        time_column (str): Name of time column
        signal_column (str): Name of signal column
        has_header (bool): Whether CSV has header row
        sampling_rate (Optional[float]): Sampling rate if not in file
    """

    def __init__(
        self,
        time_column: str = 'time',
        signal_column: str = 'signal',
        has_header: bool = True,
        sampling_rate: Optional[float] = None
    ):
        """
        Initialize CSV loader.

        Args:
            time_column: Name of time column
            signal_column: Name of signal column
            has_header: Whether CSV has header
            sampling_rate: Sampling rate if not calculable from data
        """
        self.time_column = time_column
        self.signal_column = signal_column
        self.has_header = has_header
        self.sampling_rate = sampling_rate

    def load(self, filepath: str) -> RespiratoryRecording:
        """Load recording from CSV file."""
        filepath = Path(filepath)

        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")

        # Load CSV
        if self.has_header:
            df = pd.read_csv(filepath)

            # Try to find signal column
            signal_col = None
            if self.signal_column in df.columns:
                signal_col = self.signal_column
            else:
                # Try common alternatives
                for col in ['signal', 'data', 'respiratory', 'resp', 'y', 'value']:
                    if col in df.columns:
                        signal_col = col
                        warnings.warn(f"Using column '{col}' as signal in {filepath.name}")
                        break

            if signal_col is None:
                raise ValueError(f"Could not find signal column in {filepath.name}")

            data = df[signal_col].values

            # Try to get time column for sampling rate calculation
            time_col = None
            if self.time_column in df.columns:
                time_col = self.time_column
            else:
                for col in ['time', 't', 'Time']:
                    if col in df.columns:
                        time_col = col
                        break

            if time_col is not None and self.sampling_rate is None:
                time_array = df[time_col].values
                sampling_rate = self._calculate_sampling_rate(time_array)
            else:
                sampling_rate = self.sampling_rate if self.sampling_rate else 6.0
                if self.sampling_rate is None:
                    warnings.warn(f"Sampling rate not found, using default: {sampling_rate} Hz")
        else:
            # No header, assume first column is time, second is signal
            df = pd.read_csv(filepath, header=None)
            if df.shape[1] >= 2:
                data = df.iloc[:, 1].values
                if self.sampling_rate is None:
                    time_array = df.iloc[:, 0].values
                    sampling_rate = self._calculate_sampling_rate(time_array)
                else:
                    sampling_rate = self.sampling_rate
            elif df.shape[1] == 1:
                data = df.iloc[:, 0].values
                sampling_rate = self.sampling_rate if self.sampling_rate else 6.0
                warnings.warn(f"Only one column found, using as signal. Sampling rate: {sampling_rate} Hz")
            else:
                raise ValueError(f"CSV file has no data: {filepath.name}")

        # Extract subject ID and date
        subject_id = self._extract_subject_id(filepath.name)
        recording_date = self._extract_date_from_csv_filename(filepath.name)

        recording = RespiratoryRecording(
            data=data,
            sampling_rate=sampling_rate,
            subject_id=subject_id,
            recording_date=recording_date,
            metadata={'source_file': filepath.name}
        )

        return recording

    def load_batch(self, directory: str, pattern: str = "*.csv") -> List[RespiratoryRecording]:
        """Load all CSV files from directory."""
        directory = Path(directory)
        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")

        files = sorted(directory.glob(pattern))
        recordings = []
        failed_files = []

        print(f"Loading {len(files)} CSV files from {directory}...")

        for filepath in files:
            try:
                recording = self.load(filepath)
                recordings.append(recording)
            except Exception as e:
                failed_files.append((filepath.name, str(e)))
                warnings.warn(f"Failed to load {filepath.name}: {e}")

        print(f"\nBatch loading summary:")
        print(f"  Successfully loaded: {len(recordings)} files")
        print(f"  Failed: {len(failed_files)} files")

        if failed_files:
            print(f"\nFailed files:")
            for filename, error in failed_files:
                print(f"  - {filename}: {error}")

        return recordings

    def _calculate_sampling_rate(self, time_array: np.ndarray) -> float:
        """
        Calculate sampling rate from time array.

        Args:
            time_array: Time values

        Returns:
            Estimated sampling rate in Hz
        """
        if len(time_array) < 2:
            raise ValueError("Need at least 2 time points to calculate sampling rate")

        # Calculate median time difference
        time_diffs = np.diff(time_array)
        median_dt = np.median(time_diffs)

        if median_dt <= 0:
            raise ValueError("Invalid time array (non-increasing)")

        sampling_rate = 1.0 / median_dt
        return sampling_rate

    def _extract_date_from_csv_filename(self, filename: str) -> str:
        """Extract date from CSV filename (similar to MAT loader)."""
        base_name = Path(filename).stem
        numbers = re.findall(r'\d+', base_name)

        if len(numbers) >= 3:
            day = numbers[-3].zfill(2)
            month = numbers[-2].zfill(2)
            year = numbers[-1]

            if len(year) == 2:
                year_int = int(year)
                if year_int <= 30:
                    year = '20' + year
                else:
                    year = '19' + year

            return f"{year}{month}{day}"
        else:
            return "UNKNOWN"


class BinaryDataLoader(DataLoader):
    """
    Loader for binary files (raw bytes).

    For files that store respiratory data as raw binary values.
    Requires known sampling rate and data type.

    Attributes:
        sampling_rate (float): Sampling rate in Hz
        dtype (str): Data type (e.g., 'float32', 'int16')
        byte_order (str): Byte order ('little' or 'big')
    """

    def __init__(
        self,
        sampling_rate: float,
        dtype: str = 'float32',
        byte_order: str = 'little'
    ):
        """
        Initialize binary loader.

        Args:
            sampling_rate: Sampling rate in Hz
            dtype: NumPy data type
            byte_order: Byte order
        """
        self.sampling_rate = sampling_rate
        self.dtype = dtype
        self.byte_order = byte_order

    def load(self, filepath: str) -> RespiratoryRecording:
        """Load recording from binary file."""
        filepath = Path(filepath)

        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")

        # Determine byte order prefix
        if self.byte_order == 'little':
            dtype_str = '<' + self.dtype
        elif self.byte_order == 'big':
            dtype_str = '>' + self.dtype
        else:
            dtype_str = self.dtype

        # Load binary data
        try:
            data = np.fromfile(str(filepath), dtype=dtype_str)
        except Exception as e:
            raise ValueError(f"Failed to load binary file {filepath}: {e}")

        # Extract subject ID and date
        subject_id = self._extract_subject_id(filepath.name)
        recording_date = self._extract_date_from_binary_filename(filepath.name)

        recording = RespiratoryRecording(
            data=data,
            sampling_rate=self.sampling_rate,
            subject_id=subject_id,
            recording_date=recording_date,
            metadata={'source_file': filepath.name, 'dtype': self.dtype}
        )

        return recording

    def load_batch(self, directory: str, pattern: str = "*.bin") -> List[RespiratoryRecording]:
        """Load all binary files from directory."""
        directory = Path(directory)
        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")

        files = sorted(directory.glob(pattern))
        recordings = []
        failed_files = []

        print(f"Loading {len(files)} binary files from {directory}...")

        for filepath in files:
            try:
                recording = self.load(filepath)
                recordings.append(recording)
            except Exception as e:
                failed_files.append((filepath.name, str(e)))
                warnings.warn(f"Failed to load {filepath.name}: {e}")

        print(f"\nBatch loading summary:")
        print(f"  Successfully loaded: {len(recordings)} files")
        print(f"  Failed: {len(failed_files)} files")

        if failed_files:
            print(f"\nFailed files:")
            for filename, error in failed_files:
                print(f"  - {filename}: {error}")

        return recordings

    def _extract_date_from_binary_filename(self, filename: str) -> str:
        """Extract date from binary filename."""
        base_name = Path(filename).stem
        numbers = re.findall(r'\d+', base_name)

        if len(numbers) >= 3:
            day = numbers[-3].zfill(2)
            month = numbers[-2].zfill(2)
            year = numbers[-1]

            if len(year) == 2:
                year_int = int(year)
                if year_int <= 30:
                    year = '20' + year
                else:
                    year = '19' + year

            return f"{year}{month}{day}"
        else:
            return "UNKNOWN"
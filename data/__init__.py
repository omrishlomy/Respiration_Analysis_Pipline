"""
Data Layer

Handles loading, storing, and exporting respiratory recordings and clinical labels.
"""

from .recording import RespiratoryRecording
from .loaders import (
    DataLoader,
    MATDataLoader,
    CSVDataLoader,
    BinaryDataLoader
)
from .clinical_labels import ClinicalLabels
from .exporters import (
    ExcelExporter,
    CSVExporter,
    ResultsExporter
)

__all__ = [
    # Core data structures
    'RespiratoryRecording',
    
    # Loaders
    'DataLoader',
    'MATDataLoader',
    'CSVDataLoader',
    'BinaryDataLoader',
    
    # Labels
    'ClinicalLabels',
    
    # Exporters
    'ExcelExporter',
    'CSVExporter',
    'ResultsExporter',
]

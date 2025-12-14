"""
Preprocessing Layer

Signal cleaning, quality assessment, and windowing.
"""

from .cleaner import (
    SignalCleaner,
    QualityIssue,
    QualityReport
)
from .windowing import (
    WindowConfig,
    SignalWindow,
    WindowGenerator,
    BatchWindowProcessor
)

__all__ = [
    # Cleaning
    'SignalCleaner',
    'QualityIssue',
    'QualityReport',

    # Windowing
    'WindowConfig',
    'SignalWindow',
    'WindowGenerator',
    'BatchWindowProcessor',
]
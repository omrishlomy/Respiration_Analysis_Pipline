"""
Respiratory Analysis Toolkit

A comprehensive Python package for analyzing respiratory data and predicting
clinical outcomes in Disorders of Consciousness (DoC).

Main Components:
    - Data loading and management
    - Signal preprocessing and cleaning
    - Feature extraction from respiratory signals
    - Statistical analysis and feature selection
    - Dimensionality reduction (PCA, t-SNE)
    - Machine learning classification
    - Interactive and static visualization
    - Pipeline orchestration
"""

__version__ = '0.1.0'
__author__ = 'Your Name'

# Core classes for easy import
from .data.recording import RespiratoryRecording
from .data.loaders import MATDataLoader
from .data.clinical_labels import ClinicalLabels
# Pipeline classes are defined in pipeline/__init__.py, not separate modules
try:
    from .pipeline import AnalysisPipeline, AnalysisConfig
except ImportError:
    AnalysisPipeline = None
    AnalysisConfig = None

__all__ = [
    'RespiratoryRecording',
    'MATDataLoader',
    'ClinicalLabels',
    'AnalysisPipeline',
    'AnalysisConfig',
]

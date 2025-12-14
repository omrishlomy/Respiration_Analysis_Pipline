"""
Feature Extractors

Classes for extracting features from respiratory signals.
"""

from .base import (
    FeatureExtractor,
    CompositeFeatureExtractor,
    CachedFeatureExtractor
)

from .breathing import (
    BreathingParameterExtractor,
    BreathPeak
)

__all__ = [
    # Base classes
    'FeatureExtractor',
    'CompositeFeatureExtractor',
    'CachedFeatureExtractor',

    # Breathing extractor
    'BreathingParameterExtractor',
    'BreathPeak',
]
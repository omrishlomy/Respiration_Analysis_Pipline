"""
Features Layer

Feature extraction and aggregation from respiratory signals.

Module structure:
    extractors/     - Feature extraction algorithms
    aggregator.py   - Window → Recording level aggregation
    collection.py   - Feature storage and manipulation
"""

# Import from extractors subpackage
from .extractors import (
    FeatureExtractor,
    CompositeFeatureExtractor,
    CachedFeatureExtractor,
    BreathingParameterExtractor,
    BreathPeak,
)

# Import aggregation and collection
from .aggregator import (
    FeatureAggregator,
    SimpleAggregator,
    aggregate_to_recording_level,
)

from .collection import FeatureCollection

__all__ = [
    # Feature Extractors (from extractors/)
    'FeatureExtractor',
    'CompositeFeatureExtractor',
    'CachedFeatureExtractor',
    'BreathingParameterExtractor',
    'BreathPeak',

    # Aggregation
    'FeatureAggregator',
    'SimpleAggregator',
    'aggregate_to_recording_level',

    # Collection
    'FeatureCollection',
]
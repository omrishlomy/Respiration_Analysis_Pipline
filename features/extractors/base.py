"""
Base Feature Extractor

Abstract base class for all feature extractors.
Defines the interface that all extractors must implement.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
import numpy as np
import warnings


class FeatureExtractor(ABC):
    """
    Abstract base class for feature extraction.

    All feature extractors must implement:
    - extract(): Extract features from a signal window
    - get_feature_names(): Return list of feature names

    Attributes:
        name (str): Name of this feature extractor
        prefix (str): Prefix for feature names (for namespacing)
    """

    def __init__(self, name: str, prefix: Optional[str] = None):
        """
        Initialize feature extractor.

        Args:
            name: Name of the extractor
            prefix: Optional prefix for feature names
        """
        self.name = name
        self.prefix = prefix or ""
        self.is_fitted = False

    @abstractmethod
    def extract(
        self,
        signal: np.ndarray,
        sampling_rate: float,
        **kwargs
    ) -> Dict[str, float]:
        """
        Extract features from a signal window.

        Args:
            signal: Input signal array
            sampling_rate: Sampling rate in Hz
            **kwargs: Additional parameters

        Returns:
            Dictionary mapping feature names to values
        """
        pass

    @abstractmethod
    def get_feature_names(self) -> List[str]:
        """
        Get list of feature names produced by this extractor.

        Returns:
            List of feature names
        """
        pass

    def validate_signal(
        self,
        signal: np.ndarray,
        min_length: Optional[int] = None
    ) -> bool:
        """
        Validate input signal.

        Args:
            signal: Input signal
            min_length: Minimum required length

        Returns:
            True if valid, False otherwise
        """
        if not isinstance(signal, np.ndarray):
            return False

        if len(signal) == 0:
            return False

        if np.all(np.isnan(signal)):
            return False

        if min_length is not None and len(signal) < min_length:
            return False

        return True

    def get_config(self) -> Dict[str, Any]:
        """
        Get configuration dictionary for this extractor.

        Returns:
            Configuration dictionary
        """
        return {
            'name': self.name,
            'prefix': self.prefix,
            'is_fitted': self.is_fitted
        }

    def __repr__(self) -> str:
        """String representation."""
        return f"{self.__class__.__name__}(name='{self.name}')"


class CompositeFeatureExtractor(FeatureExtractor):
    """
    Combine multiple feature extractors.

    Allows extracting features from multiple extractors in one call.
    Useful for building feature extraction pipelines.

    Example:
        breathing_extractor = BreathingParameterExtractor(config)
        composite = CompositeFeatureExtractor([breathing_extractor])
        features = composite.extract(signal, sampling_rate)
    """

    def __init__(
        self,
        extractors: List[FeatureExtractor],
        name: str = "composite"
    ):
        """
        Initialize composite extractor.

        Args:
            extractors: List of feature extractors to combine
            name: Name of composite extractor
        """
        super().__init__(name)
        self.extractors = {e.name: e for e in extractors}

    def add_extractor(self, extractor: FeatureExtractor) -> None:
        """
        Add an extractor to the composite.

        Args:
            extractor: Feature extractor to add
        """
        if extractor.name in self.extractors:
            warnings.warn(f"Extractor '{extractor.name}' already exists, overwriting")
        self.extractors[extractor.name] = extractor

    def remove_extractor(self, name: str) -> None:
        """
        Remove an extractor by name.

        Args:
            name: Name of extractor to remove
        """
        if name in self.extractors:
            del self.extractors[name]
        else:
            warnings.warn(f"Extractor '{name}' not found")

    def extract(
        self,
        signal: np.ndarray,
        sampling_rate: float,
        **kwargs
    ) -> Dict[str, float]:
        """
        Extract features using all contained extractors.

        Args:
            signal: Input signal
            sampling_rate: Sampling rate
            **kwargs: Additional parameters

        Returns:
            Combined dictionary of all features
        """
        combined_features = {}

        for extractor_name, extractor in self.extractors.items():
            try:
                features = extractor.extract(signal, sampling_rate, **kwargs)

                # Add prefix to feature names if specified
                if extractor.prefix:
                    features = {
                        f"{extractor.prefix}_{k}": v
                        for k, v in features.items()
                    }

                combined_features.update(features)
            except Exception as e:
                warnings.warn(f"Error extracting with {extractor_name}: {e}")

        return combined_features

    def get_feature_names(self) -> List[str]:
        """
        Get all feature names from all extractors.

        Returns:
            Combined list of feature names
        """
        all_names = []

        for extractor_name, extractor in self.extractors.items():
            names = extractor.get_feature_names()

            # Add prefix if specified
            if extractor.prefix:
                names = [f"{extractor.prefix}_{n}" for n in names]

            all_names.extend(names)

        return all_names

    def get_extractor(self, name: str) -> Optional[FeatureExtractor]:
        """
        Get an extractor by name.

        Args:
            name: Extractor name

        Returns:
            FeatureExtractor or None if not found
        """
        return self.extractors.get(name)


class CachedFeatureExtractor(FeatureExtractor):
    """
    Wrapper that caches extracted features.

    Useful for expensive feature computation when processing
    the same signal multiple times.

    Example:
        base_extractor = BreathingParameterExtractor(config)
        cached = CachedFeatureExtractor(base_extractor, cache_size=100)
        features = cached.extract(signal, sampling_rate)
    """

    def __init__(
        self,
        extractor: FeatureExtractor,
        cache_size: int = 100
    ):
        """
        Initialize cached extractor.

        Args:
            extractor: Base feature extractor to wrap
            cache_size: Maximum cache size
        """
        super().__init__(f"cached_{extractor.name}")
        self.extractor = extractor
        self.cache_size = cache_size
        self._cache = {}
        self._cache_hits = 0
        self._cache_misses = 0

    def extract(
        self,
        signal: np.ndarray,
        sampling_rate: float,
        **kwargs
    ) -> Dict[str, float]:
        """
        Extract features with caching.

        Args:
            signal: Input signal
            sampling_rate: Sampling rate
            **kwargs: Additional parameters

        Returns:
            Feature dictionary
        """
        # Create cache key from signal hash and sampling rate
        cache_key = (hash(signal.tobytes()), sampling_rate)

        if cache_key in self._cache:
            self._cache_hits += 1
            return self._cache[cache_key].copy()

        # Compute features
        self._cache_misses += 1
        features = self.extractor.extract(signal, sampling_rate, **kwargs)

        # Store in cache (with size limit)
        if len(self._cache) >= self.cache_size:
            # Remove oldest item (FIFO)
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]

        self._cache[cache_key] = features.copy()

        return features

    def get_feature_names(self) -> List[str]:
        """Get feature names from wrapped extractor."""
        return self.extractor.get_feature_names()

    def clear_cache(self) -> None:
        """Clear the feature cache."""
        self._cache.clear()
        self._cache_hits = 0
        self._cache_misses = 0

    def get_cache_stats(self) -> Dict[str, int]:
        """
        Get cache statistics.

        Returns:
            Dictionary with hits, misses, size
        """
        total_requests = self._cache_hits + self._cache_misses
        hit_rate = self._cache_hits / total_requests if total_requests > 0 else 0.0

        return {
            'hits': self._cache_hits,
            'misses': self._cache_misses,
            'size': len(self._cache),
            'hit_rate': hit_rate
        }
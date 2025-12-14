"""
Dimensionality Reduction

Reduce feature dimensionality for visualization and analysis.
Supports PCA, t-SNE, and UMAP (optional).

All reducers automatically scale data before transformation.
"""

from typing import Optional, Tuple, List, Dict, Any, Union
import numpy as np
import pandas as pd
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Try to import UMAP (optional dependency)
try:
    from umap import UMAP
    HAS_UMAP = True
except (ImportError, TypeError, ValueError, Exception) as e:
    HAS_UMAP = False
    warnings.warn(
        f"UMAP could not be imported (Error: {e}). UMAPReducer will be unavailable. "
        "This is likely due to a Numba/NumPy version mismatch."
    )


class DimensionalityReducer:
    """
    Base class for dimensionality reduction.

    All reducers:
    - Automatically scale data using StandardScaler (safe for pre-scaled data)
    - Return reduced data for visualization
    - Provide component names for plotting
    """

    def __init__(
        self,
        n_components: int = 2,
        name: str = "reducer",
        scale_data: bool = True,
        random_state: Optional[int] = 42
    ):
        """
        Initialize dimensionality reducer.

        Args:
            n_components: Number of components to reduce to
            name: Name of reducer
            scale_data: Whether to standardize data before reduction
            random_state: Random seed for reproducibility
        """
        self.n_components = n_components
        self.name = name
        self.scale_data = scale_data
        self.random_state = random_state

        self._scaler = StandardScaler() if scale_data else None
        self._is_fitted = False
        self._reducer = None
        self._feature_names = None

    def _prepare_data(
        self,
        X: Union[np.ndarray, pd.DataFrame, 'FeatureCollection'],
        fit_scaler: bool = True
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Prepare data for reduction (handle different input types, scale if needed).

        Args:
            X: Input data
            fit_scaler: Whether to fit the scaler (True for fit, False for transform)

        Returns:
            Tuple of (scaled_data, feature_names)
        """
        # Handle FeatureCollection
        if hasattr(X, 'features_df'):
            feature_names = X.feature_names
            X_array = X.get_features_array()
        elif isinstance(X, pd.DataFrame):
            # Exclude metadata columns
            metadata_cols = ['SubjectID', 'RecordingDate', 'WindowIndex',
                           'StartTime', 'EndTime', 'Duration', 'N_Windows', 'N_Valid_Windows']
            numeric_cols = [col for col in X.columns
                          if col not in metadata_cols and X[col].dtype in [np.float64, np.float32, np.int64, np.int32]]
            feature_names = numeric_cols
            X_array = X[numeric_cols].values
        else:
            feature_names = [f"feature_{i}" for i in range(X.shape[1])]
            X_array = np.asarray(X)

        # Handle NaN values
        X_array = np.asarray(X_array, dtype=float)

        # Check for NaN rows
        nan_rows = np.any(np.isnan(X_array), axis=1)
        if np.any(nan_rows):
            n_nan = np.sum(nan_rows)
            warnings.warn(
                f"{n_nan} samples contain NaN values and will be removed for dimensionality reduction. "
                f"Consider imputing missing values first."
            )
            X_array = X_array[~nan_rows]

        # Scale data
        if self._scaler is not None:
            if fit_scaler:
                X_scaled = self._scaler.fit_transform(X_array)
            else:
                X_scaled = self._scaler.transform(X_array)
        else:
            X_scaled = X_array

        return X_scaled, feature_names

    def fit(
        self,
        X: Union[np.ndarray, pd.DataFrame, 'FeatureCollection'],
        y: Optional[np.ndarray] = None
    ) -> 'DimensionalityReducer':
        """
        Fit reducer to data.

        Args:
            X: Feature matrix
            y: Optional labels (ignored by most methods)

        Returns:
            self
        """
        raise NotImplementedError("Subclasses must implement fit()")

    def transform(
        self,
        X: Union[np.ndarray, pd.DataFrame, 'FeatureCollection']
    ) -> np.ndarray:
        """
        Transform data to reduced dimensions.

        Args:
            X: Feature matrix

        Returns:
            Reduced feature matrix (n_samples, n_components)
        """
        raise NotImplementedError("Subclasses must implement transform()")

    def fit_transform(
        self,
        X: Union[np.ndarray, pd.DataFrame, 'FeatureCollection'],
        y: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Fit and transform.

        Args:
            X: Feature matrix
            y: Optional labels

        Returns:
            Reduced feature matrix
        """
        self.fit(X, y)
        return self.transform(X)

    def get_component_names(self) -> List[str]:
        """
        Get names of components.

        Returns:
            List like ['PC1', 'PC2', ...] or ['t-SNE1', 't-SNE2', ...]
        """
        raise NotImplementedError("Subclasses must implement get_component_names()")

    def prepare_visualization_data(
        self,
        X: Union[np.ndarray, pd.DataFrame, 'FeatureCollection'],
        labels: Optional[np.ndarray] = None,
        subject_ids: Optional[List[str]] = None,
        outcome: Optional[str] = None,
        clinical_labels: Optional['ClinicalLabels'] = None
    ) -> pd.DataFrame:
        """
        Prepare reduced data for visualization.

        Creates a DataFrame ready for plotting with component values,
        labels (if provided), and subject IDs.

        Args:
            X: Feature matrix
            labels: Optional label array
            subject_ids: Optional subject ID list
            outcome: Outcome name (required if using clinical_labels)
            clinical_labels: ClinicalLabels object

        Returns:
            DataFrame with columns:
            - Component columns (PC1, PC2, ... or t-SNE1, t-SNE2, ...)
            - 'label' (if labels provided)
            - 'subject_id' (if provided)
        """
        if not self._is_fitted:
            raise ValueError("Reducer must be fitted first. Call fit() or fit_transform().")

        # Transform data
        X_reduced = self.transform(X)

        # Create DataFrame with component names
        component_names = self.get_component_names()
        df = pd.DataFrame(X_reduced, columns=component_names)

        # Handle subject IDs
        if subject_ids is None and hasattr(X, 'subject_ids'):
            subject_ids = X.subject_ids

        if subject_ids is not None:
            # Handle NaN removal - align subject_ids with reduced data
            if hasattr(X, 'features_df'):
                X_check = X.get_features_array()
            elif isinstance(X, pd.DataFrame):
                metadata_cols = ['SubjectID', 'RecordingDate', 'WindowIndex',
                               'StartTime', 'EndTime', 'Duration', 'N_Windows', 'N_Valid_Windows']
                numeric_cols = [col for col in X.columns
                              if col not in metadata_cols and X[col].dtype in [np.float64, np.float32, np.int64, np.int32]]
                X_check = X[numeric_cols].values
            else:
                X_check = np.asarray(X)

            nan_rows = np.any(np.isnan(X_check), axis=1)
            subject_ids_clean = [sid for sid, is_nan in zip(subject_ids, nan_rows) if not is_nan]

            if len(subject_ids_clean) == len(df):
                df['subject_id'] = subject_ids_clean

        # Handle labels
        if labels is not None:
            labels = np.asarray(labels)
            # Handle NaN removal alignment
            if hasattr(X, 'features_df'):
                X_check = X.get_features_array()
            elif isinstance(X, pd.DataFrame):
                metadata_cols = ['SubjectID', 'RecordingDate', 'WindowIndex',
                               'StartTime', 'EndTime', 'Duration', 'N_Windows', 'N_Valid_Windows']
                numeric_cols = [col for col in X.columns
                              if col not in metadata_cols and X[col].dtype in [np.float64, np.float32, np.int64, np.int32]]
                X_check = X[numeric_cols].values
            else:
                X_check = np.asarray(X)

            nan_rows = np.any(np.isnan(X_check), axis=1)
            labels_clean = labels[~nan_rows]

            if len(labels_clean) == len(df):
                df['label'] = labels_clean

        elif clinical_labels is not None and outcome is not None and 'subject_id' in df.columns:
            # Get labels from ClinicalLabels
            labels_list = []
            for sid in df['subject_id']:
                try:
                    label = clinical_labels.get_label(sid, outcome)
                    labels_list.append(label)
                except KeyError:
                    labels_list.append(np.nan)
            df['label'] = labels_list

        return df


class PCAReducer(DimensionalityReducer):
    """
    Principal Component Analysis.

    Linear dimensionality reduction using SVD.
    Preserves global structure and variance.

    Provides:
    - Explained variance analysis
    - Feature loadings (contribution of each feature to components)
    - Optimal component selection based on variance threshold
    """

    def __init__(
        self,
        n_components: int = 2,
        whiten: bool = False,
        scale_data: bool = True,
        random_state: Optional[int] = 42
    ):
        """
        Initialize PCA.

        Args:
            n_components: Number of principal components
            whiten: Whether to whiten (normalize variance of components)
            scale_data: Whether to standardize data before PCA
            random_state: Random seed
        """
        super().__init__(n_components, "PCA", scale_data, random_state)
        self.whiten = whiten
        self._pca = None

    def fit(
        self,
        X: Union[np.ndarray, pd.DataFrame, 'FeatureCollection'],
        y: Optional[np.ndarray] = None
    ) -> 'PCAReducer':
        """
        Fit PCA to data.

        Args:
            X: Feature matrix
            y: Ignored

        Returns:
            self
        """
        X_scaled, self._feature_names = self._prepare_data(X, fit_scaler=True)

        self._pca = PCA(
            n_components=min(self.n_components, X_scaled.shape[1]),
            whiten=self.whiten,
            random_state=self.random_state
        )
        self._pca.fit(X_scaled)
        self._is_fitted = True

        return self

    def transform(
        self,
        X: Union[np.ndarray, pd.DataFrame, 'FeatureCollection']
    ) -> np.ndarray:
        """
        Transform data to principal components.

        Args:
            X: Feature matrix

        Returns:
            Reduced feature matrix (n_samples, n_components)
        """
        if not self._is_fitted:
            raise ValueError("PCAReducer must be fitted first")

        X_scaled, _ = self._prepare_data(X, fit_scaler=False)
        return self._pca.transform(X_scaled)

    def get_component_names(self) -> List[str]:
        """Get component names (PC1, PC2, ...)."""
        return [f"PC{i+1}" for i in range(self.n_components)]

    def get_explained_variance(self) -> np.ndarray:
        """
        Get explained variance ratio for each component.

        Returns:
            Array of explained variance ratios (sums to ≤1)
        """
        if not self._is_fitted:
            raise ValueError("PCAReducer must be fitted first")
        return self._pca.explained_variance_ratio_

    def get_cumulative_variance(self) -> np.ndarray:
        """
        Get cumulative explained variance.

        Returns:
            Cumulative variance explained (increasing, ends at total explained)
        """
        return np.cumsum(self.get_explained_variance())

    def get_loadings(self, feature_names: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Get feature loadings for each component.

        Loadings show how much each original feature contributes
        to each principal component.

        Args:
            feature_names: Names of original features (uses stored names if None)

        Returns:
            DataFrame with features as rows, components as columns
        """
        if not self._is_fitted:
            raise ValueError("PCAReducer must be fitted first")

        if feature_names is None:
            feature_names = self._feature_names

        loadings = self._pca.components_.T
        component_names = self.get_component_names()

        return pd.DataFrame(
            loadings,
            index=feature_names,
            columns=component_names
        )

    def get_top_features_per_component(
        self,
        n_features: int = 10,
        feature_names: Optional[List[str]] = None
    ) -> Dict[str, List[Tuple[str, float]]]:
        """
        Get top contributing features for each component.

        Args:
            n_features: Number of top features per component
            feature_names: Feature names

        Returns:
            Dict mapping component name -> list of (feature, loading) tuples
        """
        loadings_df = self.get_loadings(feature_names)

        result = {}
        for component in loadings_df.columns:
            # Sort by absolute loading
            sorted_loadings = loadings_df[component].abs().sort_values(ascending=False)
            top_features = [
                (feat, loadings_df.loc[feat, component])
                for feat in sorted_loadings.head(n_features).index
            ]
            result[component] = top_features

        return result

    def select_n_components(
        self,
        X: Union[np.ndarray, pd.DataFrame, 'FeatureCollection'],
        variance_threshold: float = 0.95
    ) -> int:
        """
        Determine number of components needed to explain variance threshold.

        Args:
            X: Feature matrix
            variance_threshold: Desired cumulative variance (e.g., 0.95 for 95%)

        Returns:
            Number of components needed
        """
        # Fit full PCA to analyze
        X_scaled, _ = self._prepare_data(X, fit_scaler=True)

        full_pca = PCA(random_state=self.random_state)
        full_pca.fit(X_scaled)

        cumvar = np.cumsum(full_pca.explained_variance_ratio_)
        n_components = np.searchsorted(cumvar, variance_threshold) + 1

        return min(n_components, len(cumvar))

    def prepare_variance_plot_data(
        self,
        X: Union[np.ndarray, pd.DataFrame, 'FeatureCollection'],
        max_components: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Prepare data for scree plot / variance explained visualization.

        Args:
            X: Feature matrix
            max_components: Maximum components to include (None = all)

        Returns:
            DataFrame with columns:
            - component: Component number
            - explained_variance: Variance explained by this component
            - cumulative_variance: Cumulative variance explained
        """
        X_scaled, _ = self._prepare_data(X, fit_scaler=True)

        n_comp = min(max_components or X_scaled.shape[1], X_scaled.shape[1])
        full_pca = PCA(n_components=n_comp, random_state=self.random_state)
        full_pca.fit(X_scaled)

        variance = full_pca.explained_variance_ratio_
        cumvar = np.cumsum(variance)

        return pd.DataFrame({
            'component': range(1, len(variance) + 1),
            'explained_variance': variance,
            'cumulative_variance': cumvar
        })

    def summary(self) -> str:
        """Generate summary of PCA results."""
        if not self._is_fitted:
            return "PCAReducer: Not fitted yet"

        lines = [
            "PCA Summary",
            "=" * 40,
            f"Number of components: {self.n_components}",
            f"Whitened: {self.whiten}",
            f"Data scaled: {self.scale_data}",
            "",
            "Explained Variance:",
        ]

        variance = self.get_explained_variance()
        cumvar = self.get_cumulative_variance()

        for i, (var, cum) in enumerate(zip(variance, cumvar)):
            lines.append(f"  PC{i+1}: {var:.4f} ({var*100:.1f}%) | Cumulative: {cum:.4f} ({cum*100:.1f}%)")

        lines.append(f"\nTotal variance explained: {cumvar[-1]*100:.1f}%")

        return "\n".join(lines)


class TSNEReducer(DimensionalityReducer):
    """
    t-Distributed Stochastic Neighbor Embedding.

    Non-linear dimensionality reduction for visualization.
    Preserves local structure (clusters).

    Note: t-SNE does not support transform() on new data.
    Each call to fit_transform() produces a new embedding.
    """

    def __init__(
        self,
        n_components: int = 2,
        perplexity: float = 30.0,
        learning_rate: Union[float, str] = 'auto',
        n_iter: int = 1000,
        scale_data: bool = True,
        random_state: Optional[int] = 42,
        metric: str = 'euclidean',
        init: str = 'pca'
    ):
        """
        Initialize t-SNE.

        Args:
            n_components: Number of dimensions (typically 2 or 3)
            perplexity: Related to number of nearest neighbors (5-50 typical)
            learning_rate: Learning rate for optimization ('auto' recommended)
            n_iter: Number of iterations
            scale_data: Whether to standardize data before t-SNE
            random_state: Random seed
            metric: Distance metric ('euclidean', 'cosine', etc.)
            init: Initialization method ('pca' recommended for stability)
        """
        super().__init__(n_components, "t-SNE", scale_data, random_state)
        self.perplexity = perplexity
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.metric = metric
        self.init = init

        self._tsne = None
        self._embedding = None
        self._kl_divergence = None

    def fit(
        self,
        X: Union[np.ndarray, pd.DataFrame, 'FeatureCollection'],
        y: Optional[np.ndarray] = None
    ) -> 'TSNEReducer':
        """
        Fit t-SNE and compute embedding.

        Args:
            X: Feature matrix
            y: Ignored

        Returns:
            self
        """
        X_scaled, self._feature_names = self._prepare_data(X, fit_scaler=True)

        # Adjust perplexity if too high for sample size
        max_perplexity = (X_scaled.shape[0] - 1) / 3
        perplexity = min(self.perplexity, max_perplexity)

        if perplexity < self.perplexity:
            warnings.warn(
                f"Perplexity {self.perplexity} too high for {X_scaled.shape[0]} samples. "
                f"Reduced to {perplexity:.1f}"
            )

        self._tsne = TSNE(
            n_components=self.n_components,
            perplexity=perplexity,
            learning_rate=self.learning_rate,
            max_iter=self.n_iter,  # sklearn >= 1.4 uses max_iter instead of n_iter
            random_state=self.random_state,
            metric=self.metric,
            init=self.init
        )

        self._embedding = self._tsne.fit_transform(X_scaled)
        self._kl_divergence = self._tsne.kl_divergence_
        self._is_fitted = True

        return self

    def transform(
        self,
        X: Union[np.ndarray, pd.DataFrame, 'FeatureCollection']
    ) -> np.ndarray:
        """
        Return the stored embedding.

        Note: t-SNE cannot transform new data. This returns the embedding
        computed during fit(). For new data, call fit_transform().

        Args:
            X: Ignored (returns stored embedding)

        Returns:
            Stored embedding from fit()
        """
        if not self._is_fitted:
            raise ValueError("TSNEReducer must be fitted first")

        warnings.warn(
            "t-SNE does not support transforming new data. "
            "Returning the embedding computed during fit(). "
            "For new data, call fit_transform() instead."
        )
        return self._embedding

    def fit_transform(
        self,
        X: Union[np.ndarray, pd.DataFrame, 'FeatureCollection'],
        y: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Fit t-SNE and return embedding.

        Args:
            X: Feature matrix
            y: Ignored

        Returns:
            t-SNE embedding (n_samples, n_components)
        """
        self.fit(X, y)
        return self._embedding

    def get_component_names(self) -> List[str]:
        """Get component names (t-SNE1, t-SNE2, ...)."""
        return [f"t-SNE{i+1}" for i in range(self.n_components)]

    def get_kl_divergence(self) -> float:
        """
        Get final KL divergence (loss).

        Lower values indicate better fit.

        Returns:
            KL divergence value
        """
        if not self._is_fitted:
            raise ValueError("TSNEReducer must be fitted first")
        return self._kl_divergence

    def summary(self) -> str:
        """Generate summary of t-SNE results."""
        lines = [
            "t-SNE Summary",
            "=" * 40,
            f"Number of components: {self.n_components}",
            f"Perplexity: {self.perplexity}",
            f"Learning rate: {self.learning_rate}",
            f"Iterations: {self.n_iter}",
            f"Metric: {self.metric}",
            f"Data scaled: {self.scale_data}",
        ]

        if self._is_fitted:
            lines.append(f"\nFinal KL divergence: {self._kl_divergence:.4f}")
            lines.append(f"Embedding shape: {self._embedding.shape}")
        else:
            lines.append("\nNot fitted yet")

        return "\n".join(lines)


class UMAPReducer(DimensionalityReducer):
    """
    Uniform Manifold Approximation and Projection.

    Non-linear dimensionality reduction, faster than t-SNE.
    Preserves both local and global structure.

    Requires umap-learn package: pip install umap-learn
    """

    def __init__(
        self,
        n_components: int = 2,
        n_neighbors: int = 15,
        min_dist: float = 0.1,
        metric: str = 'euclidean',
        scale_data: bool = True,
        random_state: Optional[int] = 42
    ):
        """
        Initialize UMAP.

        Args:
            n_components: Number of dimensions
            n_neighbors: Number of neighbors (controls local vs global structure)
            min_dist: Minimum distance in embedding (controls clustering tightness)
            metric: Distance metric
            scale_data: Whether to standardize data before UMAP
            random_state: Random seed
        """
        if not HAS_UMAP:
            raise ImportError(
                "UMAP is not installed. Install with: pip install umap-learn"
            )

        super().__init__(n_components, "UMAP", scale_data, random_state)
        self.n_neighbors = n_neighbors
        self.min_dist = min_dist
        self.metric = metric

        self._umap = None

    def fit(
        self,
        X: Union[np.ndarray, pd.DataFrame, 'FeatureCollection'],
        y: Optional[np.ndarray] = None
    ) -> 'UMAPReducer':
        """
        Fit UMAP to data.

        Args:
            X: Feature matrix
            y: Optional labels (can improve embedding for supervised tasks)

        Returns:
            self
        """
        X_scaled, self._feature_names = self._prepare_data(X, fit_scaler=True)

        # Adjust n_neighbors if too high
        max_neighbors = X_scaled.shape[0] - 1
        n_neighbors = min(self.n_neighbors, max_neighbors)

        if n_neighbors < self.n_neighbors:
            warnings.warn(
                f"n_neighbors {self.n_neighbors} too high for {X_scaled.shape[0]} samples. "
                f"Reduced to {n_neighbors}"
            )

        self._umap = UMAP(
            n_components=self.n_components,
            n_neighbors=n_neighbors,
            min_dist=self.min_dist,
            metric=self.metric,
            random_state=self.random_state
        )

        self._umap.fit(X_scaled, y)
        self._is_fitted = True

        return self

    def transform(
        self,
        X: Union[np.ndarray, pd.DataFrame, 'FeatureCollection']
    ) -> np.ndarray:
        """
        Transform data to UMAP embedding.

        Unlike t-SNE, UMAP can transform new data.

        Args:
            X: Feature matrix

        Returns:
            UMAP embedding (n_samples, n_components)
        """
        if not self._is_fitted:
            raise ValueError("UMAPReducer must be fitted first")

        X_scaled, _ = self._prepare_data(X, fit_scaler=False)
        return self._umap.transform(X_scaled)

    def get_component_names(self) -> List[str]:
        """Get component names (UMAP1, UMAP2, ...)."""
        return [f"UMAP{i+1}" for i in range(self.n_components)]

    def summary(self) -> str:
        """Generate summary of UMAP results."""
        lines = [
            "UMAP Summary",
            "=" * 40,
            f"Number of components: {self.n_components}",
            f"n_neighbors: {self.n_neighbors}",
            f"min_dist: {self.min_dist}",
            f"Metric: {self.metric}",
            f"Data scaled: {self.scale_data}",
        ]

        if self._is_fitted:
            lines.append("\nFitted: Yes")
        else:
            lines.append("\nNot fitted yet")

        return "\n".join(lines)


class FeatureDimensionalityAnalysis:
    """
    Analyze feature space dimensionality and structure.

    Helps determine:
    - Intrinsic dimensionality of data
    - Best reduction method for your data
    - Optimal number of components
    """

    def __init__(self, random_state: int = 42):
        """Initialize dimensionality analyzer."""
        self.random_state = random_state

    def estimate_intrinsic_dimension(
        self,
        X: Union[np.ndarray, pd.DataFrame, 'FeatureCollection'],
        method: str = 'mle'
    ) -> int:
        """
        Estimate intrinsic dimensionality of data.

        Args:
            X: Feature matrix
            method: Estimation method:
                - 'mle': Maximum Likelihood Estimation (via PCA)
                - 'variance': Based on variance threshold (95%)
                - 'elbow': Elbow method on variance curve

        Returns:
            Estimated intrinsic dimension
        """
        # Prepare data
        if hasattr(X, 'features_df'):
            X_array = X.get_features_array()
        elif isinstance(X, pd.DataFrame):
            metadata_cols = ['SubjectID', 'RecordingDate', 'WindowIndex',
                           'StartTime', 'EndTime', 'Duration', 'N_Windows', 'N_Valid_Windows']
            numeric_cols = [col for col in X.columns
                          if col not in metadata_cols and X[col].dtype in [np.float64, np.float32, np.int64, np.int32]]
            X_array = X[numeric_cols].values
        else:
            X_array = np.asarray(X)

        # Remove NaN rows
        nan_rows = np.any(np.isnan(X_array), axis=1)
        X_clean = X_array[~nan_rows]

        # Scale
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_clean)

        # Fit full PCA
        pca = PCA(random_state=self.random_state)
        pca.fit(X_scaled)

        variance_ratio = pca.explained_variance_ratio_
        cumvar = np.cumsum(variance_ratio)

        if method == 'mle':
            # MLE-based estimation using PCA eigenvalues
            # Count components with eigenvalue > average
            eigenvalues = pca.explained_variance_
            mean_eigenvalue = np.mean(eigenvalues)
            return int(np.sum(eigenvalues > mean_eigenvalue))

        elif method == 'variance':
            # Components needed for 95% variance
            return int(np.searchsorted(cumvar, 0.95) + 1)

        elif method == 'elbow':
            # Find elbow in variance curve
            # Use second derivative to find point of maximum curvature
            if len(variance_ratio) < 3:
                return len(variance_ratio)

            second_derivative = np.diff(np.diff(variance_ratio))
            elbow = np.argmax(second_derivative) + 2  # +2 because of two diff operations
            return max(2, elbow)  # At least 2 dimensions

        else:
            raise ValueError(f"Unknown method: {method}")

    def compare_reduction_methods(
        self,
        X: Union[np.ndarray, pd.DataFrame, 'FeatureCollection'],
        y: Optional[np.ndarray] = None,
        methods: List[str] = ['pca', 'tsne'],
        n_components: int = 2
    ) -> pd.DataFrame:
        """
        Compare different reduction methods.

        For each method, computes:
        - Trustworthiness (how well local structure is preserved)
        - Time taken
        - Method-specific metrics

        Args:
            X: Feature matrix
            y: Labels (optional, for visual evaluation)
            methods: Methods to compare ('pca', 'tsne', 'umap')
            n_components: Number of components

        Returns:
            DataFrame with comparison metrics
        """
        import time
        from sklearn.manifold import trustworthiness

        # Prepare data
        if hasattr(X, 'features_df'):
            X_array = X.get_features_array()
        elif isinstance(X, pd.DataFrame):
            metadata_cols = ['SubjectID', 'RecordingDate', 'WindowIndex',
                           'StartTime', 'EndTime', 'Duration', 'N_Windows', 'N_Valid_Windows']
            numeric_cols = [col for col in X.columns
                          if col not in metadata_cols and X[col].dtype in [np.float64, np.float32, np.int64, np.int32]]
            X_array = X[numeric_cols].values
        else:
            X_array = np.asarray(X)

        # Remove NaN rows
        nan_rows = np.any(np.isnan(X_array), axis=1)
        X_clean = X_array[~nan_rows]

        # Scale
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_clean)

        results = []

        for method in methods:
            method = method.lower()

            start_time = time.time()

            try:
                if method == 'pca':
                    reducer = PCAReducer(n_components=n_components, random_state=self.random_state)
                    X_reduced = reducer.fit_transform(X_scaled)

                    variance_explained = np.sum(reducer.get_explained_variance())
                    specific_metric = variance_explained
                    metric_name = 'variance_explained'

                elif method == 'tsne':
                    reducer = TSNEReducer(n_components=n_components, random_state=self.random_state)
                    X_reduced = reducer.fit_transform(X_scaled)

                    specific_metric = reducer.get_kl_divergence()
                    metric_name = 'kl_divergence'

                elif method == 'umap':
                    if not HAS_UMAP:
                        warnings.warn("UMAP not available, skipping")
                        continue

                    reducer = UMAPReducer(n_components=n_components, random_state=self.random_state)
                    X_reduced = reducer.fit_transform(X_scaled)

                    specific_metric = np.nan  # UMAP doesn't have a standard metric
                    metric_name = 'n/a'

                else:
                    warnings.warn(f"Unknown method: {method}")
                    continue

                elapsed_time = time.time() - start_time

                # Compute trustworthiness
                n_neighbors = min(5, X_scaled.shape[0] - 1)
                trust = trustworthiness(X_scaled, X_reduced, n_neighbors=n_neighbors)

                results.append({
                    'method': method.upper(),
                    'n_components': n_components,
                    'trustworthiness': trust,
                    'time_seconds': elapsed_time,
                    metric_name: specific_metric,
                    'n_samples': X_scaled.shape[0],
                })

            except Exception as e:
                warnings.warn(f"Failed to run {method}: {e}")
                results.append({
                    'method': method.upper(),
                    'n_components': n_components,
                    'trustworthiness': np.nan,
                    'time_seconds': np.nan,
                    'error': str(e),
                })

        return pd.DataFrame(results)

    def prepare_variance_plot_data(
        self,
        X: Union[np.ndarray, pd.DataFrame, 'FeatureCollection'],
        max_components: int = 20
    ) -> pd.DataFrame:
        """
        Prepare data for PCA variance explained plot.

        Args:
            X: Feature matrix
            max_components: Maximum components to analyze

        Returns:
            DataFrame ready for plotting with:
            - component: Component number
            - explained_variance: Individual variance
            - cumulative_variance: Cumulative variance
        """
        pca = PCAReducer(n_components=max_components, random_state=self.random_state)
        return pca.prepare_variance_plot_data(X, max_components)


# Convenience function for creating reducer from config
def create_reducer_from_config(config: Dict, method: str = 'pca') -> DimensionalityReducer:
    """
    Create a dimensionality reducer from configuration dictionary.

    Args:
        config: Configuration dictionary
        method: Reduction method ('pca', 'tsne', 'umap')

    Returns:
        Configured DimensionalityReducer
    """
    dim_config = config.get('analysis', {}).get('dimensionality', {})

    n_components = dim_config.get('n_components', 2)
    random_state = config.get('models', {}).get('evaluation', {}).get('random_state', 42)

    method = method.lower()

    if method == 'pca':
        return PCAReducer(
            n_components=n_components,
            whiten=dim_config.get('pca_whiten', False),
            random_state=random_state
        )

    elif method == 'tsne':
        return TSNEReducer(
            n_components=n_components,
            perplexity=dim_config.get('tsne_perplexity', 30.0),
            n_iter=dim_config.get('tsne_n_iter', 1000),
            random_state=random_state
        )

    elif method == 'umap':
        if not HAS_UMAP:
            raise ImportError("UMAP is not installed")
        return UMAPReducer(
            n_components=n_components,
            n_neighbors=dim_config.get('umap_n_neighbors', 15),
            min_dist=dim_config.get('umap_min_dist', 0.1),
            random_state=random_state
        )

    else:
        raise ValueError(f"Unknown method: {method}")
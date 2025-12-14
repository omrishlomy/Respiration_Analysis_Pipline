"""
Feature Selection

Select most relevant features for classification/analysis.

Features:
- Statistical selection (from StatisticalAnalyzer results)
- Variance-based filtering
- Correlation-based redundancy removal
- Model-based selection (RFE, feature importance)
- Mutual information selection
"""

from typing import List, Optional, Tuple, Dict, Any, Union
import numpy as np
import pandas as pd
import warnings


class FeatureSelector:
    """
    Base class for feature selection.

    Supports multiple selection methods:
    - Statistical (based on p-values from compare_groups results)
    - Model-based (feature importance from models)
    - Variance-based (remove low variance features)
    - Correlation-based (remove redundant features)

    Can consume results directly from StatisticalAnalyzer.compare_groups().
    """

    def __init__(
        self,
        method: str = 'statistical',
        n_features: Optional[int] = None,
        threshold: Optional[float] = None,
        variance_threshold: float = 0.01,
        config: Optional[Dict] = None
    ):
        """
        Initialize feature selector.

        Args:
            method: Selection method ('statistical', 'variance', 'correlation', 'combined')
            n_features: Number of features to select (None = auto based on significance)
            threshold: Selection threshold (meaning depends on method):
                      - 'statistical': p-value threshold (default 0.05)
                      - 'variance': variance threshold (default 0.01)
                      - 'correlation': correlation threshold (default 0.95)
            variance_threshold: Minimum variance to keep a feature
            config: Optional configuration dictionary
        """
        self.method = method.lower()
        self.n_features = n_features
        self.threshold = threshold
        self.variance_threshold = variance_threshold
        self.config = config or {}

        # Load from config if provided
        fs_config = self.config.get('analysis', {}).get('feature_selection', {})
        if fs_config:
            self.method = fs_config.get('method', self.method)
            self.n_features = fs_config.get('n_features', self.n_features)
            self.variance_threshold = fs_config.get('variance_threshold', self.variance_threshold)

        # State
        self._selected_indices: Optional[List[int]] = None
        self._selected_names: Optional[List[str]] = None
        self._feature_scores: Optional[np.ndarray] = None
        self._variance_mask: Optional[np.ndarray] = None
        self._is_fitted = False

    def fit(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None,
        statistical_results: Optional[pd.DataFrame] = None
    ) -> 'FeatureSelector':
        """
        Fit feature selector.

        Args:
            X: Feature matrix (n_samples, n_features)
            y: Labels (required for statistical method)
            feature_names: Optional feature names
            statistical_results: Pre-computed results from StatisticalAnalyzer.compare_groups()

        Returns:
            self
        """
        # Convert to numpy if DataFrame
        if isinstance(X, pd.DataFrame):
            if feature_names is None:
                feature_names = X.columns.tolist()
            X = X.values

        n_samples, n_features = X.shape

        if feature_names is None:
            feature_names = [f'feature_{i}' for i in range(n_features)]

        self._all_feature_names = feature_names

        # Step 1: Variance filtering (always applied first)
        self._variance_mask = self._compute_variance_mask(X)

        # Step 2: Method-specific selection
        if self.method == 'statistical':
            self._fit_statistical(X, y, feature_names, statistical_results)
        elif self.method == 'variance':
            self._fit_variance_only(X, feature_names)
        elif self.method == 'correlation':
            self._fit_correlation(X, feature_names)
        elif self.method == 'combined':
            self._fit_combined(X, y, feature_names, statistical_results)
        else:
            raise ValueError(f"Unknown method: {self.method}")

        self._is_fitted = True
        return self

    def _compute_variance_mask(self, X: np.ndarray) -> np.ndarray:
        """Compute mask for features above variance threshold."""
        # Handle NaN values
        variances = np.nanvar(X, axis=0)
        return variances >= self.variance_threshold

    def _fit_statistical(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray],
        feature_names: List[str],
        statistical_results: Optional[pd.DataFrame]
    ) -> None:
        """Fit using statistical significance."""
        if statistical_results is None:
            if y is None:
                raise ValueError("Either 'y' or 'statistical_results' must be provided for statistical selection")

            # Compute statistics using internal method
            statistical_results = self._compute_statistics(X, y, feature_names)

        # Get threshold
        p_threshold = self.threshold if self.threshold is not None else 0.05

        # Use corrected p-values if available
        p_col = 'p_value_corrected' if 'p_value_corrected' in statistical_results.columns else 'p_value'

        # Filter by variance first
        variance_passed = [name for i, name in enumerate(feature_names) if self._variance_mask[i]]

        # Then filter by significance
        significant_mask = statistical_results[p_col] < p_threshold
        significant_features = statistical_results.loc[significant_mask, 'feature_name'].tolist()

        # Intersection with variance-passed features
        selected = [f for f in significant_features if f in variance_passed]

        # If n_features is specified, limit selection
        if self.n_features is not None and len(selected) > self.n_features:
            # Sort by p-value and take top n
            sorted_results = statistical_results[
                statistical_results['feature_name'].isin(selected)
            ].sort_values(p_col)
            selected = sorted_results['feature_name'].head(self.n_features).tolist()

        # If no significant features but n_features specified, take top n by p-value
        if len(selected) == 0 and self.n_features is not None:
            sorted_results = statistical_results[
                statistical_results['feature_name'].isin(variance_passed)
            ].sort_values(p_col)
            selected = sorted_results['feature_name'].head(self.n_features).tolist()

        self._selected_names = selected
        self._selected_indices = [feature_names.index(f) for f in selected]

        # Store scores (1 - p_value for ranking)
        self._feature_scores = np.zeros(len(feature_names))
        for _, row in statistical_results.iterrows():
            if row['feature_name'] in feature_names:
                idx = feature_names.index(row['feature_name'])
                p_val = row[p_col]
                self._feature_scores[idx] = 1 - p_val if not np.isnan(p_val) else 0

    def _fit_variance_only(
        self,
        X: np.ndarray,
        feature_names: List[str]
    ) -> None:
        """Fit using only variance threshold."""
        selected_indices = np.where(self._variance_mask)[0]

        if self.n_features is not None and len(selected_indices) > self.n_features:
            # Sort by variance and take top n
            variances = np.nanvar(X, axis=0)
            top_indices = np.argsort(variances)[::-1][:self.n_features]
            selected_indices = sorted(top_indices)

        self._selected_indices = selected_indices.tolist()
        self._selected_names = [feature_names[i] for i in selected_indices]
        self._feature_scores = np.nanvar(X, axis=0)

    def _fit_correlation(
        self,
        X: np.ndarray,
        feature_names: List[str]
    ) -> None:
        """Fit by removing highly correlated features."""
        corr_threshold = self.threshold if self.threshold is not None else 0.95

        # Start with variance-filtered features
        variance_indices = np.where(self._variance_mask)[0]
        X_filtered = X[:, variance_indices]
        filtered_names = [feature_names[i] for i in variance_indices]

        # Compute correlation matrix
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            corr_matrix = np.corrcoef(X_filtered.T)

        # Handle NaN in correlation matrix
        corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)

        # Identify features to drop (keep first in each correlated pair)
        n_filtered = len(filtered_names)
        to_drop = set()

        for i in range(n_filtered):
            if i in to_drop:
                continue
            for j in range(i + 1, n_filtered):
                if j in to_drop:
                    continue
                if abs(corr_matrix[i, j]) >= corr_threshold:
                    to_drop.add(j)

        # Selected features
        selected_filtered_indices = [i for i in range(n_filtered) if i not in to_drop]
        self._selected_names = [filtered_names[i] for i in selected_filtered_indices]
        self._selected_indices = [variance_indices[i] for i in selected_filtered_indices]

        # Scores based on mean correlation (lower is better for diversity)
        mean_corr = np.nanmean(np.abs(corr_matrix), axis=1)
        self._feature_scores = 1 - mean_corr  # Higher score = less correlated

    def _fit_combined(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray],
        feature_names: List[str],
        statistical_results: Optional[pd.DataFrame]
    ) -> None:
        """Combined approach: variance -> correlation -> statistical."""
        # Step 1: Variance filtering (already done)
        variance_indices = np.where(self._variance_mask)[0]

        if len(variance_indices) == 0:
            self._selected_indices = []
            self._selected_names = []
            self._feature_scores = np.zeros(len(feature_names))
            return

        # Step 2: Correlation filtering
        corr_threshold = 0.95
        X_filtered = X[:, variance_indices]
        filtered_names = [feature_names[i] for i in variance_indices]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            corr_matrix = np.corrcoef(X_filtered.T)
        corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)

        to_drop = set()
        for i in range(len(filtered_names)):
            if i in to_drop:
                continue
            for j in range(i + 1, len(filtered_names)):
                if j in to_drop:
                    continue
                if abs(corr_matrix[i, j]) >= corr_threshold:
                    to_drop.add(j)

        decorr_indices = [variance_indices[i] for i in range(len(filtered_names)) if i not in to_drop]
        decorr_names = [feature_names[i] for i in decorr_indices]

        # Step 3: Statistical filtering
        if statistical_results is not None or y is not None:
            if statistical_results is None:
                statistical_results = self._compute_statistics(X, y, feature_names)

            p_threshold = self.threshold if self.threshold is not None else 0.05
            p_col = 'p_value_corrected' if 'p_value_corrected' in statistical_results.columns else 'p_value'

            significant_mask = statistical_results[p_col] < p_threshold
            significant_features = statistical_results.loc[significant_mask, 'feature_name'].tolist()

            # Intersection
            selected = [f for f in significant_features if f in decorr_names]

            if len(selected) == 0 and self.n_features is not None:
                # Take top n by p-value from decorrelated features
                sorted_results = statistical_results[
                    statistical_results['feature_name'].isin(decorr_names)
                ].sort_values(p_col)
                selected = sorted_results['feature_name'].head(self.n_features).tolist()

            self._selected_names = selected
            self._selected_indices = [feature_names.index(f) for f in selected]

            # Scores
            self._feature_scores = np.zeros(len(feature_names))
            for _, row in statistical_results.iterrows():
                if row['feature_name'] in feature_names:
                    idx = feature_names.index(row['feature_name'])
                    p_val = row[p_col]
                    self._feature_scores[idx] = 1 - p_val if not np.isnan(p_val) else 0
        else:
            # No statistical filtering, just use decorrelated features
            self._selected_names = decorr_names
            self._selected_indices = decorr_indices
            self._feature_scores = np.nanvar(X, axis=0)

    def _compute_statistics(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str]
    ) -> pd.DataFrame:
        """Compute statistics internally when StatisticalAnalyzer results not provided."""
        from scipy import stats

        results = []
        unique_labels = np.unique(y[~pd.isna(y)])

        for i, name in enumerate(feature_names):
            values = X[:, i]

            if len(unique_labels) == 2:
                # Binary: Mann-Whitney U
                g1 = values[y == unique_labels[0]]
                g2 = values[y == unique_labels[1]]
                g1 = g1[~np.isnan(g1)]
                g2 = g2[~np.isnan(g2)]

                if len(g1) > 0 and len(g2) > 0:
                    try:
                        _, pval = stats.mannwhitneyu(g1, g2)
                    except ValueError:
                        pval = 1.0
                else:
                    pval = np.nan
            else:
                # Multi-group: Kruskal-Wallis
                groups = []
                for label in unique_labels:
                    g = values[y == label]
                    g = g[~np.isnan(g)]
                    if len(g) > 0:
                        groups.append(g)

                if len(groups) >= 2:
                    try:
                        _, pval = stats.kruskal(*groups)
                    except ValueError:
                        pval = 1.0
                else:
                    pval = np.nan

            results.append({
                'feature_name': name,
                'p_value': pval
            })

        return pd.DataFrame(results)

    def transform(
        self,
        X: Union[np.ndarray, pd.DataFrame]
    ) -> Union[np.ndarray, pd.DataFrame]:
        """
        Transform features (select subset).

        Args:
            X: Feature matrix

        Returns:
            Reduced feature matrix
        """
        if not self._is_fitted:
            raise ValueError("Selector not fitted. Call fit() first.")

        return_df = isinstance(X, pd.DataFrame)

        if return_df:
            original_cols = X.columns.tolist()
            X_array = X.values
        else:
            X_array = X

        if len(self._selected_indices) == 0:
            warnings.warn("No features selected, returning empty array")
            if return_df:
                return pd.DataFrame()
            return np.array([]).reshape(len(X_array), 0)

        X_selected = X_array[:, self._selected_indices]

        if return_df:
            return pd.DataFrame(X_selected, columns=self._selected_names)
        return X_selected

    def fit_transform(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None,
        statistical_results: Optional[pd.DataFrame] = None
    ) -> Union[np.ndarray, pd.DataFrame]:
        """
        Fit and transform in one step.

        Args:
            X: Feature matrix
            y: Labels
            feature_names: Feature names
            statistical_results: Pre-computed statistical results

        Returns:
            Reduced feature matrix
        """
        self.fit(X, y, feature_names, statistical_results)
        return self.transform(X)

    def get_selected_features(self) -> List[int]:
        """
        Get indices of selected features.

        Returns:
            List of feature indices
        """
        if not self._is_fitted:
            raise ValueError("Selector not fitted. Call fit() first.")
        return self._selected_indices.copy()

    def get_selected_feature_names(
        self,
        feature_names: Optional[List[str]] = None
    ) -> List[str]:
        """
        Get names of selected features.

        Args:
            feature_names: Original feature names (uses stored names if None)

        Returns:
            Selected feature names
        """
        if not self._is_fitted:
            raise ValueError("Selector not fitted. Call fit() first.")

        if self._selected_names is not None:
            return self._selected_names.copy()

        if feature_names is None:
            raise ValueError("Feature names not available")

        return [feature_names[i] for i in self._selected_indices]

    def get_feature_scores(self) -> np.ndarray:
        """
        Get feature importance scores.

        Returns:
            Array of scores (higher = more important)
        """
        if not self._is_fitted:
            raise ValueError("Selector not fitted. Call fit() first.")
        return self._feature_scores.copy()

    def get_feature_ranking(self, feature_names: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Get features ranked by importance score.

        Args:
            feature_names: Feature names

        Returns:
            DataFrame with feature rankings
        """
        if not self._is_fitted:
            raise ValueError("Selector not fitted. Call fit() first.")

        if feature_names is None:
            feature_names = self._all_feature_names

        ranking = pd.DataFrame({
            'feature_name': feature_names,
            'score': self._feature_scores,
            'selected': [i in self._selected_indices for i in range(len(feature_names))],
            'variance_passed': self._variance_mask
        })

        return ranking.sort_values('score', ascending=False).reset_index(drop=True)

    @staticmethod
    def from_statistical_results(
        results: pd.DataFrame,
        alpha: float = 0.05,
        n_features: Optional[int] = None,
        use_corrected: bool = True
    ) -> 'FeatureSelector':
        """
        Create selector directly from StatisticalAnalyzer results.

        Convenience method for quick feature selection based on statistical results.

        Args:
            results: DataFrame from StatisticalAnalyzer.compare_groups()
            alpha: Significance threshold
            n_features: Maximum features to select
            use_corrected: Use corrected p-values

        Returns:
            Fitted FeatureSelector
        """
        selector = FeatureSelector(
            method='statistical',
            threshold=alpha,
            n_features=n_features
        )

        p_col = 'p_value_corrected' if use_corrected and 'p_value_corrected' in results.columns else 'p_value'

        # Get significant features
        significant_mask = results[p_col] < alpha
        significant_features = results.loc[significant_mask, 'feature_name'].tolist()

        if n_features is not None and len(significant_features) > n_features:
            sorted_results = results[significant_mask].sort_values(p_col)
            significant_features = sorted_results['feature_name'].head(n_features).tolist()

        # Create pseudo-fit state
        all_features = results['feature_name'].tolist()
        selector._all_feature_names = all_features
        selector._selected_names = significant_features
        selector._selected_indices = [all_features.index(f) for f in significant_features]
        selector._variance_mask = np.ones(len(all_features), dtype=bool)
        selector._feature_scores = np.array([
            1 - results.loc[results['feature_name'] == f, p_col].values[0]
            if f in all_features else 0
            for f in all_features
        ])
        selector._is_fitted = True

        return selector


class RecursiveFeatureElimination:
    """
    Recursive feature elimination with cross-validation.

    Iteratively removes least important features.
    Wrapper around sklearn's RFECV.
    """

    def __init__(
        self,
        estimator,
        n_features_to_select: Optional[int] = None,
        step: int = 1,
        cv: int = 5,
        scoring: str = 'accuracy'
    ):
        """
        Initialize RFE.

        Args:
            estimator: Model with feature_importances_ or coef_ attribute
            n_features_to_select: Number of features to select (None = auto via CV)
            step: Number of features to remove each iteration
            cv: Cross-validation folds
            scoring: Scoring metric
        """
        self.estimator = estimator
        self.n_features_to_select = n_features_to_select
        self.step = step
        self.cv = cv
        self.scoring = scoring

        self._rfe = None
        self._selected_indices = None
        self._selected_names = None
        self._is_fitted = False

    def fit(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: np.ndarray,
        feature_names: Optional[List[str]] = None
    ) -> 'RecursiveFeatureElimination':
        """Fit RFE."""
        try:
            from sklearn.feature_selection import RFECV, RFE
        except ImportError:
            raise ImportError("scikit-learn required for RFE. Install with: pip install scikit-learn")

        if isinstance(X, pd.DataFrame):
            if feature_names is None:
                feature_names = X.columns.tolist()
            X = X.values

        if feature_names is None:
            feature_names = [f'feature_{i}' for i in range(X.shape[1])]

        self._all_feature_names = feature_names

        # Handle NaN in features
        X_clean = np.nan_to_num(X, nan=0.0)

        if self.n_features_to_select is None:
            # Use RFECV for automatic selection
            self._rfe = RFECV(
                estimator=self.estimator,
                step=self.step,
                cv=self.cv,
                scoring=self.scoring,
                min_features_to_select=1
            )
        else:
            # Use RFE with specified number
            self._rfe = RFE(
                estimator=self.estimator,
                n_features_to_select=self.n_features_to_select,
                step=self.step
            )

        self._rfe.fit(X_clean, y)

        self._selected_indices = np.where(self._rfe.support_)[0].tolist()
        self._selected_names = [feature_names[i] for i in self._selected_indices]
        self._is_fitted = True

        return self

    def transform(self, X: Union[np.ndarray, pd.DataFrame]) -> Union[np.ndarray, pd.DataFrame]:
        """Transform features."""
        if not self._is_fitted:
            raise ValueError("RFE not fitted. Call fit() first.")

        return_df = isinstance(X, pd.DataFrame)
        if return_df:
            X = X.values

        X_clean = np.nan_to_num(X, nan=0.0)
        X_selected = self._rfe.transform(X_clean)

        if return_df:
            return pd.DataFrame(X_selected, columns=self._selected_names)
        return X_selected

    def fit_transform(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: np.ndarray,
        feature_names: Optional[List[str]] = None
    ) -> Union[np.ndarray, pd.DataFrame]:
        """Fit and transform."""
        self.fit(X, y, feature_names)
        return self.transform(X)

    def get_selected_features(self) -> List[int]:
        """Get selected feature indices."""
        if not self._is_fitted:
            raise ValueError("RFE not fitted. Call fit() first.")
        return self._selected_indices.copy()

    def get_selected_feature_names(self) -> List[str]:
        """Get selected feature names."""
        if not self._is_fitted:
            raise ValueError("RFE not fitted. Call fit() first.")
        return self._selected_names.copy()

    def get_feature_ranking(self) -> np.ndarray:
        """Get feature ranking (1 = best)."""
        if not self._is_fitted:
            raise ValueError("RFE not fitted. Call fit() first.")
        return self._rfe.ranking_


class MutualInformationSelector:
    """
    Select features based on mutual information with target.
    """

    def __init__(
        self,
        n_features: Optional[int] = None,
        discrete_target: bool = True,
        random_state: Optional[int] = None
    ):
        """
        Initialize MI selector.

        Args:
            n_features: Number of features to select
            discrete_target: Whether target is discrete (classification)
            random_state: Random seed
        """
        self.n_features = n_features
        self.discrete_target = discrete_target
        self.random_state = random_state

        self._mi_scores = None
        self._selected_indices = None
        self._selected_names = None
        self._is_fitted = False

    def fit(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: np.ndarray,
        feature_names: Optional[List[str]] = None
    ) -> 'MutualInformationSelector':
        """Fit MI selector."""
        try:
            from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
        except ImportError:
            raise ImportError("scikit-learn required. Install with: pip install scikit-learn")

        if isinstance(X, pd.DataFrame):
            if feature_names is None:
                feature_names = X.columns.tolist()
            X = X.values

        if feature_names is None:
            feature_names = [f'feature_{i}' for i in range(X.shape[1])]

        self._all_feature_names = feature_names

        # Handle NaN
        X_clean = np.nan_to_num(X, nan=0.0)

        # Compute mutual information
        if self.discrete_target:
            self._mi_scores = mutual_info_classif(
                X_clean, y, random_state=self.random_state
            )
        else:
            self._mi_scores = mutual_info_regression(
                X_clean, y, random_state=self.random_state
            )

        # Select top features
        if self.n_features is None:
            # Select features with MI > 0
            self._selected_indices = np.where(self._mi_scores > 0)[0].tolist()
        else:
            # Select top n
            top_indices = np.argsort(self._mi_scores)[::-1][:self.n_features]
            self._selected_indices = sorted(top_indices.tolist())

        self._selected_names = [feature_names[i] for i in self._selected_indices]
        self._is_fitted = True

        return self

    def transform(self, X: Union[np.ndarray, pd.DataFrame]) -> Union[np.ndarray, pd.DataFrame]:
        """Transform features."""
        if not self._is_fitted:
            raise ValueError("Selector not fitted. Call fit() first.")

        return_df = isinstance(X, pd.DataFrame)
        if return_df:
            X = X.values

        X_selected = X[:, self._selected_indices]

        if return_df:
            return pd.DataFrame(X_selected, columns=self._selected_names)
        return X_selected

    def fit_transform(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: np.ndarray,
        feature_names: Optional[List[str]] = None
    ) -> Union[np.ndarray, pd.DataFrame]:
        """Fit and transform."""
        self.fit(X, y, feature_names)
        return self.transform(X)

    def get_selected_features(self) -> List[int]:
        """Get selected feature indices."""
        if not self._is_fitted:
            raise ValueError("Selector not fitted. Call fit() first.")
        return self._selected_indices.copy()

    def get_selected_feature_names(self) -> List[str]:
        """Get selected feature names."""
        if not self._is_fitted:
            raise ValueError("Selector not fitted. Call fit() first.")
        return self._selected_names.copy()

    def get_feature_scores(self) -> np.ndarray:
        """Get mutual information scores."""
        if not self._is_fitted:
            raise ValueError("Selector not fitted. Call fit() first.")
        return self._mi_scores.copy()

    def get_feature_ranking(self) -> pd.DataFrame:
        """Get features ranked by MI score."""
        if not self._is_fitted:
            raise ValueError("Selector not fitted. Call fit() first.")

        return pd.DataFrame({
            'feature_name': self._all_feature_names,
            'mi_score': self._mi_scores,
            'selected': [i in self._selected_indices for i in range(len(self._all_feature_names))]
        }).sort_values('mi_score', ascending=False).reset_index(drop=True)
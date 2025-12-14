"""
Statistical Analysis

Statistical tests and comparisons for respiratory features across groups/outcomes.

Features:
- Binary and multi-class comparison (auto-detected)
- Mann-Whitney U (default) or t-test (configurable)
- Multiple comparison correction (FDR, Bonferroni)
- Effect size computation (Cohen's d, rank-biserial correlation)
- Power analysis
- Correlation analysis
- Violin plot data preparation
- Feature matrix organization for window-level visualization
"""

from typing import List, Dict, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
import pandas as pd
import numpy as np
from scipy import stats
import warnings


@dataclass
class ViolinPlotData:
    """
    Data structure for violin plot visualization.

    Prepared by StatisticalAnalyzer for use by visualization layer.
    """
    feature_name: str
    group_data: Dict[Any, np.ndarray]  # group_label -> values
    group_labels: List[Any]
    outcome_name: str
    statistics: Dict[str, float] = field(default_factory=dict)  # p_value, effect_size, etc.


@dataclass
class FeatureMatrixData:
    """
    Data structure for feature matrix visualization (features x time windows).

    Shows feature values across time windows for a single recording or subject.
    """
    subject_id: str
    feature_names: List[str]
    window_times: List[Tuple[float, float]]  # (start_time, end_time) for each window
    values: np.ndarray  # Shape: (n_features, n_windows)
    window_duration: float = 300.0  # Default 5 minutes
    recording_date: Optional[str] = None


class StatisticalTest:
    """
    Base class for statistical tests.

    Wraps scipy.stats tests with consistent interface.
    """

    def __init__(self, name: str, parametric: bool = True):
        """
        Initialize statistical test.

        Args:
            name: Name of the test
            parametric: Whether test is parametric
        """
        self.name = name
        self.parametric = parametric

    def test(
        self,
        group1: np.ndarray,
        group2: np.ndarray,
        **kwargs
    ) -> Tuple[float, float]:
        """
        Perform statistical test.

        Args:
            group1: First group data
            group2: Second group data

        Returns:
            Tuple of (test_statistic, p_value)
        """
        raise NotImplementedError("Subclasses must implement test()")


class TTestIndependent(StatisticalTest):
    """Independent samples t-test."""

    def __init__(self, equal_var: bool = False):
        """
        Initialize t-test.

        Args:
            equal_var: Assume equal variance (False = Welch's t-test)
        """
        super().__init__("t-test", parametric=True)
        self.equal_var = equal_var

    def test(
        self,
        group1: np.ndarray,
        group2: np.ndarray,
        **kwargs
    ) -> Tuple[float, float]:
        """Perform independent samples t-test."""
        # Remove NaN values
        g1 = group1[~np.isnan(group1)]
        g2 = group2[~np.isnan(group2)]

        if len(g1) < 2 or len(g2) < 2:
            return np.nan, np.nan

        stat, pval = stats.ttest_ind(g1, g2, equal_var=self.equal_var)
        return stat, pval


class MannWhitneyU(StatisticalTest):
    """Mann-Whitney U test (non-parametric)."""

    def __init__(self, alternative: str = 'two-sided'):
        """
        Initialize Mann-Whitney U test.

        Args:
            alternative: 'two-sided', 'less', or 'greater'
        """
        super().__init__("Mann-Whitney U", parametric=False)
        self.alternative = alternative

    def test(
        self,
        group1: np.ndarray,
        group2: np.ndarray,
        **kwargs
    ) -> Tuple[float, float]:
        """Perform Mann-Whitney U test."""
        # Remove NaN values
        g1 = group1[~np.isnan(group1)]
        g2 = group2[~np.isnan(group2)]

        if len(g1) < 1 or len(g2) < 1:
            return np.nan, np.nan

        try:
            stat, pval = stats.mannwhitneyu(g1, g2, alternative=self.alternative)
            return stat, pval
        except ValueError:
            # All values are identical
            return np.nan, 1.0


class KruskalWallis(StatisticalTest):
    """Kruskal-Wallis H-test for multiple groups (non-parametric)."""

    def __init__(self):
        super().__init__("Kruskal-Wallis", parametric=False)

    def test(self, *groups, **kwargs) -> Tuple[float, float]:
        """
        Perform Kruskal-Wallis test.

        Args:
            *groups: Variable number of group arrays

        Returns:
            Tuple of (H_statistic, p_value)
        """
        # Remove NaN values from each group
        clean_groups = []
        for g in groups:
            g_array = np.asarray(g)
            clean = g_array[~np.isnan(g_array)]
            if len(clean) > 0:
                clean_groups.append(clean)

        if len(clean_groups) < 2:
            return np.nan, np.nan

        try:
            stat, pval = stats.kruskal(*clean_groups)
            return stat, pval
        except ValueError:
            return np.nan, 1.0


class ANOVA(StatisticalTest):
    """One-way ANOVA for multiple groups (parametric)."""

    def __init__(self):
        super().__init__("ANOVA", parametric=True)

    def test(self, *groups, **kwargs) -> Tuple[float, float]:
        """
        Perform one-way ANOVA.

        Args:
            *groups: Variable number of group arrays

        Returns:
            Tuple of (F_statistic, p_value)
        """
        # Remove NaN values from each group
        clean_groups = []
        for g in groups:
            g_array = np.asarray(g)
            clean = g_array[~np.isnan(g_array)]
            if len(clean) >= 2:
                clean_groups.append(clean)

        if len(clean_groups) < 2:
            return np.nan, np.nan

        try:
            stat, pval = stats.f_oneway(*clean_groups)
            return stat, pval
        except ValueError:
            return np.nan, 1.0


class StatisticalAnalyzer:
    """
    Perform statistical analysis of features across groups.

    Main operations:
    - Compare feature distributions between outcome groups
    - Identify significant features
    - Multiple comparison correction
    - Effect size computation
    - Prepare data for violin plots
    - Generate statistical reports

    Default: Mann-Whitney U test (non-parametric)
    Can switch to t-test via configuration.
    """

    def __init__(
        self,
        test: str = 'mannwhitney',
        correction_method: str = 'fdr_bh',
        alpha: float = 0.05,
        config: Optional[Dict] = None
    ):
        """
        Initialize statistical analyzer.

        Args:
            test: Statistical test ('mannwhitney', 'ttest', 'auto')
                  - 'mannwhitney': Mann-Whitney U (default, non-parametric)
                  - 'ttest': Independent samples t-test (parametric)
                  - 'auto': Choose based on normality test
            correction_method: Multiple comparison correction
                              ('bonferroni', 'fdr_bh', 'fdr_by', 'holm', None)
            alpha: Significance threshold
            config: Optional configuration dictionary
        """
        self.config = config or {}

        # Override with config if provided
        stat_config = self.config.get('analysis', {}).get('statistical', {})
        self.test_name = stat_config.get('test', test)
        self.correction_method = stat_config.get('correction', correction_method)
        self.alpha = stat_config.get('alpha', alpha)

        # Initialize test objects
        self._binary_test = self._create_test(self.test_name)
        self._multi_test_parametric = ANOVA()
        self._multi_test_nonparametric = KruskalWallis()

    def _create_test(self, test_name: str) -> StatisticalTest:
        """Create appropriate statistical test object."""
        if test_name.lower() in ['mannwhitney', 'mannwhitneyu', 'mann-whitney']:
            return MannWhitneyU()
        elif test_name.lower() in ['ttest', 't-test', 't_test']:
            return TTestIndependent(equal_var=False)
        else:
            # Default to Mann-Whitney
            warnings.warn(f"Unknown test '{test_name}', using Mann-Whitney U")
            return MannWhitneyU()

    def _detect_n_groups(self, labels: np.ndarray) -> int:
        """Detect number of unique groups in labels."""
        unique_labels = np.unique(labels[~pd.isna(labels)])
        return len(unique_labels)

    def _is_binary(self, labels: np.ndarray) -> bool:
        """Check if outcome is binary."""
        return self._detect_n_groups(labels) == 2

    def compare_groups(
        self,
        features_df: pd.DataFrame,
        labels: np.ndarray,
        outcome_name: str = 'outcome',
        feature_names: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Compare feature distributions between outcome groups.

        Automatically detects binary vs multi-class and applies
        appropriate test.

        Args:
            features_df: DataFrame with features (rows=samples, cols=features)
            labels: Array of group labels (same length as features_df rows)
            outcome_name: Name of outcome for reporting
            feature_names: Specific features to analyze (None = all numeric)

        Returns:
            DataFrame with columns:
            - feature_name
            - group1_mean, group1_std, group1_n (for binary)
            - group2_mean, group2_std, group2_n (for binary)
            - test_statistic
            - p_value
            - p_value_corrected
            - effect_size
            - significant (bool)
            - test_used
        """
        # Get feature columns
        if feature_names is None:
            feature_names = features_df.select_dtypes(include=[np.number]).columns.tolist()

        # Validate inputs
        if len(labels) != len(features_df):
            raise ValueError(
                f"Labels length ({len(labels)}) must match features length ({len(features_df)})"
            )

        # Detect number of groups
        n_groups = self._detect_n_groups(labels)
        is_binary = n_groups == 2

        if n_groups < 2:
            raise ValueError(f"Need at least 2 groups for comparison, got {n_groups}")

        # Get unique group labels
        unique_labels = sorted([l for l in np.unique(labels) if not pd.isna(l)])

        # Perform comparisons
        results = []

        for feature_name in feature_names:
            if feature_name not in features_df.columns:
                warnings.warn(f"Feature '{feature_name}' not in DataFrame, skipping")
                continue

            feature_values = features_df[feature_name].values

            if is_binary:
                result = self._compare_binary(
                    feature_values, labels, feature_name,
                    unique_labels[0], unique_labels[1]
                )
            else:
                result = self._compare_multi_group(
                    feature_values, labels, feature_name, unique_labels
                )

            result['outcome'] = outcome_name
            results.append(result)

        # Create results DataFrame
        results_df = pd.DataFrame(results)

        # Apply multiple comparison correction
        if len(results_df) > 0 and self.correction_method:
            p_values = results_df['p_value'].values
            corrected = self.correct_multiple_comparisons(p_values)
            results_df['p_value_corrected'] = corrected
            results_df['significant'] = corrected < self.alpha
        elif len(results_df) > 0:
            results_df['p_value_corrected'] = results_df['p_value']
            results_df['significant'] = results_df['p_value'] < self.alpha

        return results_df

    def _compare_binary(
        self,
        feature_values: np.ndarray,
        labels: np.ndarray,
        feature_name: str,
        label1: Any,
        label2: Any
    ) -> Dict[str, Any]:
        """Compare feature between two groups."""
        # Split by group
        mask1 = labels == label1
        mask2 = labels == label2

        group1 = feature_values[mask1]
        group2 = feature_values[mask2]

        # Clean NaN
        group1_clean = group1[~np.isnan(group1)]
        group2_clean = group2[~np.isnan(group2)]

        # Compute statistics
        stat, pval = self._binary_test.test(group1, group2)

        # Compute effect size
        effect_size = self.compute_effect_size(group1_clean, group2_clean)

        return {
            'feature_name': feature_name,
            'group1_label': label1,
            'group1_mean': np.nanmean(group1) if len(group1) > 0 else np.nan,
            'group1_std': np.nanstd(group1) if len(group1) > 0 else np.nan,
            'group1_median': np.nanmedian(group1) if len(group1) > 0 else np.nan,
            'group1_n': len(group1_clean),
            'group2_label': label2,
            'group2_mean': np.nanmean(group2) if len(group2) > 0 else np.nan,
            'group2_std': np.nanstd(group2) if len(group2) > 0 else np.nan,
            'group2_median': np.nanmedian(group2) if len(group2) > 0 else np.nan,
            'group2_n': len(group2_clean),
            'test_statistic': stat,
            'p_value': pval,
            'effect_size': effect_size,
            'test_used': self._binary_test.name,
            'n_groups': 2
        }

    def _compare_multi_group(
        self,
        feature_values: np.ndarray,
        labels: np.ndarray,
        feature_name: str,
        unique_labels: List[Any]
    ) -> Dict[str, Any]:
        """Compare feature across multiple groups."""
        # Split by group
        groups = []
        group_stats = {}

        for label in unique_labels:
            mask = labels == label
            group = feature_values[mask]
            group_clean = group[~np.isnan(group)]
            groups.append(group)

            group_stats[f'group_{label}_mean'] = np.nanmean(group) if len(group) > 0 else np.nan
            group_stats[f'group_{label}_std'] = np.nanstd(group) if len(group) > 0 else np.nan
            group_stats[f'group_{label}_n'] = len(group_clean)

        # Use non-parametric test by default for multi-group
        if self._binary_test.parametric:
            test = self._multi_test_parametric
        else:
            test = self._multi_test_nonparametric

        stat, pval = test.test(*groups)

        # Effect size for multi-group (eta-squared approximation)
        effect_size = self._compute_eta_squared(groups)

        result = {
            'feature_name': feature_name,
            'test_statistic': stat,
            'p_value': pval,
            'effect_size': effect_size,
            'test_used': test.name,
            'n_groups': len(unique_labels),
            'group_labels': unique_labels
        }
        result.update(group_stats)

        return result

    def compare_feature(
        self,
        feature_values: np.ndarray,
        labels_array: np.ndarray,
        feature_name: str
    ) -> Dict[str, Any]:
        """
        Compare single feature across groups.

        Args:
            feature_values: Feature values
            labels_array: Group labels
            feature_name: Name of feature

        Returns:
            Dictionary with test results
        """
        unique_labels = sorted([l for l in np.unique(labels_array) if not pd.isna(l)])

        if len(unique_labels) == 2:
            return self._compare_binary(
                feature_values, labels_array, feature_name,
                unique_labels[0], unique_labels[1]
            )
        else:
            return self._compare_multi_group(
                feature_values, labels_array, feature_name, unique_labels
            )

    def select_significant_features(
        self,
        results: pd.DataFrame,
        alpha: Optional[float] = None,
        use_corrected: bool = True
    ) -> List[str]:
        """
        Select features with significant p-values.

        Args:
            results: Results DataFrame from compare_groups()
            alpha: Significance threshold (None = use self.alpha)
            use_corrected: Use corrected p-values

        Returns:
            List of significant feature names
        """
        if alpha is None:
            alpha = self.alpha

        p_col = 'p_value_corrected' if use_corrected and 'p_value_corrected' in results.columns else 'p_value'

        mask = results[p_col] < alpha
        return results.loc[mask, 'feature_name'].tolist()

    def correct_multiple_comparisons(
        self,
        p_values: np.ndarray,
        method: Optional[str] = None
    ) -> np.ndarray:
        """
        Apply multiple comparison correction.

        Args:
            p_values: Array of p-values
            method: Correction method (None = use self.correction_method)
                   Options: 'bonferroni', 'fdr_bh', 'fdr_by', 'holm', 'sidak'

        Returns:
            Corrected p-values
        """
        if method is None:
            method = self.correction_method

        if method is None:
            return p_values

        # Handle NaN values
        p_vals = np.asarray(p_values).copy()
        valid_mask = ~np.isnan(p_vals)

        if not np.any(valid_mask):
            return p_vals

        valid_p = p_vals[valid_mask]

        try:
            from statsmodels.stats.multitest import multipletests
            _, corrected, _, _ = multipletests(valid_p, method=method)
            p_vals[valid_mask] = corrected
        except ImportError:
            warnings.warn(
                "statsmodels not installed, using manual Bonferroni correction. "
                "Install statsmodels for better correction methods: pip install statsmodels"
            )
            # Manual Bonferroni
            n_tests = len(valid_p)
            corrected = np.minimum(valid_p * n_tests, 1.0)
            p_vals[valid_mask] = corrected

        return p_vals

    def compute_effect_size(
        self,
        group1: np.ndarray,
        group2: np.ndarray,
        method: str = 'auto'
    ) -> float:
        """
        Compute effect size.

        Args:
            group1: First group (cleaned, no NaN)
            group2: Second group (cleaned, no NaN)
            method: 'cohen_d', 'hedges_g', 'glass_delta', 'rank_biserial', 'auto'
                   'auto' uses rank-biserial for non-parametric, Cohen's d for parametric

        Returns:
            Effect size value
        """
        if len(group1) < 1 or len(group2) < 1:
            return np.nan

        if method == 'auto':
            method = 'rank_biserial' if not self._binary_test.parametric else 'cohen_d'

        if method == 'cohen_d':
            return self._cohens_d(group1, group2)
        elif method == 'hedges_g':
            return self._hedges_g(group1, group2)
        elif method == 'glass_delta':
            return self._glass_delta(group1, group2)
        elif method == 'rank_biserial':
            return self._rank_biserial(group1, group2)
        else:
            return self._cohens_d(group1, group2)

    def _cohens_d(self, group1: np.ndarray, group2: np.ndarray) -> float:
        """Compute Cohen's d effect size."""
        n1, n2 = len(group1), len(group2)

        if n1 < 1 or n2 < 1:
            return np.nan

        mean1, mean2 = np.mean(group1), np.mean(group2)
        var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)

        # Pooled standard deviation
        pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))

        if pooled_std == 0:
            return 0.0

        return (mean1 - mean2) / pooled_std

    def _hedges_g(self, group1: np.ndarray, group2: np.ndarray) -> float:
        """Compute Hedges' g (bias-corrected Cohen's d)."""
        d = self._cohens_d(group1, group2)
        n = len(group1) + len(group2)

        # Correction factor
        correction = 1 - (3 / (4 * n - 9))
        return d * correction

    def _glass_delta(self, group1: np.ndarray, group2: np.ndarray) -> float:
        """Compute Glass's delta (using control group std)."""
        if len(group2) < 2:
            return np.nan

        mean1, mean2 = np.mean(group1), np.mean(group2)
        std2 = np.std(group2, ddof=1)

        if std2 == 0:
            return 0.0

        return (mean1 - mean2) / std2

    def _rank_biserial(self, group1: np.ndarray, group2: np.ndarray) -> float:
        """Compute rank-biserial correlation (for Mann-Whitney U)."""
        n1, n2 = len(group1), len(group2)

        if n1 < 1 or n2 < 1:
            return np.nan

        try:
            u_stat, _ = stats.mannwhitneyu(group1, group2, alternative='two-sided')
            # Rank-biserial correlation: r = 1 - (2U)/(n1*n2)
            r = 1 - (2 * u_stat) / (n1 * n2)
            return r
        except ValueError:
            return np.nan

    def _compute_eta_squared(self, groups: List[np.ndarray]) -> float:
        """Compute eta-squared for multi-group comparison."""
        # Combine all groups
        all_data = np.concatenate([g[~np.isnan(g)] for g in groups])

        if len(all_data) == 0:
            return np.nan

        grand_mean = np.mean(all_data)

        # Between-group sum of squares
        ss_between = 0
        ss_total = np.sum((all_data - grand_mean) ** 2)

        for g in groups:
            g_clean = g[~np.isnan(g)]
            if len(g_clean) > 0:
                group_mean = np.mean(g_clean)
                ss_between += len(g_clean) * (group_mean - grand_mean) ** 2

        if ss_total == 0:
            return 0.0

        return ss_between / ss_total

    def prepare_violin_plot_data(
        self,
        features_df: pd.DataFrame,
        labels: np.ndarray,
        outcome_name: str,
        feature_names: Optional[List[str]] = None,
        include_statistics: bool = True
    ) -> List[ViolinPlotData]:
        """
        Prepare data for violin plot visualization.

        Args:
            features_df: DataFrame with features
            labels: Group labels
            outcome_name: Name of outcome
            feature_names: Features to include (None = all numeric)
            include_statistics: Include p-values and effect sizes

        Returns:
            List of ViolinPlotData objects for visualization layer
        """
        if feature_names is None:
            feature_names = features_df.select_dtypes(include=[np.number]).columns.tolist()

        unique_labels = sorted([l for l in np.unique(labels) if not pd.isna(l)])

        # Get statistics if requested
        if include_statistics:
            stats_df = self.compare_groups(features_df, labels, outcome_name, feature_names)
            stats_dict = {row['feature_name']: row.to_dict() for _, row in stats_df.iterrows()}
        else:
            stats_dict = {}

        violin_data_list = []

        for feature_name in feature_names:
            if feature_name not in features_df.columns:
                continue

            feature_values = features_df[feature_name].values

            # Split by group
            group_data = {}
            for label in unique_labels:
                mask = labels == label
                values = feature_values[mask]
                clean_values = values[~np.isnan(values)]
                group_data[label] = clean_values

            # Get statistics for this feature
            feature_stats = stats_dict.get(feature_name, {})

            violin_data = ViolinPlotData(
                feature_name=feature_name,
                group_data=group_data,
                group_labels=unique_labels,
                outcome_name=outcome_name,
                statistics={
                    'p_value': feature_stats.get('p_value', np.nan),
                    'p_value_corrected': feature_stats.get('p_value_corrected', np.nan),
                    'effect_size': feature_stats.get('effect_size', np.nan),
                    'significant': feature_stats.get('significant', False),
                    'test_used': feature_stats.get('test_used', '')
                }
            )

            violin_data_list.append(violin_data)

        return violin_data_list

    def prepare_feature_matrix_data(
        self,
        window_features: List[Dict[str, float]],
        subject_id: str,
        window_duration: float = 300.0,
        feature_names: Optional[List[str]] = None,
        recording_date: Optional[str] = None
    ) -> FeatureMatrixData:
        """
        Prepare feature matrix data for visualization.

        Organizes feature values across time windows for a single recording.

        Args:
            window_features: List of feature dictionaries (one per window)
            subject_id: Subject identifier
            window_duration: Duration of each window in seconds (default 5 minutes)
            feature_names: Features to include (None = all numeric)
            recording_date: Optional recording date

        Returns:
            FeatureMatrixData for visualization layer
        """
        if len(window_features) == 0:
            raise ValueError("No window features provided")

        # Get feature names
        if feature_names is None:
            # Get all numeric feature names from first window
            feature_names = [
                k for k, v in window_features[0].items()
                if isinstance(v, (int, float, np.number)) and k not in [
                    'SubjectID', 'RecordingDate', 'WindowIndex',
                    'StartTime', 'EndTime', 'Duration', 'N_Windows'
                ]
            ]

        n_features = len(feature_names)
        n_windows = len(window_features)

        # Build matrix
        values = np.zeros((n_features, n_windows))
        window_times = []

        for w_idx, window in enumerate(window_features):
            # Get window times
            start_time = window.get('StartTime', w_idx * window_duration)
            end_time = window.get('EndTime', start_time + window_duration)
            window_times.append((start_time, end_time))

            # Get feature values
            for f_idx, fname in enumerate(feature_names):
                values[f_idx, w_idx] = window.get(fname, np.nan)

        return FeatureMatrixData(
            subject_id=subject_id,
            feature_names=feature_names,
            window_times=window_times,
            values=values,
            window_duration=window_duration,
            recording_date=recording_date
        )

    def generate_report(
        self,
        results: pd.DataFrame,
        output_path: Optional[str] = None
    ) -> str:
        """
        Generate human-readable statistical report.

        Args:
            results: Results DataFrame from compare_groups()
            output_path: Optional path to save report

        Returns:
            Report text
        """
        lines = []
        lines.append("=" * 70)
        lines.append("STATISTICAL ANALYSIS REPORT")
        lines.append("=" * 70)
        lines.append("")

        # Summary
        n_features = len(results)
        n_significant = results['significant'].sum() if 'significant' in results.columns else 0

        lines.append(f"Total features analyzed: {n_features}")
        lines.append(f"Significant features (α = {self.alpha}): {n_significant}")
        lines.append(f"Statistical test: {self.test_name}")
        lines.append(f"Multiple comparison correction: {self.correction_method}")
        lines.append("")

        # Significant features
        if n_significant > 0:
            lines.append("-" * 70)
            lines.append("SIGNIFICANT FEATURES")
            lines.append("-" * 70)

            sig_features = results[results['significant']].sort_values('p_value_corrected')

            for _, row in sig_features.iterrows():
                lines.append(f"\n{row['feature_name']}:")
                lines.append(f"  p-value (corrected): {row['p_value_corrected']:.4e}")
                lines.append(f"  Effect size: {row['effect_size']:.3f}")

                if 'group1_mean' in row:
                    lines.append(f"  Group {row.get('group1_label', 1)}: "
                               f"mean={row['group1_mean']:.3f}, std={row['group1_std']:.3f}, n={row['group1_n']}")
                    lines.append(f"  Group {row.get('group2_label', 2)}: "
                               f"mean={row['group2_mean']:.3f}, std={row['group2_std']:.3f}, n={row['group2_n']}")

        lines.append("")
        lines.append("=" * 70)

        report = "\n".join(lines)

        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(report)

        return report


class PowerAnalysis:
    """
    Perform statistical power analysis.

    Useful for determining sample size requirements and
    interpreting non-significant results.
    """

    def __init__(self):
        """Initialize power analysis."""
        self._has_statsmodels = False
        try:
            from statsmodels.stats.power import TTestIndPower
            self._power_analysis = TTestIndPower()
            self._has_statsmodels = True
        except ImportError:
            warnings.warn(
                "statsmodels not installed. Power analysis will use approximations. "
                "Install with: pip install statsmodels"
            )

    def compute_power(
        self,
        effect_size: float,
        n_samples: int,
        alpha: float = 0.05,
        test_type: str = 'ttest'
    ) -> float:
        """
        Compute statistical power.

        Args:
            effect_size: Expected effect size (Cohen's d)
            n_samples: Sample size per group
            alpha: Significance level
            test_type: Type of test ('ttest', 'mannwhitney')

        Returns:
            Statistical power (0-1)
        """
        if self._has_statsmodels:
            from statsmodels.stats.power import TTestIndPower
            power_analysis = TTestIndPower()

            # For Mann-Whitney, use relative efficiency adjustment
            if test_type.lower() in ['mannwhitney', 'mann-whitney']:
                # ARE of Mann-Whitney relative to t-test is ~0.955 for normal data
                # Adjust sample size for equivalent power
                effective_n = n_samples * 0.955
                power = power_analysis.power(
                    effect_size=effect_size,
                    nobs1=effective_n,
                    alpha=alpha,
                    ratio=1.0
                )
            else:
                power = power_analysis.power(
                    effect_size=effect_size,
                    nobs1=n_samples,
                    alpha=alpha,
                    ratio=1.0
                )
            return power
        else:
            # Approximation using normal distribution
            return self._approximate_power(effect_size, n_samples, alpha)

    def required_sample_size(
        self,
        effect_size: float,
        power: float = 0.8,
        alpha: float = 0.05,
        test_type: str = 'ttest'
    ) -> int:
        """
        Calculate required sample size.

        Args:
            effect_size: Expected effect size (Cohen's d)
            power: Desired power
            alpha: Significance level
            test_type: Type of test

        Returns:
            Required sample size per group
        """
        if self._has_statsmodels:
            from statsmodels.stats.power import TTestIndPower
            power_analysis = TTestIndPower()

            n = power_analysis.solve_power(
                effect_size=effect_size,
                power=power,
                alpha=alpha,
                ratio=1.0
            )

            # Adjust for Mann-Whitney
            if test_type.lower() in ['mannwhitney', 'mann-whitney']:
                n = n / 0.955

            return int(np.ceil(n))
        else:
            # Approximation
            return self._approximate_sample_size(effect_size, power, alpha)

    def _approximate_power(self, effect_size: float, n: int, alpha: float) -> float:
        """Approximate power using normal distribution."""
        se = np.sqrt(2 / n)
        ncp = effect_size / se  # Non-centrality parameter
        z_alpha = stats.norm.ppf(1 - alpha / 2)
        power = 1 - stats.norm.cdf(z_alpha - ncp) + stats.norm.cdf(-z_alpha - ncp)
        return power

    def _approximate_sample_size(self, effect_size: float, power: float, alpha: float) -> int:
        """Approximate sample size using iterative approach."""
        for n in range(5, 10000):
            p = self._approximate_power(effect_size, n, alpha)
            if p >= power:
                return n
        return 10000


class CorrelationAnalyzer:
    """
    Analyze correlations between features and outcomes.
    """

    def __init__(self, method: str = 'spearman'):
        """
        Initialize correlation analyzer.

        Args:
            method: 'pearson', 'spearman', or 'kendall'
        """
        self.method = method

    def compute_feature_outcome_correlations(
        self,
        features_df: pd.DataFrame,
        labels: np.ndarray,
        outcome_name: str = 'outcome'
    ) -> pd.DataFrame:
        """
        Compute correlations between features and outcome.

        Works for both continuous and ordinal outcomes.

        Args:
            features_df: DataFrame with features
            labels: Outcome values (numeric)
            outcome_name: Name of outcome

        Returns:
            DataFrame with feature correlations and p-values
        """
        feature_names = features_df.select_dtypes(include=[np.number]).columns.tolist()

        results = []

        for feature_name in feature_names:
            feature_values = features_df[feature_name].values

            # Remove NaN
            valid_mask = ~(np.isnan(feature_values) | pd.isna(labels))
            x = feature_values[valid_mask]
            y = np.asarray(labels)[valid_mask]

            if len(x) < 3:
                results.append({
                    'feature_name': feature_name,
                    'correlation': np.nan,
                    'p_value': np.nan
                })
                continue

            if self.method == 'pearson':
                corr, pval = stats.pearsonr(x, y)
            elif self.method == 'spearman':
                corr, pval = stats.spearmanr(x, y)
            elif self.method == 'kendall':
                corr, pval = stats.kendalltau(x, y)
            else:
                corr, pval = stats.spearmanr(x, y)

            results.append({
                'feature_name': feature_name,
                'correlation': corr,
                'p_value': pval,
                'abs_correlation': abs(corr)
            })

        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values('abs_correlation', ascending=False)
        results_df['outcome'] = outcome_name

        return results_df

    def compute_feature_correlations(
        self,
        features_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Compute pairwise feature correlations.

        Args:
            features_df: DataFrame with features

        Returns:
            Correlation matrix DataFrame
        """
        numeric_df = features_df.select_dtypes(include=[np.number])
        return numeric_df.corr(method=self.method)

    def find_highly_correlated_pairs(
        self,
        features_df: pd.DataFrame,
        threshold: float = 0.9
    ) -> List[Tuple[str, str, float]]:
        """
        Find pairs of highly correlated features.

        Args:
            features_df: DataFrame with features
            threshold: Correlation threshold

        Returns:
            List of (feature1, feature2, correlation) tuples
        """
        corr_matrix = self.compute_feature_correlations(features_df)

        pairs = []
        feature_names = corr_matrix.columns.tolist()

        for i in range(len(feature_names)):
            for j in range(i + 1, len(feature_names)):
                corr = corr_matrix.iloc[i, j]
                if abs(corr) >= threshold:
                    pairs.append((feature_names[i], feature_names[j], corr))

        # Sort by absolute correlation
        pairs.sort(key=lambda x: abs(x[2]), reverse=True)

        return pairs
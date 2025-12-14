import unittest
import numpy as np
import pandas as pd
from analysis.statistical import StatisticalAnalyzer
from analysis.feature_selector import FeatureSelector
from analysis.dimensionality import PCAReducer, TSNEReducer


class TestAnalysisLayer(unittest.TestCase):

    def setUp(self):
        """Create robust synthetic data for testing."""
        np.random.seed(42)
        self.n_samples = 100
        self.n_features = 20

        # --- FIX: Stronger Signal Generation ---
        # Instead of probabalistic generation, we explicitly shift the means
        # of the first 5 features for the second group.

        # Class 0: Mean 0, Std 1
        X_class0 = np.random.normal(0, 1, (self.n_samples // 2, self.n_features))
        y_class0 = np.zeros(self.n_samples // 2, dtype=int)

        # Class 1: Mean 0, Std 1 (but shift first 5 features)
        X_class1 = np.random.normal(0, 1, (self.n_samples // 2, self.n_features))
        # Huge shift (Effect Size > 2.0) to survive Bonferroni correction
        X_class1[:, :5] += 3.0
        y_class1 = np.ones(self.n_samples // 2, dtype=int)

        # Combine
        self.X = np.vstack([X_class0, X_class1])
        self.y = np.concatenate([y_class0, y_class1])

        # Shuffle to ensure randomness isn't just split by index
        idx = np.random.permutation(self.n_samples)
        self.X = self.X[idx]
        self.y = self.y[idx]

        self.feature_names = [f'feat_{i}' for i in range(self.n_features)]
        self.df = pd.DataFrame(self.X, columns=self.feature_names)

    def test_statistical_analyzer_binary(self):
        """Test binary group comparison and stats generation."""
        analyzer = StatisticalAnalyzer(test='mannwhitney', alpha=0.05)

        results = analyzer.compare_groups(
            self.df,
            self.y,
            outcome_name='test_outcome',
            feature_names=self.feature_names
        )

        # Check structure
        self.assertIsInstance(results, pd.DataFrame)
        self.assertIn('p_value', results.columns)
        self.assertIn('effect_size', results.columns)

        # Check logic: feat_0 should be significant (Mean 0 vs Mean 3 is huge)
        feat_0_row = results[results['feature_name'] == 'feat_0'].iloc[0]

        # Using bool() to convert np.bool_ to standard python bool for assertion messages
        is_significant = bool(feat_0_row['significant'])
        self.assertTrue(is_significant, f"Feature 0 should be significant. P-value: {feat_0_row['p_value']}")
        self.assertLess(feat_0_row['p_value'], 0.05)

    def test_feature_selector_pipeline(self):
        """Test the flow from Stats -> Selection -> Transformation."""
        # 1. Run Stats
        analyzer = StatisticalAnalyzer()
        stats_results = analyzer.compare_groups(self.df, self.y)

        # 2. Select Features
        # We explicitly ask for 5. Since we engineered 5 strong features,
        # it should find exactly these 5 (or at least 5 significant ones).
        selector = FeatureSelector(method='statistical', n_features=5)

        # Fit with pre-computed stats
        selector.fit(self.df, self.y, statistical_results=stats_results)

        selected_features = selector.get_selected_feature_names()

        # Debug print if it fails
        if len(selected_features) != 5:
            print(f"DEBUG: Selected features: {selected_features}")

        self.assertEqual(len(selected_features), 5)
        self.assertIn('feat_0', selected_features)

        # 3. Transform
        X_reduced = selector.transform(self.df)
        self.assertEqual(X_reduced.shape, (self.n_samples, 5))

    def test_dimensionality_reduction_pca(self):
        """Test PCA reduction and variance properties."""
        pca = PCAReducer(n_components=2)
        X_reduced = pca.fit_transform(self.df)

        self.assertEqual(X_reduced.shape, (self.n_samples, 2))

        # Check explained variance
        variance = pca.get_explained_variance()
        self.assertEqual(len(variance), 2)
        self.assertTrue(np.all(variance > 0))

        # Check checking logic (cumulative variance)
        n_needed = pca.select_n_components(self.df, variance_threshold=0.8)

        # Handle both Python int and NumPy integer types
        self.assertTrue(isinstance(n_needed, (int, np.integer)))

    def test_dimensionality_reduction_tsne(self):
        """Test t-SNE reduction (integrity check)."""
        # Low perplexity and iter for speed/stability on small test data
        tsne = TSNEReducer(n_components=2, perplexity=5, n_iter=250)
        X_reduced = tsne.fit_transform(self.df)

        self.assertEqual(X_reduced.shape, (self.n_samples, 2))


if __name__ == '__main__':
    unittest.main()
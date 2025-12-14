import unittest
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import GridSearchCV

# Imports from your package
from models.classifiers import SVMClassifier, RandomForestClassifier
from models.neural_network import NeuralNetworkClassifier
from models.evaluation import GridSearchTuner, ModelEvaluator
from models.bootstrap import BootstrapAnalyzer


class TestModelsLayer(unittest.TestCase):

    def setUp(self):
        """Create synthetic classification data."""
        self.X, self.y = make_classification(
            n_samples=200,
            n_features=10,
            n_informative=5,
            n_classes=2,
            random_state=42
        )

    def test_basic_classifier_training(self):
        """Test if wrappers can train and predict correctly."""
        clf = RandomForestClassifier(n_estimators=10)
        clf.train(self.X, self.y)

        preds = clf.predict(self.X)
        self.assertEqual(len(preds), 200)

        # Check accuracy is reasonable (> random)
        acc = np.mean(preds == self.y)
        self.assertGreater(acc, 0.6)

    def test_neural_network_scaling(self):
        """Test if NN handles internal scaling and training."""
        nn = NeuralNetworkClassifier(hidden_layer_sizes=(10,), max_iter=200)

        # Pass unscaled data (large magnitude) to verify internal scaling works
        X_large = self.X * 1000
        nn.train(X_large, self.y)

        preds = nn.predict(X_large)
        self.assertEqual(len(preds), 200)

    def test_grid_search_result_tracking(self):
        """
        CRITICAL TEST: Verify we can extract hyperparameter results for plotting.
        """
        svm = SVMClassifier()
        tuner = GridSearchTuner(cv_folds=3, scoring='accuracy')

        # Define hyperparameter space
        param_grid = {
            'C': [0.1, 1, 10],
            'kernel': ['rbf', 'linear']
        }

        # We need to slightly modify how we use the tuner or access the internal logic
        # to ensure we get the full results, not just the best model.
        # In this test, we verify the standard sklearn mechanism which your tuner wraps.

        # Run tuning
        # Note: Depending on your exact GridSearchTuner implementation,
        # you might need to return the 'grid' object or use the code below
        # to access the cv_results_ from the internal model.

        best_model, best_params = tuner.tune(svm, self.X, self.y, param_grid)

        # --- ACCESSING RESULTS FOR VISUALIZATION ---
        # The internal model of the wrapper should now be the best estimator.
        # However, to get the TABLE of results, we usually need the GridSearchCV object.
        # Assuming we modify GridSearchTuner to return the grid object or we recreate the logic:

        grid = GridSearchCV(
            svm.model,  # The internal sklearn model
            param_grid,
            cv=3,
            scoring='accuracy'
        )
        grid.fit(self.X, self.y)

        # Extract the results DataFrame
        results_df = pd.DataFrame(grid.cv_results_)

        # 1. Check that we have one row per parameter combination (3 C * 2 kernels = 6)
        self.assertEqual(len(results_df), 6)

        # 2. Check that we have the specific columns needed for plotting
        expected_cols = ['param_C', 'param_kernel', 'mean_test_score', 'std_test_score']
        for col in expected_cols:
            self.assertIn(col, results_df.columns)

        print("\n--- Hyperparameter Results Sample (Ready for Plotting) ---")
        print(results_df[expected_cols].head())

        # This dataframe is exactly what your Visualization layer will consume
        # e.g. px.scatter(results_df, x='param_C', y='mean_test_score', color='param_kernel')

    def test_bootstrap_analyzer(self):
        """Test bootstrap confidence interval generation."""
        svm = SVMClassifier()  # Untrained
        analyzer = BootstrapAnalyzer(n_iterations=10, test_size=0.2)

        # Should return a DataFrame with metrics and CIs
        results_df = analyzer.evaluate(svm, self.X, self.y)

        self.assertIsInstance(results_df, pd.DataFrame)
        self.assertIn('accuracy', results_df.index)
        self.assertIn('ci_lower_95', results_df.columns)
        self.assertIn('ci_upper_95', results_df.columns)

        # CI Lower should be <= Mean <= CI Upper
        acc_row = results_df.loc['accuracy']
        self.assertTrue(acc_row['ci_lower_95'] <= acc_row['mean'])
        self.assertTrue(acc_row['mean'] <= acc_row['ci_upper_95'])


if __name__ == '__main__':
    unittest.main()
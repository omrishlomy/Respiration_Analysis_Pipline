"""
Bootstrap Analysis

Performs bootstrap resampling to estimate model stability and
confidence intervals for performance metrics.
"""

from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import pandas as pd
from sklearn.utils import resample
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score
from copy import deepcopy
from .classifiers import Classifier


class BootstrapAnalyzer:
    """
    Evaluates classifier performance using Bootstrap resampling.

    Instead of a single train/test split, we train/test on N resampled datasets
    to generate distributions of performance metrics.
    """

    def __init__(
            self,
            n_iterations: int = 1000,
            test_size: float = 0.2,
            random_state: int = 42
    ):
        """
        Args:
            n_iterations: Number of bootstrap samples
            test_size: Proportion of data to leave out for testing in each iteration
            random_state: Seed for reproducibility
        """
        self.n_iterations = n_iterations
        self.test_size = test_size
        self.random_state = random_state
        self.results: List[Dict[str, float]] = []

    def evaluate(
            self,
            classifier: Classifier,
            X: np.ndarray,
            y: np.ndarray
    ) -> pd.DataFrame:
        """
        Run bootstrap evaluation.

        Args:
            classifier: An instance of a Classifier (will be cloned)
            X: Feature matrix
            y: Labels

        Returns:
            DataFrame containing summary statistics (mean, std, 95% CI) for each metric.
        """
        self.results = []
        n_samples = len(X)
        n_test = int(n_samples * self.test_size)
        n_train = n_samples - n_test

        print(f"Starting {self.n_iterations} bootstrap iterations for {classifier.name}...")

        for i in range(self.n_iterations):
            # 1. Resample indices
            # We construct a specific train/test split via resampling
            indices = np.arange(n_samples)

            # Stratified resampling for training set
            train_idx = resample(
                indices,
                n_samples=n_train,
                replace=True,
                stratify=y,
                random_state=self.random_state + i
            )

            # Use remaining samples for testing (Out-of-Bag approximate)
            # Or explicit new resampling if strict test set size needed
            mask = np.ones(n_samples, dtype=bool)
            mask[train_idx] = False
            test_idx = indices[mask]

            # If we don't have enough test samples due to overlap, fill up
            if len(test_idx) < 10:
                # Fallback: Standard Stratified Shuffle Split logic if OOB is too small
                from sklearn.model_selection import train_test_split
                train_idx, test_idx = train_test_split(
                    indices, test_size=self.test_size, stratify=y, random_state=self.random_state + i
                )

            X_train, y_train = X[train_idx], y[train_idx]
            X_test, y_test = X[test_idx], y[test_idx]

            # 2. Clone and Train Classifier
            # We use deepcopy to ensure a fresh model every time
            clf_iter = deepcopy(classifier)
            clf_iter.train(X_train, y_train)

            # 3. Predict
            y_pred = clf_iter.predict(X_test)
            try:
                y_prob = clf_iter.predict_proba(X_test)[:, 1]
            except (AttributeError, IndexError, NotImplementedError):
                # Some classifiers don't support predict_proba
                y_prob = None

            # 4. Calculate Metrics
            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'f1': f1_score(y_test, y_pred, average='weighted'),
                'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
                'recall': recall_score(y_test, y_pred, average='weighted')
            }

            if y_prob is not None and len(np.unique(y)) == 2:
                try:
                    metrics['auc'] = roc_auc_score(y_test, y_prob)
                except ValueError:
                    metrics['auc'] = np.nan

            self.results.append(metrics)

        return self._summarize_results()

    def _summarize_results(self) -> pd.DataFrame:
        """Calculate Mean, Std, and 95% Confidence Intervals."""
        df = pd.DataFrame(self.results)
        summary = []

        for col in df.columns:
            values = df[col].dropna()
            if len(values) == 0:
                continue

            mean_val = np.mean(values)
            std_val = np.std(values)
            # 95% CI
            ci_lower = np.percentile(values, 2.5)
            ci_upper = np.percentile(values, 97.5)

            summary.append({
                'metric': col,
                'mean': mean_val,
                'std': std_val,
                'ci_lower_95': ci_lower,
                'ci_upper_95': ci_upper
            })

        return pd.DataFrame(summary).set_index('metric')
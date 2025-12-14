"""
Model Evaluation and Tuning

Tools for hyperparameter tuning and standard model evaluation.
"""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from .classifiers import Classifier


class GridSearchTuner:
    """Hyperparameter tuner using Grid Search."""

    def __init__(self, cv_folds: int = 5, scoring: str = 'accuracy'):
        self.cv_folds = cv_folds
        self.scoring = scoring

    def tune(
            self,
            classifier: Classifier,
            X: np.ndarray,
            y: np.ndarray,
            param_grid: Dict[str, List[Any]]
    ) -> Tuple[Classifier, Dict[str, Any]]:
        """
        Perform grid search to find best parameters.

        Args:
            classifier: Classifier instance to tune
            X: Feature matrix
            y: Labels
            param_grid: Dictionary of parameters to try (e.g. {'n_neighbors': [3, 5, 7]})

        Returns:
            Tuple of (Trained Best Classifier, Best Params Dict)
        """
        print(f"Tuning {classifier.name}...")

        # We need to access the internal sklearn model
        # Note: We prefix params with 'model__' if we were using a Pipeline,
        # but here we are wrapping the model directly.
        # However, our Wrapper class doesn't inherit from BaseEstimator,
        # so we tune the internal model directly.

        grid = GridSearchCV(
            estimator=classifier.model,
            param_grid=param_grid,
            cv=self.cv_folds,
            scoring=self.scoring,
            n_jobs=-1,
            verbose=1
        )

        grid.fit(X, y)

        # Update the classifier with best estimator
        classifier.model = grid.best_estimator_
        classifier.is_trained = True
        classifier.best_params = grid.best_params_
        classifier.classes = grid.best_estimator_.classes_

        return classifier, grid.best_params_


class ModelEvaluator:
    """Standard model evaluator."""

    def evaluate(self, classifier: Classifier, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """
        Compute comprehensive performance metrics.
        """
        if not classifier.is_trained:
            raise ValueError("Classifier not trained")

        y_pred = classifier.predict(X_test)

        # Basic metrics
        report = classification_report(y_test, y_pred, output_dict=True)

        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)

        results = {
            'accuracy': report['accuracy'],
            'macro_f1': report['macro avg']['f1-score'],
            'confusion_matrix': cm
        }

        # AUC if applicable
        try:
            if hasattr(classifier, 'predict_proba'):
                y_prob = classifier.predict_proba(X_test)
                if len(classifier.classes) == 2:
                    results['auc'] = roc_auc_score(y_test, y_prob[:, 1])
                else:
                    results['auc'] = roc_auc_score(y_test, y_prob, multi_class='ovr')
        except Exception as e:
            results['auc'] = np.nan

        return results
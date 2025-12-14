"""
Classifiers

Wrappers for machine learning classifiers with a consistent interface.
"""

from abc import ABC, abstractmethod
from typing import Dict, Optional, Any, List, Union
import numpy as np
import warnings
import joblib

# Sklearn imports
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
# Alias RandomForest to avoid name collision with our wrapper class
from sklearn.ensemble import RandomForestClassifier as SkRandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator
from sklearn.model_selection import cross_validate

# Optional XGBoost
try:
    from xgboost import XGBClassifier as XGB
    HAS_XGB = True
except ImportError:
    HAS_XGB = False


class Classifier(ABC):
    """Abstract base class for all classifiers."""

    def __init__(self, name: str):
        self.name = name
        self.model: Optional[BaseEstimator] = None
        self.is_trained = False
        self.feature_names: Optional[List[str]] = None
        self.classes: Optional[np.ndarray] = None
        self.best_params: Optional[Dict] = None

    @abstractmethod
    def train(self, X: np.ndarray, y: np.ndarray, **kwargs) -> None:
        """Train the classifier."""
        pass

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make class predictions."""
        if not self.is_trained:
            raise ValueError(f"{self.name} is not trained.")
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        if not self.is_trained:
            raise ValueError(f"{self.name} is not trained.")

        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X)
        else:
            warnings.warn(f"{self.name} does not support probability prediction.")
            return np.zeros((X.shape[0], len(self.classes)))

    def cross_validate(self, X: np.ndarray, y: np.ndarray, cv: int = 5, scoring: str = 'accuracy') -> Dict[str, float]:
        """Perform cross-validation and return mean scores."""
        if self.model is None:
            raise ValueError("Model not initialized.")

        scores = cross_validate(self.model, X, y, cv=cv, scoring=scoring)
        return {
            f"mean_{scoring}": np.mean(scores['test_score']),
            f"std_{scoring}": np.std(scores['test_score'])
        }

    def save(self, filepath: str) -> None:
        """Save trained model to disk."""
        if not self.is_trained:
            warnings.warn("Saving untrained model.")
        joblib.dump(self, filepath)

    @staticmethod
    def load(filepath: str) -> 'Classifier':
        """Load model from disk."""
        return joblib.load(filepath)


class SVMClassifier(Classifier):
    """Support Vector Machine classifier."""

    def __init__(self, kernel: str = 'rbf', C: float = 1.0, gamma: Union[str, float] = 'scale', probability: bool = True, **kwargs):
        super().__init__("SVM")
        self.model = SVC(kernel=kernel, C=C, gamma=gamma, probability=probability, random_state=42, **kwargs)

    def train(self, X: np.ndarray, y: np.ndarray, feature_names: List[str] = None) -> None:
        self.model.fit(X, y)
        self.is_trained = True
        self.classes = self.model.classes_
        if feature_names:
            self.feature_names = feature_names


class KNNClassifier(Classifier):
    """K-Nearest Neighbors classifier."""

    def __init__(self, n_neighbors: int = 5, weights: str = 'uniform', algorithm: str = 'auto', **kwargs):
        super().__init__("KNN")
        self.model = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights, algorithm=algorithm, **kwargs)

    def train(self, X: np.ndarray, y: np.ndarray, feature_names: List[str] = None) -> None:
        self.model.fit(X, y)
        self.is_trained = True
        self.classes = self.model.classes_
        if feature_names:
            self.feature_names = feature_names


class RandomForestClassifier(Classifier):
    """Random Forest classifier."""

    def __init__(self, n_estimators: int = 100, max_depth: Optional[int] = None, class_weight: str = 'balanced', **kwargs):
        super().__init__("RandomForest")
        # FIXED: Using the aliased SkRandomForestClassifier
        self.model = SkRandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            class_weight=class_weight,
            random_state=42,
            **kwargs
        )

    def train(self, X: np.ndarray, y: np.ndarray, feature_names: List[str] = None) -> None:
        self.model.fit(X, y)
        self.is_trained = True
        self.classes = self.model.classes_
        if feature_names:
            self.feature_names = feature_names

    def get_feature_importance(self) -> np.ndarray:
        """Get feature importances."""
        if not self.is_trained:
            raise ValueError("Model not trained")
        return self.model.feature_importances_


class XGBoostClassifier(Classifier):
    """XGBoost classifier."""

    def __init__(self, n_estimators: int = 100, learning_rate: float = 0.1, max_depth: int = 3, **kwargs):
        super().__init__("XGBoost")
        if not HAS_XGB:
            raise ImportError("XGBoost not installed. pip install xgboost")

        self.model = XGB(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            use_label_encoder=False,
            eval_metric='logloss',
            random_state=42,
            **kwargs
        )

    def train(self, X: np.ndarray, y: np.ndarray, feature_names: List[str] = None) -> None:
        self.model.fit(X, y)
        self.is_trained = True
        self.classes = self.model.classes_
        if feature_names:
            self.feature_names = feature_names


class LogisticRegressionClassifier(Classifier):
    """Logistic Regression classifier (baseline)."""

    def __init__(self, C: float = 1.0, penalty: str = 'l2', solver: str = 'lbfgs', **kwargs):
        super().__init__("LogisticRegression")
        self.model = LogisticRegression(C=C, penalty=penalty, solver=solver, max_iter=1000, random_state=42, **kwargs)

    def train(self, X: np.ndarray, y: np.ndarray, feature_names: List[str] = None) -> None:
        self.model.fit(X, y)
        self.is_trained = True
        self.classes = self.model.classes_
        if feature_names:
            self.feature_names = feature_names
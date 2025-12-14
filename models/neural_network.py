"""
Neural Network Classifier

Implementation of Multi-Layer Perceptron (MLP) for respiratory analysis.
"""

from typing import List, Tuple, Union, Optional
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from .classifiers import Classifier


class NeuralNetworkClassifier(Classifier):
    """
    Feed-forward Neural Network (Multi-Layer Perceptron).

    Includes automatic internal scaling, as NNs require standardized inputs.
    """

    def __init__(
            self,
            hidden_layer_sizes: Union[int, Tuple[int]] = (100,),
            activation: str = 'relu',
            solver: str = 'adam',
            alpha: float = 0.0001,
            learning_rate_init: float = 0.001,
            max_iter: int = 500,
            early_stopping: bool = True,
            **kwargs
    ):
        """
        Initialize Neural Network.

        Args:
            hidden_layer_sizes: Tuple determining architecture (e.g., (100, 50) for 2 layers)
            activation: 'identity', 'logistic', 'tanh', 'relu'
            solver: 'lbfgs' (good for small data), 'sgd', 'adam' (default)
            alpha: L2 penalty (regularization term) parameter
            learning_rate_init: Initial learning rate
        """
        super().__init__("NeuralNetwork")
        self.scaler = StandardScaler()
        self.model = MLPClassifier(
            hidden_layer_sizes=hidden_layer_sizes,
            activation=activation,
            solver=solver,
            alpha=alpha,
            learning_rate_init=learning_rate_init,
            max_iter=max_iter,
            early_stopping=early_stopping,
            random_state=42,
            **kwargs
        )

    def train(self, X: np.ndarray, y: np.ndarray, feature_names: List[str] = None) -> None:
        """
        Train the NN.

        Automatically scales input data (Z-score normalization) before training.
        """
        # Scale data
        X_scaled = self.scaler.fit_transform(X)

        self.model.fit(X_scaled, y)
        self.is_trained = True
        self.classes = self.model.classes_
        if feature_names:
            self.feature_names = feature_names

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict classes using scaled data."""
        if not self.is_trained:
            raise ValueError("Model not trained")
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities using scaled data."""
        if not self.is_trained:
            raise ValueError("Model not trained")
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)
"""Visualization Package - Interactive and static plotting"""

from abc import ABC, abstractmethod
from typing import List, Optional, Any
import numpy as np
import pandas as pd

# base.py
class BasePlotter(ABC):
    """Abstract base plotter."""
    
    def __init__(self, output_dir: str = "plots", backend: str = "plotly"):
        self.output_dir = output_dir
        self.backend = backend
    
    @abstractmethod
    def plot_confusion_matrix(self, y_true, y_pred, labels: List[str]):
        pass
    
    @abstractmethod
    def plot_roc_curves(self, classifiers: list, X_test, y_test):
        pass
    
    @abstractmethod
    def plot_feature_importance(self, feature_names: List[str], importances: np.ndarray):
        pass
    
    @abstractmethod
    def plot_scatter_2d(self, X: np.ndarray, y: np.ndarray, labels: Optional[list] = None):
        pass


# plotly_plotter.py
class PlotlyPlotter(BasePlotter):
    """Interactive plots using Plotly (DEFAULT)."""
    
    def __init__(self, output_dir: str = "plots"):
        super().__init__(output_dir, "plotly")
    
    def plot_confusion_matrix(self, y_true, y_pred, labels: List[str]):
        """Interactive confusion matrix heatmap."""
        pass
    
    def plot_roc_curves(self, classifiers: list, X_test, y_test):
        """ROC curves for multiple classifiers."""
        pass
    
    def plot_feature_importance(self, feature_names: List[str], importances: np.ndarray):
        """Bar chart of feature importances."""
        pass
    
    def plot_scatter_2d(self, X: np.ndarray, y: np.ndarray, labels: Optional[list] = None):
        """2D scatter plot (for PCA/t-SNE)."""
        pass
    
    def plot_violin(self, features: pd.DataFrame, outcome: str, feature_name: str):
        """Violin plot for feature distribution across groups."""
        pass
    
    def plot_box(self, features: pd.DataFrame, outcome: str, feature_name: str):
        """Box plot for feature distribution."""
        pass
    
    def plot_learning_curves(self, train_sizes, train_scores, val_scores):
        """Learning curves showing model performance vs training size."""
        pass
    
    def plot_parameter_matrix(self, features: pd.DataFrame, subject_id: str):
        """Matrix showing all features across time windows."""
        pass


# matplotlib_plotter.py
class MatplotlibPlotter(BasePlotter):
    """Static publication-quality plots using Matplotlib."""
    
    def __init__(self, output_dir: str = "plots"):
        super().__init__(output_dir, "matplotlib")
    
    def plot_confusion_matrix(self, y_true, y_pred, labels: List[str]):
        """Static confusion matrix."""
        pass
    
    def plot_roc_curves(self, classifiers: list, X_test, y_test):
        """Static ROC curves."""
        pass
    
    # ... similar methods as PlotlyPlotter


# comparison.py
class ModelComparisonPlotter:
    """Specialized plots for comparing multiple models."""
    
    def __init__(self, plotter: BasePlotter):
        self.plotter = plotter
    
    def plot_model_comparison_bar(self, results_df: pd.DataFrame):
        """Bar chart comparing metrics across models."""
        pass
    
    def plot_model_comparison_radar(self, results_df: pd.DataFrame):
        """Radar chart comparing models on multiple metrics."""
        pass
    
    def plot_sensitivity_specificity(self, classifiers: list, X_test, y_test):
        """Sensitivity vs specificity for all classifiers."""
        pass

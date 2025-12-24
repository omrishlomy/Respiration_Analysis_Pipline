import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
from .classifiers import SVMClassifier
from .evaluation import GridSearchTuner
from .bootstrap import BootstrapAnalyzer


class ExperimentManager:
    """
    Manages training with extensive debugging prints to diagnose classifier behavior.
    """

    def __init__(self, config):
        self.config = config
        self.model_cfg = config['models']

    def run_experiments(self, X_df, y, significant_features_list, plotter=None):
        all_features = X_df.columns.tolist()
        experiments = {'All Features': all_features}

        if significant_features_list:
            experiments['Significant Features'] = significant_features_list
            if len(significant_features_list) > 1:
                for feature in significant_features_list:
                    subset = [f for f in significant_features_list if f != feature]
                    experiments[f'LOO: -{feature}'] = subset

        results_list = []
        predictions = {}

        classes, counts = np.unique(y, return_counts=True)
        n_recordings = len(y)
        print(f"    🔍 CLASSIFIER DATA: N = {n_recordings} recordings (samples) | Balance: {dict(zip(classes, counts))}")

        n_folds = max(2, min(self.model_cfg['cv_folds'], min(counts)))

        print(f"    Running {len(experiments)} model variants...")

        for exp_name, feature_subset in tqdm(experiments.items(), desc="Training"):
            try:
                # 1. Subset Data
                X_sub = X_df[feature_subset].values

                # 2. Tune
                clf = SVMClassifier()
                tuner = GridSearchTuner(cv_folds=n_folds)
                best_model, best_params = tuner.tune(clf, X_sub, y, self.model_cfg['tuning']['svm'])

                # 3. Bootstrap Eval
                bootstrapper = BootstrapAnalyzer(n_iterations=self.model_cfg['bootstrap_iterations'])
                metrics = bootstrapper.evaluate(best_model, X_sub, y)

                # --- DEBUG: Print Keys First Time ---
                if len(results_list) == 0:
                    print(f"    🔍 DEBUG Metrics Keys Found: {list(metrics.keys())}")

                # 4. Final Train
                best_model.train(X_sub, y)
                y_prob = best_model.predict_proba(X_sub)

                predictions[exp_name] = y_prob

                # Store results
                results_list.append({
                    'Experiment': exp_name,
                    'Dimension': len(feature_subset),
                    'Hyperparameters': str(best_params),
                    'Accuracy': self._get_metric(metrics, ['accuracy', 'test_accuracy', 'acc']),
                    'Sensitivity': self._get_metric(metrics, ['sensitivity', 'recall', 'test_recall']),
                    'Specificity': self._get_metric(metrics, ['specificity']),
                    'AUC': self._get_metric(metrics, ['roc_auc', 'auc', 'test_roc_auc']),
                    'Features_Used': ", ".join(feature_subset)
                })

                if plotter:
                    safe_name = "".join([c for c in exp_name if c.isalnum()]).strip()
                    y_pred = (y_prob > 0.5).astype(int) if y_prob.ndim == 1 else np.argmax(y_prob, axis=1)
                    plotter.plot_confusion_matrix(y, y_pred, title=f"CM: {exp_name}", filename=f"cm_{safe_name}.html")

            except Exception as e:
                print(f"    ❌ Experiment '{exp_name}' failed: {e}")
                import traceback
                traceback.print_exc()

        if plotter and predictions:
            plotter.plot_comparison_roc(predictions, y, filename="ROC_Comparison_All_Experiments.html")

        return pd.DataFrame(results_list), predictions

    def _get_metric(self, metrics, keys):
        """Helper to find a metric using multiple possible keys."""
        for k in keys:
            if k in metrics:
                # Check if it's a dict with 'mean' or just a value
                val = metrics[k]
                if isinstance(val, dict) and 'mean' in val:
                    return val['mean']
                return val
        return "N/A"
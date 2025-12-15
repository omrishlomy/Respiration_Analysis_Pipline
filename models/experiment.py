import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
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
        test_predictions = {}  # Store test set predictions (not training data!)

        classes, counts = np.unique(y, return_counts=True)
        print(f"    🔍 DATA DEBUG: Total Samples: {len(y)} | Balance: {dict(zip(classes, counts))}")

        # Create a proper train/test split (80/20) to avoid overfitting
        # Bootstrap will use training set only
        # Use 20% for test set, but ensure at least 2 samples per class for stratification
        min_class_count = min(np.unique(y, return_counts=True)[1])
        if min_class_count >= 10:
            test_size = 0.2  # Standard 20% split
        else:
            # For very small datasets, use at least 2 samples per class in test set
            test_size = max(0.2, (2 * len(np.unique(y))) / len(y))

        X_indices = np.arange(len(y))
        train_idx, test_idx = train_test_split(
            X_indices,
            test_size=test_size,
            stratify=y,
            random_state=42
        )

        y_train = y[train_idx]
        y_test = y[test_idx]
        print(f"    📊 Train/Test Split: {len(train_idx)} train, {len(test_idx)} test samples")

        n_folds = max(2, min(self.model_cfg['cv_folds'], min(np.unique(y_train, return_counts=True)[1])))

        print(f"    Running {len(experiments)} model variants...")

        for exp_name, feature_subset in tqdm(experiments.items(), desc="Training"):
            try:
                # 1. Subset Data (TRAIN and TEST separately!)
                X_train = X_df.iloc[train_idx][feature_subset].values
                X_test = X_df.iloc[test_idx][feature_subset].values

                # 2. Tune (on training data only)
                clf = SVMClassifier()
                tuner = GridSearchTuner(cv_folds=n_folds)
                best_model, best_params = tuner.tune(clf, X_train, y_train, self.model_cfg['tuning']['svm'])

                # 3. Bootstrap Eval (on training data only)
                bootstrapper = BootstrapAnalyzer(n_iterations=self.model_cfg['bootstrap_iterations'])
                metrics = bootstrapper.evaluate(best_model, X_train, y_train)

                # --- DEBUG: Print Keys First Time ---
                if len(results_list) == 0:
                    print(f"    🔍 DEBUG Metrics Keys Found: {list(metrics.keys())}")

                # 4. Final Train on ALL training data, predict on TEST set
                best_model.train(X_train, y_train)
                y_test_prob = best_model.predict_proba(X_test)

                test_predictions[exp_name] = y_test_prob

                # Store results (from bootstrap cross-validation on training set)
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

                # Generate confusion matrix from TEST set predictions (not training!)
                if plotter:
                    safe_name = "".join([c for c in exp_name if c.isalnum()]).strip()
                    y_test_pred = (y_test_prob > 0.5).astype(int) if y_test_prob.ndim == 1 else np.argmax(y_test_prob, axis=1)
                    plotter.plot_confusion_matrix(
                        y_test, y_test_pred,
                        title=f"CM (Test Set): {exp_name}",
                        filename=f"cm_{safe_name}.html"
                    )

            except Exception as e:
                print(f"    ❌ Experiment '{exp_name}' failed: {e}")
                import traceback
                traceback.print_exc()

        # Generate ROC curves from TEST set predictions (not training!)
        if plotter and test_predictions:
            plotter.plot_comparison_roc(
                test_predictions, y_test,
                filename="ROC_Comparison_All_Experiments_TestSet.html"
            )

        return pd.DataFrame(results_list), test_predictions

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
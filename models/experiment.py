import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, roc_auc_score, accuracy_score
from .classifiers import SVMClassifier
from .evaluation import GridSearchTuner, LeaveOneRecordingOut, LeaveOneSubjectOut
from copy import deepcopy


class ExperimentManager:
    """
    Manages classifier training experiments with:
    - Incremental feature training (2, 4, 6, 8... features)
    - LORO (Leave-One-Recording-Out) cross-validation
    - LOSO (Leave-One-Subject-Out) cross-validation
    """

    def __init__(self, config):
        self.config = config
        self.model_cfg = config['models']

    def run_experiments(self, X_df, y, significant_features_list, plotter=None, subject_ids=None):
        """
        Run all configured experiments.

        Args:
            X_df: DataFrame with features
            y: Labels array
            significant_features_list: List of significant feature names (sorted by p-value)
            plotter: Optional plotter for visualizations
            subject_ids: List of subject IDs for each recording (required for LORO/LOSO)

        Returns:
            Tuple of (results_df, predictions_dict)
        """
        all_features = X_df.columns.tolist()
        results_list = []
        predictions = {}  # Collect predictions for comparison plots

        classes, counts = np.unique(y, return_counts=True)
        n_recordings = len(y)
        print(f"    🔍 CLASSIFIER DATA: N = {n_recordings} recordings (samples) | Balance: {dict(zip(classes, counts))}")

        # --- 1. LORO Cross-Validation (if enabled) ---
        if self.model_cfg.get('loro', {}).get('enabled', False) and subject_ids is not None:
            print(f"    Running LORO Cross-Validation...")

            # LORO uses top 6 significant features
            top_n = self.model_cfg.get('loso', {}).get('top_n_features', 6)
            loro_features = significant_features_list[:top_n] if len(significant_features_list) >= top_n else significant_features_list

            loro_results, loro_preds = self._run_loro_cv(X_df, y, loro_features, subject_ids)
            results_list.extend(loro_results)
            if loro_preds is not None:
                predictions['LORO'] = loro_preds

        # --- 2. LOSO Cross-Validation (if enabled) ---
        if self.model_cfg.get('loso', {}).get('enabled', False) and subject_ids is not None:
            print(f"    Running LOSO Cross-Validation...")

            # LOSO uses top N significant features
            top_n = self.model_cfg.get('loso', {}).get('top_n_features', 6)
            loso_features = significant_features_list[:top_n] if len(significant_features_list) >= top_n else significant_features_list

            loso_results, loso_preds = self._run_loso_cv(X_df, y, loso_features, subject_ids)
            results_list.extend(loso_results)
            if loso_preds is not None:
                predictions['LOSO'] = loso_preds

        # --- 3. Incremental Feature Training (2, 4, 6, 8, ... features) ---
        if significant_features_list:
            print(f"    Running Incremental Feature Training...")
            incremental_results, incr_preds = self._run_incremental_features(X_df, y, significant_features_list, plotter)
            results_list.extend(incremental_results)
            predictions.update(incr_preds)

        # --- 4. Create Comparison ROC Plot ---
        if plotter:
            plotter.plot_comparison_roc(predictions, y, filename="ROC_Comparison_All_Experiments.html")

        return pd.DataFrame(results_list), predictions

    def _run_loro_cv(self, X_df, y, feature_subset, subject_ids):
        """Run Leave-One-Recording-Out cross-validation."""
        results = []

        X_sub = X_df[feature_subset].values
        loro = LeaveOneRecordingOut(subject_ids)

        print(f"      LORO: {len(feature_subset)} features, {loro.get_n_splits()} folds")

        all_y_true = []
        all_y_pred = []
        all_y_prob = []
        fold_results = []

        for fold_idx, (train_idx, test_idx) in enumerate(loro.split(X_sub, y)):
            X_train, X_test = X_sub[train_idx], X_sub[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            # Train classifier
            clf = SVMClassifier()
            tuner = GridSearchTuner(cv_folds=min(5, len(np.unique(y_train))))

            try:
                best_model, best_params = tuner.tune(clf, X_train, y_train, self.model_cfg['tuning']['svm'])
                best_model.train(X_train, y_train)

                y_pred = best_model.predict(X_test)
                y_prob = best_model.predict_proba(X_test)

                all_y_true.extend(y_test)
                all_y_pred.extend(y_pred)
                if y_prob.ndim > 1:
                    all_y_prob.extend(y_prob[:, 1])
                else:
                    all_y_prob.extend(y_prob)

                fold_results.append({
                    'n_train': len(train_idx),
                    'n_test': len(test_idx),
                    'hyperparams': str(best_params)
                })

            except Exception as e:
                print(f"        Fold {fold_idx + 1} failed: {e}")
                continue

        # Calculate overall metrics
        if len(all_y_true) > 0:
            all_y_true = np.array(all_y_true)
            all_y_pred = np.array(all_y_pred)
            all_y_prob = np.array(all_y_prob)

            accuracy = accuracy_score(all_y_true, all_y_pred)
            cm = confusion_matrix(all_y_true, all_y_pred)

            if cm.shape == (2, 2):
                tn, fp, fn, tp = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]
                sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            else:
                sensitivity = np.nan
                specificity = np.nan

            try:
                auc = roc_auc_score(all_y_true, all_y_prob)
            except:
                auc = np.nan

            # Calculate average train/test sizes
            avg_n_train = np.mean([r['n_train'] for r in fold_results])
            avg_n_test = np.mean([r['n_test'] for r in fold_results])
            hyperparams = fold_results[0]['hyperparams'] if fold_results else "N/A"

            results.append({
                'Experiment': 'LORO',
                'Dimension': len(feature_subset),
                'Features_Used': ", ".join(feature_subset),
                'Hyperparameters': hyperparams,
                'N_samples_train': int(avg_n_train),
                'N_samples_test': int(avg_n_test),
                'Accuracy': accuracy,
                'Sensitivity': sensitivity,
                'Specificity': specificity,
                'AUC': auc
            })

            # Return predictions for ROC comparison plot
            return results, all_y_prob
        else:
            return results, None

    def _run_loso_cv(self, X_df, y, feature_subset, subject_ids):
        """Run Leave-One-Subject-Out cross-validation."""
        results = []

        X_sub = X_df[feature_subset].values
        loso = LeaveOneSubjectOut(subject_ids)

        print(f"      LOSO: {len(feature_subset)} features, {loso.get_n_splits()} folds")

        all_y_true = []
        all_y_pred = []
        all_y_prob = []
        fold_results = []

        for fold_idx, (train_idx, test_idx) in enumerate(loso.split(X_sub, y)):
            X_train, X_test = X_sub[train_idx], X_sub[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            # Train classifier
            clf = SVMClassifier()
            tuner = GridSearchTuner(cv_folds=min(5, len(np.unique(y_train))))

            try:
                best_model, best_params = tuner.tune(clf, X_train, y_train, self.model_cfg['tuning']['svm'])
                best_model.train(X_train, y_train)

                y_pred = best_model.predict(X_test)
                y_prob = best_model.predict_proba(X_test)

                all_y_true.extend(y_test)
                all_y_pred.extend(y_pred)
                if y_prob.ndim > 1:
                    all_y_prob.extend(y_prob[:, 1])
                else:
                    all_y_prob.extend(y_prob)

                fold_results.append({
                    'n_train': len(train_idx),
                    'n_test': len(test_idx),
                    'hyperparams': str(best_params)
                })

            except Exception as e:
                print(f"        Fold {fold_idx + 1} failed: {e}")
                continue

        # Calculate overall metrics
        if len(all_y_true) > 0:
            all_y_true = np.array(all_y_true)
            all_y_pred = np.array(all_y_pred)
            all_y_prob = np.array(all_y_prob)

            accuracy = accuracy_score(all_y_true, all_y_pred)
            cm = confusion_matrix(all_y_true, all_y_pred)

            if cm.shape == (2, 2):
                tn, fp, fn, tp = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]
                sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            else:
                sensitivity = np.nan
                specificity = np.nan

            try:
                auc = roc_auc_score(all_y_true, all_y_prob)
            except:
                auc = np.nan

            # Calculate average train/test sizes
            avg_n_train = np.mean([r['n_train'] for r in fold_results])
            avg_n_test = np.mean([r['n_test'] for r in fold_results])
            hyperparams = fold_results[0]['hyperparams'] if fold_results else "N/A"

            results.append({
                'Experiment': 'LOSO',
                'Dimension': len(feature_subset),
                'Features_Used': ", ".join(feature_subset),
                'Hyperparameters': hyperparams,
                'N_samples_train': int(avg_n_train),
                'N_samples_test': int(avg_n_test),
                'Accuracy': accuracy,
                'Sensitivity': sensitivity,
                'Specificity': specificity,
                'AUC': auc
            })

            # Return predictions for ROC comparison plot
            return results, all_y_prob
        else:
            return results, None

    def _run_incremental_features(self, X_df, y, significant_features, plotter):
        """Train with incremental number of features: 2, 4, 6, 8, ..."""
        results = []
        predictions = {}  # Collect predictions for each feature count

        # Generate feature counts: 2, 4, 6, 8, ... up to all significant features
        n_significant = len(significant_features)
        feature_counts = list(range(2, n_significant + 1, 2))

        # If the last count doesn't include all features, add all features
        if feature_counts[-1] != n_significant:
            feature_counts.append(n_significant)

        print(f"      Training with feature counts: {feature_counts}")

        for n_features in tqdm(feature_counts, desc="Incremental Features"):
            feature_subset = significant_features[:n_features]

            try:
                # Subset data
                X_sub = X_df[feature_subset].values

                # Train-test split (simple 80-20)
                from sklearn.model_selection import train_test_split
                X_train, X_test, y_train, y_test = train_test_split(
                    X_sub, y, test_size=0.2, stratify=y, random_state=42
                )

                # Tune and train
                clf = SVMClassifier()
                tuner = GridSearchTuner(cv_folds=5)
                best_model, best_params = tuner.tune(clf, X_train, y_train, self.model_cfg['tuning']['svm'])
                best_model.train(X_train, y_train)

                # Predict on full dataset for ROC comparison
                y_prob_full = best_model.predict_proba(X_sub)

                # Store predictions for this experiment
                experiment_name = f'Top {n_features} Features'
                predictions[experiment_name] = y_prob_full

                # Predict on test set for metrics
                y_pred = best_model.predict(X_test)
                y_prob = best_model.predict_proba(X_test)

                # Calculate metrics
                accuracy = accuracy_score(y_test, y_pred)
                cm = confusion_matrix(y_test, y_pred)

                if cm.shape == (2, 2):
                    tn, fp, fn, tp = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]
                    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
                else:
                    sensitivity = np.nan
                    specificity = np.nan

                try:
                    if y_prob.ndim > 1:
                        auc = roc_auc_score(y_test, y_prob[:, 1])
                    else:
                        auc = roc_auc_score(y_test, y_prob)
                except:
                    auc = np.nan

                results.append({
                    'Experiment': experiment_name,
                    'Dimension': n_features,
                    'Features_Used': ", ".join(feature_subset),
                    'Hyperparameters': str(best_params),
                    'N_samples_train': len(X_train),
                    'N_samples_test': len(X_test),
                    'Accuracy': accuracy,
                    'Sensitivity': sensitivity,
                    'Specificity': specificity,
                    'AUC': auc
                })

            except Exception as e:
                print(f"        Top {n_features} features failed: {e}")
                continue

        return results, predictions

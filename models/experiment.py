import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from .classifiers import SVMClassifier
from .neural_network import NeuralNetworkClassifier
from .evaluation import GridSearchTuner
from .bootstrap import BootstrapAnalyzer
from preprocessing.recording_length import RecordingLengthManager
from features.collection import FeatureCollection


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

    def run_experiments_with_length_prefix(
        self, pipeline_context, X_df, y, significant_features, recording_lengths
    ):
        """
        Run experiments across different recording length prefixes.

        Args:
            pipeline_context: Dict with keys: recordings, labels_df, outcome, cleaner,
                             win_gen, extractor, aggregator
            X_df: Feature dataframe (not used directly, just for reference)
            y: Labels (not used directly, just for reference)
            significant_features: List of significant feature names to use
            recording_lengths: List of recording lengths in minutes (or None for full)

        Returns:
            (results_df, best_model_name)
        """
        from features.collection import FeatureCollection

        length_manager = RecordingLengthManager()
        results_list = []

        # Unpack context
        recordings = pipeline_context['recordings']
        labels_df = pipeline_context['labels_df']
        outcome = pipeline_context['outcome']
        cleaner = pipeline_context['cleaner']
        win_gen = pipeline_context['win_gen']
        extractor = pipeline_context['extractor']
        aggregator = pipeline_context['aggregator']

        for prefix_length in tqdm(recording_lengths, desc="    Testing Prefixes"):
            prefix_name = RecordingLengthManager.format_length_name(prefix_length)
            print(f"\n    📏 Processing prefix: {prefix_name}")

            try:
                # Extract features with this prefix length
                all_subject_features = []

                for rec in recordings:
                    try:
                        rec_truncated = length_manager.truncate_recording(rec, prefix_length)
                        rec_clean = cleaner.clean(rec_truncated)
                        windows = win_gen.generate_windows(rec_clean)

                        rec_feats = []
                        for w in windows:
                            f = extractor.extract(w.data, w.sampling_rate)
                            rec_feats.append(f)

                        if rec_feats:
                            all_subject_features.append(
                                aggregator.aggregate(rec_feats, subject_id=rec.subject_id)
                            )
                    except:
                        continue

                if not all_subject_features:
                    print(f"       ⚠️ No features extracted for {prefix_name}")
                    continue

                # Create feature dataframe and merge with labels
                features_df = pd.DataFrame(all_subject_features)
                features_df['SubjectID'] = features_df['SubjectID'].astype(str).str.strip()

                collection = FeatureCollection(
                    features_df,
                    subject_ids=features_df['SubjectID'].tolist()
                )
                X_df_prefix, y_prefix = collection.merge_with_labels(
                    labels_df, on='SubjectID', outcome=outcome
                )

                if 'SubjectID' in X_df_prefix.columns:
                    X_df_prefix = X_df_prefix.set_index('SubjectID')

                X_df_prefix = X_df_prefix.select_dtypes(include=[np.number]).fillna(0)
                y_prefix = np.array(y_prefix, dtype=int)

                if len(np.unique(y_prefix)) < 2:
                    print(f"       ⚠️ Insufficient classes for {prefix_name}")
                    continue

                # Build experiment variants (All Features, Significant, LOO)
                all_features = X_df_prefix.columns.tolist()
                experiments = {'All Features': all_features}

                if significant_features and all(f in X_df_prefix.columns for f in significant_features):
                    experiments['Significant Features'] = significant_features

                    # Leave-One-Out experiments for each significant feature
                    if len(significant_features) > 1:
                        for feature in significant_features:
                            subset = [f for f in significant_features if f != feature]
                            experiments[f'LOO: -{feature}'] = subset

                # Train and evaluate each experiment variant
                for exp_name, feature_subset in experiments.items():
                    metrics = self._train_and_evaluate_prefix(
                        X_df_prefix, y_prefix, feature_subset, prefix_name, exp_name
                    )

                    if metrics:
                        results_list.append(metrics)

            except Exception as e:
                print(f"       ❌ Failed {prefix_name}: {e}")
                import traceback
                traceback.print_exc()

        return pd.DataFrame(results_list), "SVM"

    def _train_and_evaluate_prefix(self, X_df, y, feature_subset, prefix_name, experiment_name):
        """
        Train and evaluate model for a single prefix length and experiment variant.

        Returns:
            Dict of metrics or None if failed
        """
        try:
            # Train/test split
            min_class_count = min(np.unique(y, return_counts=True)[1])
            test_size = 0.2 if min_class_count >= 10 else max(0.2, (2 * len(np.unique(y))) / len(y))

            X_indices = np.arange(len(y))
            train_idx, test_idx = train_test_split(
                X_indices, test_size=test_size, stratify=y, random_state=42
            )

            X_train = X_df.iloc[train_idx][feature_subset].values
            X_test = X_df.iloc[test_idx][feature_subset].values
            y_train = y[train_idx]
            y_test = y[test_idx]

            # Train model
            clf = SVMClassifier()
            n_folds = max(2, min(self.model_cfg['cv_folds'], min(np.unique(y_train, return_counts=True)[1])))
            tuner = GridSearchTuner(cv_folds=n_folds)
            best_model, best_params = tuner.tune(clf, X_train, y_train, self.model_cfg['tuning']['svm'])

            # Bootstrap evaluation
            bootstrapper = BootstrapAnalyzer(n_iterations=self.model_cfg['bootstrap_iterations'])
            metrics = bootstrapper.evaluate(best_model, X_train, y_train)

            # Compile results
            result = {
                'Recording_Length': prefix_name,
                'Experiment': experiment_name,
                'N_Samples': len(y),
                'N_Train': len(y_train),
                'N_Test': len(y_test),
                'N_Features': len(feature_subset),
                'Hyperparameters': str(best_params),
                'Accuracy': self._get_metric(metrics, ['accuracy', 'test_accuracy', 'acc']),
                'Sensitivity': self._get_metric(metrics, ['sensitivity', 'recall', 'test_recall']),
                'Specificity': self._get_metric(metrics, ['specificity']),
                'AUC': self._get_metric(metrics, ['roc_auc', 'auc', 'test_roc_auc']),
            }

            print(f"       ✅ {prefix_name} - {experiment_name}: Acc={result['Accuracy']:.3f}")
            return result

        except Exception as e:
            print(f"       ❌ Training failed for {prefix_name} - {experiment_name}: {e}")
            return None

    def run_neural_network_experiments_with_length_prefix(self, pipeline_context, X_df, y, significant_features, recording_lengths):
        """
        Run Neural Network experiments across recording length prefixes.
        Parallel to SVM experiments but uses MLPClassifier.

        Args:
            pipeline_context: Dict with recordings, cleaner, extractor, etc.
            X_df: Full feature DataFrame
            y: Labels
            significant_features: List of significant feature names
            recording_lengths: List of recording length prefixes to test

        Returns:
            DataFrame with Neural Network results
        """
        print(f"\n[MODELS] Training Neural Networks")
        all_results = []

        for prefix_length in tqdm(recording_lengths, desc="    Testing Prefixes (NN)"):
            # Truncate recordings to prefix length
            length_manager = RecordingLengthManager()
            truncated_recordings = [
                length_manager.truncate_recording(rec, prefix_length)
                for rec in pipeline_context['recordings']
            ]

            # Re-extract features for this prefix
            X_df_prefix, y_prefix = self._extract_features_for_prefix(
                truncated_recordings, pipeline_context, X_df.index
            )

            if X_df_prefix is None or len(X_df_prefix) == 0:
                continue

            prefix_name = RecordingLengthManager.format_length_name(prefix_length)

            # Build experiment variants (All Features, Significant, LOO)
            all_features = X_df_prefix.columns.tolist()
            experiments = {'All Features': all_features}

            if significant_features and all(f in X_df_prefix.columns for f in significant_features):
                experiments['Significant Features'] = significant_features

                # Leave-One-Out experiments for each significant feature
                if len(significant_features) > 1:
                    for feature in significant_features:
                        subset = [f for f in significant_features if f != feature]
                        experiments[f'LOO: -{feature}'] = subset

            # Train each experiment variant
            for exp_name, feature_subset in experiments.items():
                metrics = self._train_and_evaluate_nn_prefix(
                    X_df_prefix, y_prefix, feature_subset, prefix_name, exp_name
                )
                if metrics:
                    all_results.append(metrics)

        if not all_results:
            print("    ⚠️  No Neural Network results generated")
            return pd.DataFrame()

        results_df = pd.DataFrame(all_results)
        return results_df

    def _train_and_evaluate_nn_prefix(self, X_df, y, feature_subset, prefix_name, experiment_name):
        """
        Train and evaluate Neural Network for a specific prefix and experiment.

        Args:
            X_df: Feature DataFrame for this prefix
            y: Labels for this prefix
            feature_subset: List of features to use
            prefix_name: Name of recording length prefix (e.g., "10min")
            experiment_name: Name of experiment variant (e.g., "All Features", "LOO: -feature_x")

        Returns:
            Dict with metrics, or None if training failed
        """
        try:
            X = X_df[feature_subset].values

            # Split data (80/20)
            min_class_count = min(np.unique(y, return_counts=True)[1])
            if min_class_count < 2:
                return None

            test_size = 0.2 if min_class_count >= 10 else max(0.2, (2 * len(np.unique(y))) / len(y))

            train_idx, test_idx = train_test_split(
                np.arange(len(y)),
                test_size=test_size,
                stratify=y,
                random_state=42
            )

            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            # Create Neural Network classifier
            nn_clf = NeuralNetworkClassifier(
                hidden_layer_sizes=(100, 50),  # 2 hidden layers
                activation='relu',
                solver='adam',
                alpha=0.0001,
                learning_rate_init=0.001,
                max_iter=500,
                early_stopping=True,
                random_state=42
            )

            # Train on training set
            nn_clf.train(X_train, y_train, feature_names=feature_subset)

            # Evaluate on test set
            y_pred = nn_clf.predict(X_test)
            y_prob = nn_clf.predict_proba(X_test)

            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)

            # Sensitivity (recall for positive class)
            sensitivity = recall_score(y_test, y_pred, average='binary', pos_label=1)

            # Specificity (recall for negative class)
            cm = confusion_matrix(y_test, y_pred)
            if cm.shape[0] == 2:
                tn, fp, fn, tp = cm.ravel()
                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            else:
                specificity = 0

            # AUC
            if y_prob.shape[1] == 2:
                auc = roc_auc_score(y_test, y_prob[:, 1])
            else:
                auc = 0

            return {
                'Recording_Length': prefix_name,
                'Experiment': experiment_name,
                'N_Samples': len(y),
                'N_Train': len(y_train),
                'N_Test': len(y_test),
                'N_Features': len(feature_subset),
                'Architecture': '(100,50)',  # Hidden layer sizes
                'Accuracy': accuracy,
                'Sensitivity': sensitivity,
                'Specificity': specificity,
                'AUC': auc,
            }

        except Exception as e:
            print(f"       ❌ NN training failed for {prefix_name} - {experiment_name}: {e}")
            return None
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
from analysis.statistical import StatisticalAnalyzer


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

        IMPORTANT: To avoid data leakage, this method:
        1. Extracts features from ALL recordings (keeping global feature extraction as requested)
        2. Splits data into train/test FIRST (subject-level split)
        3. Performs statistical feature selection on TRAINING data only
        4. Trains NN on training data with selected features
        5. Evaluates on held-out test data

        This prevents the "peeking" problem where feature selection uses test data.

        Args:
            pipeline_context: Dict with recordings, cleaner, extractor, etc.
            X_df: Full feature DataFrame (not used - kept for compatibility)
            y: Labels (not used - kept for compatibility)
            significant_features: List of significant feature names (not used - will compute on train data)
            recording_lengths: List of recording length prefixes to test

        Returns:
            DataFrame with Neural Network results
        """
        print(f"\n[MODELS] Training Neural Networks (with proper train/test split)")
        print(f"    ⚠️  Feature selection will be done on TRAINING data only to avoid leakage")
        all_results = []

        # Extract pipeline components
        recordings = pipeline_context['recordings']
        labels_df = pipeline_context['labels_df']
        outcome = pipeline_context['outcome']
        cleaner = pipeline_context['cleaner']
        win_gen = pipeline_context['win_gen']
        extractor = pipeline_context['extractor']
        aggregator = pipeline_context['aggregator']

        # Get statistical test configuration
        stats_config = self.config.get('analysis', {}).get('statistical', {})
        test_method = stats_config.get('test', 'mannwhitney')
        correction_method = stats_config.get('correction', 'fdr_bh')

        for prefix_length in tqdm(recording_lengths, desc="    Testing Prefixes (NN)"):
            try:
                # Truncate recordings to prefix length
                length_manager = RecordingLengthManager()
                truncated_recordings = [
                    length_manager.truncate_recording(rec, prefix_length)
                    for rec in recordings
                ]

                prefix_name = RecordingLengthManager.format_length_name(prefix_length)

                # Extract features for this prefix (from ALL recordings as requested)
                all_subject_features = []
                for rec in truncated_recordings:
                    try:
                        rec_clean = cleaner.clean(rec)
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

                # Keep subject IDs before converting to numeric-only
                subject_ids = X_df_prefix['SubjectID'] if 'SubjectID' in X_df_prefix.columns else X_df_prefix.index

                if 'SubjectID' in X_df_prefix.columns:
                    X_df_prefix = X_df_prefix.set_index('SubjectID')

                X_df_prefix = X_df_prefix.select_dtypes(include=[np.number]).fillna(0)
                y_prefix = np.array(y_prefix, dtype=int)

                if len(np.unique(y_prefix)) < 2:
                    print(f"       ⚠️ Insufficient classes for {prefix_name}")
                    continue

                # STEP 1: Split data into train/test FIRST (subject-level)
                min_class_count = min(np.unique(y_prefix, return_counts=True)[1])
                if min_class_count < 2:
                    continue

                test_size = 0.2 if min_class_count >= 10 else max(0.2, (2 * len(np.unique(y_prefix))) / len(y_prefix))

                train_idx, test_idx = train_test_split(
                    np.arange(len(y_prefix)),
                    test_size=test_size,
                    stratify=y_prefix,
                    random_state=42
                )

                # Split features and labels
                X_train_full = X_df_prefix.iloc[train_idx]
                X_test_full = X_df_prefix.iloc[test_idx]
                y_train = y_prefix[train_idx]
                y_test = y_prefix[test_idx]

                # STEP 2: Perform statistical feature selection on TRAINING data only
                stats_analyzer = StatisticalAnalyzer(test=test_method, correction_method=correction_method)
                stats_df_train = stats_analyzer.compare_groups(
                    X_train_full, y_train,
                    outcome_name=outcome,
                    feature_names=X_train_full.columns.tolist()
                )

                # Extract significant features from training data
                feat_col = 'feature_name' if 'feature_name' in stats_df_train.columns else 'feature'
                significant_features_train = stats_df_train[stats_df_train['significant'] == True][feat_col].tolist() if feat_col in stats_df_train.columns else []

                # STEP 3: Build experiment variants
                all_features = X_train_full.columns.tolist()
                experiments = {'All Features': all_features}

                if significant_features_train:
                    experiments['Significant Features (Train)'] = significant_features_train

                    # Leave-One-Out experiments for each significant feature
                    if len(significant_features_train) > 1:
                        for feature in significant_features_train:
                            subset = [f for f in significant_features_train if f != feature]
                            experiments[f'LOO: -{feature}'] = subset

                # STEP 4: Train and evaluate each experiment variant
                for exp_name, feature_subset in experiments.items():
                    metrics = self._train_and_evaluate_nn_no_leakage(
                        X_train_full, X_test_full, y_train, y_test,
                        feature_subset, prefix_name, exp_name
                    )
                    if metrics:
                        all_results.append(metrics)

            except Exception as e:
                print(f"       ❌ NN Failed {prefix_name}: {e}")
                import traceback
                traceback.print_exc()

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
            # Note: random_state=42 is set internally in NeuralNetworkClassifier
            nn_clf = NeuralNetworkClassifier(
                hidden_layer_sizes=(100, 50),  # 2 hidden layers
                activation='relu',
                solver='adam',
                alpha=0.0001,
                learning_rate_init=0.001,
                max_iter=500,
                early_stopping=True
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

    def _train_and_evaluate_nn_no_leakage(self, X_train_df, X_test_df, y_train, y_test, feature_subset, prefix_name, experiment_name):
        """
        Train and evaluate Neural Network with pre-split train/test data (NO LEAKAGE).

        This method receives data that has ALREADY been split, ensuring that
        feature selection was performed only on training data.

        Args:
            X_train_df: Training features DataFrame
            X_test_df: Test features DataFrame
            y_train: Training labels
            y_test: Test labels
            feature_subset: List of features to use
            prefix_name: Name of recording length prefix (e.g., "10min")
            experiment_name: Name of experiment variant (e.g., "All Features", "Significant Features (Train)")

        Returns:
            Dict with metrics, or None if training failed
        """
        try:
            # Extract feature values
            X_train = X_train_df[feature_subset].values
            X_test = X_test_df[feature_subset].values

            # Create Neural Network classifier
            # Note: random_state=42 is set internally in NeuralNetworkClassifier
            nn_clf = NeuralNetworkClassifier(
                hidden_layer_sizes=(100, 50),  # 2 hidden layers
                activation='relu',
                solver='adam',
                alpha=0.0001,
                learning_rate_init=0.001,
                max_iter=500,
                early_stopping=True
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

            result = {
                'Recording_Length': prefix_name,
                'Experiment': experiment_name,
                'N_Samples': len(y_train) + len(y_test),
                'N_Train': len(y_train),
                'N_Test': len(y_test),
                'N_Features': len(feature_subset),
                'Architecture': '(100,50)',  # Hidden layer sizes
                'Accuracy': accuracy,
                'Sensitivity': sensitivity,
                'Specificity': specificity,
                'AUC': auc,
            }

            print(f"       ✅ {prefix_name} - {experiment_name}: Acc={accuracy:.3f}, N_Train={len(y_train)}, N_Test={len(y_test)}")
            return result

        except Exception as e:
            print(f"       ❌ NN training failed for {prefix_name} - {experiment_name}: {e}")
            return None
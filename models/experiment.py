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
            (results_df, best_model_name, removal_tracking_list)
        """
        from features.collection import FeatureCollection

        length_manager = RecordingLengthManager()
        results_list = []

        # TRACKING: Recording removal tracking for diagnostic report
        removal_tracking = []

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
                successfully_processed_ids = set()

                for rec in recordings:
                    recording_id = f"{rec.subject_id}_{rec.recording_date}" if hasattr(rec, 'recording_date') else rec.subject_id

                    try:
                        rec_truncated = length_manager.truncate_recording(rec, prefix_length)
                    except Exception as e:
                        removal_tracking.append({
                            'Outcome': outcome,
                            'Recording_Length': prefix_name,
                            'Recording_ID': recording_id,
                            'Subject_ID': rec.subject_id,
                            'Step': 'Truncation',
                            'Reason': str(e) if str(e) else 'Failed to truncate recording'
                        })
                        continue

                    try:
                        rec_clean = cleaner.clean(rec_truncated)
                    except Exception as e:
                        removal_tracking.append({
                            'Outcome': outcome,
                            'Recording_Length': prefix_name,
                            'Recording_ID': recording_id,
                            'Subject_ID': rec.subject_id,
                            'Step': 'Cleaning',
                            'Reason': str(e) if str(e) else 'Failed during signal cleaning'
                        })
                        continue

                    try:
                        windows = win_gen.generate_windows(rec_clean)
                    except Exception as e:
                        removal_tracking.append({
                            'Outcome': outcome,
                            'Recording_Length': prefix_name,
                            'Recording_ID': recording_id,
                            'Subject_ID': rec.subject_id,
                            'Step': 'Windowing',
                            'Reason': str(e) if str(e) else 'Failed to generate windows'
                        })
                        continue

                    try:
                        rec_feats = []
                        for w in windows:
                            f = extractor.extract(w.data, w.sampling_rate)
                            rec_feats.append(f)

                        if rec_feats:
                            all_subject_features.append(
                                aggregator.aggregate(rec_feats, subject_id=rec.subject_id, recording_date=rec.recording_date)
                            )
                            successfully_processed_ids.add(recording_id)
                        else:
                            removal_tracking.append({
                                'Outcome': outcome,
                                'Recording_Length': prefix_name,
                                'Recording_ID': recording_id,
                                'Subject_ID': rec.subject_id,
                                'Step': 'Feature_Extraction',
                                'Reason': 'No features extracted (empty rec_feats)'
                            })
                    except Exception as e:
                        removal_tracking.append({
                            'Outcome': outcome,
                            'Recording_Length': prefix_name,
                            'Recording_ID': recording_id,
                            'Subject_ID': rec.subject_id,
                            'Step': 'Feature_Extraction',
                            'Reason': str(e) if str(e) else 'Failed during feature extraction or aggregation'
                        })
                        continue

                if not all_subject_features:
                    print(f"       ⚠️ No features extracted for {prefix_name}")
                    continue

                # Create feature dataframe and merge with labels
                features_df = pd.DataFrame(all_subject_features)
                features_df['SubjectID'] = features_df['SubjectID'].astype(str).str.strip()

                # Create unique RecordingID to prevent duplicates
                if 'RecordingDate' in features_df.columns:
                    features_df['RecordingDate'] = features_df['RecordingDate'].astype(str).str.strip()
                    features_df['RecordingID'] = features_df['SubjectID'] + '_' + features_df['RecordingDate']
                else:
                    features_df['RecordingID'] = features_df['SubjectID']

                # DIAGNOSTIC: Check for duplicates
                n_recordings_extracted = len(features_df)
                n_unique_recording_ids = features_df['RecordingID'].nunique()
                n_unique_subjects = features_df['SubjectID'].nunique()

                print(f"       📊 Extracted features from {n_recordings_extracted} recordings")
                print(f"          Unique RecordingIDs: {n_unique_recording_ids}, Unique Subjects: {n_unique_subjects}")

                if n_recordings_extracted != n_unique_recording_ids:
                    duplicate_ids = features_df[features_df.duplicated(subset=['RecordingID'], keep=False)]['RecordingID'].unique()
                    print(f"          ⚠️  WARNING: {n_recordings_extracted - n_unique_recording_ids} duplicate RecordingIDs found!")
                    print(f"          Duplicate IDs: {list(duplicate_ids[:5])}{'...' if len(duplicate_ids) > 5 else ''}")

                # Track RecordingIDs before merge
                recording_ids_before_merge = set(features_df['RecordingID'].tolist())

                collection = FeatureCollection(
                    features_df,
                    subject_ids=features_df['SubjectID'].tolist()
                )
                X_df_prefix, y_prefix = collection.merge_with_labels(
                    labels_df, on='SubjectID', outcome=outcome
                )

                # DIAGNOSTIC: Check merge results
                n_after_merge = len(X_df_prefix)
                print(f"          After merge with labels: {n_after_merge} samples")
                if n_after_merge != n_recordings_extracted:
                    print(f"          ⚠️  Merge changed sample count: {n_recordings_extracted} → {n_after_merge}")

                    # Track which recordings were dropped during merge
                    if 'RecordingID' in X_df_prefix.columns:
                        recording_ids_after_merge = set(X_df_prefix['RecordingID'].tolist())
                    elif X_df_prefix.index.name == 'RecordingID':
                        recording_ids_after_merge = set(X_df_prefix.index.tolist())
                    else:
                        # RecordingID might have been set as index already, check before setting
                        recording_ids_after_merge = recording_ids_before_merge  # Assume all kept if can't determine

                    dropped_recording_ids = recording_ids_before_merge - recording_ids_after_merge
                    for dropped_id in dropped_recording_ids:
                        # Get SubjectID from features_df
                        subject_id = features_df[features_df['RecordingID'] == dropped_id]['SubjectID'].iloc[0] if not features_df[features_df['RecordingID'] == dropped_id].empty else 'Unknown'
                        removal_tracking.append({
                            'Outcome': outcome,
                            'Recording_Length': prefix_name,
                            'Recording_ID': dropped_id,
                            'Subject_ID': subject_id,
                            'Step': 'Merge_With_Labels',
                            'Reason': f'Subject not in labels for {outcome} outcome'
                        })

                # Use RecordingID as index to ensure uniqueness
                if 'RecordingID' in X_df_prefix.columns:
                    X_df_prefix = X_df_prefix.set_index('RecordingID')
                elif 'SubjectID' in X_df_prefix.columns:
                    X_df_prefix = X_df_prefix.set_index('SubjectID')

                # DIAGNOSTIC: Check for duplicate index
                if X_df_prefix.index.duplicated().any():
                    n_duplicates = X_df_prefix.index.duplicated().sum()
                    duplicate_ids = X_df_prefix.index[X_df_prefix.index.duplicated()].unique()
                    print(f"          ⚠️  WARNING: {n_duplicates} duplicate indices after set_index!")
                    print(f"          Duplicate indices: {list(duplicate_ids[:5])}{'...' if len(duplicate_ids) > 5 else ''}")

                X_df_prefix = X_df_prefix.select_dtypes(include=[np.number]).fillna(0)
                y_prefix = np.array(y_prefix, dtype=int)

                # DIAGNOSTIC: Final sample count
                final_n_samples = len(X_df_prefix)
                print(f"          Final N_Samples for training: {final_n_samples}")
                if final_n_samples != n_after_merge:
                    print(f"          ⚠️  Sample count changed after processing: {n_after_merge} → {final_n_samples}")

                if len(np.unique(y_prefix)) < 2:
                    print(f"       ⚠️ Insufficient classes for {prefix_name}")
                    continue

                # Run statistical analysis on THIS prefix to find significant features
                # (instead of using global significant_features which may be 0 when combining all lengths)
                from analysis.statistical import StatisticalAnalyzer
                test_method = self.config['analysis']['statistical'].get('test', 'mannwhitneyu')
                correction_method = self.config['analysis']['statistical'].get('correction', 'fdr_bh')

                stats_analyzer = StatisticalAnalyzer(test=test_method, correction_method=correction_method)
                stats_df_prefix = stats_analyzer.compare_groups(
                    X_df_prefix, y_prefix,
                    outcome_name=outcome,
                    feature_names=X_df_prefix.columns.tolist()
                )

                # Extract significant features for THIS prefix
                feat_col = 'feature_name' if 'feature_name' in stats_df_prefix.columns else 'feature'
                sig_feats_prefix = stats_df_prefix[stats_df_prefix['significant'] == True][feat_col].tolist() if feat_col in stats_df_prefix.columns else []
                print(f"       Found {len(sig_feats_prefix)} significant features for {prefix_name}")

                # Build experiment variants (All Features, Incremental Significant Features)
                all_features = X_df_prefix.columns.tolist()
                experiments = {'All Features': all_features}

                if sig_feats_prefix and all(f in X_df_prefix.columns for f in sig_feats_prefix):
                    experiments['All Significant Features'] = sig_feats_prefix

                    # Incremental feature experiments: Top 2, Top 4, Top 6, Top 8, ...
                    if len(sig_feats_prefix) > 1:
                        for n in range(2, len(sig_feats_prefix), 2):
                            top_n = sig_feats_prefix[:n]
                            experiments[f'Top {n} Features'] = top_n

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

        return pd.DataFrame(results_list), "SVM", removal_tracking

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
                                aggregator.aggregate(rec_feats, subject_id=rec.subject_id, recording_date=rec.recording_date)
                            )
                    except:
                        continue

                if not all_subject_features:
                    print(f"       ⚠️ No features extracted for {prefix_name}")
                    continue

                # Create feature dataframe and merge with labels
                features_df = pd.DataFrame(all_subject_features)
                features_df['SubjectID'] = features_df['SubjectID'].astype(str).str.strip()

                # Create unique RecordingID to prevent duplicates
                if 'RecordingDate' in features_df.columns:
                    features_df['RecordingDate'] = features_df['RecordingDate'].astype(str).str.strip()
                    features_df['RecordingID'] = features_df['SubjectID'] + '_' + features_df['RecordingDate']
                else:
                    features_df['RecordingID'] = features_df['SubjectID']

                collection = FeatureCollection(
                    features_df,
                    subject_ids=features_df['SubjectID'].tolist()
                )
                X_df_prefix, y_prefix = collection.merge_with_labels(
                    labels_df, on='SubjectID', outcome=outcome
                )

                # Keep subject IDs before converting to numeric-only
                subject_ids = X_df_prefix['SubjectID'] if 'SubjectID' in X_df_prefix.columns else X_df_prefix.index

                # Use RecordingID as index to ensure uniqueness
                if 'RecordingID' in X_df_prefix.columns:
                    X_df_prefix = X_df_prefix.set_index('RecordingID')
                elif 'SubjectID' in X_df_prefix.columns:
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

    def run_loso_experiments_with_length_prefix(self, pipeline_context, X_df, y, significant_features, recording_lengths):
        """
        Run Leave-One-Subject-Out (LOSO) cross-validation experiments.

        For each subject:
        1. Train on all OTHER subjects
        2. Test on that ONE subject
        3. Repeat for all subjects

        This ensures proper patient-level cross-validation where all recordings
        from the same patient stay together (no data leakage between train/test).

        Runs TWO experiments per recording length:
        - Top 6 significant features
        - All significant features

        Feature selection is done on training subjects only (proper LOSO, no leakage).

        Args:
            pipeline_context: Dict with recordings, cleaner, extractor, etc.
            X_df: Full feature DataFrame (not used - kept for compatibility)
            y: Labels (not used - kept for compatibility)
            significant_features: Significant features from global analysis (not used)
            recording_lengths: List of recording length prefixes to test

        Returns:
            DataFrame with LOSO results including features used
        """
        print(f"\n[MODELS] Running Leave-One-Subject-Out (LOSO) Cross-Validation")
        print(f"    Patient-level CV: Each subject's recordings used ONLY in train OR test")
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

        # Get LOSO configuration
        loso_config = self.config.get('models', {}).get('loso', {})
        top_n_features = loso_config.get('top_n_features', 6)

        for prefix_length in tqdm(recording_lengths, desc="    Testing Prefixes (LOSO)"):
            try:
                # Truncate recordings to prefix length
                length_manager = RecordingLengthManager()
                truncated_recordings = [
                    length_manager.truncate_recording(rec, prefix_length)
                    for rec in recordings
                ]

                prefix_name = RecordingLengthManager.format_length_name(prefix_length)

                # Extract features for this prefix (from ALL recordings)
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
                                aggregator.aggregate(rec_feats, subject_id=rec.subject_id, recording_date=rec.recording_date)
                            )
                    except:
                        continue

                if not all_subject_features:
                    print(f"       ⚠️ No features extracted for {prefix_name}")
                    continue

                # Create feature dataframe and merge with labels
                features_df = pd.DataFrame(all_subject_features)
                features_df['SubjectID'] = features_df['SubjectID'].astype(str).str.strip()

                # Create unique RecordingID to prevent duplicates
                if 'RecordingDate' in features_df.columns:
                    features_df['RecordingDate'] = features_df['RecordingDate'].astype(str).str.strip()
                    features_df['RecordingID'] = features_df['SubjectID'] + '_' + features_df['RecordingDate']
                else:
                    features_df['RecordingID'] = features_df['SubjectID']

                collection = FeatureCollection(
                    features_df,
                    subject_ids=features_df['SubjectID'].tolist()
                )
                X_df_prefix, y_prefix = collection.merge_with_labels(
                    labels_df, on='SubjectID', outcome=outcome
                )

                # For LOSO, keep SubjectID as a column for grouping (don't set as index yet)
                # Keep subject IDs array for LOSO splitting
                subject_ids_array = X_df_prefix['SubjectID'].values if 'SubjectID' in X_df_prefix.columns else None

                # Use RecordingID as index to ensure uniqueness, but keep SubjectID column for LOSO
                if 'RecordingID' in X_df_prefix.columns:
                    # For LOSO, we need SubjectID column to group recordings
                    # Keep SubjectID column, set RecordingID as index
                    if 'SubjectID' in X_df_prefix.columns:
                        subject_id_col = X_df_prefix['SubjectID'].copy()
                        X_df_prefix = X_df_prefix.set_index('RecordingID')
                        X_df_prefix['SubjectID'] = subject_id_col
                elif 'SubjectID' in X_df_prefix.columns:
                    X_df_prefix = X_df_prefix.set_index('SubjectID')

                # Select numeric columns only
                numeric_cols = X_df_prefix.select_dtypes(include=[np.number]).columns.tolist()
                X_df_numeric = X_df_prefix[numeric_cols]

                # Keep SubjectID for LOSO grouping
                if 'SubjectID' in X_df_prefix.columns:
                    subject_ids_for_loso = X_df_prefix['SubjectID'].values
                else:
                    subject_ids_for_loso = X_df_prefix.index.values

                X_df_prefix = X_df_numeric
                y_prefix = np.array(y_prefix, dtype=int)

                if len(np.unique(y_prefix)) < 2:
                    print(f"       ⚠️ Insufficient classes for {prefix_name}")
                    continue

                # Get unique subjects for LOSO
                unique_subjects = np.unique(subject_ids_for_loso)
                n_subjects = len(unique_subjects)

                print(f"       Running LOSO with {n_subjects} subjects for {prefix_name}")

                # Run LOSO: For each subject, train on others and test on this one
                loso_predictions_top6 = []  # For top 6 features experiment
                loso_predictions_all = []   # For all significant features experiment
                loso_true_labels = []
                features_used_top6 = None
                features_used_all = None

                for subject_id in unique_subjects:
                    try:
                        # Split by subject: train = all others, test = this subject
                        test_mask = subject_ids_for_loso == subject_id
                        train_mask = ~test_mask

                        X_train_full = X_df_prefix[train_mask]
                        X_test_full = X_df_prefix[test_mask]
                        y_train = y_prefix[train_mask]
                        y_test = y_prefix[test_mask]

                        # Skip if no samples
                        if len(X_train_full) == 0 or len(X_test_full) == 0:
                            continue

                        # Perform statistical feature selection on TRAINING subjects only
                        stats_analyzer = StatisticalAnalyzer(test=test_method, correction_method=correction_method)
                        stats_df_train = stats_analyzer.compare_groups(
                            X_train_full, y_train,
                            outcome_name=outcome,
                            feature_names=X_train_full.columns.tolist()
                        )

                        # Extract significant features from training data
                        feat_col = 'feature_name' if 'feature_name' in stats_df_train.columns else 'feature'
                        sig_feats_train = stats_df_train[stats_df_train['significant'] == True][feat_col].tolist()

                        if len(sig_feats_train) == 0:
                            # No significant features, skip this fold
                            continue

                        # Experiment 1: Top N significant features (configurable)
                        top_n_features_list = sig_feats_train[:top_n_features]
                        if features_used_top6 is None:
                            features_used_top6 = top_n_features_list  # Store for reporting

                        pred_top6 = self._train_and_test_loso_fold(
                            X_train_full, X_test_full, y_train, y_test, top_n_features_list
                        )
                        if pred_top6 is not None:
                            loso_predictions_top6.extend(pred_top6)

                        # Experiment 2: All significant features
                        if features_used_all is None:
                            features_used_all = sig_feats_train  # Store for reporting

                        pred_all = self._train_and_test_loso_fold(
                            X_train_full, X_test_full, y_train, y_test, sig_feats_train
                        )
                        if pred_all is not None:
                            loso_predictions_all.extend(pred_all)

                        # Store true labels
                        loso_true_labels.extend(y_test)

                    except Exception as e:
                        print(f"         ⚠️ LOSO fold failed for subject {subject_id}: {e}")
                        continue

                # Calculate metrics for Top N features
                if len(loso_predictions_top6) > 0 and len(loso_true_labels) > 0:
                    metrics_top6 = self._calculate_loso_metrics(
                        loso_true_labels, loso_predictions_top6,
                        prefix_name, f"LOSO: Top {top_n_features} Features",
                        features_used_top6, n_subjects
                    )
                    if metrics_top6:
                        all_results.append(metrics_top6)

                # Calculate metrics for All significant features
                if len(loso_predictions_all) > 0 and len(loso_true_labels) > 0:
                    metrics_all = self._calculate_loso_metrics(
                        loso_true_labels, loso_predictions_all,
                        prefix_name, "LOSO: All Significant Features",
                        features_used_all, n_subjects
                    )
                    if metrics_all:
                        all_results.append(metrics_all)

            except Exception as e:
                print(f"       ❌ LOSO failed for {prefix_name}: {e}")
                import traceback
                traceback.print_exc()

        if not all_results:
            print("    ⚠️  No LOSO results generated")
            return pd.DataFrame()

        results_df = pd.DataFrame(all_results)
        return results_df

    def _train_and_test_loso_fold(self, X_train_df, X_test_df, y_train, y_test, feature_subset):
        """
        Train SVM on training subjects and test on held-out subject (one LOSO fold).

        Args:
            X_train_df: Training features DataFrame
            X_test_df: Test features DataFrame
            y_train: Training labels
            y_test: Test labels
            feature_subset: List of features to use

        Returns:
            List of predictions for test samples, or None if failed
        """
        try:
            # Extract feature values
            X_train = X_train_df[feature_subset].values
            X_test = X_test_df[feature_subset].values

            # Train SVM classifier
            clf = SVMClassifier()
            n_folds = max(2, min(self.model_cfg['cv_folds'], min(np.unique(y_train, return_counts=True)[1])))
            tuner = GridSearchTuner(cv_folds=n_folds)
            best_model, best_params = tuner.tune(clf, X_train, y_train, self.model_cfg['tuning']['svm'])

            # Predict on test subject
            y_pred = best_model.predict(X_test)

            return y_pred.tolist()

        except Exception as e:
            # Silently fail for individual folds
            return None

    def _calculate_loso_metrics(self, y_true, y_pred, prefix_name, experiment_name, features_used, n_subjects):
        """
        Calculate metrics from LOSO predictions.

        Args:
            y_true: True labels (all test subjects concatenated)
            y_pred: Predicted labels (all test subjects concatenated)
            prefix_name: Recording length prefix name
            experiment_name: Experiment name (e.g., "LOSO: Top 6 Features")
            features_used: List of feature names used
            n_subjects: Number of subjects in LOSO

        Returns:
            Dict with metrics
        """
        try:
            y_true = np.array(y_true)
            y_pred = np.array(y_pred)

            # Calculate metrics
            accuracy = accuracy_score(y_true, y_pred)
            sensitivity = recall_score(y_true, y_pred, average='binary', pos_label=1, zero_division=0)

            # Specificity
            cm = confusion_matrix(y_true, y_pred)
            if cm.shape[0] == 2:
                tn, fp, fn, tp = cm.ravel()
                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            else:
                specificity = 0

            result = {
                'Recording_Length': prefix_name,
                'Experiment': experiment_name,
                'N_Subjects': n_subjects,
                'N_Samples': len(y_true),
                'N_Features': len(features_used) if features_used else 0,
                'Features_Used': ', '.join(features_used) if features_used else '',
                'Accuracy': accuracy,
                'Sensitivity': sensitivity,
                'Specificity': specificity,
            }

            print(f"       ✅ {prefix_name} - {experiment_name}: Acc={accuracy:.3f}, N_Features={len(features_used) if features_used else 0}")
            return result

        except Exception as e:
            print(f"       ❌ Metric calculation failed for {prefix_name} - {experiment_name}: {e}")
            return None

    def run_loro_experiments_with_length_prefix(self, pipeline_context, X_df, y, significant_features, recording_lengths):
        """
        Run Leave-One-Recording-Out (LOROCV) cross-validation experiments.

        For each recording individually:
        1. Test on that ONE recording
        2. Train on ALL other recordings EXCEPT any from the same participant
        3. Repeat for all recordings

        This is different from LOSO:
        - LOSO: Test on ALL recordings from one subject together
        - LORO: Test on ONE recording at a time, exclude all recordings from same subject

        Runs incremental feature experiments:
        - Top 2, Top 4, Top 6, Top 8, ..., All significant features

        Args:
            pipeline_context: Dict with recordings, cleaner, extractor, etc.
            X_df: Full feature DataFrame (not used)
            y: Labels (not used)
            significant_features: Significant features from global analysis (not used)
            recording_lengths: List of recording length prefixes to test

        Returns:
            DataFrame with LORO results including features used
        """
        print(f"\n[MODELS] Running Leave-One-Recording-Out (LORO) Cross-Validation")
        print(f"    Tests each recording individually while excluding all recordings from same participant")
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

        for prefix_length in tqdm(recording_lengths, desc="    Testing Prefixes (LORO)"):
            try:
                # Truncate recordings to prefix length
                length_manager = RecordingLengthManager()
                truncated_recordings = [
                    length_manager.truncate_recording(rec, prefix_length)
                    for rec in recordings
                ]

                prefix_name = RecordingLengthManager.format_length_name(prefix_length)

                # Extract features for this prefix (from ALL recordings)
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
                                aggregator.aggregate(rec_feats, subject_id=rec.subject_id, recording_date=rec.recording_date)
                            )
                    except:
                        continue

                if not all_subject_features:
                    print(f"       ⚠️ No features extracted for {prefix_name}")
                    continue

                # Create feature dataframe and merge with labels
                features_df = pd.DataFrame(all_subject_features)
                features_df['SubjectID'] = features_df['SubjectID'].astype(str).str.strip()

                # Create unique RecordingID
                if 'RecordingDate' in features_df.columns:
                    features_df['RecordingDate'] = features_df['RecordingDate'].astype(str).str.strip()
                    features_df['RecordingID'] = features_df['SubjectID'] + '_' + features_df['RecordingDate']
                else:
                    features_df['RecordingID'] = features_df['SubjectID']

                collection = FeatureCollection(
                    features_df,
                    subject_ids=features_df['SubjectID'].tolist()
                )
                X_df_prefix, y_prefix = collection.merge_with_labels(
                    labels_df, on='SubjectID', outcome=outcome
                )

                # Keep SubjectID and RecordingID for grouping
                subject_id_col = X_df_prefix['SubjectID'].copy() if 'SubjectID' in X_df_prefix.columns else None
                recording_id_col = X_df_prefix['RecordingID'].copy() if 'RecordingID' in X_df_prefix.columns else None

                # Set RecordingID as index
                if 'RecordingID' in X_df_prefix.columns:
                    X_df_prefix = X_df_prefix.set_index('RecordingID')
                elif 'SubjectID' in X_df_prefix.columns:
                    X_df_prefix = X_df_prefix.set_index('SubjectID')

                # Add SubjectID back as column for grouping
                if subject_id_col is not None:
                    X_df_prefix['SubjectID'] = subject_id_col.values

                # Keep SubjectID mapping for exclusion logic (before filtering to numeric)
                subject_ids_array = X_df_prefix['SubjectID'].values if 'SubjectID' in X_df_prefix.columns else None

                # Select ONLY numeric columns (fixes np.isnan() TypeError)
                X_df_numeric = X_df_prefix.select_dtypes(include=[np.number])

                X_df_prefix = X_df_numeric
                y_prefix = np.array(y_prefix, dtype=int)

                if len(np.unique(y_prefix)) < 2:
                    print(f"       ⚠️ Insufficient classes for {prefix_name}")
                    continue

                n_recordings = len(X_df_prefix)
                print(f"       Running LORO with {n_recordings} recordings for {prefix_name}")

                # Statistical analysis on full dataset to get significant features ranking
                stats_analyzer = StatisticalAnalyzer(test=test_method, correction_method=correction_method)
                stats_df_full = stats_analyzer.compare_groups(
                    X_df_prefix, y_prefix,
                    outcome_name=outcome,
                    feature_names=X_df_prefix.columns.tolist()
                )

                feat_col = 'feature_name' if 'feature_name' in stats_df_full.columns else 'feature'
                sig_feats_ranked = stats_df_full[stats_df_full['significant'] == True][feat_col].tolist()

                if len(sig_feats_ranked) < 2:
                    print(f"       ⚠️ Not enough significant features for {prefix_name}")
                    continue

                # Build incremental feature sets: Top 2, Top 4, Top 6, ..., All
                feature_sets = {}
                for n in range(2, len(sig_feats_ranked), 2):
                    feature_sets[f'Top {n} Features'] = sig_feats_ranked[:n]
                feature_sets['All Significant Features'] = sig_feats_ranked

                # For each feature set, run LORO
                for exp_name, feature_subset in feature_sets.items():
                    loro_predictions = []
                    loro_true_labels = []

                    # LORO: For each recording, test on it, train on all others except same subject
                    for idx in range(n_recordings):
                        try:
                            # Get subject of this recording
                            test_subject = subject_ids_array[idx] if subject_ids_array is not None else None

                            # Create masks
                            test_mask = np.zeros(n_recordings, dtype=bool)
                            test_mask[idx] = True

                            # Train mask: all recordings EXCEPT test recording AND any from same subject
                            if test_subject is not None and subject_ids_array is not None:
                                train_mask = (subject_ids_array != test_subject) & (~test_mask)
                            else:
                                train_mask = ~test_mask

                            X_train = X_df_prefix.iloc[train_mask][feature_subset].values
                            X_test = X_df_prefix.iloc[test_mask][feature_subset].values
                            y_train = y_prefix[train_mask]
                            y_test = y_prefix[test_mask]

                            if len(X_train) == 0 or len(X_test) == 0:
                                continue

                            # Train SVM
                            clf = SVMClassifier()
                            n_folds = max(2, min(self.model_cfg['cv_folds'], min(np.unique(y_train, return_counts=True)[1])))
                            tuner = GridSearchTuner(cv_folds=n_folds)
                            best_model, best_params = tuner.tune(clf, X_train, y_train, self.model_cfg['tuning']['svm'])

                            # Predict on test recording
                            y_pred = best_model.predict(X_test)

                            loro_predictions.extend(y_pred.tolist())
                            loro_true_labels.extend(y_test.tolist())

                        except Exception as e:
                            # Silently skip failed folds
                            continue

                    # Calculate metrics for this feature set
                    if len(loro_predictions) > 0 and len(loro_true_labels) > 0:
                        metrics = self._calculate_loro_metrics(
                            loro_true_labels, loro_predictions,
                            prefix_name, exp_name, feature_subset, n_recordings
                        )
                        if metrics:
                            all_results.append(metrics)

            except Exception as e:
                print(f"       ❌ LORO failed for {prefix_name}: {e}")
                import traceback
                traceback.print_exc()

        if not all_results:
            print("    ⚠️  No LORO results generated")
            return pd.DataFrame()

        results_df = pd.DataFrame(all_results)
        return results_df

    def _calculate_loro_metrics(self, y_true, y_pred, prefix_name, experiment_name, features_used, n_recordings):
        """
        Calculate metrics from LORO predictions.

        Args:
            y_true: True labels (all test recordings)
            y_pred: Predicted labels (all test recordings)
            prefix_name: Recording length prefix name
            experiment_name: Experiment name (e.g., "Top 6 Features")
            features_used: List of feature names used
            n_recordings: Number of recordings tested

        Returns:
            Dict with metrics
        """
        try:
            y_true = np.array(y_true)
            y_pred = np.array(y_pred)

            # Calculate metrics
            accuracy = accuracy_score(y_true, y_pred)
            sensitivity = recall_score(y_true, y_pred, average='binary', pos_label=1, zero_division=0)

            # Specificity
            cm = confusion_matrix(y_true, y_pred)
            if cm.shape[0] == 2:
                tn, fp, fn, tp = cm.ravel()
                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            else:
                specificity = 0

            result = {
                'Recording_Length': prefix_name,
                'Experiment': experiment_name,
                'N_Recordings': n_recordings,
                'N_Samples': len(y_true),
                'N_Features': len(features_used) if features_used else 0,
                'Features_Used': ', '.join(features_used) if features_used else '',
                'Accuracy': accuracy,
                'Sensitivity': sensitivity,
                'Specificity': specificity,
            }

            print(f"       ✅ {prefix_name} - {experiment_name}: Acc={accuracy:.3f}, N_Features={len(features_used) if features_used else 0}")
            return result

        except Exception as e:
            print(f"       ❌ Metric calculation failed for {prefix_name} - {experiment_name}: {e}")
            return None
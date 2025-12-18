import os
import numpy as np

# =============================================================================
# CRITICAL FIX: Monkey-patch np.bool8 here too for safety
# =============================================================================
if not hasattr(np, 'bool8'):
    np.bool8 = np.bool_
if not hasattr(np, 'float'):
    np.float = float  # Fixes the Excel loading crash
# =============================================================================
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import yaml
import argparse
import traceback
# --- Import Your Pipeline Modules ---
from data.loaders import MATDataLoader
from data.clinical_labels import ClinicalLabels
from data.exporters import ExcelExporter
from preprocessing.cleaner import SignalCleaner
from preprocessing.windowing import WindowConfig, WindowGenerator
from preprocessing.recording_length import RecordingLengthManager
from features.extractors.breathing import BreathingParameterExtractor
from features.aggregator import FeatureAggregator
from features.collection import FeatureCollection
from analysis.statistical import StatisticalAnalyzer
from models.experiment import ExperimentManager
from visualization.interactive import InteractivePlotter


def load_config(config_name="config.yaml"):
    paths = [Path(config_name), Path(__file__).resolve().parent.parent.parent / config_name]
    for p in paths:
        if p.exists():
            print(f"    Found config at: {p.absolute()}")
            with open(p, 'r') as f: return yaml.safe_load(f)
    raise FileNotFoundError(f"Config not found.")


def main():
    print(f"🚀 Starting Pipeline...")
    try:
        config = load_config()
    except Exception as e:
        return print(f"❌ CONFIG ERROR: {e}")

    data_cfg = config['data']
    base_out_dir = Path(data_cfg["output_dir"]) if os.path.isabs(data_cfg["output_dir"]) else Path(
        __file__).resolve().parent.parent.parent / data_cfg["output_dir"]
    base_out_dir.mkdir(parents=True, exist_ok=True)

    # Initialize plotter
    global_plotter = InteractivePlotter(output_dir=base_out_dir / "general_plots")

    # --- PHASE 1: DATA ---
    print("\n[PHASE 1] Data Loading & Extraction")

    try:
        labels_df = pd.read_excel(data_cfg['labels_file'], engine='openpyxl')
        id_col = data_cfg.get("subject_id_column", "SubjectID")
        if id_col not in labels_df.columns:
            for col in labels_df.columns:
                if col.strip() == id_col.strip():
                    labels_df = labels_df.rename(columns={col: "SubjectID"})
                    break
        else:
            labels_df = labels_df.rename(columns={id_col: "SubjectID"})
        labels_df['SubjectID'] = labels_df['SubjectID'].astype(str).str.strip()
    except Exception as e:
        return print(f"❌ LABEL ERROR: {e}")

    # Load recordings once (keep them for prefix experiments)
    loader = MATDataLoader(default_sampling_rate=data_cfg["default_sampling_rate"])
    recordings = loader.load_batch(str(data_cfg["data_dir"]))
    if not recordings: return print("❌ No recordings found.")

    # Get recording lengths for prefix experiments and extend dynamically
    recording_lengths = data_cfg.get('recording_lengths', [None])
    if recording_lengths is None:
        recording_lengths = [None]

    # Calculate the minimum recording duration across all recordings to extend prefix list
    length_manager = RecordingLengthManager()
    recording_durations = []
    for rec in recordings:
        duration_min = length_manager.get_recording_duration_minutes(rec)
        recording_durations.append(duration_min)

    if recording_durations:
        min_recording_length = min(recording_durations)
        max_recording_length = max(recording_durations)
        print(f"    Recording durations: min={min_recording_length:.1f}min, max={max_recording_length:.1f}min")

        # Extend prefix list to include steps up to the minimum recording length
        # This ensures all recordings have data for all prefixes
        extended_prefixes = []
        for prefix in recording_lengths:
            if prefix is None:
                # Keep None to represent full recording
                continue
            extended_prefixes.append(prefix)

        # Add 5-minute increments from 20 up to the minimum recording length
        if extended_prefixes:
            max_configured_prefix = max([p for p in extended_prefixes if p is not None])
            current_prefix = max_configured_prefix + 5
            while current_prefix < min_recording_length:
                extended_prefixes.append(current_prefix)
                current_prefix += 5

        # Add the minimum recording length as the final numeric prefix
        # (this ensures we test the maximum usable length before "full")
        if min_recording_length > 5:
            # Round to nearest integer for cleaner display
            min_length_rounded = int(min_recording_length)
            if min_length_rounded not in extended_prefixes:
                extended_prefixes.append(min_length_rounded)

        # Finally add None for full recording length
        extended_prefixes.append(None)
        recording_lengths = extended_prefixes

    print(f"\n📏 Recording Length Prefixes: {[RecordingLengthManager.format_length_name(l) for l in recording_lengths]}")

    # Initialize processing components
    cleaner = SignalCleaner(config)
    win_gen = WindowGenerator(WindowConfig(config['preprocessing']['windowing']['window_size'],
                                           config['preprocessing']['windowing']['overlap']))
    extractor = BreathingParameterExtractor(config)
    aggregator = FeatureAggregator(config)

    # Extract features using FULL recordings first
    all_subject_features = []
    feature_matrices = {}

    # Check if signal plots already exist (unless force regenerate is enabled)
    signals_dir = base_out_dir / "general_plots" / "signals"
    force_regenerate = config.get('visualization', {}).get('force_regenerate_signals', False)

    if force_regenerate:
        skip_visualization = False
        print(f"    🔄 Force regeneration enabled, will recreate signal plots...")
    else:
        skip_visualization = signals_dir.exists() and len(list(signals_dir.glob("*.html"))) > 0

    if skip_visualization:
        print(f"    ℹ️  Signal plots already exist ({len(list(signals_dir.glob('*.html')))} plots found), skipping visualization...")
    else:
        print(f"    Creating breathmetrics plots for ALL recordings (5 minutes each)...")

    viz_created_count = 0

    for idx, rec in enumerate(tqdm(recordings, desc="Extracting")):
        try:
            rec_clean = cleaner.clean(rec)

            # Extract breath peaks for visualization (ALL recordings, 5 minutes each)
            if not skip_visualization:
                try:
                    # Process first 5 minutes for visualization
                    viz_duration_sec = 300  # 5 minutes
                    viz_samples = int(viz_duration_sec * rec_clean.sampling_rate)

                    if len(rec_clean.data) > viz_samples:
                        rec_clean_viz = rec_clean.data[:viz_samples]
                        rec_raw_viz = rec.data[:viz_samples]
                    else:
                        rec_clean_viz = rec_clean.data
                        rec_raw_viz = rec.data

                    _, breath_peaks = extractor.extract_with_details(rec_clean_viz, rec_clean.sampling_rate)
                    global_plotter.plot_signal_traces(rec_raw_viz, rec_clean_viz, rec.sampling_rate, rec.subject_id, breath_peaks)
                    viz_created_count += 1
                except Exception as e:
                    print(f"    ⚠️  Visualization failed for {rec.subject_id}: {e}")

            try:
                windows = win_gen.generate_windows(rec_clean)
            except ValueError:
                continue

            rec_feats = []
            for w in windows:
                f = extractor.extract(w.data, w.sampling_rate)
                f['WindowIndex'] = w.window_index
                f['Time'] = w.start_time
                rec_feats.append(f)

            if not rec_feats: continue

            all_subject_features.append(aggregator.aggregate(rec_feats, subject_id=rec.subject_id))
            feature_matrices[rec.subject_id] = pd.DataFrame(rec_feats).set_index('WindowIndex').select_dtypes(
                include=[np.number])

        except Exception:
            pass

    if not skip_visualization and viz_created_count > 0:
        print(f"    ✅ Created {viz_created_count} signal visualizations")

    if not all_subject_features: return print("❌ No features extracted.")
    master_features_df = pd.DataFrame(all_subject_features)
    master_features_df['SubjectID'] = master_features_df['SubjectID'].astype(str).str.strip()
    master_features_df.to_csv(base_out_dir / "all_extracted_features.csv", index=False)

    # --- PHASE 2: ANALYSIS ---
    print("\n[PHASE 2] Analysis Loop")

    for outcome in data_cfg.get("outcomes", []):
        outcome = outcome.strip()
        print(f"\n🔵 ANALYZING: {outcome}")
        out_dir = base_out_dir / outcome
        out_dir.mkdir(exist_ok=True)

        try:
            # 1. Filter & Merge
            curr_labels = labels_df.copy()
            if outcome == "Recovery" and "currentConsciousness" in curr_labels.columns:
                curr_labels = curr_labels[curr_labels["currentConsciousness"] == 0]
                print(f"    ℹ️ Recovery Filter: Using {len(curr_labels)} UWS subjects.")

            if outcome not in curr_labels.columns: continue

            collection = FeatureCollection(master_features_df, subject_ids=master_features_df['SubjectID'].tolist())
            X_df, y = collection.merge_with_labels(curr_labels, on='SubjectID', outcome=outcome)

            if 'SubjectID' in X_df.columns:
                valid_ids = X_df['SubjectID'].tolist()
                X_df = X_df.set_index('SubjectID')
            else:
                valid_ids = []

            # Clean
            X_df = X_df.select_dtypes(include=[np.number]).fillna(0)
            y = np.array(y, dtype=int)
            if len(np.unique(y)) < 2: continue

            # 2. Stats
            print("    Running Stats...")
            stats = StatisticalAnalyzer(test=config['analysis']['statistical']['test'],
                                        correction_method=config['analysis']['statistical']['correction'])
            stats_df = stats.compare_groups(X_df, y, outcome_name=outcome, feature_names=X_df.columns.tolist())

            feat_col = 'feature_name' if 'feature_name' in stats_df.columns else 'feature'
            sig_feats = stats_df[stats_df['significant'] == True][
                feat_col].tolist() if feat_col in stats_df.columns else []
            print(f"    Significant features: {len(sig_feats)}")

            # 3. Setup Outcome-Specific Plotter
            plotter = InteractivePlotter(output_dir=out_dir / "plots")

            # Violin Plots
            plotter.plot_feature_violins(X_df, y, outcome)
            plotter.plot_statistical_ranking(stats_df)
            plotter.plot_feature_distributions(X_df, y, outcome, stats_df)

            # 3.5. PCA Analysis and Visualization
            print("    Running PCA Analysis...")
            from analysis.dimensionality import PCAReducer

            # Determine number of components (up to 10 or number of features)
            n_components_full = min(10, X_df.shape[1], X_df.shape[0])

            if n_components_full >= 2:
                try:
                    # Create PCA reducer for variance analysis
                    pca_full = PCAReducer(n_components=n_components_full, scale_data=True)
                    pca_full.fit(X_df)

                    # Get explained variance
                    explained_var = pca_full.get_explained_variance()
                    cumulative_var = pca_full.get_cumulative_variance()

                    print(f"    📊 PCA: {cumulative_var[0]*100:.1f}% variance explained by PC1")
                    if len(cumulative_var) > 1:
                        print(f"    📊 PCA: {cumulative_var[1]*100:.1f}% cumulative variance by PC2")

                    # Prepare 2D visualization
                    pca_2d = PCAReducer(n_components=2, scale_data=True)
                    pca_viz_df = pca_2d.fit(X_df).prepare_visualization_data(
                        X_df, labels=y, subject_ids=X_df.index.tolist()
                    )

                    # Get feature loadings (which original features contribute to each PC)
                    loadings_df = pca_2d.get_loadings()
                    top_features_pc1 = loadings_df['PC1'].abs().sort_values(ascending=False).head(10)
                    top_features_pc2 = loadings_df['PC2'].abs().sort_values(ascending=False).head(10)

                    # Print top contributing features for interpretation
                    print(f"\n    🔍 Top Features Contributing to PC1 ({cumulative_var[0]*100:.1f}% variance):")
                    for i, (feat, loading) in enumerate(zip(top_features_pc1.index, loadings_df.loc[top_features_pc1.index, 'PC1'].values), 1):
                        print(f"       {i}. {feat}: {loading:+.3f}")

                    print(f"\n    🔍 Top Features Contributing to PC2 (additional {explained_var[1]*100:.1f}% variance):")
                    for i, (feat, loading) in enumerate(zip(top_features_pc2.index, loadings_df.loc[top_features_pc2.index, 'PC2'].values), 1):
                        print(f"       {i}. {feat}: {loading:+.3f}")
                    print()  # Empty line for readability

                    # Plot 1: PCA scatter plot (PC1 vs PC2)
                    plotter.plot_pca_scatter(pca_viz_df, outcome, explained_var[:2])

                    # Plot 2: Variance explained (scree plot)
                    variance_df = pca_full.prepare_variance_plot_data(X_df, max_components=n_components_full)
                    plotter.plot_pca_variance(variance_df)

                    # Plot 3: Feature loadings (top contributors to PC1 and PC2)
                    plotter.plot_pca_loadings(loadings_df, top_features_pc1.index.tolist(), top_features_pc2.index.tolist())

                    # Prepare PCA results for Excel export
                    pca_results = pd.DataFrame({
                        'Component': [f'PC{i+1}' for i in range(len(explained_var))],
                        'Explained_Variance': explained_var,
                        'Cumulative_Variance': cumulative_var
                    })

                    # Add top contributing features summary to Excel
                    top_contributors = pd.DataFrame({
                        'PC1_Feature': top_features_pc1.index.tolist(),
                        'PC1_Loading': loadings_df.loc[top_features_pc1.index, 'PC1'].values,
                        'PC2_Feature': top_features_pc2.index.tolist(),
                        'PC2_Loading': loadings_df.loc[top_features_pc2.index, 'PC2'].values
                    })

                    # Store for later export
                    pca_loadings_export = loadings_df.copy()
                    pca_top_contributors = top_contributors

                except Exception as e:
                    print(f"    ⚠️  PCA analysis failed: {e}")
                    pca_results = None
                    pca_loadings_export = None
                    pca_top_contributors = None
            else:
                print(f"    ⚠️  Not enough components for PCA (need at least 2, have {n_components_full})")
                pca_results = None
                pca_loadings_export = None
                pca_top_contributors = None

            # 4. Save Feature Matrices (in dedicated directory)
            feature_matrix_plotter = InteractivePlotter(output_dir=out_dir / "feature_matrices")
            for sid in valid_ids:
                if sid in feature_matrices:
                    safe = "".join([c for c in sid if c.isalnum()]).strip()
                    feature_matrix_plotter.plot_feature_matrix(feature_matrices[sid], filename=f"{safe}.html")

            # 5. Run Models with Recording Length Experiments
            print(f"    Running models across {len(recording_lengths)} recording length prefixes...")
            exp_manager = ExperimentManager(config)

            # Create pipeline context for the experiment manager
            pipeline_context = {
                'recordings': recordings,
                'labels_df': curr_labels,  # Use filtered labels for this outcome
                'outcome': outcome,
                'cleaner': cleaner,
                'win_gen': win_gen,
                'extractor': extractor,
                'aggregator': aggregator
            }

            # Run SVM experiments
            svm_results_df, best_model_name = exp_manager.run_experiments_with_length_prefix(
                pipeline_context, X_df, y, sig_feats, recording_lengths
            )

            # Run Neural Network experiments (if enabled)
            nn_results_df = None
            nn_enabled = config.get('models', {}).get('neural_network', {}).get('enabled', False)

            if nn_enabled:
                print(f"    Running Neural Network models across {len(recording_lengths)} prefixes...")
                nn_results_df = exp_manager.run_neural_network_experiments_with_length_prefix(
                    pipeline_context, X_df, y, sig_feats, recording_lengths
                )
            else:
                print(f"    ⚠️  Neural Network training is DISABLED (set models.neural_network.enabled: true to enable)")

            # Run LOSO (Leave-One-Subject-Out) cross-validation (if enabled)
            loso_results_df = None
            loso_enabled = config.get('models', {}).get('loso', {}).get('enabled', False)

            if loso_enabled:
                print(f"    Running Leave-One-Subject-Out (LOSO) cross-validation...")
                loso_results_df = exp_manager.run_loso_experiments_with_length_prefix(
                    pipeline_context, X_df, y, sig_feats, recording_lengths
                )
            else:
                print(f"    ⚠️  LOSO cross-validation is DISABLED (set models.loso.enabled: true to enable)")

            # 6. Export - Multiple Sheets
            exporter = ExcelExporter(filepath=str(out_dir / f"REPORT_{outcome}.xlsx"))

            # Add SVM results
            exporter.add_sheet("SVM_Results_By_Prefix", svm_results_df)

            # Add Neural Network results (only if enabled and generated)
            if nn_enabled and nn_results_df is not None and not nn_results_df.empty:
                exporter.add_sheet("NeuralNetwork_Results", nn_results_df)

            # Add LOSO results (only if enabled and generated)
            if loso_enabled and loso_results_df is not None and not loso_results_df.empty:
                exporter.add_sheet("LOSO_Results", loso_results_df)

            # Add PCA results if available
            if pca_results is not None:
                exporter.add_sheet("PCA_Variance_Analysis", pca_results)
            if pca_top_contributors is not None:
                exporter.add_sheet("PCA_Top_Contributors", pca_top_contributors)
            if pca_loadings_export is not None:
                exporter.add_sheet("PCA_All_Feature_Loadings", pca_loadings_export)

            # Add statistical results
            exporter.add_statistical_results_sheet(stats_df)

            exporter.write()
            print(f"    ✅ Completed {outcome}")

        except Exception as e:
            print(f"    ❌ FAILED {outcome}: {e}")
            traceback.print_exc()

    print("\n✅ Pipeline Finished.")


if __name__ == "__main__":
    main()
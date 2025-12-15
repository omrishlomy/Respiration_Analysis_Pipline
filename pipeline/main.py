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

    # Get recording lengths for prefix experiments
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
    elif not force_regenerate:
        # Only visualize first N recordings to save time (can be changed in config)
        viz_limit = config.get('visualization', {}).get('max_signal_plots', 10)
        print(f"    Creating breathmetrics plots for first {viz_limit} recordings...")

    # Set viz_limit even if skipping, to avoid errors
    viz_limit = config.get('visualization', {}).get('max_signal_plots', 10)
    viz_created_count = 0

    for idx, rec in enumerate(tqdm(recordings, desc="Extracting")):
        try:
            rec_clean = cleaner.clean(rec)

            # Extract breath peaks for visualization (only for first N recordings, and only if not already done)
            if not skip_visualization and idx < viz_limit:
                try:
                    # Only process first 2 minutes for visualization (much faster!)
                    viz_duration_sec = 120  # 2 minutes
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

            results_df, best_model_name = exp_manager.run_experiments_with_length_prefix(
                pipeline_context, X_df, y, sig_feats, recording_lengths
            )

            # 6. Export
            exporter = ExcelExporter(filepath=str(out_dir / f"REPORT_{outcome}.xlsx"))
            exporter.add_sheet("Model_Experiments_By_Prefix", results_df)
            exporter.add_statistical_results_sheet(stats_df)
            exporter.write()
            print(f"    ✅ Completed {outcome}")

        except Exception as e:
            print(f"    ❌ FAILED {outcome}: {e}")
            traceback.print_exc()

    print("\n✅ Pipeline Finished.")


if __name__ == "__main__":
    main()
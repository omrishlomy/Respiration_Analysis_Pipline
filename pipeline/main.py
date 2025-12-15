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

    for rec in tqdm(recordings, desc="Extracting"):
        try:
            rec_clean = cleaner.clean(rec)

            # Extract breath peaks for visualization
            _, breath_peaks = extractor.extract_with_details(rec_clean.data, rec_clean.sampling_rate)
            global_plotter.plot_signal_traces(rec.data, rec_clean.data, rec.sampling_rate, rec.subject_id, breath_peaks)

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
            results_df, best_model_name = exp_manager.run_experiments_with_length_prefix(
                recordings, labels_df, outcome, recording_lengths,
                cleaner, win_gen, extractor, aggregator,
                sig_feats, plotter
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
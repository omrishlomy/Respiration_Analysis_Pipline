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

    # Initialize Plotter early for signal traces
    global_plotter = InteractivePlotter(output_dir=base_out_dir / "general_plots")

    # --- PHASE 1: DATA ---
    print("\n[PHASE 1] Data Loading & Extraction")

    try:
        labels_df = pd.read_excel(data_cfg['labels_file'], engine='openpyxl')
        id_col = data_cfg.get("subject_id_column", "SubjectID")
        # Normalize ID column
        if id_col not in labels_df.columns:
            for col in labels_df.columns:
                if col.strip() == id_col.strip():
                    labels_df = labels_df.rename(columns={col: "SubjectID"})
                    break
        else:
            labels_df = labels_df.rename(columns={id_col: "SubjectID"})

        # CRITICAL FIX: Normalize SubjectID to uppercase and take first 4 characters only
        # (matches filename format: ABCD - date.mat, where ABCD is the 4-letter subject ID)
        labels_df['SubjectID'] = labels_df['SubjectID'].astype(str).str.strip().str.upper()

        # Extract only first 4 characters to match filename format
        # This handles cases like "EBEB - SLEEPY NO CNC" → "EBEB"
        labels_df['SubjectID'] = labels_df['SubjectID'].str[:4]

        # CRITICAL FIX: Add RecordingDate column from second column (date column)
        # The date is in the second column (index 1) in format day.month.year
        if len(labels_df.columns) > 1:
            date_col = labels_df.columns[1]
            print(f"    Detected date column: '{date_col}'")

            # Convert date column to RecordingDate in YYYYMMDD format
            def parse_date_to_yyyymmdd(date_str):
                """Convert date from various formats to YYYYMMDD string."""
                if pd.isna(date_str):
                    return None
                date_str = str(date_str).strip()

                # Try parsing day.month.year format (e.g., "22.8.16" or "5.12.2015")
                if '.' in date_str:
                    parts = date_str.split('.')
                    if len(parts) == 3:
                        day, month, year = parts
                        # Handle 2-digit year (assume 2000s)
                        if len(year) == 2:
                            year = '20' + year
                        # Pad day and month with zeros
                        day = day.zfill(2)
                        month = month.zfill(2)
                        return f"{year}{month}{day}"

                # Try parsing as pandas datetime
                try:
                    dt = pd.to_datetime(date_str)
                    return dt.strftime('%Y%m%d')
                except:
                    pass

                return None

            labels_df['RecordingDate'] = labels_df[date_col].apply(parse_date_to_yyyymmdd)

            # Remove rows with invalid dates
            n_before = len(labels_df)
            labels_df = labels_df[labels_df['RecordingDate'].notna()]
            if len(labels_df) < n_before:
                print(f"    ⚠️  Removed {n_before - len(labels_df)} rows with invalid dates")

        # Debug: Show labels file structure
        print(f"\n📋 Labels file loaded: {len(labels_df)} rows")
        print(f"    Columns: {list(labels_df.columns)}")
        print(f"    Unique subjects: {labels_df['SubjectID'].nunique()}")
        print(f"    Unique recordings (SubjectID + Date): {labels_df.groupby(['SubjectID', 'RecordingDate']).ngroups if 'RecordingDate' in labels_df.columns else 'N/A'}")
        print(f"    First few rows of SubjectID column: {labels_df['SubjectID'].head().tolist()}")
        if 'RecordingDate' in labels_df.columns:
            print(f"    First few RecordingDate values: {labels_df['RecordingDate'].head().tolist()}")

    except Exception as e:
        return print(f"❌ LABEL ERROR: {e}")

    all_subject_features = []
    feature_matrices = {}

    loader = MATDataLoader(default_sampling_rate=data_cfg["default_sampling_rate"])
    recordings = loader.load_batch(str(data_cfg["data_dir"]))
    if not recordings: return print("❌ No recordings found.")

    # Debug: Show first few recordings' IDs and dates
    print(f"\n📁 Sample of loaded recordings:")
    for i, rec in enumerate(recordings[:5]):
        print(f"    {i+1}. SubjectID: '{rec.subject_id}', RecordingDate: '{rec.recording_date}'")

    cleaner = SignalCleaner(config)
    win_gen = WindowGenerator(WindowConfig(config['preprocessing']['windowing']['window_size'],
                                           config['preprocessing']['windowing']['overlap']))
    extractor = BreathingParameterExtractor(config)
    aggregator = FeatureAggregator(config)

    for rec in tqdm(recordings, desc="Extracting"):
        try:
            rec_clean = cleaner.clean(rec)

            # --- NEW: Plot Signal Trace ---
            global_plotter.plot_signal_traces(rec.data, rec_clean.data, rec.sampling_rate, rec.subject_id)

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

            # Pass both subject_id AND recording_date to properly distinguish multiple recordings per subject
            all_subject_features.append(aggregator.aggregate(rec_feats, subject_id=rec.subject_id, recording_date=rec.recording_date))

            # Use unique recording ID to avoid overwriting when same subject has multiple recordings
            recording_id = f"{rec.subject_id}_{rec.recording_date}"
            feature_matrices[recording_id] = pd.DataFrame(rec_feats).set_index('WindowIndex').select_dtypes(
                include=[np.number])

        except Exception:
            pass

    if not all_subject_features: return print("❌ No features extracted.")
    master_features_df = pd.DataFrame(all_subject_features)

    # CRITICAL FIX: Normalize SubjectID to uppercase for case-insensitive matching
    master_features_df['SubjectID'] = master_features_df['SubjectID'].astype(str).str.strip().str.upper()

    if 'RecordingDate' in master_features_df.columns:
        master_features_df['RecordingDate'] = master_features_df['RecordingDate'].astype(str).str.strip()

    # Debug: Show total recordings processed
    n_total_recordings = len(master_features_df)
    n_unique_subjects = master_features_df['SubjectID'].nunique()
    print(f"\n✅ Extracted features from {n_total_recordings} recordings from {n_unique_subjects} unique subjects")

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
                # Filter to only UWS subjects (currentConsciousness == 0)
                curr_labels = curr_labels[curr_labels["currentConsciousness"] == 0]
                uws_subject_ids = set(curr_labels['SubjectID'].tolist())
                n_uws_recordings = master_features_df[master_features_df['SubjectID'].isin(uws_subject_ids)].shape[0]
                print(f"    ℹ️ Recovery Filter: Using {len(curr_labels)} UWS subjects → {n_uws_recordings} recordings with currentConsciousness = 0")

            if outcome not in curr_labels.columns: continue

            # Separate metadata columns from features before creating FeatureCollection
            # Keep SubjectID and RecordingDate in the DataFrame for merging (will be removed before training)
            # Only exclude N_Windows as it's just a count
            metadata_cols = ['N_Windows']
            feature_cols = [col for col in master_features_df.columns if col not in metadata_cols]

            # Create FeatureCollection with SubjectID and RecordingDate included
            features_with_metadata = master_features_df[feature_cols].copy()
            subject_ids_list = master_features_df['SubjectID'].tolist()

            collection = FeatureCollection(features_with_metadata, subject_ids=subject_ids_list)

            # Debug: Show what we have before merge
            n_before_merge = len(features_with_metadata)
            unique_subjects_in_features = set(subject_ids_list)
            unique_subjects_in_labels = set(curr_labels['SubjectID'].astype(str).str.strip().tolist())

            print(f"    📊 Before merge: {n_before_merge} recordings from {len(unique_subjects_in_features)} unique subjects")
            print(f"    📋 Labels available for: {len(unique_subjects_in_labels)} subjects")

            # Debug: Show sample RecordingDate values to verify they match
            if 'RecordingDate' in features_with_metadata.columns and 'RecordingDate' in curr_labels.columns:
                sample_dates_features = features_with_metadata['RecordingDate'].head(3).tolist()
                sample_dates_labels = curr_labels['RecordingDate'].head(3).tolist()
                print(f"    📅 Sample dates in features: {sample_dates_features}")
                print(f"    📅 Sample dates in labels: {sample_dates_labels}")

            # Find mismatches
            subjects_with_features_no_labels = unique_subjects_in_features - unique_subjects_in_labels
            subjects_with_labels_no_features = unique_subjects_in_labels - unique_subjects_in_features

            if subjects_with_features_no_labels:
                print(f"    ⚠️  {len(subjects_with_features_no_labels)} subjects have features but no labels: {sorted(list(subjects_with_features_no_labels))[:5]}...")
            if subjects_with_labels_no_features:
                print(f"    ⚠️  {len(subjects_with_labels_no_features)} subjects have labels but no features: {sorted(list(subjects_with_labels_no_features))[:5]}...")

            # Merge features with labels on BOTH SubjectID AND RecordingDate to prevent duplicates
            # (labels file has one row per recording, so we need both to get 1:1 match)
            X_df, y = collection.merge_with_labels(curr_labels, on='SubjectID', outcome=outcome, also_on='RecordingDate')

            # Debug: Print actual counts after merge
            n_recordings = len(X_df)
            n_unique_participants = X_df['SubjectID'].nunique() if 'SubjectID' in X_df.columns else 0
            print(f"    ✅ After merge: N = {n_recordings} recordings from {n_unique_participants} unique participants")

            # Store SubjectID and RecordingDate for tracking BEFORE cleaning
            stored_subject_ids = X_df['SubjectID'].tolist() if 'SubjectID' in X_df.columns else None
            stored_recording_dates = X_df['RecordingDate'].tolist() if 'RecordingDate' in X_df.columns else None

            if 'SubjectID' in X_df.columns:
                valid_ids = X_df['SubjectID'].tolist()
                # Keep SubjectID for potential train/test splitting by subject
                # But don't use it as a feature for training
            else:
                valid_ids = []

            # Clean and handle NaN values more robustly
            X_df = X_df.select_dtypes(include=[np.number])

            # Replace infinite values with NaN first
            X_df = X_df.replace([np.inf, -np.inf], np.nan)

            # Fill NaN values with 0 (or could use median/mean imputation)
            X_df = X_df.fillna(0)

            # Verify no NaN or inf values remain
            if X_df.isnull().any().any():
                print(f"    ⚠️ WARNING: NaN values still present after cleaning!")
                X_df = X_df.fillna(0)  # Extra safety

            y = np.array(y, dtype=int)
            if len(np.unique(y)) < 2: continue

            # 1.5. Setup Outcome-Specific Plotter
            plotter = InteractivePlotter(output_dir=out_dir / "plots")

            # 1.6. Save and Visualize Classifier Input Data
            print("    Saving classifier input data and creating visualizations...")
            plotter.save_classifier_input_data(X_df, y, outcome, stored_subject_ids, stored_recording_dates)
            plotter.plot_correlation_matrix(X_df, outcome)
            plotter.plot_pca_2d(X_df, y, outcome)
            plotter.plot_pca_3d(X_df, y, outcome)

            # 2. Stats
            print("    Running Stats...")
            stats = StatisticalAnalyzer(test=config['analysis']['statistical']['test'],
                                        correction_method=config['analysis']['statistical']['correction'])
            stats_df = stats.compare_groups(X_df, y, outcome_name=outcome, feature_names=X_df.columns.tolist())

            feat_col = 'feature_name' if 'feature_name' in stats_df.columns else 'feature'
            sig_feats = stats_df[stats_df['significant'] == True][
                feat_col].tolist() if feat_col in stats_df.columns else []
            print(f"    Significant features: {len(sig_feats)}")

            # 3. Plot Feature Distributions
            plotter.plot_feature_violins(X_df, y, outcome)
            plotter.plot_statistical_ranking(stats_df)
            plotter.plot_feature_distributions(X_df, y, outcome, stats_df)

            # 4. Save Feature Matrices
            for sid in valid_ids:
                if sid in feature_matrices:
                    safe = "".join([c for c in sid if c.isalnum()]).strip()
                    plotter.plot_feature_matrix(feature_matrices[sid], filename=f"{safe}.html")

            # 5. Run Models
            exp_manager = ExperimentManager(config)
            results_df, _ = exp_manager.run_experiments(X_df, y, sig_feats, plotter=plotter, subject_ids=stored_subject_ids)

            # 6. Export
            exporter = ExcelExporter(filepath=str(out_dir / f"REPORT_{outcome}.xlsx"))
            exporter.add_sheet("Model_Experiments_Detailed", results_df)
            exporter.add_statistical_results_sheet(stats_df)
            exporter.write()
            print(f"    ✅ Completed {outcome}")

        except Exception as e:
            print(f"    ❌ FAILED {outcome}: {e}")
            traceback.print_exc()

    print("\n✅ Pipeline Finished.")


if __name__ == "__main__":
    main()
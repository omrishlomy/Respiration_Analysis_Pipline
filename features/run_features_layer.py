"""
Features Layer Main Script

Minimal orchestration script that extracts breathing features from all recordings.
Core functionality is in separate modules:
- signal_processing.py: Signal cleaning, multi-peak detection, feature extraction
- visualizations.py: All plotting functions
- clinical_labels.py: Label loading and merging
- breathing.py: BreathingParameterExtractor (existing)
- collection.py: FeatureCollection (existing)

Outputs:
1. CSV file with recording-level features (30 base x 4 aggregations = 120 features)
2. Signal plots with peak detection
3. Feature matrix heatmaps
4. Correlation matrix
5. Feature comparison plots (by clinical labels)

Usage:
    python run_features_layer.py
"""

import sys
from pathlib import Path

# Get script directory and project root
SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent

# Add paths
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

import yaml
import numpy as np
import pandas as pd
import warnings
import matplotlib.pyplot as plt

# Import from project (data layer, preprocessing layer)
try:
    from data.loaders import MATDataLoader
    from data.recording import RespiratoryRecording
    from preprocessing.windowing import WindowGenerator, WindowConfig
except ImportError:
    try:
        from loaders import MATDataLoader
        from recording import RespiratoryRecording
        from windowing import WindowGenerator, WindowConfig
    except ImportError as e:
        print(f"Import error: {e}")
        print("\nPlease ensure data layer modules are available.")
        sys.exit(1)

# Import from features layer (local modules)
try:
    # When running from within features/ package
    from extractors.breathing import BreathingParameterExtractor
    from signal_processing import (
        extract_recording_features,
        get_base_feature_names,
        get_aggregated_feature_names,
    )
    from visualizations import (
        plot_signal_with_peaks,
        plot_feature_matrix,
        plot_feature_over_time,
        plot_correlation_matrix,
        plot_feature_across_recordings,
    )
    from data.clinical_labels import ClinicalLabels
except ImportError:
    # When running as part of larger package (e.g., from project root)
    from features.extractors.breathing import BreathingParameterExtractor
    from features.signal_processing import (
        extract_recording_features,
        get_base_feature_names,
        get_aggregated_feature_names,
    )
    from features.visualizations import (
        plot_signal_with_peaks,
        plot_feature_matrix,
        plot_feature_over_time,
        plot_correlation_matrix,
        plot_feature_across_recordings,
    )
    from data.clinical_labels import ClinicalLabels

# Debug mode
DEBUG = False


def plot_feature_by_labels(features_df, feature_name, label_columns, output_path):
    """
    Create 3-subplot feature comparison plot.

    CRITICAL: This function MUST create 3 side-by-side subplots, one for each label.

    Args:
        features_df: DataFrame with aggregated features (ONE ROW per recording)
        feature_name: Name of feature column to plot
        label_columns: List of 3 labels ['Recovery', 'currentConsciousness', 'Survival']
        output_path: Where to save the plot
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    # Verify labels exist in dataframe
    missing_labels = [col for col in label_columns if col not in features_df.columns]
    if missing_labels:
        print(f"WARNING: Labels {missing_labels} not found in DataFrame. Available columns: {list(features_df.columns[:10])}")
        return

    # Create 3 side-by-side subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Define colors for each label value
    colors = {
        0: 'orange',    # Label value 0
        1: 'blue',      # Label value 1
        'missing': 'gray'  # Missing labels
    }

    for idx, (label_col, ax) in enumerate(zip(label_columns, axes)):
        # Filter data for Recovery subplot only
        if label_col == 'Recovery':
            # Recovery only applies to patients with currentConsciousness==0
            if 'currentConsciousness' in features_df.columns:
                plot_data = features_df[features_df['currentConsciousness'] == 0].copy()
            else:
                plot_data = features_df.copy()
        else:
            # All other labels: include all recordings
            plot_data = features_df.copy()

        # Get feature values
        feature_values = plot_data[feature_name].values
        label_values = plot_data[label_col].values

        # Create recording indices
        x_indices = np.arange(len(plot_data))

        # Separate by label value
        mask_0 = (label_values == 0)
        mask_1 = (label_values == 1)
        mask_missing = pd.isna(label_values)

        # Count for legend
        n_0 = mask_0.sum()
        n_1 = mask_1.sum()
        n_missing = mask_missing.sum()

        # Plot each group
        ax.scatter(x_indices[mask_0], feature_values[mask_0],
                   c=colors[0], label=f'{label_col}=0 (n={n_0})', alpha=0.6, s=30)
        ax.scatter(x_indices[mask_1], feature_values[mask_1],
                   c=colors[1], label=f'{label_col}=1 (n={n_1})', alpha=0.6, s=30)
        if n_missing > 0:
            ax.scatter(x_indices[mask_missing], feature_values[mask_missing],
                       c=colors['missing'], label=f'Missing (n={n_missing})', alpha=0.3, s=30)

        # Labels and styling
        ax.set_xlabel('Recording Index', fontsize=10)
        ax.set_ylabel(feature_name, fontsize=10)
        ax.set_title(f'{feature_name} by {label_col}', fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def load_config(config_path: Path = None) -> dict:
    """Load configuration from YAML file."""
    if config_path is None:
        possible_paths = [
            PROJECT_ROOT / 'config.yaml',
            PROJECT_ROOT / 'pipeline' / 'config.yaml',
            SCRIPT_DIR / 'config.yaml',
            SCRIPT_DIR.parent / 'config.yaml',
            SCRIPT_DIR.parent / 'pipeline' / 'config.yaml',
            Path.cwd() / 'config.yaml',
            Path.cwd() / 'pipeline' / 'config.yaml',
        ]

        print(f"  Looking for config.yaml in:")
        for path in possible_paths:
            exists = "✓" if path.exists() else "✗"
            print(f"    {exists} {path}")

        for path in possible_paths:
            if path.exists():
                config_path = path
                break
        else:
            raise FileNotFoundError(
                f"config.yaml not found in any of:\n" +
                "\n".join(f"  - {p}" for p in possible_paths)
            )

    print(f"  Using: {config_path}")
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    """Main entry point for features layer processing."""
    print("=" * 70)
    print("FEATURES LAYER")
    print("=" * 70)

    # === Step 1: Load configuration ===
    print("\n[INFO] Step 1: Loading configuration...")
    config = load_config()

    data_config = config.get('data', {})
    preprocessing_config = config.get('preprocessing', {})
    features_config = config.get('features', {})

    # === Step 2: Set up paths ===
    print("\n[INFO] Step 2: Setting up paths...")
    data_dir = Path(data_config.get('data_dir') or data_config.get('data_directory', '.'))
    output_base = Path(data_config.get('output_dir') or data_config.get('output_directory', './results'))
    output_dir = output_base / 'features'

    # Create output directories
    output_dir.mkdir(parents=True, exist_ok=True)
    peaks_dir = output_dir / 'signal_with_peaks'
    matrix_dir = output_dir / 'feature_matrices'
    scatter_dir = output_dir / 'feature_scatter'
    comparison_dir = output_dir / 'feature_comparison'

    for d in [peaks_dir, matrix_dir, scatter_dir, comparison_dir]:
        d.mkdir(parents=True, exist_ok=True)

    print(f"  Data directory: {data_dir}")
    print(f"  Output directory: {output_dir}")

    # === Step 3: Show configuration summary ===
    windowing_config = preprocessing_config.get('windowing', {})
    cleaning_config = preprocessing_config.get('cleaning', {})
    multipeak_config = features_config.get('multipeak_detection', {})

    print(f"\n[INFO] Step 3: Configuration summary:")
    print(f"  Window size: {windowing_config.get('window_size', 300)}s")
    print(f"  Overlap: {windowing_config.get('overlap', 60)}s")
    print(f"  Cleaning - remove outliers: {cleaning_config.get('remove_outliers', True)}")
    print(f"  Cleaning - apply filter: {cleaning_config.get('apply_filter', False)}")
    print(f"  Multi-peak - min_distance_sec: {multipeak_config.get('min_distance_sec', 0.4)}")

    # === Step 4: Initialize components ===
    print(f"\n[INFO] Step 4: Initializing components...")
    loader = MATDataLoader(
        data_key=data_config.get('data_key', 'data'),
        fs_key=data_config.get('fs_key', 'fs'),
        auto_detect_keys=data_config.get('auto_detect_keys', True),
        default_sampling_rate=data_config.get('default_sampling_rate', 6.0)
    )

    window_config = WindowConfig(
        window_size=windowing_config.get('window_size', 300),
        overlap=windowing_config.get('overlap', 60)
    )
    window_generator = WindowGenerator(window_config)
    extractor = BreathingParameterExtractor(features_config)

    # Get feature names
    base_feature_names = get_base_feature_names(extractor)
    aggregated_feature_names = get_aggregated_feature_names(base_feature_names)

    print(f"\n[INFO] Feature extraction plan:")
    print(f"  - {len(base_feature_names)} base features (25 breathing + 5 multi-peak)")
    print(f"  - Aggregation: mean, std, min, max per feature")
    print(f"  - Total aggregated features: {len(aggregated_feature_names)}")

    # === Step 5: Load recordings ===
    print(f"\n[INFO] Step 5: Loading recordings...")
    recordings = loader.load_batch(str(data_dir), pattern='*.mat')

    # Filter out non-subject recordings (e.g., "RESP_*" files)
    valid_recordings = [
        rec for rec in recordings
        if not rec.subject_id.upper().startswith('RESP')
    ]

    print(f"  Loaded {len(recordings)} total files")
    print(f"  Valid recordings: {len(valid_recordings)}")

    if len(valid_recordings) == 0:
        print("\nERROR: No valid recordings found!")
        sys.exit(1)

    valid_recordings.sort(key=lambda r: (r.subject_id, r.recording_date))

    # === Step 6: Process each recording ===
    print(f"\n[INFO] Step 6: Processing recordings...")
    print("=" * 70)

    all_recording_features = []
    all_window_features = {}

    for i, rec in enumerate(valid_recordings, 1):
        rec_id = f"{rec.subject_id}_{rec.recording_date}"
        print(f"\n[{i}/{len(valid_recordings)}] Processing: {rec_id}")

        try:
            # Extract features using signal_processing module
            agg_features, window_features, cleaned_data, inhale_idx, exhale_idx, multipeak_idx = \
                extract_recording_features(rec, window_generator, extractor, config, debug=DEBUG)

            # Add metadata
            agg_features['RecordingID'] = rec_id
            agg_features['SubjectID'] = rec.subject_id
            agg_features['RecordingDate'] = rec.recording_date
            agg_features['Duration_min'] = rec.duration / 60
            agg_features['N_Inhales'] = len(inhale_idx)
            agg_features['N_Exhales'] = len(exhale_idx)
            agg_features['N_MultiPeaks'] = len(multipeak_idx)

            all_recording_features.append(agg_features)
            all_window_features[rec_id] = window_features

            # Generate plots using visualizations module
            peaks_path = peaks_dir / f"{i:03d}_{rec_id}.png"
            plot_signal_with_peaks(rec, cleaned_data, inhale_idx, exhale_idx, multipeak_idx, peaks_path)

            if len(window_features) > 0:
                matrix_path = matrix_dir / f"{i:03d}_{rec_id}.png"
                plot_feature_matrix(window_features, rec, base_feature_names, matrix_path)

                rec_scatter_dir = scatter_dir / f"{i:03d}_{rec_id}"
                rec_scatter_dir.mkdir(parents=True, exist_ok=True)

                for feat_name in base_feature_names:
                    scatter_path = rec_scatter_dir / f"{feat_name}.png"
                    plot_feature_over_time(window_features, feat_name, rec, window_config, scatter_path)

            print(f"  ✓ Completed: {rec_id}")

        except Exception as e:
            print(f"  ✗ FAILED: {rec_id} - {e}")
            warnings.warn(f"Failed to process {rec_id}: {e}")
            continue

    print("\n" + "=" * 70)

    # === Step 7: Create features DataFrame ===
    print(f"\n[INFO] Step 7: Creating feature matrix CSV...")
    features_df = pd.DataFrame(all_recording_features)

    # Reorder columns (metadata first, then features)
    metadata_cols = ['RecordingID', 'SubjectID', 'RecordingDate', 'Duration_min',
                     'N_Windows', 'N_Inhales', 'N_Exhales', 'N_MultiPeaks']
    ordered_cols = [c for c in metadata_cols if c in features_df.columns]
    for c in aggregated_feature_names:
        if c in features_df.columns and c not in ordered_cols:
            ordered_cols.append(c)
    features_df = features_df[ordered_cols]

    # Save features CSV
    csv_path = output_dir / 'recording_features.csv'
    features_df.to_csv(csv_path, index=False)
    print(f"  Saved: {csv_path}")
    print(f"  Shape: {features_df.shape[0]} recordings x {features_df.shape[1]} columns")

    # === Step 8: Plot correlation matrix ===
    print(f"\n[INFO] Step 8: Plotting correlation matrix...")
    corr_path = output_dir / 'correlation_matrix.png'
    mean_features = [f for f in features_df.columns if f.endswith('_mean')]
    plot_correlation_matrix(features_df, mean_features, corr_path)
    print(f"  Saved: {corr_path}")

    # === Step 9: Load and merge clinical labels ===
    print(f"\n[INFO] Step 9: Loading clinical labels...")
    labels_file = data_config.get('labels_file')
    label_columns = data_config.get('outcomes', [])
    subject_column = data_config.get('subject_id_column', 'Name')

    if labels_file and Path(labels_file).exists():
        try:
            labels = ClinicalLabels.from_excel(
                labels_file,
                subject_id_column=subject_column
            )
            print(f"  Loaded labels from: {labels_file}")
            print(f"  Subject column: {subject_column}")
            print(f"  Label columns: {label_columns}")

            # Merge labels with features using clinical_labels module
            # CORRECT - No filter_recovery parameter
            features_df = labels.aggregate_and_add_labels(
                features_df,
                label_columns=['Recovery', 'currentConsciousness', 'Survival'],
                subject_id_column='SubjectID',
                aggregation='mean'
            )

            # Save updated CSV with labels
            features_df.to_csv(csv_path, index=False)
            print(f"  Updated CSV with labels: {csv_path}")

        except Exception as e:
            print(f"  WARNING: Could not load labels: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"  No labels file configured or file not found")
        if labels_file:
            print(f"  (looked for: {labels_file})")

    # === Step 10: Plot feature comparisons ===
    print(f"\n[INFO] Step 10: Plotting feature comparisons...")
    comparison_features = [f for f in features_df.columns if f.endswith('_mean')]
    available_labels = [c for c in label_columns if c in features_df.columns]

    if len(available_labels) >= 1:
        print(f"  Using labels: {available_labels}")
        # Use the 3-subplot plotting function
        for feat_name in comparison_features:
            comparison_path = comparison_dir / f"{feat_name}.png"
            plot_feature_by_labels(features_df, feat_name, available_labels, comparison_path)
        print(f"  Saved {len(comparison_features)} label comparison plots")
    else:
        print(f"  No labels available, using simple comparison plots")
        for feat_name in comparison_features:
            comparison_path = comparison_dir / f"{feat_name}.png"
            plot_feature_across_recordings(features_df, feat_name, comparison_path)
        print(f"  Saved {len(comparison_features)} comparison plots")

    # === Step 11: Write summary ===
    print(f"\n[INFO] Step 11: Writing summary...")
    summary_path = output_dir / '_summary.txt'
    with open(summary_path, 'w') as f:
        f.write("Features Layer Summary\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Total recordings processed: {len(all_recording_features)}\n")
        f.write(f"Unique subjects: {features_df['SubjectID'].nunique()}\n")
        f.write(f"Base features: {len(base_feature_names)}\n")
        f.write(f"Aggregated features: {len(aggregated_feature_names)}\n\n")

        f.write("Aggregation method:\n")
        f.write("  - Features extracted per window (default 5 min)\n")
        f.write("  - Aggregated across windows: mean, std, min, max\n\n")

        f.write("Output files:\n")
        f.write(f"  - recording_features.csv: {len(features_df)} rows x {len(features_df.columns)} cols\n")
        f.write(f"  - correlation_matrix.png\n")
        f.write(f"  - signal_with_peaks/: {len(list(peaks_dir.glob('*.png')))} plots\n")
        f.write(f"  - feature_matrices/: {len(list(matrix_dir.glob('*.png')))} plots\n")
        f.write(f"  - feature_scatter/: {len(list(scatter_dir.iterdir()))} directories\n")
        f.write(f"  - feature_comparison/: {len(list(comparison_dir.glob('*.png')))} plots\n\n")

        f.write("Base features (30 total):\n")
        f.write("-" * 40 + "\n")
        f.write("\n25 Breathing Metrics:\n")
        for name in extractor.get_feature_names():
            f.write(f"  - {name}\n")
        f.write("\n5 Multi-Peak Detection Features:\n")
        f.write("  - MultiPeak_Mean_NumPeaks\n")
        f.write("  - MultiPeak_Max_NumPeaks\n")
        f.write("  - MultiPeak_Percent_MultipeakBreaths\n")
        f.write("  - MultiPeak_Percent_Inhale_Multipeak\n")
        f.write("  - MultiPeak_Percent_Exhale_Multipeak\n")

    # === Final summary ===
    print(f"\n" + "=" * 70)
    print("DONE!")
    print("=" * 70)
    print(f"\nOutputs saved to: {output_dir}")
    print(f"  - recording_features.csv ({len(features_df)} recordings x {len(ordered_cols)} columns)")
    print(f"  - correlation_matrix.png")
    print(f"  - signal_with_peaks/ ({len(list(peaks_dir.glob('*.png')))} plots)")
    print(f"  - feature_matrices/ ({len(list(matrix_dir.glob('*.png')))} plots)")
    print(f"  - feature_scatter/ ({len(list(scatter_dir.iterdir()))} directories)")
    print(f"  - feature_comparison/ ({len(list(comparison_dir.glob('*.png')))} plots)")
    print(f"  - _summary.txt")


if __name__ == '__main__':
    main()

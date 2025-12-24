"""
Standalone Preprocessing Test Script

This script allows you to test the preprocessing/cleaning layer independently
with signal visualization for manual inspection.

Usage:
    python -m preprocessing.test_cleaning

Or from project root:
    python preprocessing/test_cleaning.py
"""

import sys
import os
from pathlib import Path
import yaml
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Add project root to path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from data.loaders import load_recordings_from_directory
from preprocessing.cleaner import SignalCleaner
from data.recording import RespiratoryRecording


def create_signal_visualization(recording_raw, recording_clean, output_path):
    """
    Create interactive HTML visualization comparing raw and cleaned signals.

    Args:
        recording_raw: Raw RespiratoryRecording
        recording_clean: Cleaned RespiratoryRecording
        output_path: Path to save HTML file
    """
    # Create time arrays
    time_raw = np.arange(len(recording_raw.data)) / recording_raw.sampling_rate
    time_clean = np.arange(len(recording_clean.data)) / recording_clean.sampling_rate

    # Create subplots: Raw signal + Cleaned signal
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=(
            f'Raw: {recording_raw.subject_id}',
            'Cleaned'
        ),
        vertical_spacing=0.1,
        shared_xaxes=True
    )

    # Raw signal
    fig.add_trace(
        go.Scatter(
            x=time_raw,
            y=recording_raw.data,
            mode='lines',
            name='Raw',
            line=dict(color='gray', width=1),
            hovertemplate='Time: %{x:.2f}s<br>Value: %{y:.4f}<extra></extra>'
        ),
        row=1, col=1
    )

    # Cleaned signal
    fig.add_trace(
        go.Scatter(
            x=time_clean,
            y=recording_clean.data,
            mode='lines',
            name='Cleaned',
            line=dict(color='cyan', width=1.5),
            hovertemplate='Time: %{x:.2f}s<br>Value: %{y:.4f}<extra></extra>'
        ),
        row=2, col=1
    )

    # Update layout
    fig.update_xaxes(title_text="Time (s)", row=2, col=1)
    fig.update_yaxes(title_text="Amplitude", row=1, col=1)
    fig.update_yaxes(title_text="Amplitude", row=2, col=1)

    fig.update_layout(
        title=f"Signal: {recording_raw.subject_id}_{getattr(recording_raw, 'recording_date', 'unknown')}",
        height=600,
        showlegend=True,
        hovermode='x unified',
        font=dict(family="monospace", size=12)
    )

    # Save to HTML
    fig.write_html(str(output_path))
    print(f"    ✅ Visualization saved: {output_path.name}")


def test_cleaning(config_path=None, data_dir=None, output_dir=None, max_recordings=None):
    """
    Test the cleaning pipeline on recordings.

    Args:
        config_path: Path to config.yaml (default: pipeline/config.yaml)
        data_dir: Path to recordings directory (default: from config)
        output_dir: Path to save visualizations (default: preprocessing_test/)
        max_recordings: Maximum number of recordings to process (default: all)
    """
    print("🔬 Preprocessing Test Script")
    print("=" * 60)

    # Load configuration
    if config_path is None:
        config_path = project_root / "pipeline" / "config.yaml"

    print(f"\n📋 Loading config from: {config_path}")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Get data directory
    if data_dir is None:
        data_dir = Path(config['data']['data_directory'])
    else:
        data_dir = Path(data_dir)

    print(f"📂 Data directory: {data_dir}")

    # Create output directory
    if output_dir is None:
        output_dir = project_root / "preprocessing_test"
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(exist_ok=True)
    print(f"📁 Output directory: {output_dir}")

    # Initialize cleaner
    print("\n🧹 Initializing SignalCleaner...")
    cleaner = SignalCleaner(config)

    print("\n   Cleaning Parameters:")
    print(f"     • Remove outliers: {cleaner.remove_outliers}")
    print(f"     • Apply filter: {cleaner.apply_filter}")
    if cleaner.apply_filter:
        print(f"       - Type: {cleaner.filter_type}")
        print(f"       - Lowcut: {cleaner.lowcut} Hz")
        print(f"       - Highcut: {cleaner.highcut} Hz")
        print(f"       - Order: {cleaner.filter_order}")
    print(f"     • Detrend: {cleaner.detrend_method}")
    print(f"     • Quality checks: {cleaner.check_quality}")
    if cleaner.check_quality:
        print(f"       - Min valid %: {cleaner.min_valid_percent}%")
        print(f"       - Auto clean: {cleaner.auto_clean}")

    # Load recordings
    print(f"\n📥 Loading recordings from: {data_dir}")
    recordings, failed = load_recordings_from_directory(
        data_dir,
        data_key=config['data'].get('data_key', 'data'),
        fs_key=config['data'].get('fs_key', 'fs'),
        default_fs=config['data'].get('default_sampling_rate', 6.0)
    )

    print(f"    ✅ Loaded: {len(recordings)} recordings")
    if failed:
        print(f"    ⚠️  Failed: {failed} recordings")

    # Limit recordings if specified
    if max_recordings and len(recordings) > max_recordings:
        print(f"    ℹ️  Limiting to first {max_recordings} recordings")
        recordings = recordings[:max_recordings]

    # Process each recording
    print(f"\n🔄 Processing {len(recordings)} recordings...")
    print("=" * 60)

    success_count = 0
    error_count = 0

    for i, rec in enumerate(recordings, 1):
        recording_id = f"{rec.subject_id}_{getattr(rec, 'recording_date', 'unknown')}"
        print(f"\n[{i}/{len(recordings)}] {recording_id}")

        try:
            # Create a copy for raw data (before cleaning)
            rec_raw = RespiratoryRecording(
                data=rec.data.copy(),
                sampling_rate=rec.sampling_rate,
                subject_id=rec.subject_id,
                recording_date=getattr(rec, 'recording_date', None)
            )

            # Clean the recording
            print("    🧹 Cleaning...")
            rec_clean = cleaner.clean(rec)

            # Statistics
            print(f"    📊 Statistics:")
            print(f"       Raw:     len={len(rec_raw.data)}, mean={np.mean(rec_raw.data):.4f}, std={np.std(rec_raw.data):.4f}")
            print(f"       Cleaned: len={len(rec_clean.data)}, mean={np.mean(rec_clean.data):.4f}, std={np.std(rec_clean.data):.4f}")

            # Check for potential issues
            if len(rec_clean.data) < len(rec_raw.data) * 0.5:
                print(f"    ⚠️  WARNING: Cleaned signal is < 50% of original length")

            if np.std(rec_clean.data) < 0.001:
                print(f"    ⚠️  WARNING: Cleaned signal has very low variance (std={np.std(rec_clean.data):.6f})")

            # Create visualization
            print("    📊 Creating visualization...")
            safe_id = "".join([c if c.isalnum() or c in ['_', '-'] else '_' for c in recording_id])
            viz_path = output_dir / f"{safe_id}.html"
            create_signal_visualization(rec_raw, rec_clean, viz_path)

            success_count += 1

        except Exception as e:
            print(f"    ❌ ERROR: {e}")
            import traceback
            traceback.print_exc()
            error_count += 1

    # Summary
    print("\n" + "=" * 60)
    print("📊 Processing Summary")
    print("=" * 60)
    print(f"✅ Successfully processed: {success_count}/{len(recordings)}")
    print(f"❌ Errors: {error_count}/{len(recordings)}")
    print(f"\n📁 Visualizations saved to: {output_dir}")
    print(f"   Open the .html files in a browser to inspect signals")

    return success_count, error_count


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test preprocessing/cleaning pipeline")
    parser.add_argument('--config', type=str, help='Path to config.yaml')
    parser.add_argument('--data-dir', type=str, help='Path to data directory')
    parser.add_argument('--output-dir', type=str, help='Path to output directory')
    parser.add_argument('--max', type=int, help='Maximum number of recordings to process')

    args = parser.parse_args()

    test_cleaning(
        config_path=args.config,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        max_recordings=args.max
    )

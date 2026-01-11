#!/usr/bin/env python3
"""
Debug script to identify why 3 recordings are missing labels in the features layer analysis.
"""

import os
import numpy as np

# Fix numpy compatibility issues
if not hasattr(np, 'bool8'):
    np.bool8 = np.bool_
if not hasattr(np, 'float'):
    np.float = float

import pandas as pd
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from features.collection import FeatureCollection

def debug_label_merge(features_csv, labels_file):
    """
    Debug the label merging process to identify missing labels.

    Args:
        features_csv: Path to the extracted features CSV
        labels_file: Path to the labels Excel file
    """
    print("\n" + "="*80)
    print("DEBUGGING LABEL MERGE PROCESS")
    print("="*80)

    # Load extracted features
    print(f"\n1. Loading features from: {features_csv}")
    if not Path(features_csv).exists():
        print(f"❌ ERROR: Features file not found!")
        return

    master_features_df = pd.read_csv(features_csv)
    print(f"   ✓ Loaded {len(master_features_df)} feature records")

    # Normalize SubjectID in features
    if 'SubjectID' in master_features_df.columns:
        master_features_df['SubjectID'] = master_features_df['SubjectID'].astype(str).str.strip()
        feature_subjects = set(master_features_df['SubjectID'].unique())
        print(f"   ✓ Found {len(feature_subjects)} unique subjects in features:")
        for sid in sorted(feature_subjects):
            count = len(master_features_df[master_features_df['SubjectID'] == sid])
            print(f"      - {sid}: {count} recording(s)")
    else:
        print(f"   ❌ ERROR: No SubjectID column in features!")
        return

    # Load labels
    print(f"\n2. Loading labels from: {labels_file}")
    if not Path(labels_file).exists():
        print(f"❌ ERROR: Labels file not found!")
        return

    labels_df = pd.read_excel(labels_file, engine='openpyxl')
    print(f"   ✓ Loaded {len(labels_df)} label records")
    print(f"   ✓ Label columns: {list(labels_df.columns)}")

    # Normalize SubjectID in labels
    id_col = "SubjectID"
    if id_col not in labels_df.columns:
        for col in labels_df.columns:
            if col.strip() == id_col.strip() or 'subject' in col.lower() or 'id' in col.lower():
                labels_df = labels_df.rename(columns={col: "SubjectID"})
                print(f"   ℹ️  Renamed column '{col}' to 'SubjectID'")
                break

    if 'SubjectID' in labels_df.columns:
        labels_df['SubjectID'] = labels_df['SubjectID'].astype(str).str.strip()
        label_subjects = set(labels_df['SubjectID'].unique())
        print(f"   ✓ Found {len(label_subjects)} unique subjects in labels:")
        for sid in sorted(label_subjects):
            print(f"      - {sid}")
    else:
        print(f"   ❌ ERROR: Could not find SubjectID column in labels!")
        return

    # Compare sets
    print(f"\n3. Comparing Subject IDs:")
    print(f"   Features: {len(feature_subjects)} subjects")
    print(f"   Labels:   {len(label_subjects)} subjects")

    # Subjects in features but NOT in labels
    missing_labels = feature_subjects - label_subjects
    if missing_labels:
        print(f"\n   ❌ FOUND {len(missing_labels)} SUBJECTS IN FEATURES WITHOUT LABELS:")
        for sid in sorted(missing_labels):
            count = len(master_features_df[master_features_df['SubjectID'] == sid])
            print(f"      - {sid}: {count} recording(s)")
            print(f"        This is likely the source of the missing labels bug!")
    else:
        print(f"   ✓ All feature subjects have labels")

    # Subjects in labels but NOT in features
    missing_features = label_subjects - feature_subjects
    if missing_features:
        print(f"\n   ℹ️  Found {len(missing_features)} subjects in labels without features:")
        for sid in sorted(missing_features):
            print(f"      - {sid}")
    else:
        print(f"   ✓ All label subjects have features")

    # Matching subjects
    matching = feature_subjects & label_subjects
    print(f"\n   ✓ {len(matching)} subjects have both features and labels:")
    for sid in sorted(matching):
        count = len(master_features_df[master_features_df['SubjectID'] == sid])
        print(f"      - {sid}: {count} recording(s)")

    # Test merge with FeatureCollection
    print(f"\n4. Testing FeatureCollection.merge_with_labels():")

    # Find an outcome column to test
    outcome_cols = [c for c in labels_df.columns if c != 'SubjectID' and labels_df[c].dtype in [np.int64, np.float64, int, float]]
    if not outcome_cols:
        print(f"   ⚠️  No numeric outcome columns found to test merge")
        return

    test_outcome = outcome_cols[0]
    print(f"   Testing with outcome: {test_outcome}")

    collection = FeatureCollection(master_features_df, subject_ids=master_features_df['SubjectID'].tolist())

    print(f"   Before merge: {len(collection.features_df)} records")
    X_df, y = collection.merge_with_labels(labels_df, on='SubjectID', outcome=test_outcome)
    print(f"   After merge:  {len(X_df)} records")
    print(f"   Lost {len(collection.features_df) - len(X_df)} records during merge")

    if len(collection.features_df) - len(X_df) > 0:
        print(f"\n   🔍 DIAGNOSIS:")
        print(f"   The merge uses an 'inner' join, which only keeps records")
        print(f"   where SubjectID exists in BOTH features and labels.")
        print(f"   The {len(missing_labels)} subject(s) without labels are being dropped.")

    # Detailed SubjectID comparison
    print(f"\n5. Detailed SubjectID Analysis:")
    print(f"\n   Sample feature SubjectIDs (first 5):")
    for sid in sorted(feature_subjects)[:5]:
        repr_val = repr(sid)
        print(f"      {repr_val} (type: {type(sid).__name__}, len: {len(sid)})")

    print(f"\n   Sample label SubjectIDs (first 5):")
    for sid in sorted(label_subjects)[:5]:
        repr_val = repr(sid)
        print(f"      {repr_val} (type: {type(sid).__name__}, len: {len(sid)})")

    # Check for whitespace or formatting issues
    print(f"\n6. Checking for whitespace/formatting issues:")
    for sid in sorted(missing_labels)[:3]:  # Check first 3 missing
        # Try to find similar IDs in labels
        similar = [lbl_id for lbl_id in label_subjects
                  if lbl_id.replace(' ', '').replace('-', '').replace('_', '').lower()
                  == sid.replace(' ', '').replace('-', '').replace('_', '').lower()]
        if similar:
            print(f"   ⚠️  '{sid}' from features might match '{similar[0]}' from labels")
            print(f"       Feature ID: {repr(sid)}")
            print(f"       Label ID:   {repr(similar[0])}")
        else:
            print(f"   ❌ '{sid}' has no similar match in labels")

    print("\n" + "="*80)
    print("DEBUG COMPLETE")
    print("="*80 + "\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Debug label merge issues')
    parser.add_argument('--features', type=str, help='Path to features CSV file')
    parser.add_argument('--labels', type=str, help='Path to labels Excel file')

    args = parser.parse_args()

    # Try to find files if not specified
    if not args.features or not args.labels:
        print("Searching for features and labels files...")

        # Look for features CSV
        if not args.features:
            possible_features = list(Path('.').rglob('*extracted_features*.csv'))
            if possible_features:
                args.features = str(possible_features[0])
                print(f"Found features: {args.features}")

        # Look for labels Excel
        if not args.labels:
            possible_labels = list(Path('.').rglob('*.xlsx'))
            possible_labels = [p for p in possible_labels if 'label' in p.name.lower() or 'clinical' in p.name.lower()]
            if possible_labels:
                args.labels = str(possible_labels[0])
                print(f"Found labels: {args.labels}")

    if not args.features or not args.labels:
        print("\n❌ ERROR: Please specify --features and --labels paths")
        print("\nExample:")
        print("  python debug_labels.py --features output/all_extracted_features.csv --labels data/labels.xlsx")
        sys.exit(1)

    debug_label_merge(args.features, args.labels)

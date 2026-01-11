#!/usr/bin/env python3
"""
Quick script to identify which SubjectIDs from recordings are missing in labels.
Run this in your Respiration_Analysis_Pipline_2 directory.
"""

import pandas as pd
from pathlib import Path
import sys

# Common patterns for SubjectID extraction (first 4 chars)
def extract_subject_id(filename):
    """Extract subject ID from filename (first 4 chars, uppercase)"""
    stem = Path(filename).stem
    if len(stem) >= 4:
        return stem[:4].upper()
    return stem.upper()

print("="*80)
print("MISSING LABELS CHECKER")
print("="*80)

# 1. Get labels file
labels_path = input("\nEnter path to labels Excel file: ").strip().strip('"')
if not Path(labels_path).exists():
    print(f"❌ File not found: {labels_path}")
    sys.exit(1)

# 2. Load labels
print(f"\n📋 Loading labels from: {labels_path}")
labels_df = pd.read_excel(labels_path, engine='openpyxl')

# Find SubjectID column (usually first column or named 'SubjectID')
subject_col = None
if 'SubjectID' in labels_df.columns:
    subject_col = 'SubjectID'
else:
    # Assume first column is SubjectID
    subject_col = labels_df.columns[0]
    print(f"   Using first column as SubjectID: '{subject_col}'")
    labels_df = labels_df.rename(columns={subject_col: 'SubjectID'})
    subject_col = 'SubjectID'

# Normalize SubjectIDs (strip whitespace, uppercase)
labels_df['SubjectID'] = labels_df['SubjectID'].astype(str).str.strip().str.upper()
label_ids = set(labels_df['SubjectID'].unique())
print(f"   ✓ Found {len(label_ids)} unique SubjectIDs in labels")

# 3. Get recording files
data_dir = input("\nEnter path to data directory (with .mat files): ").strip().strip('"')
if not Path(data_dir).exists():
    print(f"❌ Directory not found: {data_dir}")
    sys.exit(1)

mat_files = list(Path(data_dir).glob("*.mat"))
if not mat_files:
    print(f"❌ No .mat files found in: {data_dir}")
    sys.exit(1)

print(f"\n📂 Found {len(mat_files)} .mat files")

# Extract SubjectIDs from filenames
recording_ids = {}
for mat_file in mat_files:
    subj_id = extract_subject_id(mat_file.name)
    if subj_id not in recording_ids:
        recording_ids[subj_id] = []
    recording_ids[subj_id].append(mat_file.name)

print(f"   ✓ Found {len(recording_ids)} unique SubjectIDs in recordings")

# 4. Find missing
missing_in_labels = set(recording_ids.keys()) - label_ids

print("\n" + "="*80)
print("RESULTS")
print("="*80)

if missing_in_labels:
    print(f"\n⚠️  FOUND {len(missing_in_labels)} SubjectID(s) in recordings but NOT in labels:\n")
    total_recordings = 0
    for subj_id in sorted(missing_in_labels):
        files = recording_ids[subj_id]
        total_recordings += len(files)
        print(f"   SubjectID: '{subj_id}'")
        print(f"   Recordings: {len(files)}")
        for fname in files:
            print(f"      - {fname}")
        print()

    print(f"   Total recordings that will be EXCLUDED: {total_recordings}\n")
    print("="*80)
    print("\nRECOMMENDATIONS:")
    print("1. Add these SubjectIDs to your labels Excel file")
    print("2. OR check if filenames have typos/different format")
    print("3. OR verify these recordings should be included")
else:
    print("\n✅ All recording SubjectIDs found in labels!")
    print("   No missing labels detected.\n")

# 5. Also check reverse (labels without recordings)
missing_in_recordings = label_ids - set(recording_ids.keys())
if missing_in_recordings:
    print(f"\nℹ️  {len(missing_in_recordings)} SubjectIDs in labels but NOT in recordings:")
    print(f"   (These are OK - just means you have labels for subjects you haven't recorded)")
    for subj_id in sorted(list(missing_in_recordings)[:10]):  # Show first 10
        print(f"      - {subj_id}")
    if len(missing_in_recordings) > 10:
        print(f"      ... and {len(missing_in_recordings) - 10} more")

print("\n" + "="*80)

# Bug Fix: Missing Labels in Features Analysis

## Problem Summary

**Issue**: 3 recordings are missing labels in the comparison feature visualization

**Root Cause**: Subject ID mismatch between recording filenames and labels file

## Technical Analysis

### How Subject IDs Are Extracted

When loading `.mat` files, the `MATDataLoader` extracts subject IDs from filenames using this logic (see `data/loaders.py:237`):

```python
def _extract_subject_id(self, filename: str) -> str:
    base_name = Path(filename).stem
    if len(base_name) >= 4:
        return base_name[:4].upper()  # Takes first 4 characters
    else:
        return base_name.upper()
```

**Example**:
- Filename: `ABCD_-_4_9_16.mat`
- Extracted SubjectID: `ABCD`

### The Merge Process

When features are merged with labels in `features/collection.py:337-393`:

1. Features DataFrame has SubjectIDs extracted from filenames
2. Labels DataFrame has SubjectIDs from the Excel file
3. An **inner join** is performed on SubjectID
4. **Any SubjectID that exists in features but NOT in labels is DROPPED**

```python
merged = features_with_ids.merge(labels_df, on='SubjectID', how='inner')
```

### Why 3 Recordings Are Missing

The 3 recordings have SubjectIDs (extracted from their filenames) that don't exist in the labels file. Possible reasons:

1. **Different format**: Filename uses `ABCD` but labels file has `ABCD123` or `A-BCD`
2. **Case sensitivity**: Filename extracts `ABCD` but labels file has `abcd` (though both are normalized to uppercase)
3. **Truncation**: First 4 characters of filename don't match the actual subject ID
4. **Missing entries**: Those subjects genuinely aren't in the labels file

## Debug Enhancements Added

### 1. Label Loading Debug (main.py:72-74)

Now shows which subject IDs are in the labels file:

```
📋 Labels loaded: 25 subjects
   Subject IDs in labels: ['ABCD', 'EFGH', 'IJKL', ...]
```

### 2. Feature Extraction Debug (main.py:125-131)

Now shows which subject IDs were extracted from recordings:

```
📊 Features extracted: 28 recordings
   Subject IDs in features: ['ABCD', 'EFGH', 'IJKL', 'UNKN', ...]
   Recordings per subject:
      - ABCD: 2 recording(s)
      - EFGH: 1 recording(s)
      - UNKN: 3 recording(s)  ⚠️ This subject might be missing from labels!
```

### 3. Merge Warning (collection.py:358-368)

Now identifies EXACTLY which recordings are being excluded:

```
⚠️  WARNING: 1 subject ID(s) in features have NO labels:
    - 'UNKN': 3 recording(s)
    These 3 recordings will be EXCLUDED from analysis!
```

## How to Identify the Problem

Run the pipeline and look for the debug output:

```bash
python -m pipeline.main
```

The output will show:
1. All subject IDs from labels file
2. All subject IDs extracted from recordings
3. Which subject IDs are missing (causing the 3 recordings to be excluded)

## Solutions

### Option 1: Fix Labels File (Recommended)

Add the missing subject IDs to your labels Excel file.

**Example**: If the debug shows:
```
⚠️  WARNING: 1 subject ID(s) in features have NO labels:
    - 'UNKN': 3 recording(s)
```

Then add a row to your labels file with `SubjectID = UNKN` and appropriate outcome values.

### Option 2: Custom Subject ID Extraction

If your filenames have a different pattern, override the `_extract_subject_id` method:

**Example**: If your filenames are `Patient_ABCD123_date.mat` and you want the full `ABCD123`:

```python
# In data/loaders.py
class CustomMATLoader(MATDataLoader):
    def _extract_subject_id(self, filename: str) -> str:
        """Extract full subject ID from Patient_XXX_date.mat format"""
        import re
        match = re.search(r'Patient_([A-Z0-9]+)_', filename)
        if match:
            return match.group(1).upper()
        else:
            # Fallback to default
            return super()._extract_subject_id(filename)
```

Then use `CustomMATLoader` instead of `MATDataLoader` in `pipeline/main.py:77`.

### Option 3: Subject ID Mapping

Create a mapping file if the IDs differ systematically:

```python
# In pipeline/main.py, after loading labels:
id_mapping = {
    'UNKN': 'UNKNOWN_PATIENT',
    'TEMP': 'TEMPORARY_ID',
    # ... etc
}

# Apply mapping to features
master_features_df['SubjectID'] = master_features_df['SubjectID'].map(
    lambda x: id_mapping.get(x, x)
)
```

### Option 4: Use Recording Date as Secondary Key

If multiple recordings per subject, ensure dates match:

```python
# Merge on both SubjectID and RecordingDate
merged = features_with_ids.merge(
    labels_df,
    on=['SubjectID', 'RecordingDate'],
    how='inner'
)
```

## Verification

After applying a fix, verify:

1. Run pipeline again
2. Check that the warning disappears:
   ```
   ⚠️  WARNING: 0 subject ID(s) in features have NO labels:
   ```

3. Confirm all recordings are included in analysis

4. Check violin plots show data for all expected subjects

## Files Modified

1. **features/collection.py** (lines 358-368)
   - Added debug output showing which SubjectIDs are missing labels

2. **pipeline/main.py** (lines 72-74, 125-131)
   - Added debug output showing SubjectIDs from labels and features
   - Shows count of recordings per subject

3. **debug_labels.py** (new file)
   - Standalone script for debugging label merges
   - Can be run independently to diagnose issues

## Running the Debug Script

Alternatively, use the standalone debug script:

```bash
python debug_labels.py --features output/all_extracted_features.csv --labels path/to/labels.xlsx
```

This will show detailed comparison of SubjectIDs between features and labels.

## Next Steps

1. Run the pipeline with the debug enhancements
2. Identify the 3 missing SubjectIDs from the output
3. Choose and apply one of the solutions above
4. Verify all recordings are included
5. Remove or disable debug print statements (if desired) once fixed

---

**Note**: The debug print statements can be left in production as they provide valuable feedback about data quality and merge success rates.

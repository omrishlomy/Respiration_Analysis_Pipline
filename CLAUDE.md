# Respiration Analysis Pipeline

## Project Overview

Machine learning pipeline for analyzing respiratory recordings to predict clinical outcomes (Recovery, currentConsciousness, Survival).

**Key Features:**
- Signal processing with multi-peak detection
- Breathing parameter extraction (25 metrics)
- Window-based feature aggregation
- Statistical analysis and ML classification
- ROC curve comparisons across models

## CRITICAL REQUIREMENTS - DO NOT CHANGE

**Feature Comparison Plot Format (MUST ALWAYS BE 3 SUBPLOTS):**

⚠️ **NEVER create single combined plots. ALWAYS create 3 side-by-side subplots.**

For EACH feature being plotted, create exactly 3 subplots showing:
1. **Left subplot**: Feature values by Recovery label (Recovery=0 vs Recovery=1 vs Missing)
   - Filter: Only include recordings where `currentConsciousness==0`
   - Rationale: Recovery only applies to patients starting at consciousness level 0
2. **Middle subplot**: Feature values by currentConsciousness label (0 vs 1 vs Missing)
   - No filtering - include all recordings
3. **Right subplot**: Feature values by Survival label (Survival=0 vs Survival=1 vs Missing)
   - No filtering - include all recordings

**Plot Layout:**
- Use matplotlib's `fig, axes = plt.subplots(1, 3, figsize=(width, 5))`
- Each subplot shows scatter points colored by label value (0, 1, Missing)
- Include n= counts in legend for each group
- X-axis: Recording index (0 to ~149)
- Y-axis: Feature value
- Title: `{Feature_Name} by {Label_Name}`

**Data Preparation (CRITICAL):**
- Use `aggregate_and_add_labels()` to get ONE ROW per recording (not per window)
- Include ALL 149 recordings in the aggregated DataFrame
- Apply Recovery filter (currentConsciousness==0) ONLY in plotting code for Recovery subplot
- Never filter during data aggregation phase

This format has been requested multiple times and must be maintained in all future changes.

## Architecture

```
data/              - Data loading, clinical labels, recording objects
preprocessing/     - Signal cleaning, windowing
features/          - Feature extraction, aggregation, visualization
analysis/          - Statistical tests, feature selection
models/            - SVM classifiers, LORO/LOSO cross-validation
visualization/     - Interactive plots (Plotly), ROC curves
pipeline/          - Main orchestration (main.py)
```

## Current Work

**Branch:** `claude/add-roc-comparison-plots-pHWPk`

**Recent Fixes:**
1. ✅ Removed incorrect filter_recovery logic that was losing recordings
   - Issue: Only 131 recordings instead of expected 149 in plots
   - Root Cause: `filter_recovery=True` filtered labels BEFORE merging, losing 18 recordings
   - Solution: Removed filter_recovery parameter from `add_labels_to_features()` and `aggregate_and_add_labels()`
   - Impact: All recordings now included; filtering should be done in visualization layer
   - Files: `data/clinical_labels.py`
   - Note: For Recovery plots, filter by `currentConsciousness==0` in plotting code, not data prep

2. ✅ Added diagnostic logging to track data flow
   - Shows: input rows/windows, unique SubjectIDs, aggregated rows, label counts
   - Helps identify where recordings are being lost
   - Files: `data/clinical_labels.py` (aggregate_and_add_labels method)

3. ✅ Fixed inflated recording counts in feature comparison plots
   - Issue: Plots showing 755 recordings instead of 149 (window-level data, not recording-level)
   - Root Cause: Features DataFrame has multiple rows per recording (one per time window)
   - Solution: Added `aggregate_and_add_labels()` method to group by SubjectID and aggregate features
   - Impact: Plots now show correct n= counts (one point per recording, not per window)
   - Files: `data/clinical_labels.py` (lines 478-570)
   - Usage: Replace `add_labels_to_features()` with `aggregate_and_add_labels()` in visualization scripts

2. ✅ Fixed missing labels in feature comparison plots
   - Issue: SubjectID mismatch due to trailing spaces in Excel file (`'ABOU '` vs `'ABOU'`)
   - Solution: Added SubjectID normalization (strip whitespace, uppercase) in `add_labels_to_features()`
   - Impact: All labels now merge correctly regardless of whitespace or case differences
   - Files: `data/clinical_labels.py` (lines 454-474)

3. ✅ Fixed ROC comparison plots not being created
   - Issue: `run_experiments()` returned empty predictions dict
   - Solution: Collect predictions from LORO, LOSO, and incremental feature experiments
   - Files: `models/experiment.py`, `visualization/interactive.py`

4. ✅ Fixed feature comparison plots not regenerating (Commit: 7b33348)
   - Issue: `merge_with_features()` expected single outcome (string), but script passed list
   - Solution: Added `add_labels_to_features()` method for visualization pipelines
   - Files: `data/clinical_labels.py`

**Known Issues:**
- None currently

## Running the Pipeline

**Full Pipeline:**
```bash
python pipeline/main.py
```

**Features Layer Only:**
```bash
python features/run_features_layer.py
```

**Configuration:** `config.yaml` (auto-detected)

## Key Files

- `pipeline/main.py` - Main entry point
- `models/experiment.py` - Experiment orchestration (LORO/LOSO/Incremental)
- `data/clinical_labels.py` - Label loading and merging
- `visualization/interactive.py` - ROC plots and confusion matrices
- `features/run_features_layer.py` - Feature extraction + visualization (local file, untracked)

## Reference Code: 3-Subplot Feature Comparison Plots

**CRITICAL: Use this template for all feature comparison plots. DO NOT create single combined plots.**

```python
def plot_feature_by_labels(features_df, feature_name, label_columns, output_path):
    """
    Create 3-subplot feature comparison plot.

    Args:
        features_df: DataFrame with aggregated features (ONE ROW per recording)
        feature_name: Name of feature column to plot
        label_columns: List of 3 labels ['Recovery', 'currentConsciousness', 'Survival']
        output_path: Where to save the plot
    """
    import matplotlib.pyplot as plt
    import numpy as np

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
            plot_data = features_df[features_df['currentConsciousness'] == 0].copy()
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
                   c=colors[0], label=f'{label_col}=0 (n={n_0})', alpha=0.6)
        ax.scatter(x_indices[mask_1], feature_values[mask_1],
                   c=colors[1], label=f'{label_col}=1 (n={n_1})', alpha=0.6)
        if n_missing > 0:
            ax.scatter(x_indices[mask_missing], feature_values[mask_missing],
                       c=colors['missing'], label=f'Missing (n={n_missing})', alpha=0.3)

        # Labels and styling
        ax.set_xlabel('Recording Index')
        ax.set_ylabel(feature_name)
        ax.set_title(f'{feature_name} by {label_col}')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
```

**Usage in run_features_layer.py:**

```python
# 1. Aggregate features (ONE ROW per recording)
# CORRECT - No filter_recovery parameter
features_df = labels.aggregate_and_add_labels(
    features_df,
    label_columns=['Recovery', 'currentConsciousness', 'Survival'],
    subject_id_column='Name',  # Use 'Name' based on config.yaml
    aggregation='mean'
)

# 2. Plot each feature with 3-subplot layout
for feature_name in feature_list:
    output_path = output_dir / f'{feature_name}_comparison.png'
    plot_feature_by_labels(features_df, feature_name,
                          ['Recovery', 'currentConsciousness', 'Survival'],
                          output_path)
```

## Common Errors and Fixes

### ❌ ERROR 1: `TypeError: got an unexpected keyword argument 'filter_recovery'`

**Wrong code:**
```python
features_df = labels.aggregate_and_add_labels(
    features_df,
    label_columns,
    subject_id_column='Name',
    aggregation='mean',
    filter_recovery=True  # ← REMOVED IN LATEST VERSION
)
```

**Correct code:**
```python
features_df = labels.aggregate_and_add_labels(
    features_df,
    label_columns=['Recovery', 'currentConsciousness', 'Survival'],
    subject_id_column='Name',
    aggregation='mean'
)
# Do filtering in plotting code, not data preparation
```

### ❌ ERROR 2: Single combined plot instead of 3 subplots

**Wrong:** Using `plot_feature_comparison()` or similar that creates one plot

**Correct:** Always use `plot_feature_by_labels()` which creates 3 subplots:
```python
# This function MUST create 3 subplots using:
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
```

### ❌ ERROR 3: Wrong number of recordings (131 instead of 149)

**Cause:** Filtering happens during data preparation instead of visualization

**Fix:**
- Remove any `filter_recovery=True` calls
- ALL 149 recordings must be in the aggregated DataFrame
- Filter by `currentConsciousness==0` ONLY in plotting code for Recovery subplot

## Complete Working Example for run_features_layer.py

```python
from pathlib import Path
from data.clinical_labels import ClinicalLabels
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load clinical labels
labels = ClinicalLabels.from_excel(
    labels_file='path/to/respiratory_analysis_all_subjects.xlsx',
    subject_id_column='Name'  # Use 'Name' per config.yaml
)

# Load features (window-level data)
features_df = pd.read_csv('path/to/features.csv')
# features_df has ~755 rows (multiple windows per recording)

# Aggregate to recording level and add labels
features_df = labels.aggregate_and_add_labels(
    features_df,
    label_columns=['Recovery', 'currentConsciousness', 'Survival'],
    subject_id_column='Name',
    aggregation='mean'
)
# features_df now has ~149 rows (one per recording) with labels

# Define 3-subplot plotting function
def plot_feature_by_labels(features_df, feature_name, label_columns, output_path):
    """Create 3-subplot feature comparison plot."""
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    colors = {0: 'orange', 1: 'blue', 'missing': 'gray'}

    for idx, (label_col, ax) in enumerate(zip(label_columns, axes)):
        # Filter Recovery subplot only
        if label_col == 'Recovery':
            plot_data = features_df[features_df['currentConsciousness'] == 0].copy()
        else:
            plot_data = features_df.copy()

        feature_values = plot_data[feature_name].values
        label_values = plot_data[label_col].values
        x_indices = np.arange(len(plot_data))

        mask_0 = (label_values == 0)
        mask_1 = (label_values == 1)
        mask_missing = pd.isna(label_values)

        n_0, n_1, n_missing = mask_0.sum(), mask_1.sum(), mask_missing.sum()

        ax.scatter(x_indices[mask_0], feature_values[mask_0],
                   c=colors[0], label=f'{label_col}=0 (n={n_0})', alpha=0.6, s=30)
        ax.scatter(x_indices[mask_1], feature_values[mask_1],
                   c=colors[1], label=f'{label_col}=1 (n={n_1})', alpha=0.6, s=30)
        if n_missing > 0:
            ax.scatter(x_indices[mask_missing], feature_values[mask_missing],
                       c=colors['missing'], label=f'Missing (n={n_missing})', alpha=0.3, s=30)

        ax.set_xlabel('Recording Index')
        ax.set_ylabel(feature_name)
        ax.set_title(f'{feature_name} by {label_col}')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

# Plot each feature
output_dir = Path('outputs')
output_dir.mkdir(exist_ok=True)

feature_list = ['BreathingRate_mean', 'Duty_Cycle_inhale_mean', ...]  # Your features
for feature_name in feature_list:
    output_path = output_dir / f'{feature_name}_comparison.png'
    plot_feature_by_labels(
        features_df,
        feature_name,
        ['Recovery', 'currentConsciousness', 'Survival'],
        output_path
    )
```

**Expected Output:**
```
[aggregate_and_add_labels] Input features:
  - Total rows (windows): 755
  - Unique SubjectIDs: 149

[aggregate_and_add_labels] After aggregation:
  - Total rows (recordings): 149
  - Aggregation method: mean

[aggregate_and_add_labels] After adding labels:
  - Total rows: 149
  - Recovery: 131 valid, 18 missing
  - currentConsciousness: 149 valid, 0 missing
  - Survival: 141 valid, 8 missing
```

Each plot will have:
- **3 subplots** (Recovery | currentConsciousness | Survival)
- **Correct counts** (~149 total across all subplots)
- **Recovery filtered** to show only currentConsciousness==0

## Development Workflow

**Git Branches:**
- Always develop on `claude/*` branches
- Branch naming: `claude/<description>-<sessionID>`
- Push when work is complete
- Never push to main directly

**Important Notes:**
- Debug logging added to identify missing SubjectIDs (commit 6653d48)
- Performance optimization: O(n²) → O(n) pause calculation (commit cc0663c)
- Feature comparison plots require LEFT JOIN to show missing labels

## Session Resumption

To continue this work in a new session:
1. Use `claude --continue` or `claude --resume`
2. Or provide context: "Continuing work on ROC comparison plots"
3. This file preserves project context automatically

---

## Instructions for Claude (Auto-Maintenance)

**IMPORTANT: Claude should automatically update this file during sessions to keep it current.**

### When to Add Issues to "Known Issues":

Add a new issue when:
- ✅ User reports a bug or problem
- ✅ Tests fail or code doesn't work as expected
- ✅ You discover an issue while working on code
- ✅ User mentions something isn't working correctly
- ✅ Error messages or warnings appear during execution

**Format:**
```
- [Short description] (affected file/module)
  - Details: [what's wrong]
  - Impact: [how it affects users]
```

### When to Remove/Update Issues from "Known Issues":

Remove an issue when:
- ✅ The fix is committed and pushed
- ✅ Tests pass confirming the fix works
- ✅ User confirms the issue is resolved

Move to "Recent Fixes" when:
- ✅ Issue is fully resolved
- ✅ Add the commit hash or branch where it was fixed

**Update Process:**
1. When you fix an issue, immediately update this file
2. Move the issue from "Known Issues" to "Recent Fixes"
3. Add solution details and affected files
4. Commit the CLAUDE.md update with your fix

### When to Update "Current Work":

Update when:
- ✅ Starting work on a new feature or fix
- ✅ Switching to a new branch
- ✅ Completing a major milestone
- ✅ User provides new requirements or changes direction

### General Maintenance:

- Keep "Recent Fixes" to last 5-7 items (move older ones to git history)
- Update "Important Notes" when significant architectural decisions are made
- Add new files to "Key Files" when they become central to the project
- Update branch name in "Current Work" when switching branches

**Proactive Updates:** Don't wait for the user to ask - update this file automatically whenever you make significant changes!

---

## Contact

Project: Respiration Analysis for Clinical Outcomes

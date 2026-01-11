# Respiration Analysis Pipeline

## Project Overview

Machine learning pipeline for analyzing respiratory recordings to predict clinical outcomes (Recovery, currentConsciousness, Survival).

**Key Features:**
- Signal processing with multi-peak detection
- Breathing parameter extraction (25 metrics)
- Window-based feature aggregation
- Statistical analysis and ML classification
- ROC curve comparisons across models

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
1. ✅ Fixed ROC comparison plots not being created
   - Issue: `run_experiments()` returned empty predictions dict
   - Solution: Collect predictions from LORO, LOSO, and incremental feature experiments
   - Files: `models/experiment.py`, `visualization/interactive.py`

2. ✅ Fixed feature comparison plots not regenerating
   - Issue: `merge_with_features()` expected single outcome (string), but script passed list
   - Solution: Added `add_labels_to_features()` method for visualization pipelines
   - Files: `data/clinical_labels.py`

**Known Issues:**
- Feature layer comparison plots show old data (need to update run_features_layer.py)
- User needs to change line 258: `merge_with_features` → `add_labels_to_features`

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

## Contact

Project: Respiration Analysis for Clinical Outcomes

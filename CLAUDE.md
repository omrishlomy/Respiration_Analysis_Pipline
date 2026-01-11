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

2. ✅ Fixed feature comparison plots not regenerating (Commit: 7b33348)
   - Issue: `merge_with_features()` expected single outcome (string), but script passed list
   - Solution: Added `add_labels_to_features()` method for visualization pipelines
   - Files: `data/clinical_labels.py`
   - Action Required: User needs to update `run_features_layer.py` line 258

**Known Issues:**
- ⚠️ User's local `run_features_layer.py` needs manual update
  - Details: Line 258 calls `merge_with_features()` but should call `add_labels_to_features()`
  - Impact: Comparison plots not regenerating with current data
  - Fix: Change `labels.merge_with_features(...)` → `labels.add_labels_to_features(...)`
  - File: `features/run_features_layer.py` (local untracked file)

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

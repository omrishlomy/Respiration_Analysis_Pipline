# Implementation Guide

## What We've Built

A complete API structure for a professional respiratory analysis toolkit with:

✅ **7 Major Layers**: Data, Preprocessing, Features, Analysis, Models, Visualization, Pipeline  
✅ **45+ Classes** with clear interfaces and responsibilities  
✅ **Zero Implementation** - pure API definitions ready for careful implementation  
✅ **Extensible Design** - easy to add new features, classifiers, and visualizations  
✅ **Configuration-Driven** - YAML-based configuration for reproducibility  

---

## Implementation Order (4 Weeks)

### Week 1: Foundation Layer
**Goal**: Load data and manage labels

#### Priority 1: Data Layer
1. **Start Here**: `respiratory_analysis/data/recording.py`
   - Implement `RespiratoryRecording` class
   - Properties: `duration`, `n_samples`, `time_axis`
   - Methods: `get_segment()`, `get_samples_range()`

2. **Then**: `respiratory_analysis/data/loaders.py`
   - Implement `MATDataLoader`
   - Focus on loading your specific .mat format
   - Test with 2-3 real files

3. **Next**: `respiratory_analysis/data/clinical_labels.py`
   - Implement `ClinicalLabels`
   - Methods: `from_excel()`, `get_label()`, `remap_labels()`

4. **Finally**: `respiratory_analysis/data/exporters.py`
   - Implement `ExcelExporter`
   - Method: `add_sheet()`, `write()`

**Deliverable**: Can load .mat files and Excel labels, export DataFrames to Excel

**Test**:
```python
loader = MATDataLoader()
rec = loader.load('test.mat')
print(f"Duration: {rec.duration}s, Samples: {rec.n_samples}")

labels = ClinicalLabels.from_excel('labels.xlsx')
print(labels.summary())
```

---

### Week 2: Processing & Features
**Goal**: Clean signals and extract breathing parameters

#### Priority 2: Preprocessing
5. **Start**: `respiratory_analysis/preprocessing/cleaner.py`
   - Implement basic `clean()` method
   - Add outlier removal and filtering
   - Keep it simple initially

6. **Then**: `respiratory_analysis/preprocessing/windowing.py`
   - Implement `WindowGenerator`
   - Method: `generate_windows()` with overlap logic
   - Test window count calculations

**Test**:
```python
cleaner = SignalCleaner()
clean_rec = cleaner.clean(rec)

generator = WindowGenerator(WindowConfig(window_size=300, overlap=60))
windows = generator.generate_windows(clean_rec)
print(f"Created {len(windows)} windows")
```

#### Priority 3: Feature Extraction (MOST IMPORTANT!)
7. **Critical**: `respiratory_analysis/features/extractors/breathing.py`
   - This is where you port your MATLAB breathing parameter code!
   - Implement `BreathingParameterExtractor`
   - Start with core parameters:
     - Respiratory rate (mean, std, CV)
     - Breath amplitude (mean, std, CV)
     - Inter-breath intervals
   - Add remaining 19 parameters gradually
   
8. **Then**: `respiratory_analysis/features/collection.py`
   - Implement `FeatureCollection`
   - Methods: `from_dict()`, `get_features_array()`, `normalize()`

9. **Finally**: `respiratory_analysis/features/aggregator.py`
   - Implement `FeatureAggregator`
   - Method: `aggregate()` with mean, std, CV

**Deliverable**: Extract 25 breathing parameters per window, aggregate to recording level

**Test**:
```python
extractor = BreathingParameterExtractor()
features = extractor.extract(windows[0].data, windows[0].sampling_rate)
print(f"Extracted {len(features)} parameters:")
print(features)
```

---

### Week 3: Analysis
**Goal**: Statistical tests and dimensionality reduction

#### Priority 4: Statistical Analysis
10. **Implement**: `respiratory_analysis/analysis/statistical.py`
    - Method: `compare_groups()` using scipy.stats
    - T-tests and Mann-Whitney U tests
    - Multiple comparison correction (statsmodels)

11. **Then**: `respiratory_analysis/analysis/dimensionality.py`
    - Implement `PCAReducer` (using sklearn.decomposition.PCA)
    - Implement `TSNEReducer` (using sklearn.manifold.TSNE)

**Deliverable**: Identify significant features, perform PCA/t-SNE

**Test**:
```python
analyzer = StatisticalAnalyzer()
stats = analyzer.compare_groups(features, labels, 'consciousness')
significant = analyzer.select_significant_features(stats)
print(f"Significant features: {significant}")
```

---

### Week 4: Models & Integration
**Goal**: Complete classification pipeline with visualization

#### Priority 5: Classification
12. **Implement**: `respiratory_analysis/models/classifiers.py`
    - Wrap sklearn classifiers:
      - `SVMClassifier` → sklearn.svm.SVC
      - `KNNClassifier` → sklearn.neighbors.KNeighborsClassifier
      - `RandomForestClassifier` → sklearn.ensemble.RandomForestClassifier
    
13. **Then**: `respiratory_analysis/models/__init__.py` (evaluator and tuner)
    - `ModelEvaluator` using sklearn.metrics
    - `GridSearchTuner` using sklearn.model_selection.GridSearchCV

**Test**:
```python
svm = SVMClassifier(kernel='rbf')
svm.train(X_train, y_train)
evaluator = ModelEvaluator()
metrics = evaluator.evaluate(svm, X_test, y_test)
print(metrics)
```

#### Priority 6: Visualization
14. **Implement**: `respiratory_analysis/visualization/__init__.py`
    - Start with Plotly for interactive plots
    - Methods: `plot_confusion_matrix()`, `plot_roc_curves()`, `plot_scatter_2d()`

**Test**:
```python
plotter = PlotlyPlotter()
plotter.plot_confusion_matrix(y_true, y_pred, ['Unconscious', 'Conscious'])
```

#### Priority 7: Pipeline (Final Integration)
15. **Implement**: `respiratory_analysis/pipeline/__init__.py`
    - `AnalysisConfig.from_yaml()`
    - `AnalysisPipeline.run()` - orchestrate all components
    
**Deliverable**: End-to-end analysis from YAML config

---

## Implementation Tips

### General Approach
1. **One class at a time**: Don't try to implement everything at once
2. **Test immediately**: Write simple tests as you go
3. **Use your MATLAB code**: Port the algorithms you know work
4. **Start simple**: Add complexity gradually
5. **Document as you code**: Docstrings help you remember intent

### Specific Recommendations

**For BreathingParameterExtractor** (Week 2):
- This is your core IP - port carefully from MATLAB
- Test each parameter extraction independently
- Compare results with MATLAB output for validation
- Use scipy.signal for peak detection

**For Statistical Analysis** (Week 3):
- Use `scipy.stats.ttest_ind()` for t-tests
- Use `statsmodels.stats.multitest.multipletests()` for FDR correction
- pandas DataFrames make life much easier here

**For Classifiers** (Week 4):
- These are just thin wrappers around sklearn
- Focus on consistent interface
- The base `Classifier` class does most of the work

### Testing Strategy
Create simple test files:
```
tests/
├── test_data/
│   ├── test_recording.py  # Test RespiratoryRecording
│   ├── test_loaders.py    # Test MATDataLoader
├── test_features/
│   ├── test_breathing_extractor.py  # Critical!
├── test_pipeline/
│   ├── test_end_to_end.py  # Final integration test
```

---

## Quick Start After Implementation

Once you've implemented Weeks 1-2, you can start using it:

```python
# Load data
from respiratory_analysis.data import MATDataLoader, ClinicalLabels
loader = MATDataLoader()
recordings = loader.load_batch('data/')
labels = ClinicalLabels.from_excel('labels.xlsx')

# Preprocess
from respiratory_analysis.preprocessing import SignalCleaner, WindowGenerator, WindowConfig
cleaner = SignalCleaner()
clean_recs = [cleaner.clean(r) for r in recordings]

config = WindowConfig(window_size=300, overlap=60)
generator = WindowGenerator(config)
windows = [w for r in clean_recs for w in generator.generate_windows(r)]

# Extract features
from respiratory_analysis.features import BreathingParameterExtractor, FeatureCollection
extractor = BreathingParameterExtractor()
features = [extractor.extract(w.data, w.sampling_rate) for w in windows]

# You now have your 25 breathing parameters!
import pandas as pd
features_df = pd.DataFrame(features)
features_df.to_excel('extracted_features.xlsx')
```

---

## Questions to Answer Before Starting

1. **MATLAB Code Location**: Where is your current working MATLAB code?
2. **Data Format**: What keys are used in your .mat files? (data, fs, etc.)
3. **Excel Structure**: Column names in your labels Excel file?
4. **Priority Outcomes**: Which outcome is most important? (consciousness, recovery, survival)
5. **Computing Environment**: Local machine or cluster?

---

## Ready to Start?

Pick one of these:

**Option A**: "Let's implement the data layer together" (Week 1)
**Option B**: "Show me how to port the breathing parameter extractor" (Week 2 - most critical)
**Option C**: "I want to implement everything myself, just clarify X"

Which would you like?

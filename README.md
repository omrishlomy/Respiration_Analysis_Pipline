# Respiratory Analysis Toolkit - API Documentation

## Project Overview

A professional, object-oriented Python package for analyzing respiratory data and predicting clinical outcomes in Disorders of Consciousness (DoC).

### Key Features
- 🔄 Modular, extensible architecture
- 📊 Comprehensive breathing parameter extraction (25+ parameters)
- 📈 Statistical analysis with multiple comparison correction
- 🤖 Multiple ML classifiers (SVM, KNN, Random Forest, XGBoost, Logistic Regression)
- 🎨 Interactive and static visualization
- 📁 Excel/CSV export with multiple sheets
- 🔧 Hyperparameter tuning (GridSearch)
- 📉 Dimensionality reduction (PCA, t-SNE, UMAP)
- 💾 Model persistence

---

## Development Order (4-Week Plan)

### Week 1: Foundation
1. Data layer (recording, loaders, labels, exporters)
2. Unit tests for data layer

### Week 2: Core Processing
3. Preprocessing layer (cleaner, windowing)
4. Feature extraction (extractors, collection, aggregator)
5. Tests with synthetic signals

### Week 3: Analysis
6. Statistical analysis
7. Feature selection
8. Dimensionality reduction
9. Integration tests

### Week 4: Models & Integration
10. Classifiers and tuning
11. Model evaluation
12. Visualization
13. Pipeline orchestration
14. End-to-end tests
15. Documentation and examples

---

## Package Structure

```
respiratory_analysis/
├── data/              # Data loading, storage, export
├── preprocessing/     # Signal cleaning and windowing
├── features/          # Feature extraction and aggregation
├── analysis/          # Statistical tests, feature selection, dim. reduction
├── models/            # Classifiers, tuning, evaluation
├── visualization/     # Plotting (Plotly + Matplotlib)
└── pipeline/          # Orchestration and configuration
```

---

## API Summary by Layer

### 1. Data Layer

**Classes:**
- `RespiratoryRecording`: Core data structure for a single recording
- `MATDataLoader`, `CSVDataLoader`, `BinaryDataLoader`: Load different formats
- `ClinicalLabels`: Manage outcome labels
- `ExcelExporter`, `CSVExporter`, `ResultsExporter`: Export results

**Key Methods:**
```python
# Loading data
loader = MATDataLoader(data_key='data', fs_key='fs')
recording = loader.load('subject_001_20231015.mat')
recordings = loader.load_batch('data_directory/')

# Managing labels
labels = ClinicalLabels.from_excel('labels.xlsx')
label = labels.get_label('subject_001', 'consciousness')
labels = labels.remap_labels('consciousness', {2: 1})  # MCS → conscious

# Exporting
exporter = ExcelExporter('results.xlsx')
exporter.add_features_sheet(features_df)
exporter.add_statistical_results_sheet(stats_df)
exporter.write()
```

### 2. Preprocessing Layer

**Classes:**
- `SignalCleaner`: Remove artifacts, filter, baseline correction
- `WindowConfig`, `SignalWindow`: Window configuration and representation
- `WindowGenerator`: Split recordings into blocks
- `BatchWindowProcessor`: Process multiple recordings

**Key Methods:**
```python
# Clean signal
cleaner = SignalCleaner(remove_outliers=True, apply_filter=True)
clean_recording = cleaner.clean(recording)

# Create windows
config = WindowConfig(window_size=300, overlap=60)  # 5min windows, 1min overlap
generator = WindowGenerator(config)
windows = generator.generate_windows(clean_recording)
```

### 3. Features Layer

**Classes:**
- `BreathingParameterExtractor`: Extract 25+ breathing parameters
- `SpectralBreathingExtractor`: Frequency-domain features
- `FeatureCollection`: Container for extracted features
- `FeatureAggregator`: Aggregate block → recording level
- `FeatureEngineering`: Create derived features

**Key Methods:**
```python
# Extract features
extractor = BreathingParameterExtractor()
features = extractor.extract(window.data, window.sampling_rate)
# Returns: {'respiratory_rate_mean': 15.2, 'breath_amplitude_std': 0.8, ...}

# Aggregate blocks to recording level
aggregator = FeatureAggregator()
recording_features = aggregator.aggregate(block_features)
```

### 4. Analysis Layer

**Classes:**
- `StatisticalAnalyzer`: Compare groups, significance testing
- `FeatureSelector`: Select important features
- `PCAReducer`, `TSNEReducer`, `UMAPReducer`: Dimensionality reduction

**Key Methods:**
```python
# Statistical comparison
analyzer = StatisticalAnalyzer(test='ttest', alpha=0.05)
results = analyzer.compare_groups(features, labels, outcome='consciousness')
significant_features = analyzer.select_significant_features(results)

# Dimensionality reduction
pca = PCAReducer(n_components=2)
reduced = pca.fit_transform(X)
explained_var = pca.get_explained_variance()
```

### 5. Models Layer

**Classes:**
- `SVMClassifier`, `KNNClassifier`, `RandomForestClassifier`, etc.
- `GridSearchTuner`: Hyperparameter optimization
- `ModelEvaluator`: Performance metrics
- `ModelPersistence`: Save/load models

**Key Methods:**
```python
# Train classifier
svm = SVMClassifier(kernel='rbf', C=1.0)
svm.train(X_train, y_train)
predictions = svm.predict(X_test)

# Hyperparameter tuning
tuner = GridSearchTuner(SVMClassifier, param_grid={'C': [0.1, 1, 10], 'kernel': ['rbf', 'linear']})
best_model = tuner.search(X_train, y_train, cv=5)

# Evaluation
evaluator = ModelEvaluator()
metrics = evaluator.evaluate(svm, X_test, y_test)
# Returns: {'accuracy': 0.85, 'precision': 0.82, 'recall': 0.88, ...}
```

### 6. Visualization Layer

**Classes:**
- `PlotlyPlotter`: Interactive plots (default)
- `MatplotlibPlotter`: Static publication-quality plots
- `ModelComparisonPlotter`: Compare multiple models

**Key Methods:**
```python
# Create visualizations
plotter = PlotlyPlotter(output_dir='plots/')
plotter.plot_confusion_matrix(y_true, y_pred, labels=['Unconscious', 'Conscious'])
plotter.plot_roc_curves(classifiers, X_test, y_test)
plotter.plot_feature_importance(feature_names, importances)
plotter.plot_scatter_2d(reduced_features, labels, title='PCA Projection')
```

### 7. Pipeline Layer

**Classes:**
- `AnalysisConfig`: Configuration dataclass
- `AnalysisPipeline`: Main orchestration
- `BatchProcessor`: Process multiple experiments

**Key Methods:**
```python
# Configure and run pipeline
config = AnalysisConfig(
    data_dir='data/',
    labels_file='labels.xlsx',
    outcomes=['consciousness', 'recovery', 'survival'],
    classifiers=['svm', 'knn', 'rf'],
    window_size=300,
    overlap=60
)

pipeline = AnalysisPipeline(config)
results = pipeline.run()
# Automatically: loads data, extracts features, runs stats, trains models, generates plots
```

---

## Usage Examples

### Example 1: Basic Analysis
```python
from respiratory_analysis import AnalysisPipeline, AnalysisConfig

config = AnalysisConfig.from_yaml('config.yaml')
pipeline = AnalysisPipeline(config)
results = pipeline.run()
```

### Example 2: Custom Feature Extractor
```python
from respiratory_analysis.features import FeatureExtractor

class MyCustomExtractor(FeatureExtractor):
    def extract(self, signal, sampling_rate):
        # Your custom logic
        return {'my_feature': value}
    
    def get_feature_names(self):
        return ['my_feature']

# Use in pipeline
pipeline.add_feature_extractor(MyCustomExtractor())
```

### Example 3: Single Recording Analysis
```python
from respiratory_analysis.data import MATDataLoader
from respiratory_analysis.preprocessing import SignalCleaner, WindowGenerator
from respiratory_analysis.features import BreathingParameterExtractor

# Load and clean
loader = MATDataLoader()
recording = loader.load('my_recording.mat')

cleaner = SignalCleaner()
clean = cleaner.clean(recording)

# Extract features
generator = WindowGenerator(WindowConfig(window_size=300))
windows = generator.generate_windows(clean)

extractor = BreathingParameterExtractor()
all_features = [extractor.extract(w.data, w.sampling_rate) for w in windows]
```

---

## Configuration File Example

```yaml
# config.yaml
data:
  data_dir: "path/to/recordings"
  labels_file: "path/to/labels.xlsx"
  output_dir: "results"

preprocessing:
  window_size: 300  # seconds
  overlap: 60
  remove_outliers: true
  apply_filter: true

features:
  extractors:
    - breathing  # BreathingParameterExtractor
    - spectral   # SpectralBreathingExtractor
  
analysis:
  outcomes:
    - consciousness
    - recovery
    - survival
  statistical_test: ttest
  alpha: 0.05
  dimensionality_reduction:
    - pca
    - tsne

models:
  classifiers:
    - svm
    - knn
    - random_forest
    - xgboost
  hyperparameter_tuning: true
  cv_folds: 5

visualization:
  backend: plotly  # or matplotlib
  plot_types:
    - confusion_matrix
    - roc_curves
    - feature_importance
    - pca_scatter
```

---

## Next Steps

1. **Start implementing**: Begin with data layer (Week 1)
2. **Test as you go**: Write unit tests for each component
3. **Use your MATLAB code as reference**: Port algorithms carefully
4. **Ask questions**: Clarify any API design before implementing

Would you like me to:
- Start implementing any specific module?
- Create test templates?
- Generate example scripts?
- Modify any API design?

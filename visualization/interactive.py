import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


class InteractivePlotter:
    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def plot_signal_traces(self, raw_data, clean_data, fs, subject_id):
        save_dir = self.output_dir / "signals"
        save_dir.mkdir(parents=True, exist_ok=True)
        time = np.arange(len(raw_data)) / fs

        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1,
                            subplot_titles=(f"Raw: {subject_id}", "Cleaned"))
        fig.add_trace(go.Scatter(x=time, y=raw_data, name="Raw", line=dict(color='gray', width=1)), row=1, col=1)
        fig.add_trace(go.Scatter(x=time, y=clean_data, name="Cleaned", line=dict(color='#00CC96', width=1)), row=2,
                      col=1)
        fig.update_layout(height=600, showlegend=False, title_text=f"Signal: {subject_id}")

        safe_name = "".join([c for c in subject_id if c.isalnum() or c in ('-', '_')]).strip()
        fig.write_html(str(save_dir / f"{safe_name}.html"))

    def plot_feature_violins(self, X_df, y, outcome_name):
        save_dir = self.output_dir / "violins"
        save_dir.mkdir(parents=True, exist_ok=True)
        plot_df = X_df.copy()
        plot_df['Outcome'] = [str(lbl) for lbl in y]

        for feature in X_df.columns:
            try:
                fig = px.violin(plot_df, x='Outcome', y=feature, color='Outcome', box=True, points="all",
                                title=f"{feature} by {outcome_name}")
                safe_feat = "".join([c for c in feature if c.isalnum() or c in ('_')]).strip()
                fig.write_html(str(save_dir / f"{safe_feat}.html"))
            except:
                pass

    def plot_statistical_ranking(self, stats_df):
        if stats_df.empty: return
        # --- FIX: Auto-detect correct column name ---
        feat_col = 'feature_name' if 'feature_name' in stats_df.columns else 'feature'

        df = stats_df.sort_values('p_value', ascending=True).head(20).iloc[::-1]
        fig = px.bar(df, x='p_value', y=feat_col,
                     title="Top 20 Features (p-value)", orientation='h', text='p_value')
        fig.add_vline(x=0.05, line_dash="dash", line_color="red")
        fig.update_traces(texttemplate='%{text:.4f}', textposition='outside')
        fig.write_html(str(self.output_dir / "statistical_ranking.html"))

    def plot_feature_distributions(self, X_df, y, outcome_name, significance_df=None):
        top_features = []
        if significance_df is not None and not significance_df.empty:
            # --- FIX: Auto-detect here too ---
            feat_col = 'feature_name' if 'feature_name' in significance_df.columns else 'feature'
            top_features = significance_df.sort_values('p_value')[feat_col].head(6).tolist()
        else:
            top_features = X_df.columns[:6].tolist()
        if not top_features: return

        plot_df = X_df[top_features].copy()
        plot_df['Outcome'] = [str(lbl) for lbl in y]
        plot_df = plot_df.melt(id_vars='Outcome', var_name='Feature', value_name='Value')

        fig = px.box(plot_df, x='Outcome', y='Value', color='Outcome',
                     facet_col='Feature', facet_col_wrap=3,
                     title=f"Top Features by {outcome_name}")
        fig.update_yaxes(matches=None)
        fig.write_html(str(self.output_dir / "feature_distributions.html"))

    def plot_feature_matrix(self, matrix_df, filename="feature_matrix.html"):
        df_norm = (matrix_df - matrix_df.mean()) / (matrix_df.std() + 1e-9)
        fig = go.Figure(data=go.Heatmap(
            z=df_norm.values.T, x=df_norm.index, y=df_norm.columns,
            colorscale='Viridis',
            hovertemplate='Time: %{x}<br>Feature: %{y}<br>Value: %{z:.2f}<extra></extra>'
        ))
        fig.update_layout(title="Feature Matrix (Normalized)", height=800)
        fig.write_html(str(self.output_dir / filename))

    def plot_comparison_roc(self, model_predictions, y_true, filename="ROC_Comparison.html"):
        fig = go.Figure()
        fig.add_shape(type='line', line=dict(dash='dash', color='gray'), x0=0, x1=1, y0=0, y1=1)
        for name, y_prob in model_predictions.items():
            try:
                score = y_prob[:, 1] if y_prob.ndim > 1 else y_prob
                fpr, tpr, _ = roc_curve(y_true, score)
                roc_auc = auc(fpr, tpr)
                fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'{name} (AUC={roc_auc:.2f})'))
            except:
                continue
        fig.update_layout(title="ROC Comparison", xaxis_title="FPR", yaxis_title="TPR")
        fig.write_html(str(self.output_dir / filename))

    def plot_confusion_matrix(self, y_true, y_pred, title="Confusion Matrix", filename="cm.html"):
        save_dir = self.output_dir / "confusion_matrices"
        save_dir.mkdir(parents=True, exist_ok=True)
        cm = confusion_matrix(y_true, y_pred)
        annotations = []
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                annotations.append(dict(x=j, y=i, text=str(cm[i, j]), showarrow=False, font=dict(color='white')))
        fig = go.Figure(data=go.Heatmap(z=cm, colorscale='Blues', showscale=False))
        fig.update_layout(title=title, xaxis_title="Predicted", yaxis_title="True", annotations=annotations)
        fig.write_html(str(save_dir / filename))

    def plot_correlation_matrix(self, X_df, outcome_name, filename="correlation_matrix.html"):
        """Plot correlation matrix heatmap of features."""
        save_dir = self.output_dir / "data_analysis"
        save_dir.mkdir(parents=True, exist_ok=True)

        # Compute correlation matrix
        corr_matrix = X_df.corr()

        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu',
            zmid=0,
            zmin=-1,
            zmax=1,
            hovertemplate='Feature 1: %{x}<br>Feature 2: %{y}<br>Correlation: %{z:.3f}<extra></extra>'
        ))
        fig.update_layout(
            title=f"Feature Correlation Matrix - {outcome_name}",
            height=800,
            width=900
        )
        fig.write_html(str(save_dir / filename))

    def plot_pca_2d(self, X_df, y, outcome_name, filename="pca_2d.html"):
        """Plot 2D PCA visualization of the data."""
        save_dir = self.output_dir / "data_analysis"
        save_dir.mkdir(parents=True, exist_ok=True)

        # Standardize the data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_df)

        # Perform PCA
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)

        # Create DataFrame for plotting
        pca_df = pd.DataFrame({
            'PC1': X_pca[:, 0],
            'PC2': X_pca[:, 1],
            'Outcome': [str(label) for label in y]
        })

        # Create scatter plot
        fig = px.scatter(
            pca_df,
            x='PC1',
            y='PC2',
            color='Outcome',
            title=f'PCA 2D Projection - {outcome_name}<br>Explained Variance: PC1={pca.explained_variance_ratio_[0]:.2%}, PC2={pca.explained_variance_ratio_[1]:.2%}',
            labels={'PC1': f'PC1 ({pca.explained_variance_ratio_[0]:.2%})',
                   'PC2': f'PC2 ({pca.explained_variance_ratio_[1]:.2%})'}
        )
        fig.update_traces(marker=dict(size=10, opacity=0.7))
        fig.write_html(str(save_dir / filename))

    def plot_pca_3d(self, X_df, y, outcome_name, filename="pca_3d.html"):
        """Plot 3D PCA visualization of the data."""
        save_dir = self.output_dir / "data_analysis"
        save_dir.mkdir(parents=True, exist_ok=True)

        # Standardize the data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_df)

        # Perform PCA
        pca = PCA(n_components=min(3, X_df.shape[1]))
        X_pca = pca.fit_transform(X_scaled)

        if X_pca.shape[1] < 3:
            print(f"    ⚠️  Not enough features for 3D PCA (only {X_pca.shape[1]} components)")
            return

        # Create DataFrame for plotting
        pca_df = pd.DataFrame({
            'PC1': X_pca[:, 0],
            'PC2': X_pca[:, 1],
            'PC3': X_pca[:, 2],
            'Outcome': [str(label) for label in y]
        })

        # Create 3D scatter plot
        fig = px.scatter_3d(
            pca_df,
            x='PC1',
            y='PC2',
            z='PC3',
            color='Outcome',
            title=f'PCA 3D Projection - {outcome_name}<br>Explained Variance: PC1={pca.explained_variance_ratio_[0]:.2%}, PC2={pca.explained_variance_ratio_[1]:.2%}, PC3={pca.explained_variance_ratio_[2]:.2%}',
            labels={'PC1': f'PC1 ({pca.explained_variance_ratio_[0]:.2%})',
                   'PC2': f'PC2 ({pca.explained_variance_ratio_[1]:.2%})',
                   'PC3': f'PC3 ({pca.explained_variance_ratio_[2]:.2%})'}
        )
        fig.update_traces(marker=dict(size=5, opacity=0.7))
        fig.write_html(str(save_dir / filename))

    def save_classifier_input_data(self, X_df, y, outcome_name, subject_ids=None, recording_dates=None):
        """Save the cleaned data that goes into the classifier."""
        save_dir = self.output_dir / "data_analysis"
        save_dir.mkdir(parents=True, exist_ok=True)

        # Create a combined DataFrame
        data_df = X_df.copy()
        data_df['Outcome'] = y

        if subject_ids is not None:
            data_df.insert(0, 'SubjectID', subject_ids)
        if recording_dates is not None:
            data_df.insert(1, 'RecordingDate', recording_dates)

        # Save to CSV
        csv_path = save_dir / f"classifier_input_data_{outcome_name}.csv"
        data_df.to_csv(csv_path, index=False)
        print(f"    💾 Saved classifier input data to: {csv_path.name}")

        # Create summary statistics
        summary_stats = X_df.describe().T
        summary_stats['feature'] = summary_stats.index
        summary_stats = summary_stats[['feature', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']]

        summary_path = save_dir / f"feature_summary_stats_{outcome_name}.csv"
        summary_stats.to_csv(summary_path, index=False)
        print(f"    📊 Saved feature summary statistics to: {summary_path.name}")
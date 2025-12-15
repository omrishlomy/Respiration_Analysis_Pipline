import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import roc_curve, auc, confusion_matrix


class InteractivePlotter:
    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def plot_signal_traces(self, raw_data, clean_data, fs, subject_id, breath_peaks=None):
        """
        Plot raw, cleaned, and breathmetrics signal traces.

        Args:
            raw_data: Raw signal array
            clean_data: Cleaned signal array
            fs: Sampling rate in Hz
            subject_id: Subject identifier
            breath_peaks: Optional list of BreathPeak objects from breathmetrics algorithm
        """
        save_dir = self.output_dir / "signals"
        save_dir.mkdir(parents=True, exist_ok=True)
        time = np.arange(len(raw_data)) / fs

        # Create 3 subplots: Raw, Cleaned, Breathmetrics
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.08,
            subplot_titles=(f"Raw: {subject_id}", "Cleaned", "Peak Detection")
        )

        fig.add_trace(go.Scatter(x=time, y=raw_data, name="Raw", line=dict(color='gray', width=1)), row=1, col=1)
        fig.add_trace(go.Scatter(x=time, y=clean_data, name="Cleaned", line=dict(color='#00CC96', width=1)), row=2, col=1)

        # Add breathmetrics overlay if provided (on third subplot)
        if breath_peaks is not None and len(breath_peaks) > 0:
            # Add cleaned signal to third subplot as base
            fig.add_trace(go.Scatter(
                x=time, y=clean_data,
                name="Signal",
                line=dict(color='#00CC96', width=1)
            ), row=3, col=1)

            # Plot inhale peaks (red circles)
            inhale_peaks = [b for b in breath_peaks if b.PeakValue > 0]
            if inhale_peaks:
                inhale_times = [b.PeakLocation / fs for b in inhale_peaks]
                inhale_values = [b.PeakValue for b in inhale_peaks]
                fig.add_trace(go.Scatter(
                    x=inhale_times,
                    y=inhale_values,
                    mode='markers',
                    name='Inhale Peaks',
                    marker=dict(size=8, color='red', symbol='circle'),
                    showlegend=True
                ), row=3, col=1)

            # Plot exhale peaks (green circles)
            exhale_peaks = [b for b in breath_peaks if b.PeakValue < 0]
            if exhale_peaks:
                exhale_times = [b.PeakLocation / fs for b in exhale_peaks]
                exhale_values = [b.PeakValue for b in exhale_peaks]
                fig.add_trace(go.Scatter(
                    x=exhale_times,
                    y=exhale_values,
                    mode='markers',
                    name='Exhale Peaks',
                    marker=dict(size=8, color='green', symbol='circle'),
                    showlegend=True
                ), row=3, col=1)

        fig.update_layout(height=800, showlegend=True, title_text=f"Signal: {subject_id}")
        fig.update_xaxes(title_text="Time (s)", row=3, col=1)
        fig.update_yaxes(title_text="Amplitude", row=1, col=1)
        fig.update_yaxes(title_text="Amplitude", row=2, col=1)
        fig.update_yaxes(title_text="Respiratory Amplitude", row=3, col=1)

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

    def plot_pca_scatter(self, pca_df, outcome_name, explained_variance):
        """
        Plot PCA scatter (PC1 vs PC2) colored by class label.

        Args:
            pca_df: DataFrame with PC1, PC2, and label columns
            outcome_name: Name of outcome for title
            explained_variance: Array of explained variance ratios for PC1, PC2
        """
        import plotly.express as px

        fig = px.scatter(
            pca_df, x='PC1', y='PC2', color='label',
            hover_data=['subject_id'] if 'subject_id' in pca_df.columns else None,
            title=f"PCA Projection: {outcome_name}",
            labels={
                'PC1': f'PC1 ({explained_variance[0]*100:.1f}% variance)',
                'PC2': f'PC2 ({explained_variance[1]*100:.1f}% variance)'
            },
            color_continuous_scale='Viridis' if pca_df['label'].dtype in ['float64', 'float32'] else None
        )
        fig.update_layout(width=800, height=600)

        filename = f"pca_scatter_{outcome_name}.html"
        fig.write_html(str(self.output_dir / filename))

    def plot_pca_variance(self, variance_df):
        """
        Plot scree plot showing variance explained by each component.

        Args:
            variance_df: DataFrame with component, explained_variance, cumulative_variance columns
        """
        from plotly.subplots import make_subplots

        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=("Variance Explained per Component", "Cumulative Variance")
        )

        # Individual variance (scree plot)
        fig.add_trace(go.Bar(
            x=variance_df['component'],
            y=variance_df['explained_variance'] * 100,
            name='Individual',
            marker_color='steelblue'
        ), row=1, col=1)

        # Cumulative variance
        fig.add_trace(go.Scatter(
            x=variance_df['component'],
            y=variance_df['cumulative_variance'] * 100,
            mode='lines+markers',
            name='Cumulative',
            line=dict(color='firebrick', width=2),
            marker=dict(size=6)
        ), row=1, col=2)

        # Add 95% threshold line
        fig.add_hline(y=95, line_dash="dash", line_color="gray",
                      annotation_text="95% threshold", row=1, col=2)

        fig.update_xaxes(title_text="Component", row=1, col=1)
        fig.update_xaxes(title_text="Component", row=1, col=2)
        fig.update_yaxes(title_text="Variance Explained (%)", row=1, col=1)
        fig.update_yaxes(title_text="Cumulative Variance (%)", row=1, col=2)

        fig.update_layout(height=400, width=1000, showlegend=False)
        fig.write_html(str(self.output_dir / "pca_variance_explained.html"))

    def plot_pca_loadings(self, loadings_df, top_pc1_features, top_pc2_features):
        """
        Plot feature loadings (contributions) for PC1 and PC2.

        Args:
            loadings_df: DataFrame with features as rows, PC1/PC2 as columns
            top_pc1_features: List of top feature names for PC1
            top_pc2_features: List of top feature names for PC2
        """
        from plotly.subplots import make_subplots

        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=("Top Features for PC1", "Top Features for PC2")
        )

        # PC1 loadings
        pc1_data = loadings_df.loc[top_pc1_features, 'PC1'].sort_values()
        fig.add_trace(go.Bar(
            y=pc1_data.index,
            x=pc1_data.values,
            orientation='h',
            name='PC1',
            marker_color='steelblue'
        ), row=1, col=1)

        # PC2 loadings
        pc2_data = loadings_df.loc[top_pc2_features, 'PC2'].sort_values()
        fig.add_trace(go.Bar(
            y=pc2_data.index,
            x=pc2_data.values,
            orientation='h',
            name='PC2',
            marker_color='firebrick'
        ), row=1, col=2)

        fig.update_xaxes(title_text="Loading", row=1, col=1)
        fig.update_xaxes(title_text="Loading", row=1, col=2)
        fig.update_layout(height=600, width=1200, showlegend=False)

        fig.write_html(str(self.output_dir / "pca_feature_loadings.html"))
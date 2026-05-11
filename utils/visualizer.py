"""
Result visualisation helpers (Plotly charts).
"""

import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from typing import List, Dict


DARK_BG   = "#0d1117"
DARK_CARD = "#16213e"
ACCENT    = "#e94560"
TEXT      = "#a8b2d8"


class ResultVisualizer:
    """Generate Plotly figures for classification results."""

    def confidence_bar(self, results: List[Dict]) -> go.Figure:
        labels = [r["label"] for r in results]
        scores = [r["confidence"] for r in results]
        fig = go.Figure(go.Bar(
            x=scores, y=labels, orientation="h",
            marker=dict(color=scores,
                        colorscale=[[0, "#0f3460"], [0.5, ACCENT], [1, "#00b894"]]),
            text=[f"{s:.1%}" for s in scores], textposition="outside",
        ))
        fig.update_layout(
            title="Top Predictions",
            xaxis=dict(range=[0, 1], tickformat=".0%"),
            yaxis=dict(autorange="reversed"),
            plot_bgcolor=DARK_BG, paper_bgcolor=DARK_BG,
            font=dict(color=TEXT), height=250,
            margin=dict(l=10, r=10, t=40, b=10),
        )
        return fig

    def role_distribution_pie(self, role_counts) -> go.Figure:
        fig = px.pie(
            values=role_counts.values,
            names=role_counts.index,
            title="Predicted Role Distribution",
            color_discrete_sequence=px.colors.sequential.RdBu,
        )
        fig.update_layout(plot_bgcolor=DARK_BG, paper_bgcolor=DARK_BG,
                          font=dict(color=TEXT))
        return fig

    def tsne_scatter(self, df) -> go.Figure:
        fig = px.scatter(df, x="x", y="y", color="label",
                         title="t-SNE of BERT Resume Embeddings (768-dim → 2-dim)")
        fig.update_layout(plot_bgcolor=DARK_BG, paper_bgcolor=DARK_BG,
                          font=dict(color=TEXT))
        return fig

    def per_class_accuracy_bar(self, categories, acc_scores) -> go.Figure:
        fig = px.bar(x=categories, y=acc_scores,
                     title="Per-Class Classification Accuracy",
                     color=acc_scores, color_continuous_scale="RdYlGn",
                     labels={"x": "Job Role", "y": "Accuracy"})
        fig.update_layout(plot_bgcolor=DARK_BG, paper_bgcolor=DARK_BG,
                          font=dict(color=TEXT), xaxis_tickangle=-45,
                          showlegend=False)
        return fig

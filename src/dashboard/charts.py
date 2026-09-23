"""Plotly figures for the dashboard (hover, zoom and legend toggling come with Plotly).

Colors follow the entity, never its rank: each asset, strategy and predictor keeps
the same color on every page (the same palette as the Phase 2-7 report figures).
Charts render with Streamlit's theme, so light and dark mode both work.
"""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

ASSET_COLORS = {"VOO": "#2a78d6", "VTI": "#eb6834", "VT": "#1baf7a", "EWJ": "#eda100",
                "BND": "#e87ba4", "BIL": "#008300"}
SERIES_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
SEGMENT_COLORS = {"train": "#86b6ef", "validation": "#3987e5", "test": "#184f95"}
MUTED = "#898781"
DIVERGING = [[0.0, "#e34948"], [0.5, "#f0efec"], [1.0, "#2a78d6"]]


def _layout(fig: go.Figure, height: int, y_title: str = "", y_format: str | None = None,
            log: bool = False, x_title: str = "") -> go.Figure:
    fig.update_layout(
        height=height, margin=dict(l=8, r=8, t=36, b=8), hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="left", x=0, title=None),
    )
    fig.update_yaxes(title=y_title, tickformat=y_format, type="log" if log else "linear")
    fig.update_xaxes(title=x_title)
    return fig


def lines(frame: pd.DataFrame, colors: dict[str, str], y_title: str, y_format: str = ".1%",
          log: bool = False, height: int = 380, dashed: tuple[str, ...] = ()) -> go.Figure:
    """One line per column; the unified hover shows every series on the hovered date."""
    fig = go.Figure()
    for col in frame.columns:
        fig.add_trace(go.Scatter(
            x=frame.index, y=frame[col], mode="lines", name=str(col),
            line=dict(color=colors.get(col, MUTED), width=1.6, dash="dot" if col in dashed else None),
            hovertemplate=f"%{{y:{y_format}}}<extra>{col}</extra>",
        ))
    return _layout(fig, height, y_title, y_format, log)


def heatmap(matrix: pd.DataFrame, height: int = 380) -> go.Figure:
    """Correlation matrix on a red-gray-blue scale fixed to [-1, 1], with the value in every cell."""
    fig = go.Figure(go.Heatmap(
        z=matrix.to_numpy(), x=list(matrix.columns), y=list(matrix.index), zmin=-1, zmax=1,
        colorscale=DIVERGING, text=matrix.round(2).to_numpy(), texttemplate="%{text:.2f}",
        hovertemplate="%{y} × %{x}: %{z:.3f}<extra></extra>", xgap=2, ygap=2,
        colorbar=dict(title="相関", thickness=12),
    ))
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(height=height, margin=dict(l=8, r=8, t=16, b=8))
    return fig


def intervals(frame: pd.DataFrame, estimate: str, lower: str, upper: str, x_title: str,
              reference: float | None = 0.0, height: int | None = None) -> go.Figure:
    """Point estimate with its confidence interval, one row per index label."""
    labels = [str(i) for i in frame.index]
    fig = go.Figure(go.Scatter(
        x=frame[estimate], y=labels, mode="markers", marker=dict(color=SERIES_COLORS[0], size=10),
        error_x=dict(type="data", symmetric=False, array=frame[upper] - frame[estimate],
                     arrayminus=frame[estimate] - frame[lower], color=SERIES_COLORS[0], thickness=2, width=6),
        customdata=frame[[lower, upper]].to_numpy(),
        hovertemplate="%{y}: %{x:.3f}<br>区間 %{customdata[0]:.3f} 〜 %{customdata[1]:.3f}<extra></extra>",
    ))
    if reference is not None:
        fig.add_vline(x=reference, line=dict(color=MUTED, dash="dash", width=1))
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(height=height or 90 + 44 * len(labels), margin=dict(l=8, r=8, t=16, b=8),
                      xaxis_title=x_title, hovermode="closest", showlegend=False)
    return fig


def histogram(returns: pd.Series, markers: dict[str, float], color: str, height: int = 360) -> go.Figure:
    """Daily return distribution with vertical lines at the given returns (e.g. -VaR)."""
    fig = go.Figure(go.Histogram(x=returns, nbinsx=120, marker=dict(color=color, line=dict(width=0)),
                                 hovertemplate="%{x:.2%}: %{y} 日<extra></extra>", name="日次リターン"))
    dashes = ["dash", "dot", "dashdot", "longdash"]
    for (label, value), dash in zip(markers.items(), dashes):
        fig.add_vline(x=value, line=dict(color=MUTED, dash=dash, width=1.2),
                      annotation_text=label, annotation_position="top left", annotation_font_size=11)
    fig.update_layout(height=height, margin=dict(l=8, r=8, t=24, b=8), bargap=0.05, showlegend=False,
                      xaxis=dict(tickformat=".1%", title="日次リターン"), yaxis_title="日数")
    return fig


def horizontal_bars(frame: pd.DataFrame, value: str, label: str, error: str | None, x_title: str,
                    color: str = SERIES_COLORS[0]) -> go.Figure:
    """Ranked horizontal bars (largest at the top)."""
    fig = go.Figure(go.Bar(
        x=frame[value], y=frame[label], orientation="h", marker=dict(color=color),
        error_x=dict(type="data", array=frame[error], color=MUTED, thickness=1.2) if error else None,
        hovertemplate="%{y}: %{x:.4f}<extra></extra>",
    ))
    fig.add_vline(x=0, line=dict(color=MUTED, width=1))
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(height=90 + 26 * len(frame), margin=dict(l=8, r=8, t=16, b=8), xaxis_title=x_title,
                      hovermode="closest", showlegend=False)
    return fig


def bars(values: pd.Series, y_title: str, reference: float | None, color: str, y_format: str = ".2f",
         height: int = 300) -> go.Figure:
    """Vertical bars (e.g. one per year) around a reference line such as ROC-AUC 0.5."""
    base = reference or 0.0
    fig = go.Figure(go.Bar(x=[str(i) for i in values.index], y=values - base, base=base, customdata=values,
                           marker=dict(color=color),
                           hovertemplate=f"%{{x}}: %{{customdata:{y_format}}}<extra></extra>"))
    if reference is not None:
        fig.add_hline(y=reference, line=dict(color=MUTED, dash="dash", width=1))
    fig.update_layout(height=height, margin=dict(l=8, r=8, t=16, b=8), yaxis=dict(title=y_title, tickformat=y_format),
                      hovermode="closest", showlegend=False, bargap=0.3)
    return fig


def stacked_area(frame: pd.DataFrame, colors: dict[str, str], y_title: str, height: int = 320) -> go.Figure:
    fig = go.Figure()
    for col in frame.columns:
        fig.add_trace(go.Scatter(x=frame.index, y=frame[col], name=col, mode="lines", stackgroup="one",
                                 line=dict(width=0.5, color=colors.get(col, MUTED)),
                                 hovertemplate=f"%{{y:.1%}}<extra>{col}</extra>"))
    return _layout(fig, height, y_title, ".0%")


def probability_by_segment(probability: pd.Series, segments: dict[str, tuple], threshold: float,
                           color: str, height: int = 340) -> go.Figure:
    """Predicted probability over time on shaded train / validation / test periods."""
    fig = go.Figure()
    for name, (start, end) in segments.items():
        fig.add_vrect(x0=start, x1=end, fillcolor=SEGMENT_COLORS[name], opacity=0.18, line_width=0,
                      annotation_text=name, annotation_position="top left", annotation_font_size=11)
    fig.add_trace(go.Scatter(x=probability.index, y=probability, mode="lines", name="予測確率(上昇)",
                             line=dict(color=color, width=1.2), hovertemplate="%{y:.3f}<extra></extra>"))
    fig.add_hline(y=threshold, line=dict(color=MUTED, dash="dash", width=1),
                  annotation_text=f"しきい値 {threshold:.2f}", annotation_position="bottom right")
    return _layout(fig, height, "予測確率", ".2f")

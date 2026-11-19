from __future__ import annotations



import json

from pathlib import Path



import numpy as np

import pandas as pd

import plotly.graph_objects as go

from plotly.subplots import make_subplots



from dash import Dash, dcc, html, Input, Output





# ============================================================

# RAIBS DYNAMIC PORTFOLIO RISK & OPTIMIZATION ENGINE

# Dashboard / Quantitative Portfolio Risk Cockpit

# ============================================================



ROOT = Path(__file__).resolve().parents[1]

SNAPSHOT = ROOT / "outputs" / "latest_snapshot.json"



app = Dash(__name__, assets_folder="assets")

app.title = "RAIBS Quantitative Portfolio Risk Platform"





# ============================================================

# THEME

# ============================================================



BG = "#0B0F14"

PANEL = "#151B23"

PANEL_2 = "#11161D"

BORDER = "#334155"

TEXT = "#E5E7EB"

MUTED = "#9CA3AF"

ACCENT = "#C5A253"

ACCENT_2 = "#D4AF37"





PLOT_LAYOUT = dict(

    template="plotly_dark",

    paper_bgcolor=BG,

    plot_bgcolor=BG,

    font=dict(color=TEXT, family="Arial"),

    margin=dict(l=50, r=30, t=65, b=45),

    hovermode="closest",

)





# ============================================================

# DATA

# ============================================================



def load_snapshot():

    if not SNAPSHOT.exists():

        return None



    try:

        return json.loads(

            SNAPSHOT.read_text(encoding="utf-8")

        )

    except (json.JSONDecodeError, OSError):

        return None





def portfolio_dataframe(snapshot):

    return pd.DataFrame(snapshot["portfolios"])





def optimization_dataframe(snapshot):

    opt = snapshot.get("optimization", {})



    all_assets = set()



    for model in opt.values():

        if isinstance(model, dict):

            all_assets.update(model.keys())



    assets = sorted(all_assets)



    data = {"asset": assets}



    for model_name, model in opt.items():

        if isinstance(model, dict):

            data[model_name] = [

                float(model.get(asset, 0.0))

                for asset in assets

            ]



    return pd.DataFrame(data)





# ============================================================

# HELPERS

# ============================================================



def card(title, value, subtitle=None):

    children = [

        html.Div(

            title,

            style={

                "fontSize": "11px",

                "letterSpacing": "1.2px",

                "color": MUTED,

                "fontWeight": "600",

            },

        ),

        html.Div(

            value,

            style={

                "fontSize": "25px",

                "fontWeight": "700",

                "marginTop": "7px",

                "color": TEXT,

            },

        ),

    ]



    if subtitle:

        children.append(

            html.Div(

                subtitle,

                style={

                    "fontSize": "11px",

                    "color": MUTED,

                    "marginTop": "5px",

                },

            )

        )



    return html.Div(

        children,

        style={

            "padding": "16px 18px",

            "backgroundColor": PANEL,

            "border": f"1px solid {BORDER}",

            "borderRadius": "6px",

            "minWidth": "170px",

            "flex": "1",

        },

    )





def section_title(title, subtitle=None):

    children = [

        html.H3(

            title,

            style={

                "marginBottom": "4px",

                "letterSpacing": "1px",

                "fontSize": "17px",

            },

        )

    ]



    if subtitle:

        children.append(

            html.Div(

                subtitle,

                style={

                    "color": MUTED,

                    "fontSize": "12px",

                    "marginBottom": "14px",

                },

            )

        )



    return html.Div(children)





def base_figure(fig, title=None):

    fig.update_layout(**PLOT_LAYOUT)



    if title:

        fig.update_layout(title=title)



    return fig





# ============================================================

# EXECUTIVE OVERVIEW

# ============================================================



def build_overview_figure(portfolios):

    ordered = portfolios.sort_values(

        "annualized_volatility"

    )



    fig = go.Figure()



    fig.add_trace(

        go.Bar(

            x=ordered["portfolio_id"],

            y=ordered["annualized_volatility"] * 100,

            customdata=ordered[

                ["portfolio_name"]

            ],

            hovertemplate=(

                "<b>%{x}</b><br>"

                "Portfolio: %{customdata[0]}<br>"

                "Annual Volatility: %{y:.2f}%"

                "<extra></extra>"

            ),

            name="Annualized Volatility",

        )

    )



    fig.update_xaxes(title="Portfolio")

    fig.update_yaxes(title="Annualized Volatility (%)")



    return base_figure(

        fig,

        "31-Portfolio Risk Ranking",

    )





def build_model_tournament(portfolios):

    models = [

        ("vol_garch_ewma", "GARCH-EWMA"),

        ("vol_rolling", "Rolling"),

        ("vol_ewma", "EWMA"),

        ("vol_ledoit_wolf", "Ledoit-Wolf"),

    ]



    fig = go.Figure()



    for column, label in models:

        fig.add_trace(

            go.Scatter(

                x=portfolios["portfolio_id"],

                y=portfolios[column] * 100,

                mode="lines+markers",

                name=label,

            )

        )



    fig.update_xaxes(title="Portfolio")

    fig.update_yaxes(

        title="Daily Volatility (%)"

    )



    return base_figure(

        fig,

        "Model Tournament — Daily Volatility",

    )





# ============================================================

# PORTFOLIO DEEP DIVE

# ============================================================



def build_selected_model_chart(row):

    models = {

        "GARCH-EWMA": row["vol_garch_ewma"],

        "Rolling": row["vol_rolling"],

        "EWMA": row["vol_ewma"],

        "Ledoit-Wolf": row["vol_ledoit_wolf"],

    }



    fig = go.Figure()



    fig.add_trace(

        go.Bar(

            x=list(models.keys()),

            y=[v * 100 for v in models.values()],

            text=[

                f"{v:.3f}%"

                for v in models.values()

            ],

            textposition="outside",

            name="Daily Volatility",

        )

    )



    fig.update_yaxes(

        title="Daily Volatility (%)"

    )



    return base_figure(

        fig,

        "Selected Portfolio — Risk Model Comparison",

    )





def build_model_dispersion_figure(portfolios):

    ordered = portfolios.sort_values(

        "dispersion",

        ascending=False,

    )



    fig = go.Figure()



    fig.add_trace(

        go.Bar(

            x=ordered["portfolio_id"],

            y=ordered["dispersion"] * 100,

            customdata=ordered[

                ["portfolio_name"]

            ],

            hovertemplate=(

                "<b>%{x}</b><br>"

                "Portfolio: %{customdata[0]}<br>"

                "Dispersion: %{y:.4f}%"

                "<extra></extra>"

            ),

            name="Model Dispersion",

        )

    )



    fig.update_yaxes(

        title="Model Dispersion (%)"

    )



    return base_figure(

        fig,

        "Model Disagreement / Estimation Dispersion",

    )





# ============================================================

# CORRELATION

# ============================================================



def build_correlation_figure(snapshot):

    correlation = snapshot.get(

        "correlation",

        {},

    )



    if not correlation:

        return base_figure(

            go.Figure(),

            "Correlation Matrix — Not Available",

        )



    assets = list(correlation.keys())



    matrix = np.array(

        [

            [

                correlation[a].get(b, np.nan)

                for b in assets

            ]

            for a in assets

        ]

    )



    fig = go.Figure(

        data=go.Heatmap(

            z=matrix,

            x=assets,

            y=assets,

            zmin=-1,

            zmax=1,

            colorscale="RdBu",

            reversescale=True,

            text=np.round(matrix, 2),

            texttemplate="%{text}",

            hovertemplate=(

                "%{y} ↔ %{x}"

                "<br>Correlation: %{z:.3f}"

                "<extra></extra>"

            ),

            colorbar=dict(

                title="ρ"

            ),

        )

    )



    return base_figure(

        fig,

        "Dynamic Correlation Matrix",

    )





def build_correlation_pairs(snapshot):

    correlation = snapshot.get(

        "correlation",

        {},

    )



    pairs = []



    assets = list(correlation.keys())



    for i, a in enumerate(assets):

        for b in assets[i + 1:]:

            value = correlation[a].get(

                b,

                np.nan,

            )



            if pd.notna(value):

                pairs.append(

                    {

                        "Asset A": a,

                        "Asset B": b,

                        "Correlation": value,

                        "Absolute Correlation": abs(value),

                    }

                )



    if not pairs:

        return pd.DataFrame(

            columns=[

                "Asset A",

                "Asset B",

                "Correlation",

            ]

        )



    return (

        pd.DataFrame(pairs)

        .sort_values(

            "Absolute Correlation",

            ascending=False,

        )

        .head(15)

    )





# ============================================================

# OPTIMIZATION

# ============================================================



def build_optimization_figure(snapshot):

    opt_df = optimization_dataframe(

        snapshot

    )



    if opt_df.empty:

        return base_figure(

            go.Figure(),

            "Optimization — Not Available",

        )



    fig = go.Figure()



    mapping = [

        ("minimum_variance", "Minimum Variance"),

        ("maximum_sharpe", "Maximum Sharpe"),

        ("risk_parity", "Risk Parity"),

    ]



    for column, label in mapping:

        if column not in opt_df:

            continue



        fig.add_trace(

            go.Bar(

                x=opt_df["asset"],

                y=opt_df[column] * 100,

                name=label,

            )

        )



    fig.update_layout(

        barmode="group"

    )



    fig.update_yaxes(

        title="Portfolio Weight (%)"

    )



    return base_figure(

        fig,

        "Optimization Allocation Comparison",

    )





def build_optimization_heatmap(snapshot):

    opt_df = optimization_dataframe(

        snapshot

    )



    if opt_df.empty:

        return base_figure(

            go.Figure(),

            "Optimization Matrix — Not Available",

        )



    matrix = opt_df.set_index(

        "asset"

    )



    matrix = matrix.rename(

        columns={

            "minimum_variance": "Minimum Variance",

            "maximum_sharpe": "Maximum Sharpe",

            "risk_parity": "Risk Parity",

        }

    )



    fig = go.Figure(

        data=go.Heatmap(

            z=matrix.values.T * 100,

            x=matrix.index,

            y=matrix.columns,

            text=np.round(

                matrix.values.T * 100,

                1,

            ),

            texttemplate="%{text}%",

            colorscale="Viridis",

            hovertemplate=(

                "Asset: %{x}<br>"

                "Strategy: %{y}<br>"

                "Weight: %{z:.2f}%"

                "<extra></extra>"

            ),

        )

    )



    return base_figure(

        fig,

        "Optimization Weight Matrix",

    )





# ============================================================



# ============================================================
# EFFICIENCY ANALYSIS
# ============================================================

def build_efficient_frontier_figure(snapshot, selected=None):
    """Build the interactive annualized risk-return efficient frontier."""
    frontier = pd.DataFrame(snapshot.get("efficient_frontier", []))
    portfolios = pd.DataFrame(snapshot.get("portfolios", []))
    metrics = snapshot.get("optimization_metrics", {})

    fig = go.Figure()

    if not frontier.empty:
        fig.add_trace(
            go.Scatter(
                x=frontier["volatility"] * 100,
                y=frontier["expected_return"] * 100,
                mode="lines",
                line=dict(color=ACCENT_2, width=3),
                name="Efficient Frontier",
                hovertemplate=(
                    "Annual Volatility: %{x:.2f}%<br>"
                    "Expected Return: %{y:.2f}%<extra></extra>"
                ),
            )
        )

    if not portfolios.empty and "expected_return" in portfolios.columns:
        fig.add_trace(
            go.Scatter(
                x=portfolios["annualized_volatility"] * 100,
                y=portfolios["expected_return"] * 100,
                mode="markers",
                marker=dict(size=8, color="#64748B", opacity=0.68),
                customdata=portfolios[["portfolio_id", "portfolio_name"]],
                name="Existing Portfolios",
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>"
                    "Portfolio: %{customdata[1]}<br>"
                    "Annual Volatility: %{x:.2f}%<br>"
                    "Expected Return: %{y:.2f}%<extra></extra>"
                ),
            )
        )

        if selected in portfolios["portfolio_id"].values:
            selected_row = portfolios[portfolios["portfolio_id"] == selected].iloc[0]
            fig.add_trace(
                go.Scatter(
                    x=[selected_row["annualized_volatility"] * 100],
                    y=[selected_row["expected_return"] * 100],
                    mode="markers",
                    marker=dict(size=13, color=ACCENT, line=dict(color=TEXT, width=1.5)),
                    name="Selected Portfolio",
                    hovertemplate=(
                        f"<b>{selected_row['portfolio_id']}</b><br>"
                        f"{selected_row['portfolio_name']}<br>"
                        "Annual Volatility: %{x:.2f}%<br>"
                        "Expected Return: %{y:.2f}%<extra></extra>"
                    ),
                )
            )

    labels = {
        "minimum_variance": "Minimum Variance",
        "maximum_sharpe": "Maximum Sharpe",
        "risk_parity": "Risk Parity",
    }
    marker_symbols = {
        "minimum_variance": "diamond",
        "maximum_sharpe": "star",
        "risk_parity": "triangle-up",
    }
    marker_colors = {
        "minimum_variance": "#A8B1BD",
        "maximum_sharpe": ACCENT_2,
        "risk_parity": "#7F8FA6",
    }

    for key, label in labels.items():
        metric = metrics.get(key)
        if not metric:
            continue
        fig.add_trace(
            go.Scatter(
                x=[float(metric["annualized_volatility"]) * 100],
                y=[float(metric["expected_return"]) * 100],
                mode="markers",
                marker=dict(
                    size=15,
                    symbol=marker_symbols[key],
                    color=marker_colors[key],
                    line=dict(color=TEXT, width=1),
                ),
                name=label,
                hovertemplate=(
                    f"<b>{label}</b><br>"
                    "Annual Volatility: %{x:.2f}%<br>"
                    "Expected Return: %{y:.2f}%<extra></extra>"
                ),
            )
        )

    fig.update_xaxes(title="Annualized Volatility (%)", zeroline=False)
    fig.update_yaxes(title="Expected Annual Return (%)", zeroline=False)
    return base_figure(fig, "Efficient Frontier — Risk / Return Analysis")


# PORTFOLIO TABLE

# ============================================================



def portfolio_table(portfolios):

    columns = [

        ("portfolio_id", "ID"),

        ("portfolio_name", "Portfolio"),

        ("annualized_volatility", "Annual Vol"),

        ("daily_volatility", "Daily Vol"),

        ("vol_garch_ewma", "GARCH-EWMA"),

        ("vol_rolling", "Rolling"),

        ("vol_ewma", "EWMA"),

        ("vol_ledoit_wolf", "Ledoit-Wolf"),

        ("dispersion", "Dispersion"),

        ("cv", "CV"),

    ]



    header = html.Tr(

        [

            html.Th(

                label,

                style={

                    "padding": "10px",

                    "borderBottom": f"1px solid {BORDER}",

                    "color": MUTED,

                    "fontSize": "11px",

                    "textAlign": "left",

                },

            )

            for _, label in columns

        ]

    )



    rows = []



    for _, row in portfolios.iterrows():

        cells = []



        for column, _ in columns:

            value = row[column]



            if column == "annualized_volatility":

                display = f"{value:.2%}"



            elif column in {

                "daily_volatility",

                "vol_garch_ewma",

                "vol_rolling",

                "vol_ewma",

                "vol_ledoit_wolf",

                "dispersion",

            }:

                display = f"{value:.4%}"



            elif column == "cv":

                display = f"{value:.2%}"



            else:

                display = str(value)



            cells.append(

                html.Td(

                    display,

                    style={

                        "padding": "9px 10px",

                        "borderBottom": f"1px solid {BORDER}",

                        "fontSize": "12px",

                    },

                )

            )



        rows.append(

            html.Tr(cells)

        )



    return html.Div(

        html.Table(

            [html.Thead(header), html.Tbody(rows)],

            style={

                "width": "100%",

                "borderCollapse": "collapse",

            },

        ),

        style={

            "overflowX": "auto",

            "backgroundColor": PANEL,

            "border": f"1px solid {BORDER}",

            "borderRadius": "6px",

        },

    )





# ============================================================

# LAYOUT

# ============================================================



app.layout = html.Div(

    style={

        "backgroundColor": BG,

        "color": TEXT,

        "minHeight": "100vh",

        "fontFamily": "Arial",

    },

    children=[



        # ----------------------------------------------------

        # HEADER

        # ----------------------------------------------------



        html.Div(

            [

                # --------------------------------------------

                # TITLE + SMALL RAIBS LOGO

                # --------------------------------------------



                html.Div(

                    [

                        html.H1(

                            "RAIBS LLC - QUANTITATIVE PORTFOLIO RISK PLATFORM",

                            style={

                                "letterSpacing": "2px",

                                "fontSize": "25px",

                                "margin": "0",

                                "fontWeight": "700",

                            },

                        ),



                        html.Img(

                            src="/assets/raibs-llc_logo.png",

                            style={

                                "height": "27px",

                                "width": "auto",

                                "marginLeft": "12px",

                                "opacity": "0.92",

                                "filter":

                                    "brightness(1.15) "

                                    "drop-shadow(0px 0px 5px rgba(212,175,55,0.35))",

                            },

                        ),

                    ],

                    style={

                        "display": "flex",

                        "alignItems": "center",

                    },

                ),



                # --------------------------------------------

                # SUBTITLE

                # --------------------------------------------



                html.Div(

                    "Dynamic Covariance • GARCH Volatility • Risk Attribution • MPT • Risk Parity",

                    style={

                        "color": MUTED,

                        "fontSize": "13px",

                        "marginTop": "6px",

                    },

                ),



                # --------------------------------------------

                # TIMESTAMP

                # --------------------------------------------



                html.Div(

                    id="timestamp",

                    style={

                        "color": ACCENT,

                        "fontSize": "11px",

                        "marginTop": "8px",

                    },

                ),

            ],

            style={

                "padding": "24px 28px 18px",

                "borderBottom": f"1px solid {BORDER}",

            },

        ),



        # ----------------------------------------------------

        # CONTROLS

        # ----------------------------------------------------



        html.Div(

            [

                html.Div(

                    [

                        html.Label(

                            "PORTFOLIO",

                            style={

                                "fontSize": "11px",

                                "color": MUTED,

                            },

                        ),



                        dcc.Dropdown(

                            id="portfolio",

                            style={

                                "color": "#111827",

                                "width": "430px",

                            },

                        ),

                    ]

                ),



                html.Div(

                    [

                        html.Label(

                            "REFRESH",

                            style={

                                "fontSize": "11px",

                                "color": MUTED,

                            },

                        ),



                        html.Div(

                            "30 seconds",

                            style={

                                "fontSize": "13px",

                                "paddingTop": "8px",

                            },

                        ),

                    ]

                ),

            ],

            style={

                "display": "flex",

                "gap": "28px",

                "alignItems": "end",

                "padding": "18px 28px",

            },

        ),



        # ----------------------------------------------------

        html.Div(
            html.Button(
                "EFFICIENCY ANALYSIS",
                id="efficiency-button",
                n_clicks=0,
                style={
                    "backgroundColor": "transparent",
                    "color": ACCENT,
                    "border": f"1px solid {ACCENT}",
                    "borderRadius": "4px",
                    "padding": "8px 14px",
                    "fontSize": "11px",
                    "fontWeight": "700",
                    "letterSpacing": "1px",
                    "cursor": "pointer",
                },
            ),
            style={"padding": "0 28px 18px"},
        ),

        html.Div(
            [
                html.Div(
                    [
                        html.Div(
                            [
                                html.Div(
                                    "EFFICIENCY ANALYSIS",
                                    style={"fontSize": "15px", "fontWeight": "700", "letterSpacing": "1px"},
                                ),
                                html.Button(
                                    "×",
                                    id="efficiency-close",
                                    n_clicks=0,
                                    style={
                                        "background": "transparent",
                                        "border": "none",
                                        "color": MUTED,
                                        "fontSize": "25px",
                                        "cursor": "pointer",
                                        "lineHeight": "1",
                                    },
                                ),
                            ],
                            style={
                                "display": "flex",
                                "justifyContent": "space-between",
                                "alignItems": "center",
                                "borderBottom": f"1px solid {BORDER}",
                                "paddingBottom": "12px",
                                "marginBottom": "8px",
                            },
                        ),
                        html.Div(
                            "Annualized historical mean return versus annualized portfolio risk. "
                            "The frontier uses the same return and covariance assumptions as the optimization engine.",
                            style={"color": MUTED, "fontSize": "12px", "lineHeight": "1.5", "marginBottom": "4px"},
                        ),
                        dcc.Graph(id="efficient-frontier", style={"height": "600px"}),
                    ],
                    style={
                        "backgroundColor": PANEL,
                        "border": f"1px solid {BORDER}",
                        "borderRadius": "6px",
                        "padding": "18px",
                        "width": "min(1100px, 92vw)",
                        "maxHeight": "88vh",
                        "overflowY": "auto",
                        "boxShadow": "0 18px 50px rgba(0,0,0,0.55)",
                    },
                )
            ],
            id="efficiency-modal",
            style={
                "display": "none",
                "position": "fixed",
                "inset": "0",
                "zIndex": "9999",
                "backgroundColor": "rgba(3,7,12,0.78)",
                "alignItems": "center",
                "justifyContent": "center",
                "padding": "30px",
            },
        ),

        # KPI STRIP

        # ----------------------------------------------------



        html.Div(

            id="kpis",

            style={

                "display": "flex",

                "gap": "12px",

                "padding": "0 28px 22px",

                "flexWrap": "wrap",

            },

        ),



        # ----------------------------------------------------

        # TABS

        # ----------------------------------------------------



        dcc.Tabs(

            id="tabs",

            value="overview",

            colors={

                "border": BORDER,

                "primary": ACCENT,

                "background": PANEL,

            },



            children=[



                # ==================================================

                # OVERVIEW

                # ==================================================



                dcc.Tab(

                    label="EXECUTIVE OVERVIEW",

                    value="overview",

                    style={

                        "padding": "12px",

                    },

                    selected_style={

                        "padding": "12px",

                        "fontWeight": "700",

                    },



                    children=[

                        html.Div(

                            [

                                dcc.Graph(

                                    id="portfolio-ranking"

                                ),



                                dcc.Graph(

                                    id="model-tournament"

                                ),



                                dcc.Graph(

                                    id="model-dispersion"

                                ),



                                html.H3(

                                    "31-Portfolio Risk Matrix",

                                    style={

                                        "marginTop": "25px",

                                        "fontSize": "17px",

                                    },

                                ),



                                html.Div(

                                    id="portfolio-table"

                                ),

                            ],

                            style={

                                "padding": "0 28px 40px"

                            },

                        )

                    ],

                ),



                # ==================================================

                # PORTFOLIO DEEP DIVE

                # ==================================================



                dcc.Tab(

                    label="PORTFOLIO DEEP DIVE",

                    value="deepdive",

                    style={

                        "padding": "12px",

                    },

                    selected_style={

                        "padding": "12px",

                        "fontWeight": "700",

                    },



                    children=[

                        html.Div(

                            [

                                dcc.Graph(

                                    id="selected-models"

                                ),



                                html.Div(

                                    [

                                        html.H3(

                                            "Selected Portfolio Diagnostics",

                                            style={

                                                "fontSize": "17px"

                                            },

                                        ),



                                        html.Div(

                                            id="selected-diagnostics"

                                        ),

                                    ],

                                    style={

                                        "backgroundColor": PANEL,

                                        "border": f"1px solid {BORDER}",

                                        "borderRadius": "6px",

                                        "padding": "18px",

                                    },

                                ),

                            ],

                            style={

                                "padding": "0 28px 40px"

                            },

                        )

                    ],

                ),



                # ==================================================

                # CORRELATION

                # ==================================================



                dcc.Tab(

                    label="CORRELATION & DIVERSIFICATION",

                    value="correlation",

                    style={

                        "padding": "12px",

                    },

                    selected_style={

                        "padding": "12px",

                        "fontWeight": "700",

                    },



                    children=[

                        html.Div(

                            [

                                dcc.Graph(

                                    id="correlation-heatmap"

                                ),



                                html.H3(

                                    "Strongest Pairwise Relationships",

                                    style={

                                        "fontSize": "17px"

                                    },

                                ),



                                html.Div(

                                    id="correlation-table"

                                ),

                            ],

                            style={

                                "padding": "0 28px 40px"

                            },

                        )

                    ],

                ),



                # ==================================================

                # OPTIMIZATION

                # ==================================================



                dcc.Tab(

                    label="OPTIMIZATION LAB",

                    value="optimization",

                    style={

                        "padding": "12px",

                    },

                    selected_style={

                        "padding": "12px",

                        "fontWeight": "700",

                    },



                    children=[

                        html.Div(

                            [

                                dcc.Graph(

                                    id="optimization-chart"

                                ),



                                dcc.Graph(

                                    id="optimization-heatmap"

                                ),



                                html.Div(

                                    [

                                        html.H3(

                                            "Optimization Interpretation",

                                            style={

                                                "fontSize": "17px"

                                            },

                                        ),



                                        html.P(

                                            "Minimum Variance minimizes portfolio variance. "

                                            "Maximum Sharpe maximizes the estimated risk-adjusted return. "

                                            "Risk Parity seeks a more balanced distribution of portfolio risk across assets.",

                                            style={

                                                "color": MUTED,

                                                "lineHeight": "1.6",

                                            },

                                        ),



                                        html.P(

                                            "Optimization outputs should be interpreted together with "

                                            "portfolio constraints, expected-return assumptions, liquidity, "

                                            "turnover and concentration controls.",

                                            style={

                                                "color": MUTED,

                                                "lineHeight": "1.6",

                                            },

                                        ),

                                    ],

                                    style={

                                        "backgroundColor": PANEL,

                                        "border": f"1px solid {BORDER}",

                                        "borderRadius": "6px",

                                        "padding": "18px",

                                    },

                                ),

                            ],

                            style={

                                "padding": "0 28px 40px"

                            },

                        )

                    ],

                ),



                # ==================================================

                # MODEL RISK

                # ==================================================



                dcc.Tab(

                    label="MODEL RISK",

                    value="modelrisk",

                    style={

                        "padding": "12px",

                    },

                    selected_style={

                        "padding": "12px",

                        "fontWeight": "700",

                    },



                    children=[

                        html.Div(

                            [

                                html.Div(

                                    [

                                        html.H3(

                                            "Covariance Model Framework",

                                            style={

                                                "fontSize": "18px"

                                            },

                                        ),



                                        html.P(

                                            "The primary portfolio-risk estimate uses "

                                            "GARCH conditional volatility combined with "

                                            "EWMA correlation. Rolling, EWMA and Ledoit-Wolf "

                                            "estimates are retained as verification models.",

                                            style={

                                                "color": MUTED,

                                                "lineHeight": "1.6",

                                            },

                                        ),

                                    ],

                                    style={

                                        "backgroundColor": PANEL,

                                        "border": f"1px solid {BORDER}",

                                        "borderRadius": "6px",

                                        "padding": "18px",

                                        "marginBottom": "15px",

                                    },

                                ),



                                dcc.Graph(

                                    id="model-risk-overview"

                                ),

                            ],

                            style={

                                "padding": "0 28px 40px"

                            },

                        )

                    ],

                ),

            ],

        ),



        # ----------------------------------------------------

        # REFRESH ENGINE

        # ----------------------------------------------------



        dcc.Interval(

            id="refresh",

            interval=30_000,

            n_intervals=0,

        ),

    ],

)





# ============================================================

# CALLBACK

# ============================================================



@app.callback(

    Output("portfolio", "options"),

    Output("portfolio", "value"),

    Output("kpis", "children"),

    Output("timestamp", "children"),

    Output("portfolio-ranking", "figure"),

    Output("model-tournament", "figure"),

    Output("model-dispersion", "figure"),

    Output("portfolio-table", "children"),

    Output("selected-models", "figure"),

    Output("selected-diagnostics", "children"),

    Output("correlation-heatmap", "figure"),

    Output("correlation-table", "children"),

    Output("optimization-chart", "figure"),

    Output("optimization-heatmap", "figure"),

    Output("model-risk-overview", "figure"),

    Input("refresh", "n_intervals"),

    Input("portfolio", "value"),

)

def refresh(_, selected):



    snapshot = load_snapshot()



    if not snapshot:



        empty = base_figure(

            go.Figure(),

            "No snapshot available — run: python run_once.py",

        )



        return (

            [],

            None,

            [],

            "No snapshot available",

            empty,

            empty,

            empty,

            html.Div(

                "No snapshot available.",

                style={"padding": "20px"},

            ),

            empty,

            html.Div(

                "No portfolio data available.",

                style={"padding": "20px"},

            ),

            empty,

            html.Div(

                "No correlation data available.",

                style={"padding": "20px"},

            ),

            empty,

            empty,

            empty,

        )



    portfolios = portfolio_dataframe(

        snapshot

    )



    options = [

        {

            "label": (

                f"{row.portfolio_id} — "

                f"{row.portfolio_name}"

            ),

            "value": row.portfolio_id,

        }

        for row in portfolios.itertuples()

    ]



    if (

        selected is None

        or selected not in portfolios[

            "portfolio_id"

        ].values

    ):

        selected = portfolios[

            "portfolio_id"

        ].iloc[0]



    row = portfolios[

        portfolios["portfolio_id"] == selected

    ].iloc[0]



    # --------------------------------------------------------

    # KPI CARDS

    # --------------------------------------------------------



    kpis = [

        card(

            "ANNUALIZED VOLATILITY",

            f"{row.annualized_volatility:.2%}",

            "252-day annualization",

        ),



        card(

            "DAILY VOLATILITY",

            f"{row.daily_volatility:.4%}",

            "Primary portfolio risk",

        ),



        card(

            "GARCH-EWMA",

            f"{row.vol_garch_ewma:.4%}",

            "Primary model",

        ),



        card(

            "MODEL DISPERSION",

            f"{row.dispersion:.4%}",

            "Max − Min model estimate",

        ),



        card(

            "MODEL CV",

            f"{row.cv:.2%}",

            "Relative dispersion",

        ),



        card(

            "OBSERVATIONS",

            f"{snapshot.get('observations', 'N/A'):,}",

            "Historical observations",

        ),

    ]



    timestamp = (

        f"DATA SNAPSHOT • "

        f"{snapshot.get('generated_at_utc', 'N/A')}"

    )



    # --------------------------------------------------------

    # SELECTED DIAGNOSTICS

    # --------------------------------------------------------



    diagnostics = html.Div(

        [

            html.Div(

                f"Portfolio: {row.portfolio_id} — {row.portfolio_name}",

                style={

                    "fontWeight": "700",

                    "fontSize": "16px",

                    "marginBottom": "12px",

                },

            ),



            html.Div(

                [

                    html.Div(

                        f"Risk contribution reconciliation: "

                        f"{row.risk_contribution_sum:.6%}",

                        style={

                            "padding": "8px 0",

                            "color": TEXT,

                        },

                    ),



                    html.Div(

                        f"Minimum model volatility: "

                        f"{row.min_vol:.4%}",

                        style={

                            "padding": "8px 0",

                            "color": TEXT,

                        },

                    ),



                    html.Div(

                        f"Maximum model volatility: "

                        f"{row.max_vol:.4%}",

                        style={

                            "padding": "8px 0",

                            "color": TEXT,

                        },

                    ),



                    html.Div(

                        f"Model range: "

                        f"{(row.max_vol - row.min_vol):.4%}",

                        style={

                            "padding": "8px 0",

                            "color": TEXT,

                        },

                    ),



                    html.Div(

                        "Risk contribution sum should reconcile "

                        "to portfolio volatility.",

                        style={

                            "padding": "8px 0",

                            "color": MUTED,

                            "fontSize": "12px",

                        },

                    ),

                ]

            ),

        ]

    )



    # --------------------------------------------------------

    # CORRELATION TABLE

    # --------------------------------------------------------



    pairs = build_correlation_pairs(

        snapshot

    )



    if pairs.empty:



        correlation_table = html.Div(

            "No correlation data available."

        )



    else:



        header = html.Tr(

            [

                html.Th(

                    name,

                    style={

                        "padding": "9px",

                        "color": MUTED,

                    },

                )

                for name in [

                    "Asset A",

                    "Asset B",

                    "Correlation",

                ]

            ]

        )



        body = []



        for _, pair in pairs.iterrows():



            body.append(

                html.Tr(

                    [

                        html.Td(

                            pair["Asset A"],

                            style={

                                "padding": "9px"

                            },

                        ),



                        html.Td(

                            pair["Asset B"],

                            style={

                                "padding": "9px"

                            },

                        ),



                        html.Td(

                            f"{pair['Correlation']:.3f}",

                            style={

                                "padding": "9px",

                                "fontWeight": "700",

                            },

                        ),

                    ]

                )

            )



        correlation_table = html.Div(

            html.Table(

                [

                    html.Thead(header),

                    html.Tbody(body),

                ],

                style={

                    "width": "100%",

                    "borderCollapse": "collapse",

                },

            ),

            style={

                "backgroundColor": PANEL,

                "border": f"1px solid {BORDER}",

                "borderRadius": "6px",

                "overflowX": "auto",

            },

        )



    # --------------------------------------------------------

    # FIGURES

    # --------------------------------------------------------



    return (

        options,

        selected,

        kpis,

        timestamp,

        build_overview_figure(

            portfolios

        ),

        build_model_tournament(

            portfolios

        ),

        build_model_dispersion_figure(

            portfolios

        ),

        portfolio_table(

            portfolios

        ),

        build_selected_model_chart(

            row

        ),

        diagnostics,

        build_correlation_figure(

            snapshot

        ),

        correlation_table,

        build_optimization_figure(

            snapshot

        ),

        build_optimization_heatmap(

            snapshot

        ),

        build_model_tournament(

            portfolios

        ),

    )





# ============================================================

@app.callback(
    Output("efficiency-modal", "style"),
    Output("efficient-frontier", "figure"),
    Input("efficiency-button", "n_clicks"),
    Input("efficiency-close", "n_clicks"),
    Input("portfolio", "value"),
)
def toggle_efficiency_modal(open_clicks, close_clicks, selected):
    open_clicks = open_clicks or 0
    close_clicks = close_clicks or 0
    is_open = open_clicks > close_clicks
    snapshot = load_snapshot()

    if snapshot is None:
        figure = base_figure(go.Figure(), "Efficient Frontier — No Snapshot Available")
    else:
        figure = build_efficient_frontier_figure(snapshot, selected)

    style = {
        "display": "flex" if is_open else "none",
        "position": "fixed",
        "inset": "0",
        "zIndex": "9999",
        "backgroundColor": "rgba(3,7,12,0.78)",
        "alignItems": "center",
        "justifyContent": "center",
        "padding": "30px",
    }
    return style, figure


# MAIN

# ============================================================



if __name__ == "__main__":

    app.run(debug=True)
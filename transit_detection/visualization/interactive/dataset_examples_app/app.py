from dash import (
    Dash,
    dcc,
    html,
    Input,
    Output,
    State,
    dash_table,
    ALL,
)
import logging
import plotly.graph_objects as go
import yaml
from pathlib import Path
from flask_caching import Cache
import numpy as np

from utils_app import (
    build_index,
    load_index,
    get_tfrec_fp_from_index,
    load_t_sr_data,
    get_preview_fig,
    get_color,
    get_sector_color,
    detect_set,
)

# ─────────────── Logging ───────────────
logging.basicConfig(level=logging.INFO)
print("test")
logger = logging.getLogger(__name__)

# ─────────────── Load config & index ───────────────
config_fp = Path(__file__).with_name("config_app.yaml")
with open(config_fp) as f:
    config = yaml.safe_load(f)

tfrec_dir = Path(config["setup"]["data"]["paths"]["tfrec_dir"])
storage_dir = Path(config["setup"]["data"]["paths"]["storage_dir"])
storage_dir.mkdir(parents=True, exist_ok=True)

index_fp = storage_dir / "index.json"
if not index_fp.exists():
    build_index(tfrec_dir, index_fp)
index = load_index(index_fp)

# ─────────────── Dash + Cache ───────────────
app = Dash(
    __name__,
    assets_folder=str(Path(__file__).parent / "assets"),
    suppress_callback_exceptions=True,
)
cache = Cache(app.server, config=config["setup"]["data"]["flask"]["cache_config"])

# ─────────────── Layout ───────────────
app.layout = html.Div(
    style={"padding": 0, "margin": 0},
    children=[
        html.H2("Search Dataset TFRecords by Target / Sector Run"),
        # Target dropdown
        html.Label("Target", htmlFor="target-dropdown"),
        dcc.Dropdown(
            id="target-dropdown",
            options=[{"label": t, "value": t} for t in sorted(index.keys())],
            placeholder="Pick a target…",
        ),
        # Sector-run dropdown
        html.Label("Sector Run", htmlFor="sectorrun-dropdown"),
        dcc.Dropdown(
            id="sectorrun-dropdown",
            options=[],
            value=None,
            placeholder="Select a target first",
            disabled=True,
        ),
        # TCE table
        html.H3("TCE Selection"),
        dash_table.DataTable(
            id="tce-table",
            columns=[
                {"name": "UID", "id": "tce_uid"},
                {"name": "PERIOD (days)", "id": "tce_period"},
                {"name": "DURATION (hours)", "id": "tce_duration"},
                {"name": "TIME0BK (BTJD)", "id": "tce_time0bk"},
                {"name": "DISPOSITION", "id": "disposition"},
            ],
            row_selectable="multi",
            data=[],
            selected_rows=[],
        ),
        # Display / filter controls
        html.Label("View Mode", htmlFor="view-mode"),
        dcc.RadioItems(
            id="view-mode",
            options=[
                {"label": "Dataset", "value": "tabs"},
                {"label": "Epehemerides", "value": "tabs"},
            ],
            value="tabs",
            inline=True,
        ),
        html.Label("Filter Labels", htmlFor="label-filter"),
        dcc.Checklist(
            id="label-filter",
            options=[
                {"label": "Transit", "value": "1"},
                {"label": "Non-Transit", "value": "0"},
            ],
            value=["0", "1"],
            inline=True,
        ),
        # Light-curve area
        html.H3("Light Curve Plot"),
        html.Div(
            id="plot-section",
            style={"position": "relative", "padding": 0, "margin": 0},
            children=[
                dcc.Loading(
                    id="loading-lightcurve",
                    type="circle",
                    fullscreen=False,
                    children=html.Div(
                        id="lightcurve-loading-wrapper",
                        children=[
                            html.Div(
                                "Loading light curve data...",
                                id="lightcurve-loading-message",
                                style={"display": "none", "padding": "10px"},
                            ),
                            html.Div(
                                id="lightcurve-output",
                                style={"minHeight": "300px", "padding": "10px"},
                            ),
                        ],
                    ),
                ),
            ],
        ),
        # Hover tooltip
        html.Div(
            id="hover-tooltip",
            children=dcc.Graph(
                id="hover-fig",
                config={"displayModeBar": False},
                style={"height": "200px"},
            ),
            style={
                "position": "absolute",
                "display": "none",
                "zIndex": "1000",
                "background": "white",
                "border": "1px solid #ccc",
                "padding": "4px",
                "boxShadow": "0px 2px 6px rgba(0,0,0,0.2)",
            },
        ),
        # Detailed view
        html.H3("Detailed View (Click)"),
        dcc.Loading(
            id="loading-detail",
            type="circle",
            children=html.Div(id="example-detail"),
        ),
        # Store
        dcc.Store(id="cached-t-sr-data"),
        # Error banner
        html.Div(id="error-banner", style={"color": "red", "marginTop": "10px"}),
    ],
)

# ─────────────── Callbacks ───────────────


@app.callback(
    Output("sectorrun-dropdown", "options"),
    Output("sectorrun-dropdown", "value"),
    Output("sectorrun-dropdown", "disabled"),
    Input("target-dropdown", "value"),
)
def update_sector_runs(target):
    if not target:
        return [], None, True
    opts = [{"label": sr, "value": sr} for sr in sorted(index[target].keys())]
    return opts, None, False


@app.callback(
    Output("cached-t-sr-data", "data"),
    Output("lightcurve-loading-message", "style"),
    Input("target-dropdown", "value"),
    Input("sectorrun-dropdown", "value"),
    prevent_initial_call=True,
)
def load_and_cache_t_sr_data(target, sector_run):
    if not target or not sector_run:
        return {}, {"display": "none"}

    logger.info(f"Loading TFRecord for target={target}, sector_run={sector_run}")
    cache.clear()

    entry = get_tfrec_fp_from_index(index, target, sector_run)
    tfrec_fp = entry.get("path", "")
    set_name = detect_set(tfrec_fp)

    t_sr_data = load_t_sr_data(target, sector_run, tfrec_fp)
    for tce in t_sr_data["tces"]:
        tce["set"] = set_name

    logger.info(f"  → Loaded {len(t_sr_data['tces'])} TCEs")
    cache.set(f"{target}_{sector_run}", t_sr_data)

    return t_sr_data, {"display": "none"}


@app.callback(
    Output("tce-table", "data"),
    Output("tce-table", "selected_rows"),
    Input("cached-t-sr-data", "data"),
    Input("sectorrun-dropdown", "value"),
)
def update_tce_table(t_sr_data, sector_run):
    count = len(t_sr_data.get("tces", []))
    logger.info(f"Updating table for sector_run={sector_run}: {count} rows")
    if count == 0:
        return [], []
    return t_sr_data["tces"], list(range(count))


@app.callback(
    Output("lightcurve-output", "children"),
    Input("cached-t-sr-data", "data"),
    Input("sectorrun-dropdown", "value"),
    Input("tce-table", "selected_rows"),
    Input("tce-table", "data"),
    Input("view-mode", "value"),
    Input("label-filter", "value"),
)
def render_lightcurves(
    t_sr_data, sector_run, selected_rows, table_data, mode, label_filter
):
    if not t_sr_data or not table_data:
        return "No data loaded."

    selected_uids = [table_data[i]["tce_uid"] for i in selected_rows]

    def make_fig_for(sector_key):
        fig = go.Figure()
        sectors = (
            list(t_sr_data["sectors"].keys())
            if sector_key == "combined"
            else [sector_key]
        )
        for sector in sectors:
            d = t_sr_data["sectors"][sector]
            time = np.array(d["lightcurve"]["time"])
            flux = np.array(d["lightcurve"]["flux"])
            fig.add_trace(
                go.Scatter(
                    x=time,
                    y=flux,
                    mode="lines",
                    name=f"Sector {sector}",
                    line=dict(color=get_sector_color(sector)),
                    hoverinfo="skip",
                )
            )
            for uid in selected_uids:
                for ex in d["examples_by_tce"].get(uid, []):
                    if ex.get("label", "-1") not in label_filter:
                        continue
                    t0 = ex["t"]
                    y0 = flux[np.argmin(np.abs(time - t0))]
                    fig.add_trace(
                        go.Scatter(
                            x=[t0],
                            y=[y0],
                            mode="markers",
                            marker=dict(color=get_color(ex.get("label")), size=10),
                            name=f"{round(t0,3)} — TCE {uid} — S{sector}",
                            customdata=[{"t": t0, "tce_uid": uid, "sector": sector}],
                            hovertemplate="t = %{x:.4f}<extra></extra>",
                        )
                    )
        title = "All Sectors" if sector_key == "combined" else f"Sector {sector_key}"
        fig.update_layout(title=title, xaxis_title="Time", yaxis_title="Flux")
        return fig

    if mode == "single":
        return dcc.Graph(
            id={"type": "sector-graph", "index": "combined"},
            figure=make_fig_for("combined"),
            clear_on_unhover=True,
        )

    # tabs mode
    tabs = []
    # combined tab
    tabs.append(
        dcc.Tab(
            label="All Sectors",
            value="combined",
            children=dcc.Graph(
                id={"type": "sector-graph", "index": "combined"},
                figure=make_fig_for("combined"),
                clear_on_unhover=True,
            ),
        )
    )
    for sector in t_sr_data["sectors"]:
        tabs.append(
            dcc.Tab(
                label=f"Sector {sector}",
                value=str(sector),
                children=dcc.Graph(
                    id={"type": "sector-graph", "index": str(sector)},
                    figure=make_fig_for(str(sector)),
                    clear_on_unhover=True,
                ),
            )
        )
    return dcc.Tabs(id="sector-tabs", children=tabs)


@app.callback(
    Output("hover-fig", "figure"),
    Output("hover-tooltip", "style"),
    Input({"type": "sector-graph", "index": ALL}, "hoverData"),
    State("cached-t-sr-data", "data"),
    State("tce-table", "data"),
    prevent_initial_call=True,
)
def update_hover_tooltip(all_hovers, t_sr_data, table_data):
    if not any(h and h.get("points") for h in all_hovers):
        return go.Figure(), {"display": "none"}

    for hd in all_hovers:
        if hd and hd.get("points"):
            pt = hd["points"][0]
            cd = pt.get("customdata") or {}
            sec, uid, t0 = cd.get("sector"), cd.get("tce_uid"), cd.get("t")
            if not (sec and uid and t0 is not None):
                continue

            for ex in t_sr_data["sectors"][sec]["examples_by_tce"].get(uid, []):
                if np.isclose(ex["t"], t0, atol=1e-3):
                    duration = next(
                        (r["tce_duration"] for r in table_data if r["tce_uid"] == uid),
                        None,
                    )
                    fig = get_preview_fig(
                        uid, ex["t"], tuple(ex["flux"]), duration / 24
                    )

                    x0, x1 = pt["bbox"]["x0"], pt["bbox"]["x1"]
                    y0 = pt["bbox"]["y0"]
                    x_center = (x0 + x1) / 2

                    style = {
                        "position": "absolute",
                        "left": f"{x_center}px",
                        "top": f"{y0 + 10}px",
                        "transform": "translate(-50%, -90%)",
                        "display": "block",
                        "zIndex": "1000",
                        "background": "white",
                        "border": "1px solid #ccc",
                        "padding": "4px",
                        "boxShadow": "0px 2px 6px rgba(0,0,0,0.2)",
                    }
                    return fig, style

    return go.Figure(), {"display": "none"}


@app.callback(
    Output("example-detail", "children"),
    Input({"type": "sector-graph", "index": ALL}, "clickData"),
    State("cached-t-sr-data", "data"),
)
def detailed_view(all_clicks, t_sr_data):
    for cd in all_clicks:
        if cd and cd.get("points"):
            pt = cd["points"][0].get("customdata", {})
            sec, t0, uid = pt.get("sector"), pt.get("t"), pt.get("tce_uid")
            if sec and t0 is not None:
                for ex in t_sr_data["sectors"][sec]["examples_by_tce"].get(uid, []):
                    if np.isclose(ex["t"], t0, atol=1e-3):
                        flux = np.array(ex["flux"])
                        time = np.linspace(
                            t0 - 2, t0 + 2, len(flux)
                        )  # TODO: update duration
                        fig = go.Figure(go.Scatter(x=time, y=flux, mode="markers"))
                        fig.update_layout(
                            title=f"TCE {uid} @ t = {t0:.4f} Flux ",
                            height=400,
                        )
                        flux_norm = np.array(ex["flux_norm"])
                        fig2 = go.Figure(
                            go.Scatter(x=time, y=flux_norm, mode="markers")
                        )
                        fig2.update_layout(
                            title=f"TCE {uid} @ t = {t0:.4f} Norm Flux",
                            height=400,
                        )
                        return html.Div(
                            [
                                dcc.Graph(figure=fig),
                                html.Button("Download Data", id="download-btn"),
                                dcc.Graph(figure=fig2),
                                html.Button("Download Data", id="download-btn"),
                            ]
                        )
    return "Click a marker to see detail."


if __name__ == "__main__":
    app.run(debug=True)

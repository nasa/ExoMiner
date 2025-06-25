# %%
# third-party
from dash import Dash, dcc, html, Input, Output, State, dash_table
import plotly.express as px
import plotly.graph_objects as go
import yaml
import os
from pathlib import Path
import json
import ijson
from flask_caching import Cache
import numpy as np

# local
from utils_app import (
    build_index,
    load_index,
    search_index_as_json_stream,
    load_t_sr_tce_list_from_tfrec,
    load_t_sr_data,
    get_tfrec_fp_from_index,
)

# App Config
config = None
config_fp = Path(__file__).with_name("config_app.yaml")
with open(config_fp, "r") as file:
    config = yaml.unsafe_load(file)
assert config != None, f"ERROR Loading config file from {config_fp}"

# Init app and cache
app = Dash(__name__)
cache = Cache(app.server, config=config["setup"]["data"]["flask"]["cache_config"])

# Path Setup
tfrec_dir = Path(config["setup"]["data"]["paths"]["tfrec_dir"])
assert tfrec_dir.exists(), f"tfrec_dir: {tfrec_dir} does not exist."

storage_dir = Path(config["setup"]["data"]["paths"]["storage_dir"])
storage_dir.mkdir(parents=True, exist_ok=True)
assert storage_dir.exists(), f"storage_dir: {tfrec_dir} does not exist."

# build index for target, sector_run to tfrec_fp mapping
index_fp = storage_dir / "index.json"

recompute_index = False
if not index_fp.exists() or recompute_index:
    app.logger.info(f"Building Index")
    build_index(tfrec_dir=tfrec_dir, index_json_fp=index_fp)

app.logger.info(f"Loading Index")
index = load_index(index_json_fp=index_fp)

# General Layout
app.layout = [
    html.Div(
        [
            html.H2(f"Search Dataset TFRecords by Target / Sector Run"),
            html.Label("Target"),
            dcc.Dropdown(
                id="target-dropdown",
                options=[{"label": t, "value": t} for t in index.keys()],
            ),
            html.Label("Sector Run"),
            dcc.Dropdown(id="sectorrun-dropdown"),
            html.Label("Sector"),
            dcc.Dropdown(id="sector-dropdown"),
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
                # style_table?
            ),
        ],  # style={"width": "30%", "display": "inline-block", "verticalAlign": "top"}
    ),
    html.Div(
        [
            html.H4("Lightcurve"),
            dcc.Graph(id="lightcurve-plot"),
        ]
    ),
    html.Div(
        [html.H3("Example Preview (Hover)"), html.Div(id="example-hover-preview")]
    ),
    html.Div([html.H3("Detailed View (Click)"), html.Div(id="example-detail")]),
    html.Div(
        [
            dcc.Store(id="cached-t-sr-data"),
        ]
    ),
    html.Div(
        [
            dcc.Store(id="selected-example-data"),
        ]
    ),
]


# @app.callback(Output("time-series-chart", "figure"), Input("ticker", "value"))
# def display_time_series(ticker):
#     df = px.data.stocks()
#     fig = px.line(df, x="date", y=ticker)
#     return fig


@app.callback(
    Output("sectorrun-dropdown", "options"), Input("target-dropdown", "value")
)
def update_sector_runs(target):
    if not target:
        return []
    return [{"label": sr, "value": sr} for sr in sorted(index[target].keys())]


@app.callback(
    Output("cached-t-sr-data", "data"),
    Input("target-dropdown", "value"),
    Input("sectorrun-dropdown", "value"),
)
def load_and_cache_t_sr_data(target, sector_run):
    if not target or not sector_run:
        return {}

    cache_key = f"{target}_{sector_run}"
    t_sr_data = cache.get(cache_key)
    if t_sr_data:
        return t_sr_data

    tfrec_fp = get_tfrec_fp_from_index(index, target, sector_run)
    t_sr_data = load_t_sr_data(target, sector_run, tfrec_fp)
    app.logger.debug(f"USING T_SR_DATA: {t_sr_data}")
    cache.set(cache_key, t_sr_data)
    return t_sr_data


@app.callback(
    Output("sector-dropdown", "options"),
    Output("sector-dropdown", "value"),
    Input("cached-t-sr-data", "data"),
    Input("sectorrun-dropdown", "value"),
)
def update_sectors(t_sr_data, sector_run):
    if not t_sr_data or not sector_run:
        return [], None
    sectors = list(sorted(t_sr_data.get("sectors", {}).keys()))
    default_value = None
    if len(sectors) == 1:
        default_value = sectors[0]
    options = [{"label": s, "value": s} for s in sectors]
    return options, default_value


@app.callback(
    Output("tce-table", "data"),
    Output("tce-table", "selected_rows"),
    Input("cached-t-sr-data", "data"),
    # Input("sectorrun-dropdown", "value"),
)
def update_tce_table(t_sr_data):
    if not t_sr_data:
        return [], []

    return t_sr_data["tces"], [i for i in range(len(t_sr_data["tces"]))]

    # return list(sorted(tce_data["tce_uid"] for tce_data in t_sr_data["tces"])), list(
    #     sorted(tce_data["tce_uid"] for tce_data in t_sr_data["tces"])
    # )


@app.callback(
    Output("lightcurve-plot", "figure"),
    Input("cached-t-sr-data", "data"),
    Input("sector-dropdown", "value"),
    Input("tce-table", "selected_rows"),
    Input("tce-table", "data"),
)
def update_lc(t_sr_data, sector, selected_rows, table_data):
    if not t_sr_data or not sector or not selected_rows or not table_data:
        return go.Figure()

    # TODO: make sure to cache stuff? -> cache lightcurve, tce_info, ??

    # NOTE: there is a limitation in that the lightcurve that is gotten from downloading from mast does not have all possible time stamps for which examples may have been drawn.
    selected_tce_uids = [table_data[i]["tce_uid"] for i in selected_rows]

    # TODO: include selected rows filtering.
    # TODO: include cacheing

    # for now, just displaying the light curve with all examples

    # get the lightcurve for the sector
    s_data = t_sr_data["sectors"][sector]
    lc = s_data["lightcurve"]
    time, flux = np.array(lc["time"]), np.array(lc["flux"])

    example_ts = []
    examples_by_tce = s_data["examples_by_tce"]
    for tce_uid, examples in examples_by_tce.items():
        example_ts.extend(example["t"] for example in examples)

    marker_ys = [flux[np.argmin(np.abs(time - ts))] for ts in example_ts]

    fig = go.Figure(
        [
            go.Scatter(x=time, y=flux, mode="lines", name="Flux"),
            go.Scatter(
                x=example_ts,
                y=marker_ys,
                mode="markers",
                name="TCE Examples",
                marker=dict(color="red", size=8),
                customdata=example_ts,
                hovertemplate="TCE Example at %{x:.4f}<extra></extra>",
            ),
        ]
    )
    fig.update_layout(
        title=f"Light Curve - Sector {sector}", xaxis_title="Time", yaxis_title="Flux"
    )
    return fig


# @app.callback(
#     Output(""),
#     Input("lightcurve-plot", "clickData"),
#     State("cached-t-sr-data", "data"),
#     State("sector-dropdown", "value"),
# )
# def show_clicked_example(clickData, t_sr_data, sector):
#     if not clickData or not t_sr_data or not sector:
#         return "Click a TCE example market to see details"

#     ts_clicked = clickData["points"][0]["x"]
#     tce_uid = clickData["points"][0].get("customdata")

#     # # lc_fig = px.line(x=time, y=flux)

#     # return lc_fig


#     # get info for all tces

#     # get info for all examples

#     # plot examples etc

#     # display only marked examples on lightcurve for the selected rows (tces)

#     # allow hover functionality


#     s_data = t_sr_data["sectors"][sector]
#     s_examples = s_data["examples_by_tce"]
#     time = s_data["lightcurve"]["time"]
#     flux = s_data["lightcurve"]["flux"]

# ["sectors"][sector]["examples_by_tce"][tce_uid].append(
#             {"uid": uid, "t": t, "flux": flux, "diff_imgs": diff_imgs}
#         )

#     selected_tces = [table_data[i] for i in selected_rows]
#     markers = []

#     for tce in selected_tces:
#         pass

# @app.callback(
#     Output("tce-checklist", "options"),
#     Output("tce-checklist", "value"),
#     Input("cached-t-sr-data", "data"),
#     # Input("sectorrun-dropdown", "value"),
# )
# def update_tces(t_sr_data):
#     if not t_sr_data:
#         return [], []

#     tce_tbl =
#     return list(sorted(tce_data["tce_uid"] for tce_data in t_sr_data["tces"])), list(
#         sorted(tce_data["tce_uid"] for tce_data in t_sr_data["tces"])
#     )


# @app.callback(
#     Output("tce-checklist", "options"),
#     Input("target-dropdown", "value"),
#     Input("sectorrun-dropdown", "value"),
# )
# def update_tce_list(selected_target, selected_sector_run):
#     if not selected_target or not selected_sector_run:
#         return []
#     print(f"index[selected_target]: {index[selected_target]}")

#     tce_list = load_t_sr_tce_list_from_tfrec(
#         index[selected_target][selected_sector_run],
#         selected_target,
#         selected_sector_run,
#     )
#     print(f"tce_list: {tce_list}")
#     return tce_list


# @app.callback(
#     Output("cached-t-sr-data", "data"),
#     Input("target-dropdown", "value"),
#     Input("sectorrun-dropdown", "value"),
# )
# def load_and_cache_t_sr_data(selected_target, selected_sector_run):
#     if not selected_target or not selected_sector_run:
#         return {}

#     tfrec_fp = index[selected_target][selected_sector_run]

#     return load_t_sr_data(
#         selected_target,
#         selected_sector_run,
#         tfrec_fp,
#     )


# @app.callback(
#     Output("cached-tce-data", "data"),
#     Input("target-dropdown", "value"),
#     Input("sectorrun-dropdown", "value"),
# )
# def load_tce_data(selected_target, selected_sector_run):
#     if not selected_target or not selected_sector_run:
#         return []

#     tfrec_fp = index[selected_target][selected_sector_run]
#     tce_list = load_t_sr_tce_list_from_tfrec(
#         tfrec_fp,
#         selected_target,
#         selected_sector_run,
#     )
#     return {"tfrec_fp": tfrec_fp, "tces": tce_list}


# @app.callback(Output("lightcurve-plot", "figure"), Input("cached-tce-data", "data"), Input("tce-checklist", "value"))
# def display_lightcurve(all_tces, selected_tces):

#     for tce in selected_tces:

#     df = px.data.stocks()
#     fig = px.line(df, x="date", y=ticker)
#     return fig


if __name__ == "__main__":
    # run on localhost
    app.run(host="127.0.0.1", port=8050, debug=True)

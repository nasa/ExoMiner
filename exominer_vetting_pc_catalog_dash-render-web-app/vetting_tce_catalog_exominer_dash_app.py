"""
Using dash to create ExoMiner vetting TCE catalog as an interactive table in a web application.
"""

# 3rd party
import dash
from dash import dash_table
from dash.dependencies import Input, Output, State
from dash import html, dcc
import pandas as pd
from datetime import datetime
from PIL import Image

EXPORT_CSV_FILENAME_PREFIX = 'exominer_vetting_tess-spoc-2-min-s1s67'
TABLE_FP = ('exominer_vetting_pc_catalog_dash-render-web-app/data/exominer_vetting_tess-spoc-2-min-s1s67_'
            'dashtable_dvm-url_scoregt0.1.csv')
APP_TITLE = 'ExoMiner Vetting - TESS SPOC TCES 2-min S1-67'
EXOMINER_LOGO_IMG_FP = 'others/images/exominer_logo.png'

# load ExoMiner logo
exominer_logo_pil_image= Image.open(EXOMINER_LOGO_IMG_FP)

# read predictions csv table
df = pd.read_csv(TABLE_FP, usecols=[
    'TIC ID',
    'Sector Run',
    'Planet Number',
    'Orbital Period [day]',
    'Transit Duration [hour]',
    'Transit Depth [ppm]',
    'Planet Radius [Earth Radii]',
    'MES',
    'ExoMiner Score',
    'ExoMiner Unc. Score',
    'DV mini-report URL',
])

# create web app
app = dash.Dash(__name__)
app.title = f"{APP_TITLE}"  # Set the tab name
app._favicon = 'exominer_logo.png'  # Set the favicon

server = app.server  # needed for deployment in server

# set properties of different columns
cols_properties = {
    'TIC ID': {"type": "numeric"},
    'Sector Run': {"type": "numeric"},
    'Planet Number': {"type": "numeric"},
    'Orbital Period [day]': {"type": "numeric"},
    'Transit Duration [hour]': {"type": "numeric"},
    'Transit Depth [ppm]': {"type": "numeric"},
    'Planet Radius [Earth Radii]': {"type": "numeric"},
    'MES': {"type": "numeric"},
    'ExoMiner Score': {"type": "numeric"},
    'ExoMiner Unc. Score': {"type": "numeric"},
    'DV mini-report URL': {"presentation": "markdown"},
}
cols_datatable = []
for col in df.columns:
    col_datatable = {"name": col, "id": col}
    col_datatable.update(cols_properties[col])
    cols_datatable.append(col_datatable)

# define tooltips header
tooltips = {
    # 'TCE ID': 'TCE ID in the format {TIC ID}-{SPOC TCE Planet Number}-S{Sector Run ID}',
    'TIC ID': 'TIC ID of target star',
    'Sector Run': 'SPOC Sector Run ID in which the TCE was detected',
    'Planet Number': 'Planet number associated with TCE by the SPOC pipeline',
    'Orbital Period [day]': 'Orbital period [day] estimated by SPOC DV',
    'Transit Duration [hour]': 'Transit duration [hour] estimated by SPOC DV',
    # 'Transit Epoch [BTJD]': 'Transit epoch [BTJD] estimated by SPOC DV',
    'Transit Depth [ppm]': 'Transit depth [ppm] estimated by SPOC DV',
    'Planet Radius [Earth Radii]': 'Planet radius [Earth Radii] estimated by SPOC DV',
    'MES': 'Multiple Event Statistic computed by SPOC pipeline',
    # 'Transit Model SNR': 'Transit model SNR estimated by SPOC DV',
    # 'Number of transits observed': 'Number of transits observed in the data',
    # 'Gaia RUWE': 'TIC RUWE from Gaia DR2',
    'ExoMiner Score': 'ExoMiner model score in [0, 1]. The model is more confident that the TCE is a planet candidate '
                      'as the score gets closer to one. It is NOT a probability',
    'ExoMiner Unc. Score': 'Uncertainty in ExoMiner model score',
    'DV mini-report URL': 'Link to download target SPOC DV mini-report from MAST'
}

# structure html layout
app.layout = html.Div([
    html.H1([
        html.Img(src=exominer_logo_pil_image, style={"height": "100px", "vertical-align": "middle"}),
        " ExoMiner Vetting Catalog for TESS SPOC 2-min TCEs S1-S67", 
        # html.A("Paper", href='https://arxiv.org/abs/2502.09790', target="_blank")
    ],
        style={
            'color': '#79a4b5',
            'fontSize': '30px',
            'fontWeight': 'bold',
            'textAlign': 'center',
            'marginBottom': '20px',
            'backgroundColor': '#f9f9f9',
            'padding': '10px',
            'borderRadius': '10px',
            'boxShadow': '0 4px 8px rgba(0, 0, 0, 0.1)',
            'fontFamily': 'Arial',
        },

),
    html.H2(["If you make use of this catalog for vetting, validation or other studies, please acknowledge our "
             "contribution by citing ", html.A("ExoMiner's TESS 2-min paper (2025)",
                                               href='https://iopscience.iop.org/article/10.3847/1538-4357/ac4399/meta', target="_blank")],
            style={
                'color': '#79a4b5',
                'fontSize': '15px',
                'fontWeight': 'bold',
                'textAlign': 'left',
                'marginBottom': '1px',
                'backgroundColor': '#f9f9f9',
                'padding': '2px',
                'borderRadius': '2px',
                'boxShadow': '0 2px 1px rgba(0, 0, 0, 0.1)',
                'fontFamily': 'Arial',
            }, ),
    html.H2("Results from 1/16/2025 10:14am | Last web app update: 10/28/2025 1:01pm | "
            "Excluded TCEs with scores < 0.1",
            style={
                'color': '#79a4b5',
                'fontSize': '15px',
                'fontWeight': 'bold',
                'textAlign': 'left',
                'marginBottom': '1px',
                'backgroundColor': '#f9f9f9',
                'padding': '2px',
                'borderRadius': '2px',
                'boxShadow': '0 2px 1px rgba(0, 0, 0, 0.1)',
                'fontFamily': 'Arial',
            }, ),
    html.H2(["Full dataset and additional data can be found in Zenodo: ",
             html.A("here", href='https://doi.org/10.5281/zenodo.15466292', target="_blank")],
            style={
                'color': '#79a4b5',
                'fontSize': '15px',
                'fontWeight': 'bold',
                'textAlign': 'left',
                'marginBottom': '1px',
                'backgroundColor': '#f9f9f9',
                'padding': '2px',
                'borderRadius': '2px',
                'boxShadow': '0 2px 1px rgba(0, 0, 0, 0.1)',
                'fontFamily': 'Arial',
            }, ),
    dcc.Input(id='sector-run-filter', type='text', placeholder='Filter TCEs by Sector Run using regex patterns '
                                                               '(e.g., 14, 14-60, -60)',
              style={'display': 'flex', 'flexDirection': 'column', 'width': '100%'}),
    html.Button("Export CSV ⬇️", id="export-button", style={'textAlign': 'right', 'fontFamily': 'Arial'}),
    dcc.Download(id="download-dataframe-csv"),
    dash_table.DataTable(
        id='table',
        columns=cols_datatable,
        data=df.to_dict('records'),
        tooltip_header={
                column: {'value': tooltips[column], 'type': 'markdown'} for column in df.columns
              },
        css=[
            dict(selector= "p", rule= "margin: 0; text-align: center"),
            dict(selector=".dash-tooltip", rule="font-family: 'Arial'")
        ],
        tooltip_duration=None,
        filter_action="native",
        sort_action="native",
        page_size=30,
        markdown_options={'link_target': '_blank'},

        style_header={
            'backgroundColor': '#79a4b5',
            'fontFamily': 'Arial',
            'color': 'white',
            'fontWeight': 'bold',
            'textAlign': 'center',
        },
        style_data_conditional=[
            {
                'if': {'row_index': 'odd'},
                'backgroundColor': '#f2f2f2',
            },

            {
                'if': {'state': 'active'},
                'backgroundColor': '#D3D3D3',
                'border': '1px solid #000',
                'fontFamily': 'Arial',
            },

            {
                'if': {'column_id': 'DV mini-report URL'},
                'textAlign': 'center',
             }

        ]
    ),
    html.H2(["🌎 Created by ", html.A("Miguel Martinho", href='https://migmartinho.github.io/',
                                     target="_blank"), "🪐"],
            style={
                'color': '#79a4b5',
                'fontSize': '15px',
                'fontWeight': 'bold',
                'textAlign': 'center',
                'marginBottom': '1px',
                'backgroundColor': '#f9f9f9',
                'padding': '2px',
                'borderRadius': '0px',
                'boxShadow': '0 0px 0px rgba(0, 0, 0, 0.1)',
                'fontFamily': 'Arial',
            }, ),
])


@app.callback(
    Output('table', 'data'),
    [
        Input('sector-run-filter', 'value'),
     ]
)
def update_table(sector_run_filter):
    """
    Update the table data based on the filter input. Can do regex filtering.
    Args:
        sector_run_filter (str): Filter value for the Sector Run column.
    Returns:
        list: Filtered data for the table.
    """

    filtered_df = df

    if sector_run_filter:
        filtered_df = filtered_df[filtered_df['Sector Run'].str.contains(rf'{sector_run_filter}', regex=True)]

    return filtered_df.to_dict('records')


@app.callback(
    Output("download-dataframe-csv", "data"),
    Input("export-button", "n_clicks"),
     State('table', 'derived_virtual_data'),
    prevent_initial_call=True,

)
def export_table_as_csv(n_clicks, virtual_data):
    """
    Export the table as a CSV file when the button is clicked.
    Args:
        n_clicks (int): Number of clicks on the button.
        virtual_data (list): Data from the table.
    Returns:
        dcc.send_data_frame: CSV file to be downloaded.
    """


    if n_clicks:
        virtual_data_df = pd.DataFrame.from_dict(virtual_data)

        return dcc.send_data_frame(
            virtual_data_df.to_csv,
            filename=f"{EXPORT_CSV_FILENAME_PREFIX}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            index=False)

    return None


if __name__ == '__main__':

    app.run(debug=False)

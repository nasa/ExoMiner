"""
Aggregate TCE tables with URLs to DV reports hosted at the MAST.
"""

# 3rd party
from pathlib import Path
import pandas as pd
import re


def _set_default_uid(x):
    target_id, tce_plnt_num, s_sector, e_sector = x['uid'].split('-')
    if s_sector[1:] == e_sector:
        return f'{target_id}-{tce_plnt_num}-{s_sector}'
    else:
        return x['uid']


def write_to_sheet(writer, df, url_columns, sheet_name):
    """ Write table to sheet in xslx file.

    Args:
        writer: Excel writer object
        df: pandas dataframe, table to write
        url_columns: list, column names of url columns
        sheet_name: str, name of sheet to write

    Returns:

    """

    df.to_excel(writer, sheet_name=sheet_name, index=False)

    # Get the xlsxwriter workbook and worksheet objects.
    workbook = writer.book
    worksheet = writer.sheets[sheet_name]

    # Iterate through the chunk and add hyperlinks.
    for col in url_columns:
        col_idx = df.columns.get_loc(col)
        for row_num, url in enumerate(df[col], start=1):
            if not isinstance(url, str):  # np.isnan(url):
                worksheet.write(row_num, col_idx, '')
            else:
                worksheet.write_url(row_num, col_idx, url, string='click here to download from MAST')


def write_to_excel(df_lst, filename, url_columns):
    """ Write list of tables to excel file. Each table is written to one sheet.

    Args:
        df_lst: list of pandas dataframe, tables to write
        filename: Path, filepath of excel file
        url_columns: list, column names of url columns

    Returns:

    """

    # Create a Pandas Excel writer using XlsxWriter as the engine.
    writer = pd.ExcelWriter(filename, engine='xlsxwriter')

    for df in df_lst:

        sector_run = '-'.join(df.loc[0, 'uid'].split('-')[2:])
        # Write chunk to a new sheet
        sheet_name = f'{sector_run}'
        print('sheet_name: ', sheet_name, len(df))

        write_to_sheet(writer, df, url_columns, sheet_name)

    # Close the Pandas Excel writer and output the Excel file.
    writer.close()


tce_tbl_fps = list(Path('/Users/msaragoc/Projects/exoplanet_transit_classification/data/dv_reports/TESS/tess_spoc_2min_urls/tess_spoc_2min_s1-s68_1-23-2025_0947').rglob('tess_spoc_2min_s1-s68*.csv'))
# Define the columns that contain URLs.
url_columns_tbl = ['DV TCE summary report', 'Full DV report', 'DV mini-report']

save_dir = Path('/Users/msaragoc/Projects/exoplanet_transit_classification/data/dv_reports/TESS/tess_spoc_2min_urls/postprocess_results_1-31-2025_1647')

msector_tbl_fps = sorted([fp for fp in tce_tbl_fps if bool(re.search(r'_sector_run_\d+-\d+', fp.name))])
ssector_tbl_fps = {int(fp.stem.split('_')[-1]): fp for fp in tce_tbl_fps if not bool(re.search(r'_sector_run_\d+-\d+', fp.name))}
ssector_tbl_fps = list(dict(sorted(ssector_tbl_fps.items())).values())

tce_tbl_fps = ssector_tbl_fps + msector_tbl_fps

tce_tbls = [pd.read_csv(fp) for fp in tce_tbl_fps]

for tbl in tce_tbls:
    tbl['uid'] = tbl.apply(_set_default_uid, axis=1)

tce_tbls = [tbl.rename(columns={'TCE summary report': 'DV TCE summary report', 'TCERT report': 'DV mini-report'}) for tbl in tce_tbls]

save_dir.mkdir(parents=True, exist_ok=True)

# Write the DataFrame to an Excel file with hyperlinks.
write_to_excel(tce_tbls, save_dir / 'tess_spoc_2min_agg_s1-s68_1-31-2025_1647.xlsx', url_columns_tbl)

# for tbl in tce_tbls:
#
#     sector_run = '-'.join(tbl.loc[0, 'uid'].split('-')[2:])
#
#     writer = pd.ExcelWriter(save_dir / f'tess_spoc_2min_{sector_run}.xlsx', engine='xlsxwriter')
#
#     # Write chunk to a new sheet
#     sheet_name = f'{sector_run}'
#     print('sheet_name: ', sheet_name, len(tbl))
#
#     write_to_sheet(writer, tbl, url_columns, sheet_name)
#
#     tbl.to_excel(writer, sheet_name=sheet_name, index=False)
#
#     # Get the xlsxwriter workbook and worksheet objects.
#     workbook = writer.book
#     worksheet = writer.sheets[sheet_name]
#
#     # Close the Pandas Excel writer and output the Excel file.
#     writer.close()

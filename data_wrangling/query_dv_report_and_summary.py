""" Script used to download DV reports and summaries for KICs and TICs from the MAST. """

# 3rd party
import pandas as pd
from astroquery.mast import Observations


def get_kic_dv_report_and_summary(kic, download_dir, verbose=False):
    """ Download DV reports and summaries available in the MAST for a given KIC.

    :param kic: int, KIC
    :param download_dir: str, download directory
    :param verbose: bool, verbose
    :return:
        astropy Table with path/URL to downloaded products
    """

    obs_table = Observations.query_criteria(target_name=f'kplr{str(kic).zfill(9)}',
                                            obs_collection='Kepler',
                                            obs_id='*lc*',
                                            )

    if verbose:
        print(f'Number of observations queried: {len(obs_table)}')

    obs_products = Observations.get_product_list(obs_table)
    obs_products_filter = Observations.filter_products(obs_products, extension='pdf')

    prod = Observations.download_products(obs_products_filter, download_dir=download_dir)

    return prod


def get_tic_dv_report_and_summary(example_id, download_dir, reports='all', verbose=False):
    """ Download DV reports and summaries available in the MAST for a given observation, which can be either a target,
    sector run, or TCE.

    :param example_id: str, example identifier. Must follow format {tic_id}-{tce_id}-S{start_sector}-{end_sector}. To
    not condition on a specific field, use `X` as placeholder. E.g. {tic_id}-X-S{start_sector}-{end_sector}.
    :param download_dir: str, download directory
    :param reports: str, choose which reports to get.
        'dv_summary': downloads only TCE DV summary reports.
        'dv_report': downloads only full DV reports.
        'tcert': downloads only TCERT reports.
        'dv': downloads both TCE DV summary and full DV reports.
        'all': downloads DV and TCERT reports.
    :param verbose: bool, verbose
    :return:
        prods, list of astropy Tables with path/URL to downloaded products
    """

    if verbose:
        print(f'Started run for example {example_id}...')

    # maps option to name in MAST table for the data products reports
    report_type_map = {
        'all': ['TCE summary report', 'full data validation report', 'Data validation mini report'],
        'dv': ['TCE summary report', 'full data validation report'],
        'dv_summary': ['TCE summary report'],
        'dv_report': ['full data validation report'],
        'tcert': ['Data validation mini report'],
    }
    reports_to_get = report_type_map[reports]

    tic_id, tce_id, s_sector, e_sector = '.' * 12, '.' * 2, '.' * 4, '.' * 4

    ids = example_id.split('-')

    tic_id = ids[0].zfill(16) if ids[0] != 'X' else tic_id
    tce_id = ids[1].zfill(2) if ids[1] != 'X' else tce_id
    s_sector = ids[2][1:] if ids[2] != 'X' else s_sector
    e_sector = ids[3] if ids[3] != 'X' else e_sector

    # sectors = sector_run_id.split('-')
    #
    # if len(sectors) == 1:  # single-sector run
    #     s_sector, e_sector = (sectors[0][1:], ) * 2
    # if len(sectors) == 2:  # multi-sector run
    #     s_sector, e_sector = sectors[0][1:], sectors[1]

    s_sector, e_sector = s_sector.zfill(4), e_sector.zfill(4)

    product_tce_id = f's{s_sector}-s{e_sector}-{tic_id}-{tce_id}'
    product_target_id = f's{s_sector}-s{e_sector}-{tic_id}'

    #  get table of observations associated with this target
    obs_table = Observations.query_criteria(target_name=int(tic_id),
                                            obs_collection='TESS',
                                            obs_id='*',
                                            )

    if verbose:
        print(f'Number of observations queried: {len(obs_table)}')

    # get list of products available for this target's observations
    obs_products = Observations.get_product_list(obs_table)

    products_filenames = obs_products.to_pandas()['productFilename']

    prods = []

    if 'TCE summary report' in reports_to_get:  # TCE DV summary reports

        # filter products based on TCE ID provided, and type of report
        filtered_products_filenames = products_filenames.loc[products_filenames.str.contains(product_tce_id,
                                                                                         regex=True)].to_list()

        obs_products_filter = Observations.filter_products(obs_products,
                                                           extension='pdf',
                                                           description='TCE summary report',
                                                           productFilename=filtered_products_filenames)

        # download selected products
        prod_tce = Observations.download_products(obs_products_filter, download_dir=download_dir)
        prods.append(prod_tce)

    if 'full data validation report' in reports_to_get:  # DV full reports

        filtered_products_filenames = products_filenames.loc[products_filenames.str.contains(product_target_id,
                                                                                             regex=True)].to_list()

        obs_products_filter = Observations.filter_products(obs_products,
                                                           extension='pdf',
                                                           description='full data validation report',
                                                           productFilename=filtered_products_filenames)

        # download selected products
        prod_dv_full = Observations.download_products(obs_products_filter, download_dir=download_dir)
        prods.append(prod_dv_full)

    if 'Data validation mini report' in reports_to_get:  # TCERT reports

        filtered_products_filenames = products_filenames.loc[products_filenames.str.contains(product_target_id,
                                                                                             regex=True)].to_list()

        obs_products_filter = Observations.filter_products(obs_products,
                                                           extension='pdf',
                                                           description='Data validation mini report',
                                                           productFilename=filtered_products_filenames)

        # download selected products
        prod_tcert = Observations.download_products(obs_products_filter, download_dir=download_dir)
        prods.append(prod_tcert)

    return prods


if __name__ == "__main__":
    
    kic_list = []
    download_dir = '/Users/msaragoc/Projects/exoplanet_transit_classification/data/dv_reports/mastDownload/Kepler/'
    for kic in kic_list:
        get_kic_dv_report_and_summary(kic, download_dir, verbose=False)

    download_dir = '/Users/msaragoc/Projects/exoplanet_transit_classification/data/dv_reports/mastDownload/TESS/'
    tbl = pd.read_csv('/Users/msaragoc/Downloads/cv_merged_conv_8-4-2023_norm/ensemble_ranked_predictions_allfolds.csv')
    objs = tbl.loc[tbl['label'] == 'T-NTP'].sort_values('score', ascending=False)[['uid', 'score']].reset_index(drop=True).iloc[0:39]['uid']

    def _correct_sector_field(x):
        target_id, tce_id = x.split('-')[:2]
        sector_id = x.split('-')[2:]
        if len(sector_id) == 2:
            return x
        else:
            sector_id = f'{sector_id[0]}-{sector_id[0][1:]}'
            return f'{target_id}-{tce_id}-{sector_id}'

    objs_list = objs.apply(_correct_sector_field)
    for obj in objs_list:
        _ = get_tic_dv_report_and_summary(obj, download_dir, reports='all', verbose=True)

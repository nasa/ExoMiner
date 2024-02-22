""" Script used to download DV reports and summaries for KICs and TICs from the MAST. """

# 3rd party
import pandas as pd
from astroquery.mast import Observations
import numpy as np
from pathlib import Path
import multiprocessing
import os


URL_HEADER = 'https://mast.stsci.edu/api/v0.1/Download/file?uri='


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


def get_dv_dataproducts(example_id, download_dir, download_products, reports='all', verbose=False):
    """ Download DV reports and summaries available in the MAST for a given observation, which can be either a target,
    sector run, or TCE.

    :param example_id: str, example identifier. Must follow format {tic_id}-{tce_id}-S{start_sector}-{end_sector}. To
    not condition on a specific field, use `X` as placeholder. E.g. {tic_id}-X-S{start_sector}-{end_sector}.
    :param download_dir: str, download directory
    :param download_products: bool, if True products are downloaded
    :param reports: str, choose which reports to get.
        'dv_summary': downloads only TCE DV summary reports.
        'dv_report': downloads only full DV reports.
        'tcert': downloads only TCERT reports.
        'dv': downloads both TCE DV summary and full DV reports.
        'all': downloads DV and TCERT reports.
    :param verbose: bool, verbose
    :return:
        prods, list of astropy Tables with path/URL to downloaded products
        uris, list of data products URIs
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
    s_sector = ids[2][1:] if ids[2][1:] != 'X' else s_sector
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
    uris = {'TCE summary report': '', 'Full DV report': '', 'TCERT report': ''}

    if 'TCE summary report' in reports_to_get:  # TCE DV summary reports

        # filter products based on TCE ID provided, and type of report
        filtered_products_filenames = products_filenames.loc[products_filenames.str.contains(product_tce_id,
                                                                                         regex=True)].to_list()

        obs_products_filter = Observations.filter_products(obs_products,
                                                           extension='pdf',
                                                           description='TCE summary report',
                                                           productFilename=filtered_products_filenames)

        # download selected products
        if download_products:
            prod_tce = Observations.download_products(obs_products_filter, download_dir=download_dir)
            prods.append(prod_tce)
        try:
            uris['TCE summary report'] = obs_products_filter['dataURI'].tolist()[0]
        except:
            print(f'No TCE summary report found for {example_id}')

    if 'full data validation report' in reports_to_get:  # DV full reports

        filtered_products_filenames = products_filenames.loc[products_filenames.str.contains(product_target_id,
                                                                                             regex=True)].to_list()

        obs_products_filter = Observations.filter_products(obs_products,
                                                           extension='pdf',
                                                           description='full data validation report',
                                                           productFilename=filtered_products_filenames)
        try:
            uris['Full DV report'] = obs_products_filter['dataURI'].tolist()[0]
        except:
            print(f'No DV full report found for {example_id}')

        # download selected products
        if download_products:
            prod_dv_full = Observations.download_products(obs_products_filter, download_dir=download_dir)
            prods.append(prod_dv_full)

    if 'Data validation mini report' in reports_to_get:  # TCERT reports

        filtered_products_filenames = products_filenames.loc[products_filenames.str.contains(product_target_id,
                                                                                             regex=True)].to_list()

        obs_products_filter = Observations.filter_products(obs_products,
                                                           extension='pdf',
                                                           description='Data validation mini report',
                                                           productFilename=filtered_products_filenames)
        try:
            uris['TCERT report'] = obs_products_filter['dataURI'].tolist()[0]
        except:
            print(f'No TCERT report found for {example_id}')

        # download selected products
        if download_products:
            prod_tcert = Observations.download_products(obs_products_filter, download_dir=download_dir)
            prods.append(prod_tcert)

    return prods, uris


def get_dv_dataproducts_list(objs_list, data_products_lst, download_dir, download_products, reports, verbose=True):
    """ Download DV reports and summaries available in the MAST for a list of observation, which can be either a target,
    sector run, or TCE.

    Args:
        objs_list: list, objects to get data products for
        data_products_lst: list, data products column names in dataframe
        download_dir: str, download directory
        download_products: bool, if True products are downloaded
        reports: str, choose which reports to get.
            'dv_summary': downloads only TCE DV summary reports.
            'dv_report': downloads only full DV reports.
            'tcert': downloads only TCERT reports.
            'dv': downloads both TCE DV summary and full DV reports.
            'all': downloads DV and TCERT reports.
        verbose: bool, verbose

    Returns: uris_dict, dictionary that contains the URIs for the data products downloaded for each object

    """

    proc_id = os.getpid()

    uris_dict = {'uid': [''] * len(objs_list)}
    uris_dict.update({field: [''] * len(objs_list) for field in data_products_lst})
    for obj_i, obj in enumerate(objs_list):
        print(f'[{proc_id}] Getting data products for object {obj} ({obj_i + 1}/{len(objs_list)})...')
        uris_dict['uid'][obj_i] = obj
        _, uris = get_dv_dataproducts(obj, str(download_dir), download_products, reports, verbose)
        for field in data_products_lst:
            uris_dict[field][obj_i] = URL_HEADER + uris[field] if uris[field] != '' else ''

    return uris_dict


if __name__ == "__main__":

    # Kepler
    kic_list = []
    download_dir = '/Users/msaragoc/Projects/exoplanet_transit_classification/data/dv_reports/mastDownload/Kepler/'
    for kic in kic_list:
        get_kic_dv_report_and_summary(kic, download_dir, verbose=False)

    # TESS
    # tbl = pd.read_csv('/Users/msaragoc/Downloads/cv_merged_conv_8-4-2023_norm/ensemble_ranked_predictions_allfolds.csv')
    # objs = tbl.loc[tbl['label'] == 'T-NTP'].sort_values('score', ascending=False)[['uid', 'score']].reset_index(drop=True).iloc[0:39]['uid']
    # objs = tbl.loc[tbl['label'].isin(['T-FP', 'T-EB'])].sort_values('score', ascending=False)[['uid', 'score']].reset_index(drop=True).iloc[0:240]['uid']
    # objs = tbl.loc[tbl['label'].isin(['T-CP', 'T-KP'])].sort_values('score', ascending=True)[['uid', 'score']].reset_index(drop=True).iloc[0:186]['uid']

    def _correct_sector_field(x):
        target_id, tce_id = x.split('-')[:2]
        sector_id = x.split('-')[2:]
        if len(sector_id) == 2:
            return x
        else:
            sector_id = f'{sector_id[0]}-{sector_id[0][1:]}'
            return f'{target_id}-{tce_id}-{sector_id}'

    objs_list = ['142276270-X-SX-X']  # objs.apply(_correct_sector_field)

    download_dir = Path('/Users/msaragoc/Projects/exoplanet_transit_classification/data/dv_reports/mastDownload/TESS/')
    data_products_lst = ['TCE summary report', 'Full DV report', 'TCERT report']
    reports = 'all'   # 'dv_summary', 'dv_report', 'tcert', 'dv', 'all'
    download_products = True
    create_csv = False
    csv_name = 'tic_267574918_tce_num_transits_zero_2-9-2024.csv'
    verbose = True
    n_procs = 12
    n_jobs = 12
    pool = multiprocessing.Pool(processes=n_procs)
    jobs = [(objs_list_job, data_products_lst, download_dir, download_products, reports, verbose)
            for objs_list_job in np.array_split(objs_list, n_jobs)]
    async_results = [pool.apply_async(get_dv_dataproducts_list, job) for job in jobs]
    pool.close()

    if create_csv:
        uris_df = pd.concat([pd.DataFrame(async_result.get()) for async_result in async_results], axis=0,
                            ignore_index=True)
        uris_df.to_csv(download_dir / csv_name, index=False)

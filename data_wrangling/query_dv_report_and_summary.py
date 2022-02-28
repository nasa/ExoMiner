""" Script used to download DV reports and summaries for KICs and TICs from the MAST. """

# 3rd party
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


def get_tic_dv_report_and_summary(tic, download_dir, verbose=False):
    """ Download DV reports and summaries available in the MAST for a given TIC.

    :param tic: int, TIC
    :param download_dir: str, download directory
    :param verbose: bool, verbose
    :return:
        astropy Table with path/URL to downloaded products
    """

    obs_table = Observations.query_criteria(target_name=f'{tic}',
                                            obs_collection='TESS',
                                            obs_id='*-s',
                                            )

    if verbose:
        print(f'Number of observations queried: {len(obs_table)}')

    obs_products = Observations.get_product_list(obs_table)
    obs_products_filter = Observations.filter_products(obs_products, extension='pdf')

    prod = Observations.download_products(obs_products_filter, download_dir=download_dir)

    return prod


if __name__ == "__main__":

    kic_list = [100001645]  # [1028246, 5451336, 11456839]  # [8561063, 3239945, 6933567, 8416523, 9663113]
    download_dir = '/Users/msaragoc/OneDrive - NASA/Projects/exoplanet_transit_classification/data/dv_reports/mastDownload/Kepler/'
    for kic in kic_list:
        get_kic_dv_report_and_summary(kic, download_dir, verbose=False)

    # download_dir = '/Users/msaragoc/OneDrive - NASA/Projects/exoplanet_transit_classification/data/dv_reports/mastDownload/TESS/'
    # run_dir = download_dir + 'self_normalization_wks_2-22-2022/'
    # os.makedirs(run_dir, exist_ok=True)
    #
    # tbl = pd.read_csv('/Users/msaragoc/OneDrive - NASA/Projects/exoplanet_transit_classification/interns/hongbo/kepler_to_tess/merged_scores_2-19-2022.csv')
    # tbl = tbl.loc[(tbl['PC v non-PC, no transit depth, no weak secondary depth, weak secondary self normalization_score'] < 0.5) &
    #               (tbl['original_label'] == 'KP')]
    # tbl = tbl.loc
    #
    # sampled_tbl = tbl.sample(n=20, replace=False)
    # sampled_tbl.to_csv(f'{run_dir}sampled_tbl.csv', index=False)
    # tic_list = sampled_tbl['target_id'].to_numpy()  # []
    # for tic in tic_list:
    #     get_tic_dv_report_and_summary(tic, run_dir, verbose=False)

    # experimental code

    # obs_table = Observations.query_object('KIC 8197761')
    # obs_table = Observations.query_criteria(target_name='kplr012935144',
    #                                         obs_collection='Kepler',
    #                                         # dataproduct_type='timeseries'
    #                                         obs_id='*lc*',
    #                                         )
    # obs_products = Observations.get_product_list(obs_table)
    # obs_products_filter = Observations.filter_products(obs_products, extension='pdf')
    # # data_products = Observations.get_product_list(obs_table)
    #
    # prod = Observations.download_products(obs_products_filter, download_dir='/home/msaragoc/Downloads/')
    #
    # obs_table = Observations.query_criteria(target_name='50365310',
    #                                         obs_collection='TESS',
    #                                         # dataproduct_type='timeseries'
    #                                         obs_id='*-s',
    #                                         )
    # obs_products = Observations.get_product_list(obs_table)
    # obs_products_filter = Observations.filter_products(obs_products, extension='pdf')
    # # data_products = Observations.get_product_list(obs_table)
    #
    # prod = Observations.download_products(obs_products_filter, download_dir='/home/msaragoc/Downloads/')

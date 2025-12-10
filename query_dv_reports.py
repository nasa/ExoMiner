""" Script used to download DV reports and summaries for KICs and TICs from the MAST. """

# 3rd party
import pandas as pd
from astroquery.mast import Observations
import numpy as np
from pathlib import Path
import multiprocessing
from tqdm import tqdm
import time

URL_HEADER = 'https://mast.stsci.edu/api/v0.1/Download/file?uri='

Observations.enable_cloud_dataset()


def inverse_correct_sector_field(x: str) -> str:
    """
    Revert a corrected TCE unique ID back to its original format by collapsing
    the sector run start and end into a single token if possible.

    Example:
        Input:  "123456789-01-S36-S36"
        Output: "123456789-01-S36"

    :param str x: Corrected TCE unique ID
    :return str: Original TCE unique ID format
    """
    parts = x.split('-')
    if len(parts) == 4:
        target_id, tce_id, sector_start, sector_end = parts
        sector_start = sector_start[1:] # Remove 'S' prefix
        # If start and end sectors are the same, collapse them
        if sector_start == sector_end:
            return f"{target_id}-{tce_id}-S{sector_start}"
        else:
            return x  # Keep as is if different sectors
    return x  # Already in original format


def correct_sector_field(x):
    """Set TCE unique ID to match format needed to query the MAST: <target_id>-<tce_id>-<sector_run_start>-<sector_run_end>.

    :param str x: TCE unique ID
    :return str: TCE unique ID with correct sector run format
    """

    target_id, tce_id = x.split('-')[:2]

    sector_id = x.split('-')[2:]
    if len(sector_id) == 2:
        return x
    else:
        sector_id = f'{sector_id[0]}-{sector_id[0][1:]}'
        return f'{target_id}-{tce_id}-{sector_id}'


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


def get_dv_dataproducts(example_id, download_dir, download_products, reports='all', spoc_ffi=False, verbose=False):
    """ Download DV reports and summaries available in the MAST for a given observation, which can be either a target,
        sector run, or TCE.

    :param example_id: str, example identifier. Must follow format {tic_id}-{tce_id}-S{start_sector}-{end_sector}. To
        not condition on a specific field, use `X` as placeholder. E.g. {tic_id}-X-S{start_sector}-{end_sector}.
    :param download_dir: str, download directory
    :param download_products: bool, if True products are downloaded
    :param reports: str, choose which reports to get.
        'dv_summary': downloads only TCE DV summary reports.
        'dv_report': downloads only full DV reports.
        'dv_mini_report': downloads only mini-reports.
        'all': downloads all DV reports.
    :param spoc_ffi: bool, if True it gets results from HLSP TESS SPOC FFI
    :param verbose: bool, verbose
    :return:
        prods, list of astropy Tables with path/URL to downloaded products
        uris, list of data products URIs
    """

    prods = []
    uris = {'DV TCE summary report': [], 'Full DV report': [], 'DV mini-report': []}

    if verbose:
        print(f'Started run for example {example_id}...')

    # maps option to name in MAST table for the data products reports
    report_type_map = {
        'all': ['TCE summary report', 'full data validation report', 'Data validation mini report'],
        'dv_summary': ['TCE summary report'],
        'dv_report': ['full data validation report'],
        'dv_mini_report': ['Data validation mini report'],
    }
    reports_to_get = report_type_map[reports]

    tic_id, tce_id, s_sector, e_sector = '.' * 12, '[0-9][0-9]' * 2, '.' * 4, '.' * 4

    ids = example_id.split('-')

    tic_id = ids[0].zfill(16) if ids[0] != 'X' else tic_id
    tce_id = ids[1].zfill(2) if ids[1] != 'X' else tce_id
    s_sector = ids[2][1:] if ids[2][1:] != 'X' else s_sector
    e_sector = ids[3] if ids[3] != 'X' else e_sector

    s_sector, e_sector = s_sector.zfill(4), e_sector.zfill(4)

    if not spoc_ffi:
        product_tce_id = f's{s_sector}-s{e_sector}-{tic_id}-{tce_id}'
        product_target_id = f's{s_sector}-s{e_sector}-{tic_id}'
    else:
        # product_tce_id = f'{tic_id}-s{s_sector}-s{e_sector}*{tce_id}'
        product_tce_id = f'{tic_id}-s{s_sector}-s{e_sector}.*-{tce_id}'
        product_target_id = f'{tic_id}-s{s_sector}-s{e_sector}'

    #  get table of observations associated with this target
    obs_table = Observations.query_criteria(target_name=int(tic_id),
                                            obs_collection='TESS' if not spoc_ffi else 'HLSP',
                                            obs_id='*' if not spoc_ffi else 'hlsp_tess-spoc*',
                                            )

    if verbose:
        print(f'Number of observations queried: {len(obs_table)}')

    if len(obs_table) == 0:
        return prods, uris

    # get list of products available for this target's observations
    obs_products = Observations.get_product_list(obs_table)

    products_filenames = obs_products.to_pandas()['productFilename']

    if 'TCE summary report' in reports_to_get:  # TCE DV summary reports

        # filter products based on TCE ID provided, and type of report
        filtered_products_filenames = products_filenames.loc[products_filenames.str.contains(product_tce_id,
                                                                                             regex=True)].to_list()

        obs_products_filter = Observations.filter_products(obs_products,
                                                           extension='pdf',
                                                           description='TCE summary report' if not spoc_ffi else 'PDF',
                                                           productFilename=filtered_products_filenames)

        # download selected products
        if download_products:
            prod_tce = Observations.download_products(obs_products_filter, download_dir=download_dir)
            prods.append(prod_tce)
        if len(obs_products_filter) > 0:
            uris['DV TCE summary report'] = obs_products_filter['dataURI'].tolist()
        else:
            print(f'No DV TCE summary report found for {example_id}.')

    if 'full data validation report' in reports_to_get:  # DV full reports

        filtered_products_filenames = products_filenames.loc[products_filenames.str.contains(product_target_id,
                                                                                             regex=True)].to_list()
        filtered_products_filenames = [fn for fn in filtered_products_filenames if 'dvr.pdf' in fn]

        obs_products_filter = Observations.filter_products(obs_products,
                                                           extension='pdf',
                                                           description='full data validation report'
                                                           if not spoc_ffi else 'Informational PDF',
                                                           productFilename=filtered_products_filenames)
        if len(obs_products_filter) > 0:
            uris['Full DV report'] = obs_products_filter['dataURI'].tolist()
        else:
            print(f'No DV full report found for {example_id}.')

        # download selected products
        if download_products:
            prod_dv_full = Observations.download_products(obs_products_filter, download_dir=download_dir)
            prods.append(prod_dv_full)

    if 'Data validation mini report' in reports_to_get:  # DV mini-reports

        filtered_products_filenames = products_filenames.loc[products_filenames.str.contains(product_target_id,
                                                                                             regex=True)].to_list()
        filtered_products_filenames = [fn for fn in filtered_products_filenames if 'dvm.pdf' in fn]


        obs_products_filter = Observations.filter_products(obs_products,
                                                           extension='pdf',
                                                           description='Data validation mini report' if not spoc_ffi
                                                           else 'PDF',
                                                           productFilename=filtered_products_filenames)
        if len(obs_products_filter) > 0:
            uris['DV mini-report'] = obs_products_filter['dataURI'].tolist()
        else:
            print(f'No DV mini-report found for {example_id}.')

        # download selected products
        if download_products:
            prod_tcert = Observations.download_products(obs_products_filter, download_dir=download_dir)
            prods.append(prod_tcert)

    return prods, uris


def get_dv_dataproducts_list(objs_list, data_products_lst, download_dir, download_products, reports, spoc_ffi=False,
                             verbose=True, create_mast_url_csv=False, get_most_recent_products=True, job_id=0):
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
            'dv_mini_report': downloads only DV mini-reports.
            'all': downloads DV reports.
        spoc_ffi: bool, if True it gets results from HLSP TESS SPOC FFI
        verbose: bool, verbose
        create_mast_url_csv: bool, set to True to write MAST URLs for DV SPOC data products for the queried objects
        get_most_recent_products: bool, if True get only most recent products available (i.e., from the latest SPOC run)
        job_id: int, job ID for logging purposes
    """

    # Define retry parameters
    MAX_RETRIES = 3
    RETRY_DELAY = 5 # seconds

    uris_dict = {'uid': []}
    uris_dict.update({field: [] for field in data_products_lst})
    for obj in tqdm(objs_list, desc=f'[Job ID {job_id}] Getting data products for object', unit='obj', total=len(objs_list)):

        success = False
        last_error = None

        for attempt in range(MAX_RETRIES):
            try:
                _, uris = get_dv_dataproducts(obj, str(download_dir), download_products, reports, spoc_ffi, verbose)
                success = True
                break # Exit retry loop on success
            except Exception as e:
                last_error = e
                # Check if it is a network error worth retrying
                if "RemoteDisconnected" in str(e) or "Connection aborted" in str(e) or "ReadTimeout" in str(e):
                    if verbose:
                        print(f"Connection dropped for {obj} (Attempt {attempt+1}/{MAX_RETRIES}). Retrying in {RETRY_DELAY}s...")
                    time.sleep(RETRY_DELAY)
                else:
                    # If it's a logic error (not network), break immediately
                    break
        
        if not success:
            print(f"Error processing {obj} after {MAX_RETRIES} attempts: {str(last_error)}")
            continue # Skip to next object

        try:
            n_tces = len(uris[data_products_lst[0]])

            if n_tces == 0:
                if verbose:
                    print(f'No data products found for {obj}.')
                continue

            uris_dict['uid'] += [obj] * n_tces
            for tce_i in range(n_tces):
                for field in data_products_lst:
                    if uris[field]:
                        uris_dict[field].append(URL_HEADER + uris[field][tce_i] if uris[field][tce_i] != '' else '')
                    else:
                        uris_dict[field].append('')

        except Exception as e:
            print(f"Error parsing results for {obj}: {str(e)}")

    if create_mast_url_csv:
        mast_urls_tbls_dir = download_dir / 'mast_urls_tables'
        mast_urls_tbls_dir.mkdir(parents=True, exist_ok=True)
        tbl_fp = mast_urls_tbls_dir / f'spoc-dv_mast-urls_job{job_id}.csv'
        print(f'[Job ID {job_id}] Writing data products URIs for {len(uris_dict)}/{len(objs_list)} events to {str(tbl_fp)}...')
        uris_df = pd.DataFrame(uris_dict)
        uris_df.to_csv(tbl_fp, index=False)


def main(objs_list, download_dir, data_products_lst, reports, mission, spoc_ffi, download_products=True, verbose=True, 
         get_most_recent_products=True, create_mast_url_csv=False, n_procs=1, n_jobs=1):
    """Main function used to query the MAST for SPOC DV reports and summaries for a list of KICs or TICs. Objects should have format {target_id}-{tce_id}-S{start_sector}-{end_sector}.

    :param list objs_list: objects to be queried
    :param Path download_dir: results download directory
    :param list data_products_lst: list with SPOC data products to be queried
    :param list reports: list with names of reports
    :param str mission: mission, either 'kepler' or 'tess'
    :param bool spoc_ffi: set to True to query HLSP TESS SPOC FFI
    :param bool download_products: set to True to download data products, defaults to True
    :param bool verbose: set to True for verbose printed to stdout console, defaults to True
    :param bool get_most_recent_products: only get data products for the most recent SPOC runs for a given sector run, defaults to True
    :param bool create_mast_url_csv: set to True to write MAST URLs for DV SPOC data products for the queried objects, defaults to False
    :param int n_procs: number of processes to use for parallelization, defaults to 1
    :param int n_jobs: number of jobs to use for parallelization, defaults to 1
    """
    
    print(f'Found {len(objs_list)} events. Downloading DV reports...')

    download_dir.mkdir(parents=True, exist_ok=True)
    
    if create_mast_url_csv:
        mast_urls_tbls_dir = download_dir / 'mast_urls_tables'
        mast_urls_tbls_dir.mkdir(parents=True, exist_ok=True)
    
    if mission == 'kepler':
        
        jobs = [(obj, download_dir, verbose) for obj in objs_list]

        n_jobs = len(jobs)
        print(f'Split work into {n_jobs} jobs.')
        
        # parallelize jobs
        with multiprocessing.Pool(processes=n_procs) as pool:
            async_results = [pool.apply_async(get_kic_dv_report_and_summary, job) for job in jobs]
            _ = [r.get() for r in async_results]  # collect results
            
    elif mission == 'tess':
        
        objs_list_jobs = np.array_split(objs_list, n_jobs)

        # split in n_jobs
        jobs = [(objs_list_job, data_products_lst, download_dir, download_products, reports, spoc_ffi, verbose, create_mast_url_csv,
                get_most_recent_products, job_i)
                for job_i, objs_list_job in enumerate(objs_list_jobs)]
        # split per sector run
        # jobs = [(objs_list_job, data_products_lst, download_dir, download_products, reports, spoc_ffi, verbose,
        #          download_dir / f'{download_dir.stem}_sector_run_{sector_run}.csv')
        #         for sector_run, objs_list_job in objs_list_jobs.items()
        #         if not (download_dir / f'{download_dir.stem}_sector_run_{sector_run}.csv').exists()]
        
        n_jobs = len(jobs)
        print(f'Split work into {n_jobs} jobs.')
        
        # parallelize jobs
        with multiprocessing.Pool(processes=n_procs) as pool:
            async_results = [pool.apply_async(get_dv_dataproducts_list, job) for job in jobs]
            _ = [r.get() for r in async_results]  # collect results
        
        if create_mast_url_csv:
            mast_url_tbls_fps = list(mast_urls_tbls_dir.glob('spoc-dv_mast-urls_job*.csv'))
            if len(mast_url_tbls_fps) > 1:
                print('Aggregating results...')
                mast_url_tbl_agg = pd.concat([pd.read_csv(mast_url_tbl_fp) for mast_url_tbl_fp in mast_url_tbls_fps], axis=0)
                mast_url_tbl_agg.to_csv(download_dir / f'spoc-dv_mast-urls_jobs-agg.csv', index=False)
        
    print(f'Finished querying MAST for DV reports.')
    
    
if __name__ == "__main__":
    
    # set parameters
    # download_dir = Path('/Users/msaragoc/Projects/exoplanet_transit_classification/data/dv_reports/TESS/check_top_ntps_tess_ffi_paper_11-11-2025_1608')
    download_dir = Path('/Users/miguelmartinho/Projects/tess_spoc_ffi_exominer_paper/mast_urls_dv_reports_spoc_ffi_12-10-2025_0817')
    data_products_lst = ['DV TCE summary report', 'Full DV report', 'DV mini-report']
    reports = 'all'   # 'dv_summary', 'dv_report', 'dv_mini_report', 'all'
    download_products = False  # if True, products are downloaded
    verbose = False
    get_most_recent_products = True
    spoc_ffi = True
    create_mast_url_csv = True
    n_procs = 4
    n_jobs = 300
    mission = 'tess'
    
    # get objects from table
    # objs = pd.read_csv('/data3/exoplnt_dl/ephemeris_tables/tess/tess_spoc_2min/tess-spoc-2min-tces-dv_s1-s94_s1s92_9-19-2025_1518.csv', usecols=['uid', 'DV mini-report'])
    # objs = objs.loc[objs['DV mini-report'].isna()]
    objs = pd.read_csv('/Users/miguelmartinho/Projects/tess_spoc_ffi_exominer_paper/tess-spoc-ffi-tces-dv_s36-s72_multisector-s56s69_10-8-2025_exofop-sg1-tois9-11-2025_fixedtointps_nontpstois.csv', comment='#', usecols=['uid'])
    print(f'Total number of objects in input table: {len(objs)}')

    # convert uid to format required for query
    objs['uid'] = objs['uid'].apply(correct_sector_field)
    
    filt_dir = Path('/Users/miguelmartinho/Projects/tess_spoc_ffi_exominer_paper/')
    filt_tbl = pd.concat([pd.read_csv(fp) for fp in filt_dir.rglob('spoc-dv_mast-urls_job*.csv')], axis=0)
    objs = objs.loc[~objs['uid'].isin(filt_tbl['uid'])]
    objs = objs.loc[~objs['uid'].str.contains('56-69')]
    
    print(f'Number of objects to query: {len(objs)}')
    
    # run with a few objects
    # objs = pd.DataFrame({'uid': [
    #     '123213412-1-S36-36',
    # ]})
    
    # split per sector run
    # objs_list_jobs = {sector_run: np.array(objs_in_sector_run['uid']) for sector_run, objs_in_sector_run in
    #                    objs.groupby('sector_run')}
    # print(f'Number of jobs set by number of sector runs: n_jobs={len(objs_list_jobs)}')
    # n_jobs = len(objs_list_jobs)

    # convert to list or NumPy array
    objs_list = np.array(objs['uid'])
    
    main(objs_list, download_dir, data_products_lst, reports, mission, spoc_ffi, download_products, verbose, get_most_recent_products, create_mast_url_csv, n_procs, n_jobs)    

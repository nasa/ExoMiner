"""
Extract RUWE values for TICs from Gaia data releases.

This works by querying the TIC catalog for a set of TIC IDs. From the TIC catalog, we get the Gaia DR2 source ids that
can be used to query Gaia DR2 catalog and extract RUWE values for that set of TIC IDs. The input table needs to have column named 'target_id' with 
valid TIC IDs to be queried.
"""

# 3rd party
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import logging
from astroquery.mast import Catalogs

# local
from src_preprocessing.tce_tables.ruwe.query_gaia import query_gaia, pick_best_per_dr2


def query_gaiadr_for_ruwe(tce_tbl, res_dir, ruwe_source='gaiadr2', plot=False):
    """ Query Gaia DR `ruwe_source` catalog to extract RUWE values for TICs in TCE table in `tce_tbl`. `tce_tbl` must
    contain column 'target_id' which stores the TIC ID of the targets associated with the TCEs.

    :param tce_tbl: pandas DataFrame, TCE table
    :param res_dir: Path, save directory for query results
    :param plot: bool, if True, plot the histogram of RUWE values for the TICs
    :param ruwe_source: str/Path, the RUWE source to use for the queried TICs.

    :return: tce_tbl, pandas DataFrame of TCE table with RUWE values for TICs ('ruwe' column)
    """

    output_fp = res_dir / f'{ruwe_source}.csv'  # save results from query to csv file

    # set logger
    logger = logging.getLogger(name='ruwe')  # set up logger
    logger_handler = logging.FileHandler(filename=res_dir / f'ruwe_run.log', mode='w')
    logger_formatter = logging.Formatter('%(asctime)s - %(message)s')
    logger.setLevel(logging.INFO)
    logger_handler.setFormatter(logger_formatter)
    logger.addHandler(logger_handler)
    logger.info(f'Starting run...')

    # get unique list of TICs
    tic_ids = tce_tbl['target_id'].unique()

    if ruwe_source in ['gaiadr2', 'gaiaedr3', 'gaiadr3']:
        try:
            # get Gaia DR source ids from tic ids in the tce table
            logger.info('Getting Gaia DR2 source ids from TIC (TIC v8 is based on Gaia DR2)')
            catalog_data = Catalogs.query_criteria(catalog='TIC', ID=tic_ids.tolist()).to_pandas()
            catalog_data.to_csv(res_dir / 'tics.csv', index=False)

            source_ids_tic_tbl = catalog_data[['ID', 'GAIA']].rename(columns={'GAIA': 'source_id', 'ID': 'target_id'})

            # remove TICs with missing source id
            logger.info('Removing TICs with missing Gaia source ID')
            source_ids_tic_tbl = source_ids_tic_tbl.loc[~source_ids_tic_tbl['source_id'].isna()]

            source_ids_tic_tbl = source_ids_tic_tbl.astype(dtype={'source_id': np.int64, 'target_id': np.int64})

            # count number of occurrences for each source id
            source_ids_cnts = \
                source_ids_tic_tbl['source_id'].value_counts().to_frame('n_occur_source_ids').reset_index().rename(
                    columns={'index': 'source_id'})
            source_ids_tic_tbl = source_ids_tic_tbl.merge(source_ids_cnts,
                                                          on=['source_id'],
                                                          how='left',
                                                          validate='many_to_one')

            # query Gaia DR
            query_gaia(source_ids_tic_tbl[['source_id']], ruwe_source, res_dir, logger=logger)
            logger.info(f'Query finished and results saved to {str(output_fp)}.')

            ruwe_tbl = pd.read_csv(output_fp)
            
            if ruwe_source in ['gaiaedr3', 'gaiadr3']:
                
                # collapse multiple matches per DR2 source
                best_df = pick_best_per_dr2(ruwe_tbl, logger=logger)

                # merge back to the TIC linkage on DR2 id
                ruwe_tbl = (source_ids_tic_tbl[['source_id']]
                            .merge(best_df.rename(columns={'dr2_source_id': 'source_id'}),
                                on='source_id', how='left', validate='many_to_one'))
                
                # check how many TIC targets failed to map to any (E)DR3 RUWE?
                n_missing = ruwe_tbl['ruwe'].isna().sum()
                logger.info(f"(E)DR3 RUWE missing for {n_missing} of {len(ruwe_tbl)} TIC targets after cross-match filtering.")
                
            # in case there are duplicate source ids in the TIC-source id table, many_to_one
            if (source_ids_tic_tbl["source_id"].value_counts() > 1).sum() > 0:
                ruwe_tbl = source_ids_tic_tbl[['target_id', 'source_id', 'n_occur_source_ids']].merge(
                    ruwe_tbl.drop_duplicates('source_id'),
                    on=['source_id'],
                    how='left',
                    validate='many_to_one')
            else:
                ruwe_tbl = source_ids_tic_tbl[['target_id', 'source_id', 'n_occur_source_ids']].merge(
                    ruwe_tbl.drop_duplicates('source_id'),
                    on=['source_id'],
                    how='left',
                    validate='one_to_one')
        except Exception as e:
            logger.exception(f'Error occurred while querying Gaia DR2 table: {str(e)}\nSetting RUWE to missing value.')
            ruwe_tbl = pd.DataFrame({'target_id': tic_ids, 'source_id': np.nan * np.ones(len(tic_ids)),
                                     'n_occur_source_ids': np.nan * np.ones(len(tic_ids)),
                                     'ruwe': np.nan * np.ones(len(tic_ids))})
    
    elif ruwe_source == 'unavailable':
        logger.info('No RUWE catalog/source. Setting RUWE to queried TICs to missing values.')
        ruwe_tbl = pd.DataFrame({'target_id': tic_ids, 'ruwe': np.nan * np.ones(len(tic_ids))})
    elif isinstance(ruwe_source, Path):
        logger.info(f'Using RUWE catalog/source in {ruwe_source}.')
        ruwe_tbl = pd.read_csv(ruwe_source)
        # filter RUWE table for the queried TICs
        ruwe_tbl = ruwe_tbl.loc[ruwe_tbl['target_id'].isin(tic_ids)]
    else:
        raise ValueError(f'Unsupported RUWE source: {ruwe_source}.')

    ruwe_tbl.to_csv(res_dir / f'{output_fp.name}_with_ticid.csv', index=False)
    logger.info(f'Number of TICs without RUWE value: {ruwe_tbl["ruwe"].isna().sum()} ouf of {len(ruwe_tbl)}')

    if plot:
        # plot histogram of RUWE values
        bins = np.linspace(0, 2, 100)  # np.logspace(0, 2, 100)

        f, ax = plt.subplots()
        ax.hist(ruwe_tbl['ruwe'], bins=bins, edgecolor='k')
        ax.set_xlabel('RUWE')
        ax.set_ylabel('Target counts')
        # ax.legend()
        ax.set_title(f'RUWE source: {str(ruwe_source)}')
        ax.set_yscale('log')
        ax.set_xlim([0, 2])
        # ax.set_xscale('log')
        # plt.show()
        f.savefig(res_dir / 'hist_ruwe_tics.png')

    tce_tbl = tce_tbl.merge(ruwe_tbl[['target_id', 'ruwe']], on='target_id', how='left', validate='many_to_one')
    logger.info(f'Number of TCEs without RUWE value: {tce_tbl["ruwe"].isna().sum()} ouf of {len(tce_tbl)}')

    return tce_tbl


if __name__ == '__main__':

    res_dir = Path('/home6/msaragoc/work_dir/Kepler-TESS_exoplanet/data/Ephemeris_tables/TESS/tess_spoc_2min/tess_spoc_2min_tces_dv_s89-s98_s1s92_2-13-2026_1010/gaiadr3_tics_ruwe')
    res_dir.mkdir(exist_ok=True)

    # file path to TCE table
    tce_tbl_fp = Path('/home6/msaragoc/work_dir/Kepler-TESS_exoplanet/data/Ephemeris_tables/TESS/tess_spoc_2min/tess_spoc_2min_tces_dv_s89-s98_s1s92_2-13-2026_1010/tess_spoc_2min_tces_dv_s89-s98_s1s92_2-13-2026_1010_stellartic8_ruwegaiadr2_preproc.csv')
    # load TCE table
    tce_tbl = pd.read_csv(tce_tbl_fp)

    tce_tbl = query_gaiadr_for_ruwe(tce_tbl, res_dir, ruwe_source='gaiadr3')

    tce_tbl.to_csv(tce_tbl_fp.parent / f'{tce_tbl_fp.stem}_ruwe-gaiadr3.csv', index=False)

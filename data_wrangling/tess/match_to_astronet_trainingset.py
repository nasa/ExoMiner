""" Matching TESS TCEs to objects from the Astronet training set. """

# 3rd party
import pandas as pd
from pathlib import Path
import numpy as np
import multiprocessing
from datetime import datetime
import logging

# local
from data_wrangling.tess.match_to_astronet_trainingset_utils import match_tces_to_astronet_traininset

if __name__ == "__main__":
    res_dir = Path(f'/home/msaragoc/Projects/Kepler-TESS_exoplanet/Analysis/'
                   f'tess_tce_astronet_match/{datetime.now().strftime("%m-%d-%Y_%H%M")}')
    res_dir.mkdir(exist_ok=True)

    logger = logging.getLogger(name=f'match_tces_to_astronet_main')
    logger_handler = logging.FileHandler(filename=res_dir / f'match_tces_to_astronet_main.log', mode='w')
    logger_formatter = logging.Formatter('%(asctime)s - %(message)s')
    logger.setLevel(logging.INFO)
    logger_handler.setFormatter(logger_formatter)
    logger.addHandler(logger_handler)
    logger.info(f'Starting matching between TCEs and Astronet training set objects in {str(res_dir)}')

    sampling_interval = 0.00001  # approximately 1 min to build ephemerides templates
    logger.info(f'Using sampling interval: {sampling_interval}')

    # load TCE table
    tce_tbl_fp = Path('/data5/tess_project/Data/Ephemeris_tables/TESS/DV_SPOC_mat_files/11-29-2021/'
                      'tess_tces_s1-s40_11-23-2021_1409_stellarparams_updated_eb_tso_tec_label_modelchisqr.csv')
    tce_tbl_cols = ['target_id', 'tce_plnt_num', 'sector_run', 'tce_period', 'tce_time0bk', 'tce_duration',
                    'match_dist', 'TFOPWG Disposition', 'TESS Disposition', 'TOI']
    tce_tbl = pd.read_csv(tce_tbl_fp)[tce_tbl_cols]
    logger.info(f'Using TCE table ({len(tce_tbl)} TCEs): {str(tce_tbl_fp)}')
    # tce_tbl.to_csv(res_dir / tce_tbl_fp.name, index=False)

    # load EB table
    astronet_tbl_cols = ['tic', 'astronet_period', 'astronet_epoch', 'astronet_sector', 'astronet_label']
    astronet_tbl_fp = Path('/data5/tess_project/Data/Ephemeris_tables/TESS/astronet/astronet_training.csv')
    astronet_tbl = pd.read_csv(astronet_tbl_fp)
    astronet_tbl = astronet_tbl.rename(columns={'period': 'astronet_period',
                                                'epoch': 'astronet_epoch',
                                                'duration': 'astronet_duration',
                                                'sector': 'astronet_sector',
                                                'label': 'astronet_label'})[astronet_tbl_cols]
    logger.info(f'Using Astronet training set ({len(astronet_tbl)} objects): {str(astronet_tbl_fp)}')
    # for col in ['period', 'bjd0']:   # filter EBs without ephemerides
    #     eb_tbl = eb_tbl.loc[eb_tbl[col] != 'None']
    # eb_tbl = eb_tbl.astype(dtype={'period': float, 'bjd0': float})
    # eb_tbl.to_csv(res_dir / 'eb_tbl.csv', index=False)

    matching_tbl = tce_tbl.copy(deep=True)

    # add columns for data from Astronet training set table
    cols_add_to_matching_tbl = astronet_tbl_cols[1:]  # + ['match_dist_astronet']
    matching_tbl = pd.concat([matching_tbl.reset_index(drop=True),
                              pd.DataFrame(data=np.nan * np.zeros((len(matching_tbl), len(cols_add_to_matching_tbl))),
                                           columns=cols_add_to_matching_tbl)], axis=1)

    # new columns
    matching_tbl['match_dist_astronet'] = 2

    # perform candidate matching between each TCE and Astronet training set objects
    logger.info(f'Starting candidate matching between TCEs and Astronet training set objects...')
    n_processes = 11
    tbl_jobs = np.array_split(matching_tbl, n_processes)
    tbl_jobs = [tbl_job.reset_index(drop=True) for tbl_job in tbl_jobs if len(tbl_job) > 0]
    n_processes = len(tbl_jobs)
    logger.info(f'Setting number of processes to {n_processes}')
    pool = multiprocessing.Pool(processes=n_processes)
    jobs = [(tbl_job,) + (astronet_tbl, sampling_interval, cols_add_to_matching_tbl, res_dir)
            for tbl_job_i, tbl_job in enumerate(tbl_jobs)]
    async_results = [pool.apply_async(match_tces_to_astronet_traininset, job) for job in jobs]
    pool.close()

    # time.sleep(5)
    _ = [match_tbl_job.get() for match_tbl_job in async_results]
    # matching_tbl = pd.concat([match_tbl_job.get() for match_tbl_job in async_results], axis=0)
    matching_tbl = pd.concat([pd.read_csv(fp) for fp in res_dir.iterdir() if fp.stem.startswith('matching_tbl_')],
                             axis=0).reset_index()
    candidate_matching_tbl_fp = res_dir / 'matching_candidates_tbl.csv'
    matching_tbl.to_csv(candidate_matching_tbl_fp, index=False)
    logger.info(f'Finished candidate matching between TCEs and Astronet training set objects: '
                f'{str(candidate_matching_tbl_fp)}')

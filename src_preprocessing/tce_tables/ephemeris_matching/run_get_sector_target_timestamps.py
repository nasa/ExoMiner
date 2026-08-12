"""
Get start and end timestamps for each TIC observed in each sector run using multiprocessing.
"""

# 3rd party
from pathlib import Path
import pandas as pd
import multiprocessing
import argparse
from tqdm import tqdm
import logging
from datetime import datetime

# local
from src_preprocessing.tce_tables.ephemeris_matching.get_start_end_timestamps_sector_runs import \
    get_start_end_timestamps_tics_sector_runs

logger = logging.getLogger(__name__)


def parse_args():
    """Parse arguments."""
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, help='Output directory', default=None)
    parser.add_argument('--lc_dir', type=str, help='Path to root directory containing the light curve FITS files.')
    parser.add_argument('--n_procs', type=int, help='Number of processes to use for parallelization', default=1)
    parser.add_argument('--data_collection_mode', type=str, help='Either `2min` or `ffi`.')
    
    args = parser.parse_args()
    
    return args


def create_agg_table(res_dir):
    """Aggregate into a single table.
    
        :param res_dir: results directory
    """
    
    target_sector_run_timestamps_all = pd.concat([pd.read_csv(fp) for fp in res_dir.iterdir()], axis=0)
    
    target_sector_run_timestamps_all.sort_values(by=['sector', 'target'], inplace=True)
    
    target_sector_run_timestamps_all.to_csv(res_dir / f'{res_dir.name}.csv', index=False)

    
def main():
    """Run job to get light curve target timestamps."""
    
    args = parse_args()
    
    # directory used to save start/end timestamps target tables for each sector run
    res_dir = Path(args.output_dir)

    # lightcurve root directory for the target data of interest from where to get the timestamps    
    lc_root_dir = Path(args.lc_dir)
    
    n_procs = args.n_procs  # number of parallel processes to spawn
    
    res_dir.mkdir(exist_ok=True)
    
    # create logger
    logging.basicConfig(filename=res_dir / f'run_{datetime.now().strftime("%m-%d-%Y_%H%M")}.log',
                        level=logging.INFO,
                        format='%(asctime)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        filemode='w',
                        )

    # 2min data
    if args.data_collection_mode == '2min':
        sector_dirs_fps = [fp for fp in lc_root_dir.iterdir() if fp.name.startswith('sector_')]
    elif args.data_collection_mode == 'ffi':
        # ffi data
        sector_dirs_fps = [fp for fp in lc_root_dir.iterdir() if fp.name.startswith('s')]
    else:
        raise ValueError(f'Data collection mode {args.data_collection_mode} not recognized. Must be one of: `2min`, `ffi`.')

    jobs = [([sector_dir_fp], res_dir) for sector_dir_fp in sector_dirs_fps]
    
    logger.info(f'Extracting start/end timestamps for targets in {len(sector_dirs_fps)} sector runs.')

    with multiprocessing.Pool(processes=n_procs) as pool, tqdm(total=len(jobs), desc='Get targets light curve timestamps in sector', unit='sectors') as pbar:
    
        def on_end(_):
            pbar.update(1)
            
        async_results = []
        for job in jobs:
            
            ar = pool.apply_async(
                get_start_end_timestamps_tics_sector_runs, 
                job,
                callback=on_end,
                ) 
            async_results.append(ar)
        
        # propagate exceptions and dealing with them if needed
        for r in async_results:
            _ = r.get()

    logger.info(f'Aggregating start/end timestamps target lc tables...')
    # aggregate tables into a single table
    create_agg_table(res_dir)
    
    logger.info('Finished.')
    
    
if __name__ == '__main__':

    # # directory used to save start/end timestamps target tables for each sector run
    # res_dir = Path('/nobackupp19/msaragoc/work_dir/Kepler-TESS_exoplanet/experiments/ephemeris_matching/tess-spoc-2min_start-end-timestamps_tics-lc_tces_s89-s98_s1s92_3-4-2026_1132')
    # # lightcurve root directory for the target data of interest from where to get the timestamps
    # lc_root_dir = Path('/u/msaragoc/work_dir/Kepler-TESS_exoplanet/data/FITS_files/TESS/spoc_2min/lc/sectors')
    # n_procs = 36  # number of parallel processes to spawn
    # --output_dir=... 
    
    main()


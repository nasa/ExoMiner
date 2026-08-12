"""
Extracting difference image data from the TESS DV XML files.

--- Output Structure ---

extracted dictionary

     - tce_uid '12345678-1-S1' (e.g.)
        - image_number list of sectors/quarters with available data
        - quality_metric: list of dicts for quality metric in each sector/quarter, each dict with keys 'value', 'valid', and 'attempted'
        - mag: target TMag
        - image_data: list of NumPy arrays [height, width, img_type, value/uncertainty], where img_type channel is in-transit, out-of-transit, difference image, and "SNR"
        - target_ref_centroid: list of dicts that contain the value and uncertainty for the reference coordinates of the target star in the pixel domain in each observed sector/quarter
            - {row: {value: X, uncertainty: Y}, col: {value: X, uncertainty: Y}}
            - ...
        - neighbor_data: dict of dicts for each sector/quarter with neighbors data; each sector/quarter dict maps the TIC ID of a neighbor to a dictionary with the column 'col_px' and row 'row_px' coordinates of these
            objects in the CCD pixel frame of the target star along with the corresponding magnitude 'TMag' and distance to the target in arcseconds 'dst_arcsec'.
            - sectorX
                - neighbor_1
                    - ra
                    - dec
                    - target_id
                    - col_px
                    - row_px
                    - Tmag
                    - flux/transit_depth ratio (flux ratio between neighbor and target / transit depth): flux_n / flux_t / tce_depth
                - ...
            - ...
"""

# 3rd party
from pathlib import Path
import multiprocessing
import logging
import yaml
from tqdm import tqdm
import argparse

# local
from src_preprocessing.diff_img.extracting.utils_diff_img import get_data_from_tess_dv_xml_main

def extract_main(dv_xml_runs: list[Path],
                 neighbors_dir: Path,
                 lc_dir: Path,
                 run_dir: Path,
                 plot_prob: float = 0.1,
                 n_processes: int = 4,
                 append_data : bool =False,
                 cache_neighbors_data: bool =False,
                 data_collection_mode: str ='2min'):
    """Extract difference image data from TESS DV XML files.

    :param list[Path] dv_xml_runs: list of sector runs directories containing DV XML files
    :param Path neighbors_dir: neighbors information directory
    :param Path lc_dir: light curves directory
    :param Path run_dir: run directory
    :param float plot_prob: probability to generate figure of extracted data for a single sector, defaults to 0.1
    :param int n_processes: number of processes used, defaults to 4
    :param bool append_data: appends data to existing data dictionary in `save_dir` with filename tess_diffimg_<dv_xml_run.name>.npy. 
        If results already exist for target, then extraction from DV XML file is skipped. Defaults to False.
    :param bool cache_neighbors_data: if True, it will read the neighbors table once (per-process) and cache it 
        (up to maximum of `maxsize` tables - see function `_load_sector_df` decorator). Defaults to False.
    :param str data_collection_mode: either '2min' or 'ffi'. Required when extracting neighbors data
    """

    # create run directory
    run_dir.mkdir(exist_ok=True, parents=True)
    # setting up data directory
    data_dir = run_dir / 'data'
    data_dir.mkdir(exist_ok=True)
    # create plotting directory
    plot_dir = run_dir / 'plots'
    plot_dir.mkdir(exist_ok=True)
    # create log directory
    log_dir = run_dir / 'logs'
    log_dir.mkdir(exist_ok=True)

    # set up logger
    logger = logging.getLogger()
    logger_handler = logging.FileHandler(filename=log_dir / f'extract_img_data_from_tess_dv_xml_main.log', mode='a')
    logger_formatter = logging.Formatter('%(asctime)s - %(message)s')
    logger.setLevel(logging.INFO)
    logger_handler.setFormatter(logger_formatter)
    logger.addHandler(logger_handler)
    logger.info(f'Starting preprocessing run...')

    # # check if NumPy file for a given sector run already exists in the run directory
    # dv_xml_runs_res_found = [dv_xml_run for dv_xml_run in dv_xml_runs
    #                          if (run_dir / 'data' / f'tess_diffimg_{dv_xml_run.name}.npy').exists()]
    # logger.info(f'Found NumPy files for the following sector runs in {run_dir}:\n {dv_xml_runs_res_found}\n Skipping '
    #             f'those sector runs...')
    # dv_xml_runs = [dv_xml_run for dv_xml_run in dv_xml_runs if dv_xml_run not in dv_xml_runs_res_found]

    logger.info(f'Number of runs: {len(dv_xml_runs)}')
    logger.info(f'Runs set for preprocessing:')
    for dv_xml_run in dv_xml_runs:
        logger.info(f'Run {str(dv_xml_run)}')

    jobs = [(dv_xml_run, data_dir, neighbors_dir, lc_dir, plot_dir, plot_prob, log_dir, job_i, False, None, 
             append_data, cache_neighbors_data, data_collection_mode)
            for job_i, dv_xml_run in enumerate(dv_xml_runs)]
    n_jobs = len(jobs)
    logger.info(f'Setting {len(jobs)} job(s).')
    logger.info('Started running job(s).')

    if n_processes > 1:
        n_processes = min(n_processes, n_jobs)
        logger.info(f'Using {n_processes} processes...')
        with multiprocessing.Pool(processes=n_processes) as pool:
            with tqdm(desc='Sector Run Job', total=len(jobs), unit='job') as pbar:
                
                def _update_progress(_):
                    pbar.update(1)
                    
                async_results = [pool.apply_async(get_data_from_tess_dv_xml_main, job, callback=_update_progress) for job in jobs]
                for res_i, res in enumerate(async_results):
                    res.get()
                    logger.info(f'Finished job {res_i + 1} out of {n_jobs}.')
        
    else:
        for job in tqdm(jobs, desc='Sector Run Job', total=len(jobs), unit='job'):
            print('Starting job')
            get_data_from_tess_dv_xml_main(*job)

    logger.info('Finished extracting difference image data from DV xml files.')
    

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_fp', type=str, help='Filepath to configuration YAML', required=True)
    args = parser.parse_args()
    
    # load config yaml file
    config_fp = Path(args.config_fp)
    with open(config_fp, 'r') as f:
        config = yaml.safe_load(f)

    # DV XML file path
    dv_xml_root_fp = Path(config['dv_xml_root_fp'])
    # directory with light curves
    lc_dir = Path(config['lc_dir'])
    # run directory
    run_dir = Path(config['run_dir'])
    plot_prob = config['plot_prob']  # plot probability
    n_processes = config['n_processes']  # number of processes used to parallelize extraction
    # directory with neighbors information
    neighbors_dir = Path(config['neighbors_dir'])
    # data collection mode
    data_collection_mode = config['data_collection_mode']
    append_data = config['append_data']
    cache_neighbors_data = config['cache_neighbors_data']
    
    run_dir.mkdir(exist_ok=True, parents=True)
    with open(run_dir / config_fp.name, 'w') as f:
        yaml.dump(config, f)

    # get list of DV XML runs
    # if they are separated into single- and multi-sector run directories
    single_sector_runs = [fp for fp in (dv_xml_root_fp / 'single-sector').iterdir() if fp.is_dir()]  # and fp.stem in [f'sector_{s}' for s in range(89, 99)]]
    multi_sector_runs = [fp for fp in (dv_xml_root_fp / 'multi-sector').iterdir() if fp.is_dir()] # and fp.stem == 'multisector_s0014-s0086']
    dv_xml_runs = list(single_sector_runs) + list(multi_sector_runs)
    print(f'Found {len(dv_xml_runs)} DV XML runs in {dv_xml_root_fp}')
    # filter runs that were already processed and have results saved in the run directory
    dv_xml_runs_res_found = [dv_xml_run for dv_xml_run in dv_xml_runs
                             if (run_dir / 'data' / f'tess_diffimg_{dv_xml_run.name}.npy').exists()]
    print(f'Found NumPy files for the following sector runs in {run_dir}. Skipping those sector runs...')
    dv_xml_runs = [dv_xml_run for dv_xml_run in dv_xml_runs if dv_xml_run not in dv_xml_runs_res_found]
    # if they are not separated into single- and multi-sector run directories
    # dv_xml_runs = [fp for fp in dv_xml_root_fp.iterdir() if fp.is_dir() if fp.name in [f's{str(sector).zfill(4)}' for sector in range/(73, 81 + 1)]]
    # dv_xml_runs = [fp for fp in dv_xml_root_fp.iterdir() if fp.is_dir() if fp.name in [f's{str(sector).zfill(4)}' for sector in [73]]]

    extract_main(
        dv_xml_runs=dv_xml_runs,
        neighbors_dir=neighbors_dir,
        lc_dir=lc_dir,
        run_dir=run_dir,
        plot_prob=plot_prob,
        n_processes=n_processes,
        append_data=append_data,
        cache_neighbors_data=cache_neighbors_data,
        data_collection_mode=data_collection_mode,
        )

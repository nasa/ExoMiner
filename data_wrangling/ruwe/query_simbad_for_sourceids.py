"""
Query list of objects from catalog `query_cat` from SIMBAD to get their Gaia source ids (DR2 and EDR3).
Used for KIC and TIC.
"""

# 3rd party
from pathlib import Path
import pandas as pd
from astroquery.simbad import Simbad
import numpy as np
import multiprocessing
from datetime import datetime


def query_simbad(objs, query_cat, res_dir=None, job_i=0):
    """ Query list of objects from catalog `query_cat` from SIMBAD to get their Gaia source ids (DR2 and EDR3).

    :param objs: list, objects to be queried
    :param query_cat: str, objects' catalog
    :param res_dir: Path, results directory
    :param job_i: int, job id
    :return:
        pandas Dataframe, with SIMBAD query results
    """

    data_to_tbl = {f'{query_cat}': [], 'gaia dr2': [], 'gaia edr3': []}

    for obj_i, obj in enumerate(objs):

        if (obj_i + 1) % 10000 == 0:
            print(f'Querying object {obj} in catalog {query_cat} ({obj_i + 1}/{len(objs)})')

        result_table = Simbad.query_objectids(f"{query_cat} {obj}")

        data_to_tbl[f'{query_cat}'].append(obj)
        data_to_tbl['gaia dr2'].append(np.nan)
        data_to_tbl['gaia edr3'].append(np.nan)

        if result_table is None:
            continue

        result_table = result_table.to_pandas()
        result_table = result_table['ID'].str.decode('utf-8')
        result_table = result_table.loc[result_table.str.contains('Gaia')]

        for id_obj in result_table:
            if 'Gaia EDR3' in id_obj:
                data_to_tbl['gaia edr3'][-1] = id_obj.split(' ')[-1]
            elif 'Gaia DR2' in id_obj:
                data_to_tbl['gaia dr2'][-1] = id_obj.split(' ')[-1]

    data_df = pd.DataFrame(data_to_tbl)

    if res_dir is not None:
        data_df.to_csv(res_dir / f'simbad_{query_cat}_{job_i}.csv', index=False)

    return data_df


if __name__ == "__main__":

    tce_tbl = pd.read_csv(
        '/data5/tess_project/Data/Ephemeris_tables/TESS/DV_SPOC_mat_files/11-29-2021/tess_tces_s1-s40_11-23-2021_1409_stellarparams_updated_tecfluxtriage_eb_label.csv')
    # kic_tbl = pd.read_csv('/data5/tess_project/Data/Ephemeris_tables/Kepler/KIC_tables/q1_q17_dr25_stellar_plus_supp.csv', usecols=['kepid'])
    objs = tce_tbl['target_id'].unique()
    query_cat = 'tic'  # catalog from where the objects come from

    root_dir = Path('/data5/tess_project/Data/Ephemeris_tables/TESS/tic_gaia_ruwe')
    res_dir = root_dir / f'{datetime.now().strftime("%m-%d-%Y_%H%M")}'
    res_dir.mkdir(exist_ok=True)

    print(f'Total number of objects to query in {query_cat}: {len(objs)}')

    n_processes = 1
    objs_jobs = np.array_split(objs, n_processes)
    pool = multiprocessing.Pool(processes=n_processes)
    jobs = [(objs_in_job, query_cat, res_dir, job_i) for job_i, objs_in_job in enumerate(objs_jobs)]
    async_results = [pool.apply_async(query_simbad, job) for job in jobs]
    pool.close()

    # decide which source id to use
    simbad_tbls = [async_result.get() for async_result in async_results]  # get tables straight from query processes
    # simbad_tbls = [pd.read_csv(tbl) for tbl in res_dir.iterdir()]  # read from directory

    simbad_tbl = pd.concat(simbad_tbls, axis=0).reset_index()

    simbad_tbl['source_id'] = np.nan
    simbad_tbl['source_id_release'] = np.nan
    for obj_i, obj in simbad_tbl.iterrows():
        if isinstance(obj['gaia edr3'], str):  # preference for source id from Gaia EDR3
            simbad_tbl.loc[obj_i, ['source_id', 'source_id_release']] = [obj['gaia edr3'], 'gaia edr3']

        if isinstance(obj['gaia dr2'], str):
            simbad_tbl.loc[obj_i, ['source_id', 'source_id_release']] = [obj['gaia dr2'], 'gaia dr2']

    simbad_tbl.to_csv(res_dir / f'simbad_{query_cat}.csv', index=False)

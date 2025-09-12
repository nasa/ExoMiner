"""
Query Gaia data releases for RUWE values.
"""

# 3rd party
from astroquery.gaia import Gaia
from astropy.table import Table
from astropy.io.votable import from_table
import time


def query_gaia(source_ids, query_gaia_dr, res_dir, logger=None):
    """ Query Gaia data release for RUWE values for a set of source ids.

    :param source_ids: pandas DataFrame, source ids
    :param query_gaia_dr: str, gaia dr to query ('gaiadr2', 'gaiaedr3')
    :param res_dir: Path, results directory
    :param logger: logger
    :return:
    """

    Gaia.timeout = 10 * 60  # set timeout to 10 minutes
    
    # select the objects to query using their source id
    source_ids_tbl = Table.from_pandas(source_ids)
    source_ids_tbl.to_pandas().to_csv(res_dir / 'sourceids_fromtbl.csv', index=False)
    source_ids_votbl = from_table(source_ids_tbl)
    source_id_tbl_fp = res_dir / 'tics_sourceids.xml'
    source_ids_votbl.to_xml(str(source_id_tbl_fp))

    upload_resource = source_id_tbl_fp
    upload_tbl_name = 'sourceids'

    output_fp = res_dir / f'{query_gaia_dr}.csv'

    if logger is not None:
        logger.info(f'Querying {query_gaia_dr} source ids for their RUWE values...')

    if query_gaia_dr == 'gaiaedr3':
        query = f"SELECT g.source_id, g.ruwe FROM gaiaedr3.gaia_source as g JOIN tap_upload.{upload_tbl_name} " \
                f"as f ON g.source_id = f.source_id"
    elif query_gaia_dr == 'gaiadr2':
        query = f"SELECT g.source_id, g.ruwe FROM gaiadr2.ruwe as g JOIN tap_upload.{upload_tbl_name} " \
                f"as f ON g.source_id = f.source_id"
    if logger is not None:
        logger.info(f'Query to be performed: {query}')

    # j = Gaia.launch_job_async(query=query,
    #                           upload_resource=str(upload_resource),
    #                           upload_table_name=upload_tbl_name,
    #                           verbose=True,
    #                           output_file=str(output_fp),
    #                           dump_to_file=True,
    #                           output_format='csv',
    #                           )

    # r = j.get_results()
    # r.pprint()
    
    job = Gaia.launch_job_async(query=query,
                                upload_resource=str(upload_resource),
                                upload_table_name=upload_tbl_name,
                                verbose=True,
                                output_format='csv')

    # poll job status
    while True:
        phase = job.get_phase()
        if logger:
            logger.info(f"Job status: {phase}")
        if phase in ['COMPLETED', 'ERROR', 'ABORTED']:
            break
        time.sleep(60)  # wait 1 minute before polling again

    # Handle job completion or failure
    if job.get_phase() == 'COMPLETED':
        if logger:
            logger.info("Job completed successfully. Retrieving results...")
        result = job.get_results()
        result.write(output_fp, format='csv', overwrite=True)
        return result
    else:
        error_msg = f"Gaia query failed with status: {job.phase}"
        if logger:
            logger.error(error_msg)
        raise RuntimeError(error_msg)

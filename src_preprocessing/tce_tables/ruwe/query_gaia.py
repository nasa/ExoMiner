"""
Query Gaia data releases for RUWE values.
"""

# 3rd party
from astroquery.gaia import Gaia
from astropy.table import Table
from astropy.io.votable import from_table
import time
import pandas as pd
import logging


def pick_best_per_dr2(df_matches: pd.DataFrame, logger: logging.Logger | None = None) -> pd.DataFrame:
    """
    Collapse multiple matches per dr2_source_id to a single best row using the smallest angular_distance.
    Assumes df_matches has columns: ['dr2_source_id', 'dr3_id', 'ruwe', 'angular_distance', 'magnitude_difference'].
    Returns a DataFrame with one row per dr2_source_id.
    """

    if logger is None:
        logger = logging.getLogger(__name__)

    required_cols = {'dr2_source_id', 'dr3_id', 'ruwe', 'angular_distance'}
    missing = required_cols - set(df_matches.columns)
    if missing:
        raise ValueError(f"Missing required columns in matches: {missing}")

    # Ensure numeric types for ranking
    df = df_matches.copy()
    df['angular_distance'] = pd.to_numeric(df['angular_distance'], errors='coerce')
    # magnitude_difference is optional; keep if present
    if 'magnitude_difference' in df.columns:
        df['magnitude_difference'] = pd.to_numeric(df['magnitude_difference'], errors='coerce')

    # Count multiplicity before reduction
    dup_counts = df['dr2_source_id'].value_counts()
    n_multi = (dup_counts > 1).sum()
    if n_multi > 0:
        logger.info(f"{n_multi} DR2 ids have multiple (E)DR3 candidates; keeping the nearest by angular_distance.")

    # Rank by angular_distance ascending within each dr2_source_id
    df_sorted = df.sort_values(['dr2_source_id', 'angular_distance', 'magnitude_difference'], ascending=[True, True, True])

    # Drop duplicates keeping the first (i.e., smallest angular_distance; then smallest |ΔG| if included above)
    # Note: If you want |ΔG| as secondary key, sort by df_sorted['magnitude_difference'].abs() instead:
    # df_sorted = df.sort_values(['dr2_source_id', 'angular_distance', df['magnitude_difference'].abs()])
    best = df_sorted.drop_duplicates(subset=['dr2_source_id'], keep='first')

    # Quick sanity logging
    before = df['dr2_source_id'].nunique()
    after  = best['dr2_source_id'].nunique()
    logger.info(f"Reduced to one match per DR2 id: {after}/{before} unique DR2 ids retained.")

    return best


def query_gaia(source_ids, query_gaia_dr, res_dir, logger=None, match_angular_dist=100, match_mag_diff=0.2):
    """ Query Gaia data release for RUWE values for a set of source ids.

    :param source_ids: pandas DataFrame, source ids
    :param query_gaia_dr: str, gaia dr to query ('gaiadr2', 'gaiaedr3', 'gaiadr3)
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
        # ADQL to get RUWE from EDR3, mapping DR2 IDs -> EDR3 via dr2_neighbourhood
        # keep dr2_source_id in the output so we can merge back to TIC.
        
        query = f"""
        SELECT
        xm.dr2_source_id,
        g.source_id AS dr3_id,
        g.ruwe,
        xm.angular_distance,
        xm.magnitude_difference
        FROM gaiaedr3.gaia_source AS g
        JOIN gaiaedr3.dr2_neighbourhood AS xm
        ON xm.dr3_source_id = g.source_id
        JOIN tap_upload.sourceids AS u
        ON u.source_id = xm.dr2_source_id
        WHERE xm.angular_distance < {match_angular_dist}
        AND ABS(xm.magnitude_difference) <= {match_mag_diff}
        """
        
    elif query_gaia_dr == 'gaiadr3':
        
        query = f"""
        SELECT
        xm.dr2_source_id,
        g.source_id AS dr3_id,
        g.ruwe,
        xm.angular_distance,
        xm.magnitude_difference
        FROM gaiadr3.gaia_source AS g
        JOIN gaiadr3.dr2_neighbourhood AS xm
        ON xm.dr3_source_id = g.source_id
        JOIN tap_upload.sourceids AS u
        ON u.source_id = xm.dr2_source_id
        WHERE xm.angular_distance < {match_angular_dist}
        AND ABS(xm.magnitude_difference) <= {match_mag_diff}
        """


    elif query_gaia_dr == 'gaiadr2':
        query = f"SELECT g.source_id, g.ruwe FROM gaiadr2.ruwe as g JOIN tap_upload.{upload_tbl_name} " \
                f"as f ON g.source_id = f.source_id"
    if logger is not None:
        logger.info(f'Query to be performed: {query}')
    
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

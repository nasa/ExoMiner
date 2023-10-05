"""
Process TESS SPOC EBs catalog that is a subset of the TSO EB catalog.
"""

# 3rd party
from pathlib import Path
import pandas as pd


def _create_uid_for_jon_tbl(x):
    """ Create UID for Jon's SPOC EB table that matched TSO EBs with SPOC TESS TCEs.

    :param x: pandas Series, EB
    :return:
        str, '{tic_id}-{tce_plnt_num}-{sector_run}'. E.g., TCE TIC 123456-1-S3 (single-sector run),
        TCE TIC 123456-2-S1-26 (multi-sector run)
    """

    sector_run = [int(el[1:]) for el in x['sectors'].split('-')]
    if len(sector_run) == 1:
        sector_run = f'S{sector_run[0]}'
    elif len(sector_run) == 2:
        sector_run = f'S{sector_run[0]}-{sector_run[1]}'
    else:
        raise ValueError(f'Sector run does not have the expected template.')

    return f'{x["ticid"]}-{x["tce_plnt_num"]}-{sector_run}'

# add uid to Jon SPOC TCE-to-TSO EB table
jon_tbl_fp = Path('/Users/msaragoc/Projects/exoplanet_transit_classification/data/ephemeris_tables/tess/eb_catalogs/eb_catalog_tso/spocEBs.csv')
jon_tbl = pd.read_csv(jon_tbl_fp)
jon_tbl['uid'] = jon_tbl.apply(_create_uid_for_jon_tbl, axis=1)
jon_tbl.drop_duplicates(subset='uid', inplace=True)

jon_tbl.to_csv('/Users/msaragoc/Projects/exoplanet_transit_classification/data/ephemeris_tables/tess/eb_catalogs/eb_catalog_tso/spocEBs_processed.csv', index=False)

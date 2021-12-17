import re

import pandas as pd
from pathlib import Path

tso_eb_catalog_fp = Path(
    '/data5/tess_project/Data/Ephemeris_tables/TESS/eb_catalogs/eb_catalog_tso/eb_list_tso_12-16-2021.csv')
tso_eb_catalog_tbl = pd.read_csv(tso_eb_catalog_fp)

tso_eb_catalog_tbl['sector_run'] = 'N/A'

reg1 = tso_eb_catalog_tbl['vetting'].str.contains('-s[0-9][0-9]-')


def _get_sector_run_reg1(x):
    # print(x)
    re_found = re.findall('-s[0-9][0-9]-[^s]', x['vetting'])
    if len(re_found) != 1:
        print(x)
        return x['sector_run']
    return re_found[0][2:4]


tso_eb_catalog_tbl.loc[reg1, 'sector_run'] = tso_eb_catalog_tbl.loc[reg1, ['vetting', 'sector_run']].apply(
    _get_sector_run_reg1, axis=1)

""" Set TCE labels in the TCE table. """

from pathlib import Path

# 3rd party
import pandas as pd

tce_tbl_fp = Path(
    '/data5/tess_project/Data/Ephemeris_tables/TESS/DV_SPOC_mat_files/9-14-2021/tess_tces_s1-s40_09-14-2021_1754_stellarparams_updated.csv')
tce_tbl = pd.read_csv(tce_tbl_fp)

# set label to TFOPWG disposition
tce_tbl['label'] = tce_tbl['TFOPWG Disposition']

tce_tbl.to_csv(tce_tbl_fp.parent / f'{tce_tbl_fp.stem}_tfopwg_disp.csv', index=False)

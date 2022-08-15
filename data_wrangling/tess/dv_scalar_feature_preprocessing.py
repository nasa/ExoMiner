
# 3rd party
from pathlib import Path
import pandas as pd

tce_tbl_fp = Path('')
tce_tbl = pd.read_csv(tce_tbl_fp)

# scale diff img centroid offset estimates by Kepler-TESS pixel scale ratio
tess_px_scale = 21  # arcsec
kepler_px_scale = 3.98  # arcsec
tess_to_kepler_px_scale_factor = tess_px_scale / kepler_px_scale


tce_tbl['tce_dikco_msky']

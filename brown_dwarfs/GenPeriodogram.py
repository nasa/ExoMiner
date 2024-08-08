"""

"""

# 3rd party
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import lightkurve as lk
from pathlib import Path
from astropy.timeseries import LombScargle
from scipy import signal
from scipy import interpolate

#%%

lc_dir = Path('/Users/agiri1/Downloads/test_bd')
lc_dir.mkdir(exist_ok=True, parents=True)

# tce_tbl = pd.read_csv('/Users/msaragoc/Projects/exoplanet_transit_classification/data/ephemeris_tables/tess/DV_SPOC_mat_files/preprocessing_tce_tables/09-25-2023_1608/tess_2min_tces_dv_s1-s68_all_msectors_11-29-2023_2157_newlabels.csv')
# tce_tbl.set_index('uid', inplace=True)
# tce_uid = '68577662-1-S43'

# tce_data = tce_tbl.loc[tce_uid]

target_id = 11391018
# sector_run = 13
# tce_plnt_num = tce_data['tce_plnt_num']  # 1

# find lightcurve file for target
# search_lc_res = lk.search_lightcurve(target=f"tic{tic_id}", mission='TESS', author=('TESS-SPOC', 'SPOC'),
#                                      exptime=120, cadence='long', sector=sector_run)
search_lc_res = lk.search_lightcurve(target=f"kic{target_id}", mission='Kepler',  author=('Kepler',),
                                     exptime=30*60)
lcf = search_lc_res[0].download(download_dir=str(lc_dir), quality_bitmask='default', flux_column='pdcsap_flux')

# tce_time0bk = tce_data['tce_time0bk']
# period_days = tce_data['tce_period']
# tce_duration = tce_data['tce_duration'] / 24

time, flux = lcf.time.value, np.array(lcf.flux.value)
print(time[0])
print(time)
f = interpolate.interp1d(time, flux)
times = np.linspace(time[0], time[-1], len(time))
print(times)
new_flux = f(times)
print(new_flux)
frequency, power = signal.periodogram(new_flux, fs = times[1]-times[0], return_onesided=True)
plt.plot(frequency, power)
print(frequency)
print(power)
# for i in range(len(time)-1):
#     if(time[i+1]-time[i] == )
plt.show()
print(times)
# plt.switch_backend('TkAgg')
f, ax = plt.subplots()
ax.scatter(times, new_flux)
plt.show()

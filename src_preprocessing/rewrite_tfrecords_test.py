import pandas as pd
import numpy as np

tceTbl = pd.read_csv('/data5/tess_project/Data/Ephemeris_tables/Kepler/Q1-Q17 DR25/'
                     'q1_q17_dr25_tce_2020.04.15_23.19.10_cumkoi_2020.02.21_shuffled_noroguetces_norm_'
                     'noRobobvetterKOIs.csv', usecols=['target_id', 'tce_plnt_num', 'label'])

label_dict = {}

for tce_i, tce in tceTbl.iterrows():

    label_dict[(tce.target_id, tce.tce_plnt_num)] = tce.label

np.save('/home/msaragoc/Projects/Kepler-TESS_exoplanet/Data/rewritelabels_hbv_to_norobovetterkois.npy', label_dict)
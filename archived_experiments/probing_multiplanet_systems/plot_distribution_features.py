"""
Plot distribution of features in the data set.
"""

# 3rd party
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

#%%

dataset_tbl_fp = Path('/Users/msaragoc/Library/CloudStorage/OneDrive-NASA/Projects/exoplanet_transit_classification/experiments/multi_plnt_sys_probing/processed_datasets/01-23-2023_1220/confirmed_kois_gaia_stellar.csv')
dataset_tbl = pd.read_csv(dataset_tbl_fp)

plot_feats_args = {
    'koi_period': {'bins': np.logspace(-1, 3, 100), 'interval': [0.3, 800], 'log_scale_x': True, 'log_scale_y': False, 'label_x': 'KOI period (day)'},
    'koi_prad': {'bins': np.linspace(0, 13, 100), 'interval': [0, 12], 'log_scale_x': False, 'log_scale_y': False, 'label_x': 'KOI planet radius (Re)'},
    'koi_incl': {'bins': np.linspace(40, 90, 100), 'interval': [40, 90], 'log_scale_x': False, 'log_scale_y': True, 'label_x': 'KOI inclination (deg)'},
}

for feat, plot_args in plot_feats_args.items():

    f, ax = plt.subplots()
    ax.hist(dataset_tbl[feat], bins=plot_args['bins'], edgecolor='k')
    ax.set_ylabel('Counts')
    ax.set_xlabel(plot_args['label_x'])
    if plot_args['log_scale_x']:
        ax.set_xscale('log')
    if plot_args['log_scale_y']:
        ax.set_yscale('log')
    ax.set_xlim(plot_args['interval'])
    f.savefig(dataset_tbl_fp.parent / f'hist_{feat}.png')
    plt.close()

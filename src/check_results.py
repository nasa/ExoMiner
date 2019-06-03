"""
Check the performance of trained models using a particular configuration.
"""

import numpy as np
import os


path_res = '/home/msaragoc/Projects/Kepler-TESS_exoplanet/Kepler_planet_finder/trained_models/shallue'
res_files = [os.path.join(path_res, file) for file in os.listdir(path_res) if 'res_eval' in file]

centr_tend, dev = 'mean', 'std'

res_eval = np.load(res_files[0]).item()
avg_res = {dataset: {metric: {centr_tend: [], dev: []} for metric in res_eval[dataset]}
                   for dataset in res_eval}

for dataset in avg_res:
    for metric in avg_res[dataset]:
        avg_res_aux = []
        for res_file in res_files:
            res_eval = np.load(res_file).item()
            avg_res_aux.append(res_eval[dataset][metric])
        if centr_tend == 'mean':
            avg_res[dataset][metric][centr_tend] = np.mean(avg_res_aux)
        elif centr_tend == 'median':
            avg_res[dataset][metric][centr_tend] = np.median(avg_res_aux)
        if dev == 'std':
            avg_res[dataset][metric][dev] = np.std(avg_res_aux, ddof=1)
        elif dev == 'mad':
            avg_res[dataset][metric][dev] = np.median(np.abs(avg_res_aux - np.median(avg_res_aux)), axis=0)

for dataset in avg_res:
    print('Dataset {} {}'.format(dataset, '#' * 100))
    for metric in avg_res[dataset]:
        print('Metric {}: {} {}, {} {}'.format(metric, centr_tend, avg_res[dataset][metric][centr_tend], dev,
                                               avg_res[dataset][metric][dev]))

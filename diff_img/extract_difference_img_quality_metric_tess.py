"""
Get difference image quality metric for TESS data.
"""

# 3rd party
import xml.etree.cElementTree as et
from pathlib import Path
import pandas as pd
import numpy as np

#%%

dv_root_dir = Path('/Users/msaragoc/Library/CloudStorage/OneDrive-NASA/Projects/exoplanet_transit_classification/data/fits_files/tess/dv')
sector_root_dir = dv_root_dir / 'sector_runs'
single_sector_runs = [fp for fp in (sector_root_dir / 'single-sector').iterdir() if fp.is_dir()]
multi_sector_runs = [fp for fp in (sector_root_dir / 'multi-sector').iterdir() if fp.is_dir()]
sector_runs = list(single_sector_runs) + list(multi_sector_runs)

for sector_run in sector_runs:
    print(f'Iterating on {sector_run.name}...')
    data = {
        'uid': [],
    }
    if 'multisector' in sector_run.name:
        s_sector, e_sector = [int(s[1:]) for s in sector_run.name.split('_')[1].split('-')]
        # n_sectors = e_sector - s_sector + 1
        sector_run_id = f'{s_sector}-{e_sector}'
    else:
        # n_sectors = 1
        sector_run_id = sector_run.name.split('_')[1]
        s_sector, e_sector = int(sector_run_id), int(sector_run_id)

    s_arr = np.arange(s_sector, e_sector + 1)
    qual_metric_fields = ['value', 'valid', 'attempted']
    for s in s_arr:
        data.update({f's{s}_{field}': [] for field in qual_metric_fields})

    for dv_xml_run_fp in sector_run.iterdir():

        tree = et.parse(dv_xml_run_fp)
        root = tree.getroot()

        # check if there are results for more than one processing run for this TIC and sector run
        tic_id = root.attrib['ticId']
        tic_drs = [int(fp.stem.split('-')[-1][:-4]) for fp in sector_run.glob(f'*{tic_id.zfill(16)}*')]
        if len(tic_drs) > 1:
            curr_dr = int(dv_xml_run_fp.stem.split('-')[-1][:-4])
            latest_dr = sorted(tic_drs)[-1]
            if curr_dr != latest_dr:
                print(f'[{sector_run_id}] Skipping {dv_xml_run_fp.name} for TIC {tic_id} since there is '
                      f'more recent processed results (current release {curr_dr}, latest release {latest_dr})')
                continue

        planet_res_lst = [el for el in root if 'planetResults' in el.tag]

        # sectors_obs = root.attrib['sectorsObserved']
        # first_sector_obs, last_sector_obs = sectors_obs.find('1'), sectors_obs.rfind('1')
        n_sectors_expected = root.attrib['sectorsObserved'].count('1')

        # n_tces = len(planet_res_lst)
        # tce_i = 0

        for planet_res in planet_res_lst:

            # tce_i += 1

            uid = f'{root.attrib["ticId"]}-{planet_res.attrib["planetNumber"]}-S{sector_run_id}'

            data['uid'].append(uid)

            # get difference image results
            diff_imgs_res = [el for el in planet_res if 'differenceImageResults' in el.tag]

            # n_sectors = len(diff_imgs_res)
            # aaa
            s_found = []
            for diff_img_res in diff_imgs_res:  # iterate through all quarter difference images
                diff_img_s = diff_img_res.attrib['sector']  # get quarter
                s_found.append(int(diff_img_s))
                diff_img_metric = [el.attrib for el in diff_img_res if 'qualityMetric' in el.tag][0]  # find quality metric
                for field_name, field in diff_img_metric.items():
                    data[f's{diff_img_s}_{field_name}'].append(field)

            # set values for quarters not found to NaN
            s_not_found = np.setdiff1d(s_arr, s_found)
            for s in s_not_found:
                for qual_metric_field in qual_metric_fields:
                    data[f's{s}_{qual_metric_field}'].append(np.nan)

    data_df = pd.DataFrame(data)
    data_df.to_csv(f'/Users/msaragoc/Downloads/diff_img_quality_metric_tess/diff_img_quality_metric_tess_{sector_run_id}.csv', index=False)

"""
Get difference image quality metric for TESS data.
"""

# 3rd party
import xml.etree.cElementTree as et
from pathlib import Path
import pandas as pd
import numpy as np
import re

#%%

dv_root_dir = Path('/data5/tess_project/Data/tess_spoc_ffi_data/dv/xml_files/')
save_dir = Path('/data5/tess_project/Data/tess_spoc_ffi_data/dv/diff_img/extracted_data/quality_metrics/s36-s69_singlesectorsonly_7-9-2024_1244')

save_dir.mkdir(exist_ok=True)

single_sector_runs = [fp for fp in (dv_root_dir / 'single-sector').iterdir() if fp.is_dir()]
multi_sector_runs = []  # [fp for fp in (dv_root_dir / 'multi-sector').iterdir() if fp.is_dir()]
sector_runs = list(single_sector_runs) + list(multi_sector_runs)

print(f'Getting quality metrics for {len(sector_runs)} sector runs.')

for sector_run in sector_runs:

    print(f'Iterating on {sector_run.name}...')
    data = {
        'uid': [],
    }

    # get filepaths to xml files
    dv_xml_run_fps = list(sector_run.rglob("*.xml"))
    print(f'Found {len(dv_xml_run_fps)} DV xml files for sector run {sector_run.name}.')

    # get start and end sector for sector id
    s_sector, e_sector = re.findall('-s[0-9]+', dv_xml_run_fps[0].stem)
    s_sector, e_sector = int(s_sector[2:]), int(e_sector[2:])
    if s_sector != e_sector:  # multisector run
        sector_run_id = f'{s_sector}-{e_sector}'
    else:
        sector_run_id = f'{s_sector}'

    n_targets = len(dv_xml_run_fps)
    print(f'[Sector run {sector_run_id}] Found {n_targets} targets DV xml files in {sector_run}.')

    # if 'multisector' in sector_run.name:
    #     s_sector, e_sector = [int(s[1:]) for s in sector_run.name.split('_')[1].split('-')]
    #     sector_run_id = f'{s_sector}-{e_sector}'
    # else:
    #     sector_run_id = sector_run.name.split('_')[1]
    #     s_sector, e_sector = int(sector_run_id), int(sector_run_id)

    if (save_dir / f'diff_img_quality_metric_tess_{sector_run_id}.csv').exists():
        print(f'Quality metric table for {sector_run_id} already exists.')
        continue

    s_arr = np.arange(s_sector, e_sector + 1)
    qual_metric_fields = ['value', 'valid', 'attempted']
    for s in s_arr:
        data.update({f's{s}_{field}': [] for field in qual_metric_fields})

    for target_i, dv_xml_run_fp in enumerate(dv_xml_run_fps):

        try:
            tree = et.parse(dv_xml_run_fp)
        except Exception as e:
            print(f'Exception found when reading {dv_xml_run_fp}: {e}.')
            continue
        root = tree.getroot()

        # check if there are results for more than one processing run for this TIC and sector run
        tic_id = root.attrib['ticId']

        tic_drs = [fp for fp in sector_run.glob(f'*{tic_id.zfill(16)}*')]
        if len(tic_drs) > 1:
            curr_dr = int(dv_xml_run_fp.stem.split('-')[-1][:-4])
            latest_dr = sorted([int(fp.stem.split('-')[-1][:-4])
                                for fp in dv_xml_run_fp.glob(f'*{tic_id.zfill(16)}*')])[-1]
            if curr_dr != latest_dr:
                print(f'Sector run [{sector_run_id}] Skipping {dv_xml_run_fp.name} for TIC {tic_id} since there is '
                      f'more recent processed results (current release {curr_dr}, latest release {latest_dr})'
                      f'... ({target_i}/{n_targets} targets)')
                continue

        planet_res_lst = [el for el in root if 'planetResults' in el.tag]

        for planet_res in planet_res_lst:

            uid = f'{root.attrib["ticId"]}-{planet_res.attrib["planetNumber"]}-S{sector_run_id}'

            data['uid'].append(uid)

            # get difference image results
            diff_imgs_res = [el for el in planet_res if 'differenceImageResults' in el.tag]

            s_found = []
            for diff_img_res in diff_imgs_res:  # iterate through all quarter difference images
                diff_img_s = diff_img_res.attrib['sector']  # get quarter
                s_found.append(int(diff_img_s))
                # find quality metric
                diff_img_metric = [el.attrib for el in diff_img_res if 'qualityMetric' in el.tag][0]
                for field_name, field in diff_img_metric.items():
                    data[f's{diff_img_s}_{field_name}'].append(field)

            # set values for quarters not found to NaN
            s_not_found = np.setdiff1d(s_arr, s_found)
            for s in s_not_found:
                for qual_metric_field in qual_metric_fields:
                    data[f's{s}_{qual_metric_field}'].append(np.nan)

    data_df = pd.DataFrame(data)
    data_df.to_csv(save_dir / f'diff_img_quality_metric_tess_{sector_run_id}.csv', index=False)

print('Finished.')

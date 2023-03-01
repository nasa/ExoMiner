"""
Get difference image quality metric for Kepler data.
"""

# 3rd party
import xml.etree.cElementTree as et
from pathlib import Path
import pandas as pd
import numpy as np

#%%

dv_xml_fp = Path('/Users/msaragoc/Library/CloudStorage/OneDrive-NASA/Projects/exoplanet_transit_classification/data/fits_files/kepler/q1_q17_dr25/dv/kplr20160128150956_dv.xml')

# get an iterable
context = et.iterparse(dv_xml_fp, events=("start", "end"))

# get the root element
event, root = next(context)

data = {
    'uid': [],
}
n_quarters = 17
q_arr = np.arange(1, n_quarters + 1)
qual_metric_fields = ['value', 'valid', 'attempted']
for q in q_arr:
    data.update({f'q{q}_{field}': [] for field in qual_metric_fields})
tce_counter = 0
for event, elem in context:

    if event == "end" and elem.tag == "planetResults":  # iterate through each planet results container

        uid = f'{elem.attrib["keplerId"]}-{elem.attrib["planetNumber"]}'

        tce_counter += 1
        if tce_counter % 100 == 0:
            print(f'Iterating over TCE KIC {uid} (total number of TCEs iterated through: {tce_counter})...')

        data['uid'].append(uid)

        diff_imgs_res = elem.findall('differenceImageResults')

        q_found = []
        for diff_img_res in diff_imgs_res:  # iterate through all quarter difference images
            diff_img_q = diff_img_res.attrib['quarter']  # get quarter
            q_found.append(int(diff_img_q))
            diff_img_metric = diff_img_res.findall('qualityMetric')[0]  # find quality metric
            for field_name, field in diff_img_metric.attrib.items():
                data[f'q{diff_img_q}_{field_name}'].append(field)

        # set values for quarters not found to NaN
        q_not_found = np.setdiff1d(q_arr, q_found)
        for q in q_not_found:
            for qual_metric_field in qual_metric_fields:
                data[f'q{q}_{qual_metric_field}'].append(np.nan)

    # if tce_counter == 100:
    #     break

data_df = pd.DataFrame(data)
data_df.to_csv('/Users/msaragoc/Downloads/diff_img_quality_metric.csv', index=False)

# 3rd party
import pandas as pd
import numpy as np
import xml.etree.cElementTree as et

tree = et.parse('/Users/msaragoc/Downloads/tess2018206190142-s0001-s0001-0000000144048277-00106_dvr.xml')
root = tree.getroot()

centroid_res = root[5][3]
diff_img_res = root[5][4]
diff_img_px_data = diff_img_res.findall('./{http://www.nasa.gov/2018/TESS/DV}differenceImagePixelData')  # iter(root[5][4][4:-3])

img_size = (19, 11)
diff_img_px_data_imgs = {
    'meanFluxInTransit': {'values': np.nan * np.ones(img_size), 'uncertainties': np.nan * np.ones(img_size)},
    'meanFluxOutOfTransit': {'values': np.nan * np.ones(img_size), 'uncertainties': np.nan * np.ones(img_size)},
    'meanFluxDifference': {'values': np.nan * np.ones(img_size), 'uncertainties': np.nan * np.ones(img_size)},
    'meanFluxForTargetTable': {'values': np.nan * np.ones(img_size), 'uncertainties': np.nan * np.ones(img_size)},
}

row_i, col_i = 0, 0
for px_i, px in enumerate(diff_img_px_data):
    if px_i == 0:
        ccd_origin = dict(px.items())
        ccd_origin = {key: float(val) for key, val in ccd_origin.items()}
    for img_name in diff_img_px_data_imgs.keys():
        px_for_img = dict(px.findall(f'./{{http://www.nasa.gov/2018/TESS/DV}}{img_name}')[0].items())
        diff_img_px_data_imgs[img_name]['values'][row_i, col_i] = float(px_for_img['value'])
        diff_img_px_data_imgs[img_name]['uncertainties'][row_i, col_i] = float(px_for_img['uncertainty'])
    col_i += 1
    if col_i == img_size[1]:
        col_i = 0
        row_i += 1

diff_img_px_data_imgs['ccdOrigin'] = ccd_origin
diff_img_px_data_imgs['ticReferenceCentroid'] = {}
ticReferenceCentroid = diff_img_res.findall(f'./{{http://www.nasa.gov/2018/TESS/DV}}ticReferenceCentroid')[0]
for coord_px_ccd in ['row', 'column']:
    tic_coord = dict(ticReferenceCentroid.find(f'{{http://www.nasa.gov/2018/TESS/DV}}{coord_px_ccd}').items())
    diff_img_px_data_imgs['ticReferenceCentroid'][coord_px_ccd] = {key: float(val) for key, val in tic_coord.items()}
diff_img_px_data_imgs['ticReferenceCentroid']['row']['value'] -= diff_img_px_data_imgs['ccdOrigin']['ccdRow']
diff_img_px_data_imgs['ticReferenceCentroid']['column']['value'] -= diff_img_px_data_imgs['ccdOrigin']['ccdColumn']

#%%

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

f, ax = plt.subplots()
im = ax.imshow(diff_img_px_data_imgs['meanFluxDifference']['values'])
ax.scatter(diff_img_px_data_imgs['ticReferenceCentroid']['column']['value'], diff_img_px_data_imgs['ticReferenceCentroid']['row']['value'], marker='+', color='r')
ax.set_ylabel('CCD Row')
ax.set_xlabel('CCD Column')
ax.set_yticks([0, 5, 10, 15])
ax.set_xticks([0, 5, 10])
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(im, cax=cax)

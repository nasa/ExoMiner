import numpy as np
from astropy.io import fits
from astropy.table import Table
import matplotlib.pyplot as plt
from astroquery.mast import Mast, Observations

obs = Observations.query_criteria(target_name='kplr008957091',
                                  obs_collection='Kepler')

prods = Observations.get_product_list(obs)

prods_filt = Observations.filter_products(prods, obs_id='*lpd*')

download_dir = '/home/msaragoc/Downloads/'
prods_get = Observations.download_products(prods_filt, download_dir=download_dir)

with fits.open(prods_get[0]) as hdulist:
    aaaa

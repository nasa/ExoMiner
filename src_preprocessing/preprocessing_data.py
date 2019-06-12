"""
read kepler ids from the injected signals fits files.
"""

from astropy.io import fits
from tensorflow import gfile

import src_preprocessing.preprocess

ex_inj_file = '/home/msaragoc/Kepler_planet_finder/src_preprocessing/kplr011081546-2011073133259_INJECTED-inj3_llc.fits'

with fits.open(gfile.Open(ex_inj_file, "rb")) as hdu_list:

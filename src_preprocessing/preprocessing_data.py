"""
read kepler ids from the injected signals fits files.
"""

import os
from astropy.io import fits
from tensorflow import gfile
from src_preprocessing.light_curve import kepler_io
import shutil
import multiprocessing

import src_preprocessing.preprocess

# ex_inj_file = '/home/msaragoc/Kepler_planet_finder/src_preprocessing/kplr011081546-2011073133259_INJECTED-inj3_llc.fits'
dir_inj_in = '/data5/tess_project/Data/Kepler-Q1-Q17-DR25/Injected/inj3in'
dir_inj_out = '/data5/tess_project/Data/Kepler-Q1-Q17-DR25/Injected/inj3'

with fits.open(gfile.Open('/home/msaragoc/Kepler_planet_finder/src_preprocessing/example_injected.fits', "rb")) as \
        hdu_list:
    hdu_list[0].header['KEPLERID']

#
kepids = []
filenames = os.listdir(dir_inj_in)
for filename in filenames:
    print('File {}'.format(filename))

    with fits.open(gfile.Open(os.path.join(dir_inj_in, filename), "rb")) as hdu_list:
        kepids.append(hdu_list[0].header['KEPLERID'])

    print('Kepler ID: {}'.format(kepids[-1]))
    kepid_aux = str(kepids[-1])
    kepid_aux = "0" * (9 - len(kepid_aux)) + kepid_aux
    print('Kepler ID: {}'.format(kepid_aux))

    if not os.path.isdir(os.path.join(dir_inj_out, kepid_aux[0:4])):
        os.mkdir(os.path.join(dir_inj_out, kepid_aux[0:4]))

    if not os.path.isdir(os.path.join(dir_inj_out, kepid_aux[0:4], kepid_aux)):
        os.mkdir(os.path.join(dir_inj_out, kepid_aux[0:4], kepid_aux))

    shutil.copy(os.path.join(dir_inj_in, filename), os.path.join(dir_inj_out, kepid_aux[0:4], kepid_aux, filename))
    aaa

# get fits filenames
injected_group = 'inj3'
filenames_per_kepid = []
for kepid in kepids:
    print('Getting fits files for Kepler ID {}'.format(kepid))
    filenames = kepler_io.kepler_filenames(dir_inj_out, kep_id=kepid, long_cadence=True, quarters=None,
                                           injected_group=injected_group,
                                           check_existence=True)
    filenames_per_kepid.append(filenames)
    aaaa

print('filenames\n', filenames)

# get time, flux, centroid time series
all_time, all_flux, all_centroid = kepler_io.read_kepler_light_curve(filenames, light_curve_extension="LIGHTCURVE",
                                                                     scramble_type=None, interpolate_missing_time=False)

# preprocess time series and create tfrecord
ex = src_preprocessing.preprocess.generate_example_for_tce(time, flux, centroids, tce)


# def create_MASTdataset_inj(q, dir_inj_in, dir_inj_out, filename):
def create_MASTdataset_inj(filename, dir_inj_in, dir_inj_out):

    # dir_inj_in = '/data5/tess_project/Data/Kepler-Q1-Q17-DR25/Injected/inj3in'
    # dir_inj_out = '/data5/tess_project/Data/Kepler-Q1-Q17-DR25/Injected/inj3'

    # kepids = []
    # filenames = os.listdir(dir_inj_in)
    # for filename in filenames:
    print('File {}'.format(filename))

    with fits.open(gfile.Open(os.path.join(dir_inj_in, filename), "rb")) as hdu_list:
        kepids.append(hdu_list[0].header._keyword_indices['KEPLERID'][0])

    print('Kepler ID: {}'.format(kepids[-1]))
    kepid_aux = str(kepids[-1])
    kepid_aux = "0" * (9 - len(kepid_aux)) + kepid_aux
    print('Kepler ID: {}'.format(kepid_aux))

    if not os.path.isdir(os.path.join(dir_inj_out, kepid_aux[0:4])):
        os.mkdir(os.path.join(dir_inj_out, kepid_aux[0:4]))

    if not os.path.isdir(os.path.join(dir_inj_out, kepid_aux[0:4], kepid_aux)):
        os.mkdir(os.path.join(dir_inj_out, kepid_aux[0:4], kepid_aux))

    shutil.copy(os.path.join(dir_inj_in, filename), os.path.join(dir_inj_out, kepid_aux[0:4], kepid_aux, filename))

    # return kepids
    # q.put(kepid)


if __name__ == '__main__':

    dir_inj_in = '/data5/tess_project/Data/Kepler-Q1-Q17-DR25/Injected/inj3in'
    dir_inj_out = '/data5/tess_project/Data/Kepler-Q1-Q17-DR25/Injected/inj3'

    kepids = []
    filenames = os.listdir(dir_inj_in)
    nworkers = 4

    with multiprocessing.Pool(nworkers) as pool:
        # kepids = pool.map(create_MASTdataset_inj, filenames)
        kepids = pool.apply(create_MASTdataset_inj, args=(filenames, dir_inj_in, dir_inj_out))

    # qvec = []
    # pvec = []
    # for filename in filenames:
    #     for i in range(nworkers):
    #         pvec.append(multiprocessing.Process(target=create_MASTdataset_inj, args=(q, dir_inj_in, dir_inj_out, filename)))
    #         pvec[i].start()

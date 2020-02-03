"""
Configuration script used to set the parameters for preprocessing and inference in the DL pipeline.
"""

# 3rd party
import tensorflow as tf
import datetime


class Config:
    """ Class that creates configuration objects that hold parameters required for running the DL pipeline."""

    # PREPROCESSING PARAMETERS ########

    satellite = 'tess'  # choose from: ['kepler', 'tess']
    multisector = False  # True for TESS multi-sector runs

    # output directory
    output_dir = '/home/msaragoc/Projects/Kepler-TESS_exoplanet/Kepler_planet_finder/' \
                 'src_deployment/output/{}-run{}'.format(satellite, datetime.datetime.now())

    # # if True, CCD module pixel coordinates are used. If False, local CCD pixel coordinates are transformed into RA
    # # and Dec (world coordinates)
    # px_coordinates = False

    # if True, saves plots of several preprocessing steps
    plot_figures = False

    # minimum gap size( in time units) for a split
    gapWidth = 0.75

    # binning parameters
    num_bins_glob = 2001  # number of bins in the global view
    num_bins_loc = 201  # number of bins in the local view
    bin_width_factor_glob = 1 / num_bins_glob
    bin_width_factor_loc = 0.16

    # # filepath to numpy file with stats used to normalize the data
    # stats_preproc_filepath = 'stats_trainingset.npy'

    # INFERENCE PARAMETERS ########

    # models' directory
    modelDir = '/home/msaragoc/Projects/Kepler-TESS_exoplanet/Kepler_planet_finder/trained_models/' \
               'bohb_dr25tcert_spline_gapped2/models'

    # TensorFlow session configuration
    config_tfsess = tf.ConfigProto(log_device_placement=False)

    # set to True if nodes have one or more GPUs - each GPU performs inference for a model
    multiGPU = False

    # number of GPUs per node
    numGPUsPerNode = 1

    # number of processes used in a multiprocessing scenario
    numProcesses = 10  # 1 per model in the ensemble


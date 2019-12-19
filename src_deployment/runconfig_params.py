import os
import tensorflow as tf


class Config:
    """ Class that creates configuration objects that hold parameters required for running the DL pipeline."""

    # PREPROCESSING PARAMETERS ########

    satellite = 'kepler'  # choose from: ['kepler', 'tess']

    # working/output directory
    output_dir = "".format(satellite)
    # # working directory
    # w_dir = ''
    # output_dir = os.path.join(w_dir, output_dir)

    # if True, CCD module pixel coordinates are used. If False, local CCD pixel coordinates are transformed into RA and
    # Dec (world coordinates)
    px_coordinates = False

    # if True, saves plots of several preprocessing steps
    plot_figures = False

    # minimum gap size( in time units) for a split
    gapWidth = 0.75

    # gapping - remove transits from other TCEs
    gapped = True
    gap_imputed = False  # fill with noise after gapping
    gap_with_confidence_level = False  # gap only if highly confident other TCEs are planets
    gap_confidence_level = 0.75

    # binning parameters
    num_bins_glob = 2001  # number of bins in the global view
    num_bins_loc = 201  # number of bins in the local view
    bin_width_factor_glob = 1 / num_bins_glob
    bin_width_factor_loc = 0.16

    # filepath to numpy file with stats used to normalize the data
    stats_preproc_filepath = 'stats_trainingset.npy'

    # INFERENCE PARAMETERS ########

    # models' directory
    modelDir = ''

    # # filepath of tfrecord with preprocessed TCE
    # tfrecFilepath = ''

    # TensorFlow session configuration
    config_tfsess = tf.ConfigProto(log_device_placement=False)

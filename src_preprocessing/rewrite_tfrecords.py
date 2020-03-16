"""
Script used to update TCE labels in the TFRecords associated with KOIs whose disposition changed since the time the
labels were updated for the original TFRecords.
"""

# 3rd party
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import numpy as np
import multiprocessing
import pandas as pd

# local
from src_preprocessing.tf_util import example_util


def generate_new_tfrecords_updtlabels(tfrec_file_paths, new_tfrec_file_paths, label_dict):
    """ Generate tfrecords based on existing tfrecords, with additional pseudo labels.
    The pseudo labels are defined as the product of the tce period and the tce duration.
        :param tfrec_dir: str, path to tfrecords to which pseudo labels are to be added
        :param new_tfrec_dir: str, path to where the generated tfrecords are to be saved
        :param label_dict: dictionary mapping pseudo labels to tuple keys (kepler id and tce planer number)
    """

    for cnt in range(len(tfrec_file_paths)):
        # try:
        record_iterator = tf.python_io.tf_record_iterator(path=tfrec_file_paths[cnt])

        # iterate over tce's in the current tfrecord and write existing features
        # and updated labels to the new tfrecord file by overwriting previous ones
        with tf.python_io.TFRecordWriter(new_tfrec_file_paths[cnt]) as writer:
            for string_record in record_iterator:
                example = tf.train.Example()
                example.ParseFromString(string_record)
                new_example = example

                # search through dictionary for the pseudo label corresponding to the current tce
                tce_plnt_num = new_example.features.feature['tce_plnt_num'].int64_list.value[0]
                target_id = new_example.features.feature['kepid'].int64_list.value[0]

                if (target_id, tce_plnt_num) in label_dict:
                    example_util.set_feature(new_example, 'av_training_set', [label_dict[(target_id, tce_plnt_num)]],
                                             allow_overwrite=True)

                writer.write(new_example.SerializeToString())

        # except:
        #     print('Non-tfrecord found. Skipped file: {}'.format(tfrec_file_paths[cnt]))

    return


def run_main():
    """ Script to generate tfrecords with additional pseudo labels, based on existing tfrecords.
    """

    # specify path to directory holding tfrecords to which the pseudo labels are to be added
    tfrec_dir = '/data5/tess_project/Data/tfrecords/Kepler/' \
                'tfrecordkeplerdr25_flux-centroid_selfnormalized-oddeven_nonwhitened_gapped_2001-201'

    # specify path to save new tfrecords
    new_tfrec_dir = '/data5/tess_project/Data/tfrecords/Kepler/' \
                    'tfrecordkeplerdr25_flux-centroid_selfnormalized-oddeven_nonwhitened_gapped_2001-201_updtKOIs'

    # make the output directory if it doesn't already exist
    tf.gfile.MakeDirs(new_tfrec_dir)

    # specify paths to specific tfrecord files within tfrec_dir
    tfrec_file_paths = [os.path.join(tfrec_dir, file) for file in os.listdir(tfrec_dir) if 'node' in file]
    # specify paths to save the corresponding generated tfrecord files
    new_tfrec_file_paths = [os.path.join(new_tfrec_dir, file) for file in os.listdir(tfrec_dir) if 'node' in file]

    updtKoisTbl = pd.read_csv('/home/msaragoc/Projects/Kepler-TESS_exoplanet/Kepler_planet_finder/'
                              'koi_ephemeris_matching/updatedKOIsdisposition.csv')

    labels_dict = {}
    for koi_i, koi in updtKoisTbl.iterrows():
        labels_dict[(koi.kepid, koi.tce_plnt_num)] = koi.label

    n_procs = 28
    jobs = []

    print('Number of TFRecords (per process) = {} (~{})'.format(len(tfrec_file_paths),
                                                                int(len(tfrec_file_paths) / n_procs)))
    print('Number of processes = {}'.format(n_procs))

    boundaries = [int(i) for i in np.linspace(0, len(tfrec_file_paths), n_procs + 1)]

    for proc_i in range(n_procs):
        indices = [(boundaries[i], boundaries[i + 1]) for i in range(n_procs)][proc_i]
        tfrecords_proc = tfrec_file_paths[indices[0]:indices[1]]
        newtfrecords_proc = new_tfrec_file_paths[indices[0]:indices[1]]
        p = multiprocessing.Process(target=generate_new_tfrecords_updtlabels,
                                    args=(tfrecords_proc, newtfrecords_proc, labels_dict))
        jobs.append(p)
        p.start()

    map(lambda p: p.join(), jobs)


if __name__ == '__main__':

    # run_main()

    # check if all labels were correctly updated
    new_tfrec_dir = '/data5/tess_project/Data/tfrecords/Kepler/' \
                    'tfrecordkeplerdr25_flux-centroid_selfnormalized-oddeven_nonwhitened_gapped_2001-201_updtKOIs'

    # specify paths to save the corresponding generated tfrecord files
    new_tfrec_file_paths = [os.path.join(new_tfrec_dir, file) for file in os.listdir(new_tfrec_dir) if 'node' in file]

    updtKoisTbl = pd.read_csv('/home/msaragoc/Projects/Kepler-TESS_exoplanet/Kepler_planet_finder/'
                              'koi_ephemeris_matching/updatedKOIsdisposition.csv')

    labels_dict = {}
    for koi_i, koi in updtKoisTbl.iterrows():
        labels_dict[(koi.kepid, koi.tce_plnt_num)] = koi.label

    labels_matched = 0
    for cnt in range(len(new_tfrec_file_paths)):
        # try:
        record_iterator = tf.python_io.tf_record_iterator(path=new_tfrec_file_paths[cnt])

        for string_record in record_iterator:
            example = tf.train.Example()
            example.ParseFromString(string_record)

            # search through dictionary for the pseudo label corresponding to the current tce
            tce_plnt_num = example.features.feature['tce_plnt_num'].int64_list.value[0]
            target_id = example.features.feature['kepid'].int64_list.value[0]

            if (target_id, tce_plnt_num) in labels_dict:
                if labels_dict[(target_id, tce_plnt_num)] == \
                example.features.feature['av_training_set'].bytes_list.value[0].decode("utf-8"):
                    labels_matched += 1

    print('Number of KOIs updated: {}'.format(len(labels_dict.keys())))
    print('Number of labels matched to the KOIs updated: {}'.format(labels_matched))

import os
import tensorflow as tf
tf.enable_eager_execution()

from src_preprocessing.preprocess import normalize_view

tfrec_dir = '/home/msaragoc/Projects/Kepler-TESS_exoplanet/Kepler_planet_finder/src_preprocessing/tfrecords/tfrecord_dr25_manual_2d_few180k_keplernonwhitened'
tfrecs = [os.path.join(tfrec_dir, tfrec_filename) for tfrec_filename in os.listdir(tfrec_dir)]

features_set = {'global_view': {'dim': 2001, 'dtype': tf.float32},
                                 'local_view': {'dim': 201, 'dtype': tf.float32}}
data_fields = {feature_name: tf.FixedLenFeature([feature_info['dim']], feature_info['dtype'])
               for feature_name, feature_info in features_set.items()}

for tfrec_filepath in tfrecs:
    tf_dataset = tf.data.TFRecordDataset([tfrec_filepath])

    for record in tf_dataset:
        print('########33')
        print(repr(record))
        tf.parse_single_example(record, features=data_fields)
        aa
#
# for raw_record in tf_dataset.take(100):
#   print(repr(raw_record))

# writing a TFRecord file
for tfrec_filepath in tfrecs:
    with tf.python_io.TFRecordWriter(tfrec_filepath) as writer:
        for i in range(n_observations):
            example = serialize_example(feature0[i], feature1[i], feature2[i], feature3[i])
            writer.write(example)

# reading a TFRecord file
for tfrec_filepath in tfrecs:
    record_iterator = tf.python_io.tf_record_iterator(path=tfrec_filepath)

    for string_record in record_iterator:
      example = tf.train.Example()
      example.ParseFromString(string_record)

      print(example)

import os
import tensorflow as tf
import pandas as pd
from pathlib import Path
import src_preprocessing.tf_util.example_util as example_util
#Supress Warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
"""
Make a TF-Record dataset from the Kepler and TESS datasets that contain brown dwarfs. Creates Kepler shard and TESS shard with BD labels
"""
#Kepler input

src_tfrec_dir = Path('/Users/agiri1/Desktop/tfrecords_keplerq1q17dr25_splinedetrending_6-18-2024_2305_adddiffimg_perimgnormdiffimg')
src_tfrec_shards = [kepler_shard for kepler_shard in src_tfrec_dir.iterdir()]
dest_tfrec_dir = Path('/Users/agiri1/Desktop/tfrecords_keplerq1q17dr25_splinedetrending_6-18-2024_2305_adddiffimg_perimgnormdiffimg_BDsandPCs')
dest_tfrec_dir.mkdir(exist_ok=True)
df = pd.read_csv("/Users/agiri1/Documents/q1_q17_dr25_tce_3-6-2023_1734_withBDs.csv")
bds = list(df[(df["label"] == "BD")]["uid"])
total = list(df[(df["label"] == "PC") | (df["label"] == "BD")]["uid"])

#TESS Input
"""
src_tfrec_dir = Path('/Users/agiri1/Desktop/tfrecords_tess_splinedetrending_s1-s67_4-5-2024_1513_merged_adddiffimg_perimgnormdiffimg')
src_tfrec_shards = [tess_shard for tess_shard in src_tfrec_dir.iterdir() if "node" in tess_shard.name]
dest_tfrec_dir = Path('/Users/agiri1/Desktop/tfrecords_tess_splinedetrending_s1-s67_4-5-2024_1513_merged_adddiffimg_perimgnormdiffimg_BDsandPCs')
dest_tfrec_dir.mkdir(exist_ok=True)
df = pd.read_csv("/Users/agiri1/Library/CloudStorage/OneDrive-NASA/tess_2min_tces_dv_s1-s68_all_msectors_11-29-2023_2157_newlabels_only_BDs_CPs_KPs.csv")
bds = list(df[(df["label"] == "BD")]["uid"])
total = list(df[(df["label"] == "CP") | (df["label"] == "BD") | (df["label"] == "KP")]["uid"])
"""
#Find Missing Shards for TESS
"""names = [tess_shard.name for tess_shard in src_tfrec_dir.iterdir() if "node" in tess_shard.name]
names.sort()
all_shards = list(range(600))
for i in names:
    all_shards.remove(int(i[6:11]))
for i in all_shards:
    print(f"Missing Shard: {i}")
exit()"""
BD_count = 0
PC_count = 0
# get the features from the source example
#Kepler Version
#with tf.io.TFRecordWriter(str(dest_tfrec_dir / "Kepler_Dataset")) as writer:
#TESS Version
with tf.io.TFRecordWriter(str(dest_tfrec_dir / "TESS_Dataset")) as writer:
    for shard_i, src_shard in enumerate(src_tfrec_shards):

        print(f'Iterating over {src_shard} ({shard_i + 1}/{len(src_tfrec_shards)}...')

        tfrecord_dataset = tf.data.TFRecordDataset(str(src_shard))
        # iterate through the source shard
        for string_record in tfrecord_dataset.as_numpy_iterator():
            example = tf.train.Example()
            example.ParseFromString(string_record)
            example_uid = example.features.feature['uid'].bytes_list.value[0].decode("utf-8")
            if example_uid in total:
                total.remove(example_uid)
                #Overwrite bd labels
                PC_count += 1
                if example_uid in bds:
                    example_util.set_bytes_feature(example, 'label', ['BD'], allow_overwrite=True)
                    BD_count += 1
                    PC_count -= 1
                    print(f"Overwriting BD {example_uid}")
                writer.write(example.SerializeToString())
print(f"BD Count: {BD_count}")
print(f"PC Count: {PC_count}")
with open("kepler_missing.txt","w") as writer:
    for i in total:
        if i in bds:
            writer.write(f"{i} BD\n")
        writer.write(f"{i} PC\n")
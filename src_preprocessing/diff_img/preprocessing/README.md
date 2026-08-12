# Difference Image Preprocessing

## Introduction

The goal of this (sub)module is to 1) **preprocess extracted difference image data** (optionally with neighbors data) and 2) **add that data to a lightcurve TFRecord dataset**.

## Steps

1. Preprocess difference image data using script [preprocess_diff_img.py](preprocess_diff_img.py) with companion configuration file [config_preprocessing.yaml](config_preprocessing.yaml).

        python preprocess_diff_img.py --config_fp=config_preprocessing.yaml

2. Add preprocessed difference image data to a lightcurve TFRecord dataset using script [add_data_to_tfrecords.py](add_data_to_tfrecords.py) with companion configuration file [config_add_diff_img_tfrecords.yaml](config_add_diff_img_tfrecords.yaml)
    
        python add_data_to_tfrecords.py --config_fp=config_add_diff_img_tfrecords.yaml
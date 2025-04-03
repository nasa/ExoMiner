"""
Get model outputs from inner layers.
"""

# 3rd party
from pathlib import Path
import pandas as pd
import yaml
from tensorflow.keras.utils import custom_object_scope
from tensorflow.keras.models import load_model
from tensorflow.keras import Model
import argparse
import numpy as np

# local
from models.models_keras import Time2Vec, SplitLayer
from src.utils.utils_dataio import (set_tf_data_type_for_features, get_data_from_tfrecords_for_predictions_table,
                                    InputFnv2 as InputFn)


def extract_features_into_csv(config, save_dir):
    """ Extract intermediate features from a model. `config` needs to have fields

    :param config: dict, training run parameters
    :param save_dir: Path, results directory

    :return:
    """

    save_dir.mkdir(exist_ok=True)

    scalar_inputs = {
        'boot_fap',
        'flux_even_local_stat_abs_min',
        'flux_global_stat_abs_min',
        'flux_local_stat_abs_min',
        'flux_odd_local_stat_abs_min',
        'flux_trend_global_stat_max',
        'flux_trend_global_stat_min',
        'flux_weak_secondary_local_stat_abs_min',
        'mag',
        'pgram_smooth_max_power',
        'pgram_tpm_smooth_max_power',
        'ruwe',
        'tce_albedo_stat',
        'tce_cap_stat',
        'tce_dikco_msky',
        'tce_dikco_msky_err',
        'tce_hap_stat',
        'tce_max_mult_ev',
        'tce_max_sngle_ev',
        'tce_maxmes',
        'tce_model_chisq',
        'tce_num_transits',
        'tce_num_transits_obs',
        # 'tce_num_transits_norm',
        # 'tce_num_transits_obs_norm',
        'tce_period',
        'tce_prad',
        'tce_ptemp_stat',
        'tce_robstat',
        'tce_sdens',
        'tce_slogg',
        'tce_smass',
        'tce_smet',
        'tce_sradius',
        'tce_steff',
    }
    scalar_inputs = [scalar_input for scalar_input in scalar_inputs if scalar_input not in config['data_fields']]

    # get data from TFRecords files to be displayed in the table with predictions
    data = get_data_from_tfrecords_for_predictions_table(config['datasets'],
                                                         config['data_fields'] + scalar_inputs,
                                                         config['datasets_fps'])

    # set and run model pipeline to get relevant outputs
    config['features_set'] = set_tf_data_type_for_features(config['features_set'])

    # load trained model
    custom_objects = {"Time2Vec": Time2Vec, 'SplitLayer': SplitLayer}
    with custom_object_scope(custom_objects):
        model = load_model(filepath=config['model_fp'], compile=False)

    chosen_layers = [
        # # 'diff_imgs_global_max_pooling_concat',
        # 'diff_imgs_convfc_batch_norm',  # 15 (including quality metrics)
        # 'global_flux_global_max_pooling',  # 16
        # # 'global_flux_conv_1_2_batch_norm',  # 16 (after global max pooling)
        # # 'global_flux_fc',  # 3 (with norm stat)
        # 'flux_trend_global_max_pooling',  # 16
        # # 'flux_trend_conv_1_2_batch_norm',  # 16
        # # 'local_centroid_conv_1_2_batch_norm',  # 16
        # 'local_centroid_global_max_pooling',  # 16
        # # 'momentum_dump_conv_1_2_batch_norm',  # 16
        # 'momentum_dump_global_max_pooling',  # 16
        # # 'flux_periodogram_conv_1_0_batch_norm',  # 8
        # 'flux_periodogram_max_pooling',  # 8
        # 'local_fluxes_local_weak_secondary_global_max_pooling',  # 16
        # 'local_fluxes_local_flux_global_max_pooling',  # 16
        # 'local_fluxes_local_odd_even_global_max_pooling',  # 16
        # 'unfolded_flux_global_max_pooling',  # 4

        # end of all branches
        'local_fluxes_fc_prelu_local_odd_even',
        'local_fluxes_fc_prelu_local_flux',
        'local_fluxes_fc_prelu_local_weak_secondary',
        'flux_periodogram_fc_prelu',
        'momentum_dump_fc_prelu',
        'local_centroid_fc_prelu',
        'flux_trend_fc_prelu',
        'global_flux_fc_prelu',
        'unfolded_flux_fc_prelu',
        'diff_imgs_fc_prelu',
        'fc_prelu_stellar_scalar',
        'fc_prelu_dv_tce_fit_scalar',

    ]
    config['output_layers'] = [layer.name for layer in model.layers if layer.name in chosen_layers]

    # set and run model pipeline to get relevant outputs
    intermediate_model = Model(inputs=model.inputs, outputs=[model.get_layer(layer_name).output
                                                           for layer_name in config['output_layers']],
                             name=f'model_intermediate_outputs')

    # create feature names
    extracted_feature_names = []
    for output in intermediate_model.outputs:
        feature_prefix_name = output.name.split('/')[0]
        n_features = np.prod(output.shape[1:])  # get number of features (exclude batch dimension)
        extracted_feature_names += [f'{feature_prefix_name}_{feature_i}' for feature_i in range(n_features)]

    for dataset in config['datasets']:  # iterate through the datasets to extract features

        predict_input_fn = InputFn(
            file_paths=config['datasets_fps'][dataset],
            batch_size=config['inference']['batch_size'],
            mode='PREDICT',
            label_map=config['label_map'],
            features_set=config['features_set'],
            multiclass=config['config']['multi_class'],
            feature_map=config['feature_map'],
            label_field_name=config['label_field_name'],
        )

        intermediate_model_outputs = intermediate_model.predict(
            predict_input_fn(),
            batch_size=None,
            verbose=config['verbose_model'],
            steps=None,
            callbacks=None,
        )

        # flatten layer output dim
        intermediate_model_outputs = [np.reshape(el, (el.shape[0], np.prod(el.shape[1:])))
                                      for el in intermediate_model_outputs]
        # concatenate features
        intermediate_model_outputs = np.concatenate(intermediate_model_outputs, axis=1)
        intermediate_model_outputs_df = pd.DataFrame(intermediate_model_outputs, columns=extracted_feature_names)
        data_df = pd.DataFrame(data[dataset])
        results_df = pd.concat([data_df, intermediate_model_outputs_df], axis=1)
        results_df.to_csv(save_dir / f'intermediate_outputs_model_{dataset}.csv', index=False)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--config_fp', type=str, help='File path to YAML configuration file.', default=None)
    parser.add_argument('--res_dir', type=str, help='Output directory', default='')
    parser.add_argument('--model_fp', type=str, help='Path to pre-existing model', default=None)
    args = parser.parse_args()

    res_dir_fp = Path(args.res_dir)

    with(open(args.config_fp, 'r')) as file:
        train_config = yaml.unsafe_load(file)

    train_config['model_fp'] = args.model_fp

    extract_features_into_csv(train_config, res_dir_fp)
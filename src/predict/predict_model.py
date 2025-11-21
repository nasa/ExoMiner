"""
Run inference with a model.
"""

# 3rd party
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import load_model
import pandas as pd
import argparse
import yaml
from pathlib import Path
import logging

# local
from src.utils.utils_dataio import InputFnv2 as InputFn, set_tf_data_type_for_features
from src.utils.utils_dataio import get_data_from_tfrecords_for_predictions_table
import models.custom_layers
from src.utils.utils import log_info


def predict_model(config, model_path, res_dir, logger=None):
    """ Run inference with a model.

    :param config: dict, training run parameters
    :param model_path: Path, file path to trained model
    :param res_dir: Path, output directory
    :param logger: logger

    :return:
    """

    config['features_set'] = set_tf_data_type_for_features(config['features_set'])

    # get data from TFRecords files to be displayed in the table with predictions
    data = get_data_from_tfrecords_for_predictions_table(config['datasets'],
                                                         config['data_fields'],
                                                         config['datasets_fps'])

    # load models
    log_info('Loading model...', logger)
    # register_custom_objects()
    model = load_model(filepath=model_path, compile=False)

    if config['write_model_summary']:
        with open(res_dir / 'model_summary.txt', 'w') as f:
            model.summary(print_fn=lambda x: f.write(x + '\n'))

    # plot model and save the figure
    if config['plot_model']:
        plot_model(model,
                   to_file=res_dir / 'model.png',
                   show_shapes=True,
                   show_layer_names=True,
                   rankdir='TB',
                   expand_nested=False,
                   dpi=96)

    scores = {dataset: [] for dataset in config['datasets']}
    for dataset in scores:

        log_info(f'Predicting on dataset {dataset}...', logger)

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

        scores[dataset] = model.predict(
            predict_input_fn(),
            verbose=config['verbose_model'],
        )

    # write results to a csv file
    for dataset in config['datasets']:
        
        log_info(f'Writing predictions for dataset {dataset}...', logger)
        
        if not config['config']['multi_class']:
            data[dataset]['score'] = scores[dataset].ravel()
        else:
            for class_label, label_id in config['label_map'].items():
                data[dataset][f'score_{class_label}'] = scores[dataset][:, label_id]

        predictions_df = pd.DataFrame(data[dataset])

        # map labels to a label id that was used to train the model        
        predictions_df['label_id'] = predictions_df['label'].apply(lambda x: config['label_map'].get(x, -1))  # -1 as default


        # sort in descending order of output
        if not config['config']['multi_class']:
            predictions_df.sort_values(by='score', ascending=False, inplace=True)
        
        predictions_df_fp = res_dir / f'predictions_{dataset}set.csv'
        
        # add metadata
        predictions_df.attrs['experiment'] = res_dir.name
        predictions_df.attrs['dataset'] = dataset
        predictions_df.attrs['label map'] =  config['label_map']
        predictions_df.attrs['created'] = str(pd.Timestamp.now().floor('min'))
        with open(predictions_df_fp, "w") as f:
            for key, value in predictions_df.attrs.items():
                f.write(f"# {key}: {value}\n")
            predictions_df.to_csv(f, index=False)
            

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--config_fp', type=str, help='File path to YAML configuration file.', default=None)
    parser.add_argument('--model_fp', type=str, help='Model file path.', default=None)
    parser.add_argument('--output_dir', type=str, help='Output directory', default=None)
    args = parser.parse_args()

    model_fp = Path(args.model_fp)
    config_fp = Path(args.config_fp)
    output_dir = Path(args.output_dir)

    with(open(config_fp, 'r')) as file:
        predict_config = yaml.unsafe_load(file)

    # set up logger
    predict_config['logger'] = logging.getLogger(name=f'predict_model')
    logger_handler = logging.FileHandler(filename=output_dir / 'predict_model.log', mode='w')
    logger_formatter = logging.Formatter('%(asctime)s - %(message)s')
    predict_config['logger'].setLevel(logging.INFO)
    logger_handler.setFormatter(logger_formatter)
    predict_config['logger'].addHandler(logger_handler)
    predict_config['logger'].info(f'Starting running inference with model {model_fp} in {output_dir}')

    predict_model(predict_config, model_fp, output_dir, logger=predict_config['logger'])

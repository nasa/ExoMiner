"""
Evaluate model.
"""

# 3rd party
from tensorflow.keras.utils import plot_model, custom_object_scope
from tensorflow.keras.models import load_model
import numpy as np
import argparse
import yaml
from pathlib import Path
import logging


# local
from src.utils_dataio import InputFnv2 as InputFn
from src.utils_metrics import get_metrics, get_metrics_multiclass
from models.utils_models import compile_model
from models.models_keras import Time2Vec
from src.utils_train_eval_predict import write_performance_metrics_to_txt_file, set_tf_data_type_for_features


def evaluate_model(config, model_path, res_dir, logger=None):
    """ Evaluate a model.

    :param config: dict, training run parameters
    :param model_path: Path, file path to trained model
    :param res_dir: Path, output directory
    :param logger: logger

    :return:
    """

    config['features_set'] = set_tf_data_type_for_features(config['features_set'])

    # load models
    if logger is None:
        print('Loading model...')
    else:
        logger.info('Loading model...')
    custom_objects = {"Time2Vec": Time2Vec}
    with custom_object_scope(custom_objects):
        model = load_model(filepath=model_path, compile=False)

    if config['write_model_summary']:
        with open(res_dir / 'model_summary.txt', 'w') as f:
            model.summary(print_fn=lambda x: f.write(x + '\n'))

    # plot model and save the figure
    if config['plot_model']:
        plot_model(model,
                   to_file=res_dir / 'model.png',
                   show_shapes=False,
                   show_layer_names=True,
                   rankdir='TB',
                   expand_nested=False,
                   dpi=96)

    # set up metrics to be monitored
    if config['config']['multi_class']: # metrics for multiclass setting
        metrics_list = get_metrics_multiclass(config['label_map'])
    else:
        metrics_list = get_metrics(clf_threshold=config['metrics']['clf_thr'],
                                   num_thresholds=config['metrics']['num_thr'])

    # compile model - loss and metrics
    model = compile_model(model, config, metrics_list, train=False)

    # initialize results dictionary for the evaluated datasets
    res = {}
    for dataset in config['datasets']:

        if dataset == 'predict':
            continue

        if logger is None:
            print(f'Evaluating on dataset {dataset}')
        else:
            logger.info(f'Evaluating on dataset {dataset}')

        # input function for evaluating on each dataset
        eval_input_fn = InputFn(file_paths=config['datasets_fps'][dataset],
                                batch_size=config['evaluation']['batch_size'],
                                mode='EVAL',
                                label_map=config['label_map'],
                                features_set=config['features_set'],
                                online_preproc_params=None,
                                multiclass=config['config']['multi_class'],
                                feature_map=config['feature_map'],
                                label_field_name=config['label_field_name'],
                                )

        callbacks_list = []
        # for callback_name in callbacks_dict:
        #     if 'layer' in callback_name:
        #         callbacks_dict[callback_name].input_fn = eval_input_fn()

        # evaluate model in the given dataset
        res_eval = model.evaluate(x=eval_input_fn(),
                                  y=None,
                                  batch_size=None,
                                  verbose=config['verbose_model'],
                                  sample_weight=None,
                                  steps=None,
                                  callbacks=callbacks_list if dataset == 'train' else None,
                                  max_queue_size=10,
                                  workers=1,
                                  use_multiprocessing=False)

        # add evaluated dataset metrics to result dictionary
        for metric_name_i, metric_name in enumerate(model.metrics_names):
            res[f'{dataset}_{metric_name}'] = res_eval[metric_name_i]

    np.save(res_dir / 'res_eval.npy', res)

    # write results to a txt file
    write_performance_metrics_to_txt_file(res_dir, config['datasets'], res)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--config_fp', type=str, help='File path to YAML configuration file.', default=None)
    parser.add_argument('--model_fp', type=str, help='Model file path.', default=None)
    parser.add_argument('--output_dir', type=str, help='Output directory', default=None)
    args = parser.parse_args()

    model_fp = Path(args.model_fp)
    config_fp = Path(args.config_fp)
    output_dir = Path(args.output_dir)

    with(open(args.config_fp, 'r')) as file:
        eval_config = yaml.unsafe_load(file)

    # set up logger
    eval_config['logger'] = logging.getLogger(name=f'evaluate_model')
    logger_handler = logging.FileHandler(filename=output_dir / 'evaluate_model.log', mode='w')
    logger_formatter = logging.Formatter('%(asctime)s - %(message)s')
    eval_config['logger'].setLevel(logging.INFO)
    logger_handler.setFormatter(logger_formatter)
    eval_config['logger'].addHandler(logger_handler)
    eval_config['logger'].info(f'Starting evaluating model {model_fp} in {output_dir}')

    evaluate_model(eval_config, model_fp, output_dir, logger=eval_config['logger'])

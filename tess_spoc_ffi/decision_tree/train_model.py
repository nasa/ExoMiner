"""
Train, evaluate, and run inference using a gradient boosted trees model.
"""

# 3rd party
import argparse
import yaml
from pathlib import Path
import logging
import ydf
import pandas as pd

# local
from src.utils.utils_dataio import set_tf_data_type_for_features
from tess_spoc_ffi.decision_tree.utils_data_io import create_tf_data_dataset
from src.utils.utils_dataio import get_data_from_tfrecords_for_predictions_table


BATCH_SIZE = 1000


def get_auxiliary_data(config):
    """ Get auxiliary data needed for predictions table.

    :param config: dict, training run parameters

    :return: data, dict with pandas DataFrame with auxiliary data split into 'train', 'val', and 'test' sets
    """

    if config['dataset_type'] == 'tfrecord':
        # get data from TFRecords files to be displayed in the table with predictions
        data = get_data_from_tfrecords_for_predictions_table(config['datasets'],
                                                             config['data_fields'],
                                                             config['datasets_fps'])
        data = {dataset: pd.DataFrame(dataset_dict) for dataset, dataset_dict in data.items()}
    elif config['dataset_type'] == 'dataframe':
        data = {dataset: pd.read_csv(dataset_fps, usecols=config['data_fields'])
                for dataset, dataset_fps in config['datasets_fps'].items()}
    else:
        raise ValueError(f'Dataset type {config["dataset_type"]} not recognized.')

    return data


def create_dataset(config):
    """ Prepare dataset for learner.

    :param config: dict, training run parameters

    :return: ds, dictionary with data split into 'train', 'val', and 'test'
    """

    if config['dataset_type'] == 'tfrecord':

        # set tensorflow data type for features in the feature set
        config['features_set'] = set_tf_data_type_for_features(config['features_set'])

        ds = {dataset: create_tf_data_dataset(config['datasets_fps'][dataset],
                                              config['features_set'],
                                              config['label_map'],
                                              config['label_field_name'].batch(BATCH_SIZE)
                                              )
                          for dataset in config['datasets']
              }

    elif config['dataset_type'] == 'dataframe':
        ds = {dataset: pd.read_csv(config['datasets_fps'][dataset],
                                   usecols=config['features_set'],

                                   )  for dataset in config['datasets']}

    else:
        raise ValueError(f'Dataset type {config["dataset_type"]} not recognized.')

    return ds


def train_model(config, model_dir, logger=None):
    """ Train a model.

    :param config: dict, training run parameters
    :param model_dir: Path, directory used to save the trained model
    :param logger: logger

    :return:
    """

    # get auxiliary data for predictions table
    aux_data = get_auxiliary_data(config)

    # initialize learner hyperparameters
    learner = ydf.GradientBoostedTreesLearner(
        label=config['label_field_name'],
        task=ydf.Task.CLASSIFICATION,
        # Enable feature selection using the "backward selection" algorithm.
        # feature_selector=ydf.BackwardSelectionFeatureSelector(),
    )

    with open(model_dir / 'model_hyperparameters.yml', 'w') as model_hp_file:
        yaml.dump(learner.hyperparameters, model_hp_file, sort_keys=False)

    # create dataset for learner
    ds = create_dataset(config)

    # fit the model to the training data
    if logger is None:
        print('Training model...')
    else:
        logger.info('Training model...')
    model = learner.train(
        ds['train'],
        valid=ds['val'],
        verbose=config['verbose'],
    )

    # get model description after training
    with open(str(model_dir / 'model_description.txt'), 'w') as trained_model_res:
        trained_model_res.write(model.describe())

    if logger is None:
        print('Saving model...')
    else:
        logger.info('Saving model...')
    # save model
    model.save(str(model_dir / 'model'))

    # conduct analysis of each dataset
    for dataset in config['datasets']:
        analysis = model.analyze(ds[dataset], sampling=0.1)
        analysis.to_file(str(model_dir / f'analysis_{dataset}.html'))

    # evaluate and run inference using the trained model
    for dataset in config['datasets']:

        if logger is None:
            print(f'Evaluating model on dataset {dataset}...')
        else:
            logger.info(f'Evaluating model on dataset {dataset}...')

        evaluation = model.evaluate(ds[dataset])

        with open(str(model_dir / f'evaluation_{dataset}.txt'), 'w') as evaluation_file:
            evaluation_file.write(str(evaluation))

        with open(str(model_dir / f'evaluation_{dataset}.html'), 'w') as evaluation_file:
            evaluation_file.write(evaluation.html())

    # run inference
    scores = {dataset: [] for dataset in config['datasets']}
    for dataset in config['datasets']:
        scores[dataset] = model.predict(ds[dataset])

    # process multiclass results if applicable
    for dataset in config['datasets']:
        if not config['config']['multi_class']:
            aux_data[dataset]['score'] = scores[dataset].ravel()
        else:
            for class_label, label_id in config['label_map'].items():
                aux_data[dataset][f'score_{class_label}'] = scores[dataset][:, label_id]

    # write results to a csv file
    for dataset in config['datasets']:

        # sort in descending order of output
        if not config['config']['multi_class']:
            aux_data[dataset].sort_values(by='score', ascending=False, inplace=True)
        aux_data[dataset].to_csv(model_dir / f'ranked_predictions_{dataset}set.csv', index=False)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--config_fp', type=str, help='File path to YAML configuration file.',
                        default=None)
    parser.add_argument('--model_dir', type=str, help='Output directory', default=None)
    parser.add_argument('--model_fp', type=str, help='Path to pre-existing model', default=None)
    args = parser.parse_args()

    model_dir_fp = Path(args.model_dir)
    config_fp = Path(args.config_fp)

    with(open(args.config_fp, 'r')) as file:
        train_config = yaml.unsafe_load(file)

    # set up logger
    train_config['logger'] = logging.getLogger(name=f'train_model')
    logger_handler = logging.FileHandler(filename=model_dir_fp / 'train_model.log', mode='w')
    logger_formatter = logging.Formatter('%(asctime)s - %(message)s')
    train_config['logger'].setLevel(logging.INFO)
    logger_handler.setFormatter(logger_formatter)
    train_config['logger'].addHandler(logger_handler)
    train_config['logger'].info(f'Starting training model in {model_dir_fp}')

    train_config['model_fp'] = args.model_fp

    train_model(train_config, model_dir_fp, logger=train_config['logger'])

"""
Test input function.
"""

# 3rd party
import yaml

# local
from src.utils.utils_dataio import InputFnv2


def test_train_input_fn(config):
    """ Test input function by calling it to iterate on a few batches.

        Args:
            config: dict, setup parameters for the input function
        Returns:

    """

    # Initialize your InputFnv2 instance with mock_config
    input_fn = InputFnv2(
        file_paths=config['datasets_fps']['train'],
        batch_size=config['training']['batch_size'],
        mode='TRAIN',
        label_map=config['label_map'],
        data_augmentation=config['training']['data_augmentation'],
        online_preproc_params=config['training']['online_preprocessing_params'],
        features_set=config['features_set'],
        category_weights=config['training']['category_weights'],
        multiclass=config['config']['multi_class'],
        feature_map=config['feature_map'],
        shuffle_buffer_size=config['training']['shuffle_buffer_size'],
        label_field_name=config['label_field_name'],
    )

    # Call the input function to get a dataset
    dataset = input_fn()

    # Iterate over a few batches to inspect the data
    for batch in dataset.take(2):  # Take 2 batches for inspection
        features, labels = batch[0], batch[1]

        # Print the keys of the features dictionary
        print("Features keys:", list(features.keys()))

        # Optionally, print the shape and dtype of each feature
        for key, value in features.items():
            print(f"Feature '{key}': shape={value.shape}, dtype={value.dtype}")

        # Print labels shape and dtype (if applicable)
        print("Labels:", labels)

# run the test function
if __name__ == '__main__':

    # define a mock configuration for testing
    config_fp = '/home6/msaragoc/work_dir/Kepler-TESS_exoplanet/experiments/tess_spoc_ffi/cv_exominerplusplus_tess-spoc-2min-s1-s88_targetssharedffi_5-27-2025_1104/cv_iter_0/models/model0/config_cv.yaml'
    with open(config_fp, 'r') as file:
        input_fn_config = yaml.unsafe_load(file)

    # print(config)
    test_train_input_fn(input_fn_config)

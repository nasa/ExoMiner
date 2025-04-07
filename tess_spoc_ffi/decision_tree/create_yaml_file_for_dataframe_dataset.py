"""
Script used to create dataset using csv files instead of tfrecords.
"""

# 3rd party
from pathlib import Path
import pandas as pd
import yaml
import multiprocessing


def create_dataset_for_cv_iteration(cv_dir_fp, datasets):
    """ Prepare dataset for learner.

    :param cv_dir_fp: dict, for each dataset, provide Path objects to the csv files in each dataset
    :param datasets: list, 'train', 'val', 'test'

    :return: ds, dictionary with data split into 'train', 'val', and 'test'
    """

    print(f'Iterating on CV iteration {cv_dir_fp.name}...')

    cv_iteration_fps = {dataset: '' for dataset in datasets}
    for dataset in datasets:

        dataset_csv_fps = list((cv_dir_fp / 'models_ffi').glob(f'model[0-9]/extracted_learned_features/'
                                                               f'intermediate_outputs_model_{dataset}.csv'))
        print(f'Found {len(dataset_csv_fps)} csv files for dataset {dataset} in {cv_dir_fp.name}')
        # concatenating datasets across models
        csv_dataset = pd.concat([pd.read_csv(fp) for fp in dataset_csv_fps], axis=0, ignore_index=False)
        print(f'csv file for dataset {dataset} in {cv_dir_fp.name}: {len(csv_dataset)} examples.')

        # # manipulate csv file
        # for col in ['uid', 'label', 'obs_type']:
        #     csv_dataset[col] = csv_dataset[col].apply(lambda x: x[2:-1])

        save_fp = cv_dir_fp / 'models_ffi' / f'intermediate_outputs_model_{dataset}.csv'
        cv_iteration_fps[dataset] = save_fp
        csv_dataset.to_csv(save_fp, index=False)

    return cv_iteration_fps


if __name__ == "__main__":

    root_dir = Path('/home6/msaragoc/work_dir/Kepler-TESS_exoplanet/experiments/tess_spoc_ffi/cv_tess-spoc-ffi_s36-s72_multisector_s56-s69_with2mindata_exominer_newarchitecture_allfeatures_globalmaxpoolingextractedfeatures_ffi_vs_2min_3-25-2025_1810')
    dataset_config_yaml_fp = root_dir / 'cv_iterations_csv_dataset.yaml'
    datasets = ['train', 'val', 'test']
    cv_dir_fps = list(root_dir.glob('cv_iter_[0-9]'))

    print(f'Found {len(cv_dir_fps)} CV iteration directories')

    # parallel processing
    n_procs = 36  # number of parallel processes
    pool = multiprocessing.Pool(processes=n_procs)
    async_results = [pool.apply_async(create_dataset_for_cv_iteration, (cv_dir_fp, datasets))
                     for cv_dir_fp in cv_dir_fps]
    pool.close()
    pool.join()
    config_dataset_yaml_lst = [async_result.get() for async_result in async_results]

    # # sequential
    # config_dataset_yaml_lst = []
    # for cv_dir_fp in cv_dir_fps:
    #     cv_iteration_fps = create_dataset_for_cv_iteration(cv_dir_fp, datasets)
    #     config_dataset_yaml_lst.append(cv_iteration_fps)

    with open(dataset_config_yaml_fp, 'w') as dataset_config_file:
        yaml.dump(config_dataset_yaml_lst, dataset_config_file, )

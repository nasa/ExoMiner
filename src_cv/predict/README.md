# CV Predict Pipeline

This pipeline runs cross-validation prediction on TFRecord datasets using trained models.

## Prerequisites

Before running this prediction pipeline, you must have:
- Completed a CV training experiment with trained models
- Saved model checkpoints in the specified `MODELS_CV_ROOT_DIR` directory
- Generated normalization statistics for each CV fold in the expected format
- Prepared a TFRecord dataset with un-normalized features ready for inference

## Step 1: Configure Preprocessing Parameters

Edit `src_cv/preprocessing/config_preprocess_cv_folds_predict_tfrecord_dataset.yaml`:

- Set `src_tfrec_dir`: Path to the TFRecord dataset directory (with un-normalized features) for inference
- Set `cv_folds_fps`: List of paths to normalization statistics folders for each CV iteration (e.g., "/path/to/cv-train/cv_iter_[0-4]/norm_stats")
- Ensure each normalization statistics folder contains required NumPy files with feature statistics
- Configure `num_processes`: Number of parallel processes for data normalization within a CV iteration.

## Step 2: Configure Prediction Pipeline parameters

Edit `src_cv/predict/config_cv_predict.yaml`:

- Set `data_fields`: list of **scalar** features/attributes that are stored in the source TFRecord dataset for each TCE and that you want to see displayed in the prediction table
- [Optional] Set parameters related to batch size inference, verbose, plot model architecture, etc. Default values are ok.

## Step 3: Configure PBS Job Script

Edit `src_cv/predict/Submitjob_cv_predict.pbs`:

- Set `PYTHONPATH`: Root directory of the codebase
- Ensure Python environment is activated
- Set `MODELS_CV_ROOT_DIR`: Root directory containing trained CV models
- Set `N_CV_ITERS`: Number of CV iterations/models for inference
- Set `PREPROCESSED_DATA`: "true" to preprocess data if TFRecord features are un-normalized
- Set `DELETE_PREPROCESSED_DATA`: "true" to use normalized TFRecord dataset after processing
- Set `N_GPUS`: Number of GPUs to use (should align with `NUM_JOBS_PARALLEL`)
- Set `NUM_TOTAL_JOBS` and `NUM_JOBS_PARALLEL`: Configure based on cluster job array settings (e.g., PBS header "#PBS -J 0-9")

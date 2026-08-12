#!/bin/bash
# Run 

# export OUTPUT_DIR=/u/msaragoc/work_dir/Kepler-TESS_exoplanet/experiments/phot_disps/cv_tess-spoc-tces_2min-s1-s98_10-folds_strict-pcs_$(date +%Y%m%d_%H%M%S)
export OUTPUT_DIR=/u/msaragoc/work_dir/Kepler-TESS_exoplanet/experiments/phot_disps/cv_tess-spoc-tces_2min-s1-s98_10-folds_strict-pcs_20260716_095902

mkdir -p $OUTPUT_DIR

PYTHONPATH=/u/msaragoc/work_dir/Kepler-TESS_exoplanet/codebase_aux_loss_source_offset
SUBMIT_JOB_CV_TRAIN_MODELS=$PYTHONPATH/src_cv/train/Submitjob_cv_train_models.pbs
SUBMIT_JOB_EVAL_PRED_CV_ENSEMBLES=$PYTHONPATH/src_cv/train/Submitjob_cv_ensemble_folds.pbs
SUBMIT_JOB_AGGREGATE=$PYTHONPATH/src_cv/train/Submitjob_aggregate_predictions.pbs

N_CV_ITERS=10  # 5
N_MODELS_PER_CV_ITER=10
N_JOBS_ARRAY="0-$((N_CV_ITERS * N_MODELS_PER_CV_ITER - 1))"

CV_RUN_LOG_FP="$OUTPUT_DIR/cv_run.log"
touch "$CV_RUN_LOG_FP"

echo "Starting CV training run at $(date)" | tee "$CV_RUN_LOG_FP"
echo "OUTPUT_DIR=$OUTPUT_DIR" | tee -a "$CV_RUN_LOG_FP"
echo "N_CV_ITERS=$N_CV_ITERS" | tee -a "$CV_RUN_LOG_FP"
echo "N_MODELS_PER_CV_ITER=$N_MODELS_PER_CV_ITER" | tee -a "$CV_RUN_LOG_FP"

echo "Submitting training jobs array with $N_CV_ITERS CV iters and $N_MODELS_PER_CV_ITER models per iter => total $((N_CV_ITERS * N_MODELS_PER_CV_ITER)) jobs" | tee -a "$CV_RUN_LOG_FP"
TRAIN_JOBID=$(qsub \
  -J $N_JOBS_ARRAY \
  -v N_CV_ITERS=$N_CV_ITERS,N_MODELS_PER_CV_ITER=$N_MODELS_PER_CV_ITER,OUTPUT_DIR=$OUTPUT_DIR \
  $SUBMIT_JOB_CV_TRAIN_MODELS)
echo "JOBID: $TRAIN_JOBID" | tee -a "$CV_RUN_LOG_FP"

N_JOBS_ARRAY=0-$((N_CV_ITERS - 1))

echo "Submitting ensemble evaluation job array with $N_JOBS_ARRAY => total $((N_CV_ITERS)) jobs" | tee -a "$CV_RUN_LOG_FP"
RUN_ENSEMBLE_JOBID=$(qsub \
  -W depend=afterok:$TRAIN_JOBID \
  -J $N_JOBS_ARRAY \
  -v N_CV_ITERS=$N_CV_ITERS,N_MODELS_PER_CV_ITER=$N_MODELS_PER_CV_ITER,OUTPUT_DIR=$OUTPUT_DIR \
  $SUBMIT_JOB_EVAL_PRED_CV_ENSEMBLES)

RUN_ENSEMBLE_JOB_ROOT=$(echo "$RUN_ENSEMBLE_JOBID" | cut -d'[' -f1)

# wait for the entire ensemble job array to finish successfully, then run the aggregation
echo "Submitting final aggregation job" | tee -a "$CV_RUN_LOG_FP"
AGGREGATE_JOBID=$(qsub \
  -W depend=afterok:$RUN_ENSEMBLE_JOBID \
  -v OUTPUT_DIR=$OUTPUT_DIR \
  $SUBMIT_JOB_AGGREGATE)

echo "AGGREGATE JOBID: $AGGREGATE_JOBID" | tee -a "$CV_RUN_LOG_FP"
# Ran a cross-validation (CV) experiment on a local system (single/multi-GPU).

# file path to directory in which to save the Python output (std.out prints)
PYOUT_DIR=/home/msaragoc/Projects/exoplnt_dl/experiments/
# file path to Python CV main script
CV_SCRIPT=/home/msaragoc/Projects/exoplnt_dl/codebase/src_cv/cv.py
# file path to configuration file for CV experiment
CONFIG_FILE=/home/msaragoc/Projects/exoplnt_dl/codebase/src_cv/config_cv_train.yaml

# Number of total CV iterations run is min(number of folds defined in the config file, N_CV_ITERS * N_CV_ITERS_SIMULT)
N_CV_ITERS=4  # number of groups of CV iterations that are run at the same time
N_CV_ITERS_SIMULT=2  # number of CV iterations per group that are run at the same time each one using a different GPU
for ((cv_i=0; cv_i<=$N_CV_ITERS; cv_i++))
# cv_i in {0...$N_CV_ITERS}
do
    echo "Running $N_CV_ITERS_SIMULT CV iterations ($(expr $cv_i + 1) out of $(expr $N_CV_ITERS + 1))..."
    mpiexec -np $N_CV_ITERS_SIMULT python $CV_SCRIPT --job_idx=$cv_i  --config_file=$CONFIG_FILE &> "$PYOUT_DIR"cv_iter_$cv_i.txt
    echo "Finished running $N_CV_ITERS_SIMULT CV iterations ($(expr $cv_i + 1) out of $(expr $N_CV_ITERS + 1))"
done

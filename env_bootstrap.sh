#!/usr/bin/env bash

# initialize conda and activate conda environment
# module use -a /swbuild/analytix/tools/modulefiles
# module load miniconda3/v4
# source activate exoplnt_dl_tf2_13

# micromamba shell hook
# eval "$(micromamba shell hook --shell=bash)"
source /home6/msaragoc/.mamba_init.sh
micromamba activate exoplnt_dl_tf2_13_gpu

set -euo pipefail

# project pythonpath
export PYTHONPATH=/home6/msaragoc/work_dir/Kepler-TESS_exoplanet/codebase_aux_loss_source_offset/

# any other shared exports

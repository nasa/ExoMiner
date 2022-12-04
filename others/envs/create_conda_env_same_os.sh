# Script used to create environment for exoplanet project in same OS using a conda environment explicit spec file.
# Source file obtained by running command `conda list --explicit > spec-file.txt` for the source conda environment.

# conda environment's name
CONDA_ENV_NAME=exoplnt_dl
# source conda environment explicit spec file
SRC_CONDA_ENV_LIST="/Users/msaragoc/OneDrive - NASA/Projects/exoplanet_transit_classification/codebase/others/envs/conda_env_exoplnt_dl_hecc.txt"

# make conda detectable by the current bash shell
source "$HOME"/.bash_profile

# create conda environment based on source conda environment explicit spec file
conda create --name $CONDA_ENV_NAME --file "$SRC_CONDA_ENV_LIST"

# activate environment just created
conda activate $CONDA_ENV_NAME

# --- install pip modules ---
pip install hpbandster
# GPU CUDA compatibility: https://www.tensorflow.org/install/source#gpu
pip install tensorflow==2.5.0
#pip install tensorboard==2.5.0
pip install tensorflow-probability==0.11.0

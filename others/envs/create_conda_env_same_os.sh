# Script used to create environment for exoplanet project in same OS using a conda environment explicit spec file.
# Source file obtained by running command `conda list --explicit > spec-file.txt` for the source conda environment.

# conda environment's name
CONDA_ENV_NAME=exoplnt_dl
# source conda environment explicit spec file
SRC_CONDA_ENV_LIST="/Users/msaragoc/OneDrive - NASA/Projects/exoplanet_transit_classification/codebase/others/envs/conda_env_exoplnt_dl_hecc.txt"
# source file with setup to initialize conda - check which file sets up conda init (e.g., .bashrc, .bash_profile, .profile)!
SRC_CONDA_INIT="$HOME"/.bashrc

# make conda detectable by the current bash shell
source $SRC_CONDA_INIT

# create conda environment based on source conda environment explicit spec file
conda create --name $CONDA_ENV_NAME --file "$SRC_CONDA_ENV_LIST"

# activate environment just created
conda activate $CONDA_ENV_NAME

# --- install pip modules for HECC/non-M2 MacBook ---
pip install hpbandster
# GPU CUDA compatibility: https://www.tensorflow.org/install/source#gpu
pip install tensorflow==2.5.0
#pip install tensorboard==2.5.0
pip install tensorflow-probability==0.11.0
pip uninstall numpy
conda install numpy


# --- install pip modules for M2 MacBook ---
pip install hpbandster
pip install "grpcio>=1.37.0,<2.0"
pip install "numpy>=1.22.3,<1.23.0"
pip install h5py
pip install pydl
pip install pydot
pip install tensorflow-macos
pip install tensorflow-metal
pip install --upgrade tensorflow-probability

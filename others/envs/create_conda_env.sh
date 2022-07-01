# Script used to create environment for exoplanet project

# conda environment's name
CONDA_ENV_NAME=exoplnt_dl_from_sh

# make conda detectable by the current bash shell
source "$HOME"/.bash_profile
#conda init bash

# start by creating empty environment
conda create --name $CONDA_ENV_NAME

# activate environment just created
conda activate $CONDA_ENV_NAME
#source activate exoplnt_dl_from_sh

# --- install Python modules ---

# conda installations
conda install python==3.8.10
conda install scikit-learn
conda install numpy
conda install pandas
conda install matplotlib
conda install yaml
conda install jupyter
conda install astropy
conda install mpi4py
conda install scipy
conda install -c conda-forge pydot
conda install -c anaconda graphviz
#conda install -c conda-forge tensorflow
#confa install -c anaconda tensorflow

# pip installations
pip install hpbandster
#conda install -c conda-forge hpbandster
pip install tensorflow==2.3.0
#pip install tensorboard==2.5.0
pip install tensorflow-probability==0.11.0

# For HECC (comment if not using HECC)
#conda install cudnn=7.6.5
#conda install cudatoolkit=10.1.243
#conda install cupti=10.1.168
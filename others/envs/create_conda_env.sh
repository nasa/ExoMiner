# Script used to create environment for exoplanet project

# conda environment's name
CONDA_ENV_NAME=exoplnt_dl

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
conda install scikit-learn=1.1.1
conda install numpy
conda install pandas
conda install matplotlib
conda install yaml
conda install jupyter
conda install astropy
# mpich and openmpi work for Linux and Mac OSes; msmpi for Windows OS
# check which mpiexec the conda environment is using by running `which mpiexec`; the output should be coming from your conda environment directory
#conda install mpi4py  # mpich implementation by default
conda install -c conda-forge mpi4py openmpi
conda install scipy
conda install -c conda-forge pydot
conda install -c anaconda graphviz
#conda install -c conda-forge tensorflow
#conda install -c anaconda tensorflow
conda install -c anaconda pydot
conda install -c conda-forge pydl

# pip installations
pip install hpbandster
#conda install -c conda-forge hpbandster
# GPU CUDA compatibility: https://www.tensorflow.org/install/source#gpu
pip install tensorflow==2.5.0
#pip install tensorboard==2.5.0
pip install tensorflow-probability==0.11.0

# For systems using GPUs (e.g., HECC Pleiades, remembrane)
conda install cudnn  # cudnn 8.1.x, cudatoolkit 11.x, cupti 11.x - these libraries should be these versions
# #conda install cudnn=7.6.5
# #conda install cudatoolkit=10.1.243
# #conda install cupti=10.1.168

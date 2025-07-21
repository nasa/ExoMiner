FROM continuumio/miniconda3:25.3.1-1

LABEL org.opencontainers.image.description="This image contains the ExoMiner Pipeline application." \
      org.opencontainers.image.title="ExoMiner Pipeline" \
      org.opencontainers.image.version="1.0.0" \
      org.opencontainers.image.authors="Miguel Martinho, <miguel.martinho@nasa.gov>" \
      org.opencontainers.image.source="https://github.com/nasa/ExoMiner" \
      org.opencontainers.image.documentation="https://github.com/nasa/ExoMiner/tree/main/docs/index.md" \
      org.opencontainers.image.exominer.model="ExoMiner++ (TESS SPOC 2-min S1-S67, Ensemble from CV Iteration 0) | DOI: https://doi.org/10.48550/arXiv.2502.09790" \
      org.opencontainers.image.revision="" \
      org.opencontainers.image.created=""

# set working directory
WORKDIR /app

# set environment variable so Python recognizes modules in code repository
ENV PYTHONPATH="/app"

ARG CONDA_TOKEN
ARG CONDA_ENV
ARG CONDARC

# adding conda token using conda token set
#RUN conda config --remove-key default_channels
#RUN conda install --freeze-installed conda-token
#RUN conda token set $CONDA_TOKEN
#RUN conda config --add channels https://repo.anaconda.cloud/repo/main
#RUN conda config --add channels https://repo.anaconda.cloud/repo/r
#RUN conda config --add channels https://repo.anaconda.cloud/repo/msys2
#RUN conda config --show channels
# adding conda token to channels
RUN conda config --remove channels defaults && \
    conda config --add channels https://repo.anaconda.cloud/repo/main/t/${CONDA_TOKEN} && \
    conda config --add channels https://repo.anaconda.cloud/repo/r/t/${CONDA_TOKEN} && \
    conda config --add channels https://repo.anaconda.cloud/repo/msys2/t/${CONDA_TOKEN} && \
    conda config --add channels conda-forge && \
    echo "default_channels: []" >> ${CONDARC}
#    echo "default_channels: []" >> /root/.condarc
#    conda config --set channel_priority strict

# copy Conda environment YAML file
COPY exominer_pipeline/${CONDA_ENV} conda_env_exoplnt_dl.yml

# miniconda ------
# when not using NASA system
#RUN conda config --remove-key channels
#RUN conda config --add channels conda-forge

# use the modified YAML to create the environment
RUN conda env create -f conda_env_exoplnt_dl.yml --yes && \
    conda clean --all -f -y && \
    pip cache purge && \
    rm -f ${CONDARC}

# copy application code
COPY . .

# create additional folders for model and data
RUN mkdir -p /model /data

# copy ExoMiner TF-Keras model
COPY exominer_pipeline/data/model.keras /model/

# copy normalization statistics
COPY exominer_pipeline/data/norm_stats /data/norm_stats

# set image to always run ExoMiner Pipeline
ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "exoplnt_dl", "python", "exominer_pipeline/run_pipeline.py"]

# show information about the arguments if no argument is provided
CMD ["--help"]

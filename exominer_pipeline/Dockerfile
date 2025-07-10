#FROM mambaorg/micromamba:2.3.0
FROM continuumio/miniconda3
# python:3.11.5

LABEL org.opencontainers.image.description="This image contains the ExoMiner Pipeline application." \
      org.opencontainers.image.title="ExoMiner Pipeline" \
      org.opencontainers.image.version="1.0.0." \
      org.opencontainers.image.authors="Miguel Martinho <mig.js.martinho@gmail.com>, <miguel.martinho@nasa.gov>" \
      org.opencontainers.image.source="https://github.com/nasa/Exominer/tree/main" \
      org.opencontainers.image.revision="" \
      org.opencontainers.image.created="" \

#      org.opencontainers.image.licenses="MIT" \

# set working directory
WORKDIR=/app

# set environment variable so Python recognizes modules in code repository
ENV PYTHONPATH="/app"

# accept token as build argument
ARG ANACONDA_TOKEN
#ENV ANACONDA_TOKEN=${ANACONDA_TOKEN}
#ARG ANACONDA_USER

## store token in Conda's expected location
#RUN mkdir -p /root/.conda && \
#echo "${ANACONDA_TOKEN}" > /root/.conda/anaconda.token
#RUN conda install anaconda-client
#RUN conda install -c conda-forge anaconda-client
#RUN anaconda login

## Copy custom .condarc for build-time use only
#COPY exominer_pipeline/.condarc /tmp/.condarc

#RUN echo "channels:" > /root/.condarc && \
#    echo "  - https://${ANACONDA_TOKEN}:@conda.anaconda.org/<your-channel>" >> /root/.condarc

# Set up .condarc with token-authenticated channels
#RUN echo "channels:" > /root/.condarc && \
#    echo "  - https://${ANACONDA_TOKEN}@repo.anaconda.cloud/repo/main" >> /root/.condarc && \
#    echo "  - https://${ANACONDA_TOKEN}@repo.anaconda.cloud/repo/r" >> /root/.condarc && \
#    echo "  - https://${ANACONDA_TOKEN}@repo.anaconda.cloud/repo/msys2" >> /root/.condarc && \
#    echo "channel_priority: strict" >> /root/.condarc
#RUN conda config --remove channels defaults && \
#    conda config --add channels https://${ANACONDA_TOKEN}@repo.anaconda.cloud/repo/msys2 && \
#    conda config --add channels https://${ANACONDA_TOKEN}@repo.anaconda.cloud/repo/r && \
#    conda config --add channels https://${ANACONDA_TOKEN}@repo.anaconda.cloud/repo/main && \
#    conda config --set channel_priority strict
#
#ENV CONDARC=/root/.condarc

ARG ANACONDA_TOKEN
ENV ANACONDA_TOKEN=${ANACONDA_TOKEN}

# Set up authenticated channels
RUN echo "channels:" > /root/.condarc && \
    echo "  - https://${ANACONDA_TOKEN}@repo.anaconda.cloud/repo/main" >> /root/.condarc && \
    echo "default_channels:" > /root/.condarc && \
    echo "  - https://${ANACONDA_TOKEN}@repo.anaconda.cloud/repo/main" >> /root/.condarc && \
    echo "channel_priority: strict" >> /root/.condarc
ENV CONDARC=/root/.condarc

#RUN conda config --remove-key default_channels &&  \
RUN conda install --freeze-installed conda-token
RUN conda token set $ANACONDA_TOKEN

#RUN conda install anaconda::conda-token
#RUN conda token set $ANACONDA_TOKEN

# copy Conda environment YAML file
COPY exominer_pipeline/conda_env_exoplnt_dl.yml .

#SHELL ["conda", "run", "-n", "exoplnt_dl", "/bin/bash", "-c"]

# create the Conda environment
# use the custom .condarc to create the environment
#RUN conda config --file /tmp/.condarc --show && CONDARC=/tmp/.condarc conda env create -f conda_env_exoplnt_dl.yml --yes
# remove the temporary .condarc
#RUN rm -f /tmp/.condarc
#RUN conda config --show channels
#RUN conda info
#RUN conda env create -f conda_env_exoplnt_dl.yml --yes
## Copy the original YAML
#COPY exominer_pipeline/conda_env_exoplnt_dl.yml .

## Inject channels into the YAML file
#RUN ANACONDA_TOKEN=${ANACONDA_TOKEN} && \
#    echo "channels:" > tmp_channels.yml && \
#    echo "  - https://${ANACONDA_TOKEN}@repo.anaconda.cloud/repo/main" >> tmp_channels.yml && \
#    echo "  - https://${ANACONDA_TOKEN}@repo.anaconda.cloud/repo/r" >> tmp_channels.yml && \
#    echo "  - https://${ANACONDA_TOKEN}@repo.anaconda.cloud/repo/msys2" >> tmp_channels.yml && \
#    echo "  - defaults" >> tmp_channels.yml && \
#    cat tmp_channels.yml conda_env_exoplnt_dl.yml > conda_env_exoplnt_dl_with_channels.yml && \
#    rm tmp_channels.yml

# Use the modified YAML to create the environment
RUN conda env create -f conda_env_exoplnt_dl.yml --yes

# create environment using micromamba
# RUN micromamba --verbose env create -f conda_env_exoplnt_dl.yml --yes --strict-channel-priority

# activate the environment using Conda
#SHELL ["conda", "run", "-n", "myenv", "/bin/bash", "-c"]
# activate the environment using Mamba; subsequent RUN commands use this
#SHELL ["micromamba", "run", "-n", "exoplnt_dl_tf2_13", "/bin/bash", "-c"]

## remove token after use for security
#RUN rm -f /root/.conda/anaconda.token

# copy application code
COPY . .

# create additional folders for model and data
RUN mkdir -p /model /data

# copy ExoMiner TF-Keras model
COPY exominer_pipeline/data/model.keras /model/

# copy normalization statistics
COPY exominer_pipeline/data/norm_stats /data/

# this code always runs at the start of the image
#ENTRYPOINT ["top", "-b"]

# set image to always run ExoMiner Pipeline
#ENTRYPOINT ["python", "exominer_pipeline/run_pipeline.py"]
#ENTRYPOINT ["micromamba", "run", "-n", "exoplnt_dl_tf2_13", "python", "exominer_pipeline/run_pipeline.py"]
ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "exoplnt_dl", "python", "exominer_pipeline/run_pipeline.py"]

# show information about the arguments if no argument is provided
CMD ["--help"]

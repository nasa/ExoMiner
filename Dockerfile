# ------------------------------------------------------------
# Base image with micromamba
# ------------------------------------------------------------
FROM mambaorg/micromamba:2.3.0

LABEL org.opencontainers.image.description="This image contains the ExoMiner Pipeline application." \
      org.opencontainers.image.title="ExoMiner Pipeline" \
      org.opencontainers.image.authors="Miguel Martinho, <miguel.martinho@nasa.gov>" \
      org.opencontainers.image.source="https://github.com/nasa/ExoMiner" \
      org.opencontainers.image.exominer.model="ExoMiner++ (TESS SPOC 2-min) | DOI: https://doi.org/10.48550/arXiv.2502.09790"

ARG MAMBA_DOCKERFILE_ACTIVATE=1
ARG ENV_FILE=environment.yml
ARG REQS_FILE=requirements.txt
ARG GIT_COMMIT_HASH
ARG BUILD_DATE

USER root
WORKDIR /app
ENV PYTHONPATH="/app"

RUN mkdir -p /app /models /data

# ------------------------------------------------------------
# Install dependencies first
# ------------------------------------------------------------
COPY exominer_pipeline/${ENV_FILE} /app/environment.yml
COPY exominer_pipeline/${REQS_FILE} /app/requirements.txt

# Conda env & Pip stack
RUN MAMBA_EXTRACT_THREADS=1 MAMBA_NUM_THREADS=1 micromamba install -vv -y -n base -f /app/environment.yml \
 && micromamba run -n base python -m pip install uv \
 && micromamba run -n base uv pip install --system --no-cache -r /app/requirements.txt \
 && micromamba clean --all --yes \
 && rm -rf /root/.cache

# ------------------------------------------------------------
# Copy models, data and app Code 
# ------------------------------------------------------------
ADD exominer_pipeline_data/models/models.tar /models/
RUN chmod -R u+rwX /models/

COPY exominer_pipeline_data/norm_stats/phot_vetting /data/norm_stats/phot_vetting/
COPY exominer_pipeline_data/norm_stats/planet_validation /data/norm_stats/planet_validation/

# 2. Copy the pipeline AND your other necessary root scripts to /app
COPY exominer_pipeline /app/exominer_pipeline
COPY src /app/src
COPY src_preprocessing /app/src_preprocessing
COPY models /app/models
COPY query_dv_reports.py /app/query_dv_reports.py

# 4. Modify pipeline info 
RUN sed -i "s|^Git Commmit Hash:.*|Git Commmit Hash: ${GIT_COMMIT_HASH}|" /app/exominer_pipeline/pipeline_info.yaml \
 && sed -i "s|^Build Date:.*|Release Date: ${BUILD_DATE}|" /app/exominer_pipeline/pipeline_info.yaml

# ------------------------------------------------------------
# Entrypoint: run ExoMiner pipeline
# ------------------------------------------------------------
ENTRYPOINT ["/usr/local/bin/_entrypoint.sh", "python", "exominer_pipeline/run_pipeline.py"]
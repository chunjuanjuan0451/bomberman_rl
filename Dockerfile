# Reproducible tournament-compatible environment.  The original floating
# installs eventually selected Python 3.14, for which its TensorFlow command
# has no compatible wheel.  Pin the versions verified on arm64 and amd64.
FROM continuumio/miniconda3@sha256:eca594d684f495c1a02beff33a9fab53aec8c5830eaf431bb149912dc6c9e4c1

WORKDIR /home/bomberman

RUN apt-get update \
    && apt-get -y install gcc g++ \
    && rm -rf /var/lib/apt/lists/*

RUN conda install -y -c pytorch \
        python=3.10.21 \
        numpy=2.2.5 scipy=1.15.3 pandas=2.3.3 matplotlib=3.10.9 numba=0.66.0 \
        pytorch=2.10.0 torchvision=0.25.0 \
    && conda clean -afy

RUN pip install --no-cache-dir \
        scikit-learn==1.7.2 tqdm==4.70.0 tensorboardX==2.6.5 \
        xgboost==3.2.0 lightgbm==4.7.0 pathfinding==1.0.22 pyaml==26.7.0 \
        igraph==1.0.0 ujson==6.0.0 networkx==3.4.2 dill==0.4.1 \
        pyastar2d==1.1.4 easydict==1.13 sympy==1.14.0 pygame==2.6.1

# TensorFlow/Keras were in the course's floating template but are not imported
# by this engine or any submitted agent.  Omitting them avoids an unrelated,
# architecture-dependent install failure without changing agent behaviour.
COPY . .
CMD ["/bin/bash"]

FROM mambaorg/micromamba:1.5.3

USER root
# Install basic development tools
RUN apt-get update && apt-get install -y \
    git \
    curl \
    openssh-client \
    less \
    vim \
    procps \
    && rm -rf /var/lib/apt/lists/*

USER mambauser

# Copy environment file
COPY environment-dev.yml /tmp/environment-dev.yml

# Install Conda environment
RUN micromamba create -y -f /tmp/environment-dev.yml && \
    micromamba clean --all --yes

# Add environment to path
ENV PATH /opt/conda/envs/larrak-dev/bin:$PATH

# Activate the environment for shell sessions
ARG MAMBA_DOCKERFILE_ACTIVATE=1

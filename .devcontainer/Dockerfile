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
    build-essential \
    gfortran \
    ninja-build \
    gosu \
    && rm -rf /var/lib/apt/lists/*

# Install GitHub CLI
RUN curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg | \
    dd of=/usr/share/keyrings/githubcli-archive-keyring.gpg && \
    chmod go+r /usr/share/keyrings/githubcli-archive-keyring.gpg && \
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" | \
    tee /etc/apt/sources.list.d/github-cli.list > /dev/null && \
    apt-get update && \
    apt-get install -y gh && \
    rm -rf /var/lib/apt/lists/*

# Copy entrypoint script
COPY .devcontainer/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

USER mambauser

# Copy environment file
COPY environment-dev.yml /tmp/environment-dev.yml

# Install Conda environment
RUN micromamba create -y -f /tmp/environment-dev.yml && \
    micromamba clean --all --yes

# Install meson for HSL build (one-time build tool)
RUN /opt/conda/envs/larrak-dev/bin/pip install meson

# Add environment to path
ENV PATH /opt/conda/envs/larrak-dev/bin:$PATH

# Activate the environment for shell sessions
ARG MAMBA_DOCKERFILE_ACTIVATE=1

# Set entrypoint
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]

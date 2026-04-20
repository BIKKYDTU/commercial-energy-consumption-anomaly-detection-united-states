###############################################
# BASE IMAGE
###############################################
FROM ubuntu:22.04
ENV DEBIAN_FRONTEND=noninteractive

###############################################
# SYSTEM DEPENDENCIES
###############################################
RUN apt-get update && apt-get install -y \
    git \
    python3 \
    python3-pip \
    python3-setuptools \
    python-is-python3 \
    unzip \
    && rm -rf /var/lib/apt/lists/*

###############################################
# WORKING DIRECTORY + GIT SETUP
###############################################
WORKDIR /app

RUN git init \
    && git config --global user.email "agent@example.com" \
    && git config --global user.name "Agent" \
    && echo "# Workspace" > README.md \
    && git add README.md \
    && git commit -m "Initial commit"

###############################################
# PYTHON DEPENDENCIES (build-time install; no COPY/ADD of project files)
# Mirrors codebase/requirements.txt
###############################################
RUN python3 -m pip install --no-cache-dir \
    pytest \
    "pandas>=2.0,<3.0" \
    "numpy>=1.24,<2.0" \
    "plotly>=5.0,<6.0"

###############################################
# EVALUATION ASSETS DIRECTORY
###############################################
# Populated at RUNTIME by the evaluation script.
RUN mkdir -p /eval_assets

# Bind-mount the repository to /app at runtime so pipeline.py, parsing.py,
# run.sh, and tests/ are available (COPY is not used in this Dockerfile).
CMD ["/bin/bash"]

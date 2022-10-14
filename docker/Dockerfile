# Base Image
FROM anibali/pytorch:1.11.0-cuda11.3-ubuntu20.04

# Set up time zone.
ENV TZ=UTC
RUN sudo ln -snf /usr/share/zoneinfo/$TZ /etc/localtime

# Setup basic packages 
RUN sudo apt-get update -q \
    && DEBIAN_FRONTEND=noninteractive sudo apt-get install -y \
    curl \
    git \
    cmake \
    unzip \
    wget \
    && sudo apt-get clean \
    && sudo rm -rf /var/lib/apt/lists/*

# Install miniconda 
WORKDIR /user
RUN curl -LO http://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && bash Miniconda3-latest-Linux-x86_64.sh -p /user/miniconda -b
RUN rm /user/Miniconda3-latest-Linux-x86_64.sh
ENV PATH=/user/miniconda/bin:${PATH}

# Create the conda env
RUN conda create -n bayesian_dqn python=3.8
ENV PATH=/user/miniconda/envs/bayesian_dqn/bin:${PATH}

# download code (temporally use bitbucket)
RUN git clone https://github.com/GilgameshD/Bayesian-DQN.git
WORKDIR /user/bayesian-dqn

# run setup.sh
RUN /bin/bash -c "conda activate bayesian_dqn; ./setup.sh"

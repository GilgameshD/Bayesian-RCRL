# Base Image
FROM anibali/pytorch:1.11.0-cuda11.5-ubuntu20.04

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

# create directory
RUN /bin/bash -c "mkdir user; cd user"

# Install miniconda 
RUN curl -LO http://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && bash Miniconda3-latest-Linux-x86_64.sh -p ./miniconda -b
RUN rm ./Miniconda3-latest-Linux-x86_64.sh
ENV PATH=/home/user/miniconda/bin:${PATH}
RUN source /home/user/.bashrc

# Create the conda env
RUN conda create -n bayesian_dqn python=3.8
ENV PATH=/user/miniconda/envs/bayesian_dqn/bin:${PATH}

# download code from github
RUN git clone https://github.com/GilgameshD/Bayesian-DQN.git

# run setup.sh
RUN /bin/bash -c "conda activate bayesian_dqn; cd Bayesian-DQN"
RUN /bin/bash -c "chmod 777 ./setup.sh; ./setup.sh"

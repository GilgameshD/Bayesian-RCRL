# Base Image
FROM anibali/pytorch:1.11.0-cuda11.5-ubuntu20.04

# Set up time zone
ENV TZ=UTC
RUN sudo ln -snf /usr/share/zoneinfo/$TZ /etc/localtime

# Setup basic packages 
RUN sudo apt-get update -q \
    && DEBIAN_FRONTEND=noninteractive sudo apt-get install -y \
    g++ \
    curl \
    git \
    cmake \
    unzip \
    wget \
    && sudo apt-get clean \
    && sudo rm -rf /var/lib/apt/lists/*

# Install miniconda 
#RUN curl -LO https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
#    && bash Miniconda3-latest-Linux-x86_64.sh -p $HOME/miniconda -b
#RUN rm ./Miniconda3-latest-Linux-x86_64.sh
#ENV PATH=$HOME/miniconda/bin:${PATH}

# Create the conda env
#RUN conda create -n bayesian_dqn python=3.8
#ENV PATH=$HOME/miniconda/envs/bayesian_dqn/bin:${PATH}
#RUN conda activate bayesian_dqn

# download code from github
WORKDIR $HOME
RUN git clone https://github.com/GilgameshD/Bayesian-DQN.git

# run setup.sh
WORKDIR $HOME/Bayesian-DQN
RUN sh ./setup.sh

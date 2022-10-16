# Base Image
FROM nvidia/cuda:11.6.2-cudnn8-devel-ubuntu20.04

# parameters
ARG DEBIAN_FRONTEND=noninteractive

# Setup basic packages 
RUN apt-get update
RUN apt-get install -y htop screen psmisc python3.8 python3-pip python-is-python3 git wget unzip g++ cmake 
# for opencv
RUN apt-get install -y ffmpeg libsm6 libxext6 

# download code from github
WORKDIR /workspace
RUN git clone https://github.com/GilgameshD/Bayesian-DQN.git

# run setup.sh
WORKDIR /workspace/Bayesian-DQN
RUN sh ./setup.sh

# setup wandb
ENV WANDB_API_KEY f77fb7919141fba6e4019f10f5cc16503cf277a1

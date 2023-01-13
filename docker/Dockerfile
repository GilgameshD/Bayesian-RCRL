# Base Image
FROM nvidia/cuda:11.6.2-cudnn8-devel-ubuntu20.04

# parameters
ARG DEBIAN_FRONTEND=noninteractive

# Setup basic packages 
RUN apt-get update
RUN apt-get install -y htop screen psmisc python3.8 python3-pip python-is-python3 git wget unzip g++ cmake 
RUN apt-get install -y ffmpeg libsm6 libxext6 
RUN apt-get install -y libglew-dev patchelf

# install python packages
RUN pip install gym[accept-rom-license]==0.23.1
RUN pip install Cython
RUN pip install ale-py==0.7.5
RUN pip install torch==1.11.0
RUN pip install numpy>=1.22
RUN pip install tqdm
RUN pip install matplotlib
RUN pip install wandb

# clone code
RUN cd /workspace && git clone https://github.com/GilgameshD/Bayesian-DQN.git

# library path for mujoco
RUN export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/workspace/Bayesian-DQN/.mujoco/mujoco210/bin
RUN export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia

# install d3rlpy and d4rl
RUN cd /workspace/Bayesian-DQN/d3rlpy && pip install -e . && cd ../d4rl && pip install -e . && cd ../d4rl-atari && pip install -e .

# set up wandb
ENV WANDB_API_KEY f77fb7919141fba6e4019f10f5cc16503cf277a1

# set up workdir
WORKDIR /workspace/Bayesian-DQN

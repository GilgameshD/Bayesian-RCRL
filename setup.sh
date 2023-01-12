###
 # @Author: Wenhao Ding
 # @Email: wenhaod@andrew.cmu.edu
 # @Date: 2022-08-10 15:22:40
 # @LastEditTime: 2023-01-12 16:02:51
 # @Description: 
### 

sudo apt-get install libglew-dev
sudo apt-get install patchelf

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/ubuntu/.mujoco/mujoco210/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia

pip install -r requirements.txt

cd ./d3rlpy
pip install -e .

cd ../d4rl
pip install -e .

cd ../d4rl-atari
pip install -e .

cd ..

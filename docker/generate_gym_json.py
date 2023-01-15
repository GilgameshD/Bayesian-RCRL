'''
Author: Wenhao Ding
Email: wenhaod@andrew.cmu.edu
Date: 2022-08-18 03:14:33
LastEditTime: 2023-01-15 01:08:26
Description: 
'''

import os
import json


command = """ \
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/root/.mujoco/mujoco210/bin && \
    pip install -e /workspace/Bayesian-DQN/d3rlpy && \
    pip install -e /workspace/Bayesian-DQN/d4rl && \
    pip install -e /workspace/Bayesian-DQN/d4rl-atari && \
    python3 /workspace/Bayesian-DQN/run.py \
        --wandb_dir={} \
        --d4rl_dataset_dir={} \
        --env_name={} \
        --dataset_type={} \
        --model_name={} \
        --qf_name={} \
        --learning_rate={} \
        --hidden_dim={} \
        --gamma={} \
        --threshold_c={} \
        --n_quantiles={} \
        --sample_iters={} \
        --n_neg_samples={} \
"""


# save directories
wandb_dir = '/results'
d4rl_dataset_dir = '/workspace/data'

# environment name
env_name = [
    'walker2d',
    #'halfcheetah',
    #'hopper'
]

# 10, 20, 30, 40, 50
dataset_type = [
    'medium-replay',
    'medium',
    'medium-expert',
    'expert',
]

# parameters of Bayesian-DQN
learning_rate = [
    0.0005,
    #0.0001,
]

hidden_dim = [
    256,
    #512,
]

gamma = [
    #0.95,
    0.99,
]

threshold_c = [
    #0.05,
    0.1,
    #0.2,
]

n_quantiles = [
    40,
    80,
]

sample_iters = [
    5,
    10,
]

n_neg_samples = [
    256,
    512,
]

# load template json file
with open('base.json', 'r') as fp:  
    base_json = json.load(fp)

# create a new folder
folder_name = './json'
if not os.path.exists(folder_name):
    os.mkdir(folder_name)

# run Bayesian DQN
model_name = 'rcrl'
q_name = 'none'
count = 0
for e_i in env_name:
    for d_i in dataset_type:
        for lr_i in learning_rate:
            for h_i in hidden_dim:
                for g_i in gamma:
                    for t_i in threshold_c:
                        for n_i in n_quantiles:
                            for s_i in sample_iters:
                                for nn_i in n_neg_samples:
                                    job_name = folder_name + '/' + str(count) + '.json'
                                    count += 1
                                    base_json['command'] = command.format(
                                        wandb_dir, 
                                        d4rl_dataset_dir, 
                                        e_i, 
                                        d_i, 
                                        model_name, 
                                        q_name, 
                                        str(lr_i), 
                                        str(h_i),
                                        str(g_i), 
                                        str(t_i), 
                                        str(n_i), 
                                        str(s_i),
                                        str(nn_i),
                                    )
                                    with open(job_name, 'w') as fp:  
                                        json.dump(base_json, fp, indent=4)

print('generated {} json files.'.format(count))

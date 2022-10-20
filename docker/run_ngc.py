'''
Author: Wenhao Ding
Email: wenhaod@andrew.cmu.edu
Date: 2022-08-18 03:14:33
LastEditTime: 2022-10-18 16:21:37
Description: 
'''

import copy
import os
import json


command = """ \
    cd /workspace/Bayesian-DQN/d3rlpy && \
    pip install -e . && \
    cd /workspace/Bayesian-DQN/d4rl-atari && \
    pip install -e . && \
    cd /workspace/Bayesian-DQN \
    python3 run.py \
        --wandb_dir={} \
        --beta_model_base={} \
        --d4rl_dataset_dir={} \
        --env_name={} \
        --dataset_type={} \
        --model_name={} \
        --qf_name={} \
        --learning_rate={} \
        --gamma={} \
        --threshold_c={} \
        --weight_penalty={} \
        --weight_R={}"
"""


# save directories
wandb_dir = '/results'
beta_model_base = '/results'
d4rl_dataset_dir = '/workspace'

# environment name
env_name = [
    'breakout'
]
# 10, 20, 30, 40, 50
dataset_type = [
    'medium'
]


# parameters of Bayesian-DQN
learning_rate = [
    0.00025,
    0.0004,
    0.0005,
]

gamma = [
    0.99,
    0.95,
    0.90
]

threshold_c = [
    0.1,
    0.5,
    1.0
]

weight_penalty = [
    0.5,
]

weight_R = [
    20.0,
]


# load template json file
with open('base.json', 'r') as fp:  
    base_json = json.load(fp)

command_list = []

# run Bayesian DQN
# 1x1x3x3x3
model_name = 'bayes'
q_name = 'bayes'
for e_i in env_name:
    for d_i in dataset_type:
        for lr_i in learning_rate:
            for g_i in gamma:
                for t_i in threshold_c:
                    for wp_i in weight_penalty:
                        for wR_i in weight_R:
                            job_name = e_i + '_' + d_i + '_' + model_name + '_' + q_name + '_' + str(lr_i) + '_' + str(t_i) + '_' + str(g_i) + '_' + str(wp_i) + '_' + str(wR_i) + '.json'
                            base_json['command'] = command.format(wandb_dir, beta_model_base, d4rl_dataset_dir, e_i, d_i, 'bayes', 'bayes', str(lr_i), str(t_i), str(g_i), str(wp_i), str(wR_i))
                            command_list.append(copy.deepcopy(base_json))

# make groups
num_in_group = 4
for c_i in range(0, len(command_list), num_in_group):
    
    with open(job_name, 'w') as fp:  
        json.dump(base_json, fp, indent=4)

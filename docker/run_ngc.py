'''
Author: Wenhao Ding
Email: wenhaod@andrew.cmu.edu
Date: 2022-08-18 03:14:33
LastEditTime: 2022-10-13 14:20:46
Description: 
'''

import os


command = """ngc batch run \
    --name "ml-model.decision-transformer" \
    --priority NORMAL \
    --preempt RUNONCE \
    --min-timeslice 1s \
    --total-runtime 259200s \
    --ace nv-us-west-2 \
    --instance dgx1v.32g.1.norm \
    --commandline "python3 /workspace/Bayesian-DQN/run.py \
        --wandb_dir={} \
        --beta_model_base={} \
        --env_name={} \
        --dataset_type={} \
        --model_name={} \
        --qf_name={} \
        --learning_rate={} \
        --gamma={} \
        --threshold_c={} \
        --weight_penalty={} \
        --weight_R={}" \
    --result /results \
    --image "nvidian/nvr-av/dedt-img:2.0" \
    --org nvidian \
    --team nvr-av \
    --datasetid 107377:/workspace/data \
    --workspace i1ss1gHKSNCBu46DQUXBYw:/workspace:RW \
    --port 8888 \
    --order 50
"""


# save dir
wandb_dir = './wandb'
beta_model_base = './model'

# environment
env_name = [
    'breakout'
]
dataset_type = [
    'expert',
    'medium',
    'mixed'
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

# run Bayesian DQN
# 3x3x3x3
for e_i in env_name:
    for d_i in dataset_type:
        for lr_i in learning_rate:
            for g_i in gamma:
                for t_i in threshold_c:
                    for wp_i in weight_penalty:
                        for wR_i in weight_R:
                            one_command = command.format(wandb_dir, beta_model_base, e_i, d_i, 'bayes', 'bayes', str(lr_i), str(t_i), str(g_i), str(wp_i), str(wR_i))
                            #print(one_command)
                            os.system(one_command)


# RUN CQL
# 3x3 = 9
for e_i in env_name:
    for d_i in dataset_type:
        for lr_i in learning_rate:
            one_command = command.format(wandb_dir, beta_model_base, e_i, d_i, 'cql', 'none', str(lr_i), 0,0, 0.9, 0.0, 0.0)
            #print(one_command)
            os.system(one_command)

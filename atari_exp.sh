###
 # @Author: Wenhao Ding
 # @Email: wenhaod@andrew.cmu.edu
 # @Date: 2022-08-10 15:22:40
 # @LastEditTime: 2023-01-20 17:02:34
 # @Description: 
### 


CUDA_VISIBLE_DEVICES=1 

# breakout
# python3 run.py \
#     --mode=offline \
#     --d4rl_dataset_dir=/mnt/data1/wenhao/.d4rl/datasets \
#     --env_name=breakout \
#     --env_type=atari \
#     --model_name=bayes \
#     --qf_name=bayes \
#     --threshold_c=0.1 \
#     --dataset_type=small \
#     --n_epochs=50 \
#     --n_trials=10 \
#     --test_epsilon=0.01 \
#     --beta_learning_rate=0.00025 \
#     --batch_size=32 \
#     --learning_rate=0.00025 \
#     --gamma=0.95 \
#     --vmin=0 \
#     --vmax=10 \
#     --beta_epoch=2 \
#     --beta_weight_penalty=0.5 \
#     --n_quantiles=51 \
#     --weight_penalty=0.5 \
#     --weight_R=20.0 \
#     --weight_A=1.0 \
#     --target_update_interval=8000 \
#     --eval_step_interval=10000 \


# pong
# python3 run.py \
#     --mode=offline \
#     --d4rl_dataset_dir=/mnt/data1/wenhao/.d4rl/datasets \
#     --env_name=pong \
#     --env_type=atari \
#     --model_name=bayes \
#     --threshold_c=0.1 \
#     --qf_name=bayes \
#     --dataset_type=medium \
#     --n_epochs=1 \
#     --n_trials=10 \
#     --test_epsilon=0.00 \
#     --beta_learning_rate=0.00025 \
#     --batch_size=32 \
#     --learning_rate=0.00025 \
#     --gamma=0.95 \
#     --vmin=0 \
#     --vmax=10 \
#     --beta_epoch=1 \
#     --beta_weight_penalty=0.5 \
#     --n_quantiles=51 \
#     --weight_penalty=0.5 \
#     --weight_R=20.0 \
#     --weight_A=1.0 \
#     --target_update_interval=8000 \
#     --eval_step_interval=10000 \


# seaquest - medium
python3 run.py \
    --mode=offline \
    --d4rl_dataset_dir=/data/wenhao/.d4rl/datasets \
    --env_name=seaquest \
    --env_type=atari \
    --threshold_c=0.1 \
    --model_name=bayes \
    --qf_name=bayes \
    --dataset_type=medium \
    --n_epochs=200 \
    --n_trials=10 \
    --test_epsilon=0.001 \
    --beta_learning_rate=0.0002 \
    --batch_size=256 \
    --learning_rate=0.0002 \
    --gamma=0.95 \
    --vmin=0 \
    --vmax=10 \
    --beta_epoch=10 \
    --beta_weight_penalty=0.0 \
    --n_quantiles=51 \
    --weight_penalty=0.0 \
    --weight_R=1.0 \
    --weight_A=1.0 \
    --target_update_interval=8000 \
    --eval_step_interval=10000 \


# qbert - small
# python3 run.py \
#     --mode=offline \
#     --d4rl_dataset_dir=/mnt/data1/wenhao/.d4rl/datasets \
#     --env_name=qbert \
#     --env_type=atari \
#     --threshold_c=0.1 \
#     --model_name=bayes \
#     --qf_name=bayes \
#     --dataset_type=small \
#     --n_epochs=20 \
#     --n_trials=10 \
#     --test_epsilon=0.01 \
#     --beta_learning_rate=0.00025 \
#     --batch_size=128 \
#     --learning_rate=0.00025 \
#     --gamma=0.95 \
#     --vmin=0 \
#     --vmax=10 \
#     --beta_epoch=5 \
#     --beta_weight_penalty=0.5 \
#     --n_quantiles=51 \
#     --weight_penalty=0.5 \
#     --weight_R=20.0 \
#     --weight_A=1.0 \
#     --target_update_interval=8000 \
#     --eval_step_interval=5000 \

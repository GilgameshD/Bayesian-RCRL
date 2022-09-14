'''
Author: Wenhao Ding
Email: wenhaod@andrew.cmu.edu
Date: 2022-08-05 19:12:08
LastEditTime: 2022-09-13 20:01:09
Description: 
'''

import argparse
import random
import numpy as np
import torch 
import gym

import d3rlpy
from d3rlpy.models.q_functions import (
    BayesianQFunctionFactory, 
    MeanQFunctionFactory, 
    QRQFunctionFactory,
    IQNQFunctionFactory,
    FQFQFunctionFactory,
    C51QFunctionFactory, 
)

import wandb
from sklearn.model_selection import train_test_split


parser = argparse.ArgumentParser()
parser.add_argument('--env_name', type=str, default='pong', help='[cartpole, breakout, pong]')
parser.add_argument('--model_name', type=str, default='cql', help='[dqn, cql, bayes, bc')
parser.add_argument('--qf_name', type=str, default='c51', help='[mean, c51, qr, iqn, fqf, bayes')
parser.add_argument('--mode', type=str, default='offline', help='[offline, online')
args = parser.parse_args()


# select parameters
if args.env_name == 'cartpole':
    batch_size = 64
    n_quantiles = 51
    alpha = 1.0
    learning_rate = 0.00002
    target_update_interval = 2000
    n_frames = 1
    scaler = 'standard'
    encoder = 'default'
    n_epochs = 50
    eval_step_interval = 5000
    if args.mode == 'offline':
        # offline parameters
        dataset_type = 'replay' # random
        dataset, env = d3rlpy.datasets.get_cartpole(dataset_type='replay')
    else:
        # online parameter
        buffer_maxlen = 1000000
        n_steps = 1000000
        eval_env = gym.make('cartpole-v1')
elif args.env_name == 'breakout':
    batch_size = 32
    alpha = 4.0
    n_quantiles = 51
    learning_rate = 0.0001
    target_update_interval = 8000
    n_frames = 4
    scaler = 'pixel'
    encoder = 'pixel'
    n_epochs = 100
    eval_step_interval = 5000
    if args.mode == 'offline':
        # offline parameters
        dataset_type = 'expert' # mixed, medium, expert
        dataset, env = d3rlpy.datasets.get_atari('breakout-'+dataset_type+'-v0')
    else:
        # online parameter
        buffer_maxlen = 1000000
        n_steps = 1000000
        eval_env = gym.make('BreakoutNoFrameskip-v4')
elif args.env_name == 'pong':
    batch_size = 32
    alpha = 4.0
    n_quantiles = 51
    learning_rate = 0.0001
    target_update_interval = 8000
    n_frames = 4
    scaler = 'pixel'
    encoder = 'pixel'
    n_epochs = 13
    eval_step_interval = 1000
    if args.mode == 'offline':
        # offline parameters
        dataset_type = 'expert' # mixed, medium, expert
        dataset, env = d3rlpy.datasets.get_atari('pong-'+dataset_type+'-v0')
    else:
        # online parameter
        buffer_maxlen = 1000000
        n_steps = 1000000
        eval_env = gym.make('PongNoFrameskip-v4')
else:
    raise ValueError('Unknown environment name')


# select model
if args.model_name == 'bayes':
    Model = d3rlpy.algos.BayesianDiscreteDQN
elif args.model_name == 'cql':
    Model = d3rlpy.algos.DiscreteCQL
elif args.model_name == 'dqn':
    Model = d3rlpy.algos.DQN
elif args.model_name == 'bc':
    Model = d3rlpy.algos.DiscreteBC
    args.qf_name = 'none'
else:
    raise ValueError('unknown model name')


# select evaluation metrics
if args.model_name in ['bc']:
    scorers = {'environment': d3rlpy.metrics.evaluate_on_environment(env, n_trials=10, epsilon=0.0),}
else:
    scorers = {
        'environment': d3rlpy.metrics.evaluate_on_environment(env, n_trials=10, epsilon=0.0),
        'td_error': d3rlpy.metrics.td_error_scorer,
    }


# select q function
if args.qf_name == 'mean':
    q_func = MeanQFunctionFactory(n_quantiles=n_quantiles)
elif args.qf_name == 'bayes':
    q_func = BayesianQFunctionFactory(n_quantiles=n_quantiles)
elif args.qf_name == 'qr':
    q_func = QRQFunctionFactory(n_quantiles=n_quantiles)
elif args.qf_name == 'iqn':
    q_func = IQNQFunctionFactory(n_quantiles=n_quantiles)
elif args.qf_name == 'fqf':
    q_func = FQFQFunctionFactory(n_quantiles=n_quantiles)
elif args.qf_name == 'c51':
    q_func = C51QFunctionFactory(n_quantiles=n_quantiles)
elif args.qf_name == 'none':
    q_func = None
else:
    raise ValueError('unknown q function name')


# set up wandb
config = {
    'env': args.env_name,
    'm': args.model_name,
    'qf': args.qf_name,
    'bs': batch_size,
    'nc': n_quantiles,
    'lr': learning_rate,
    'tui': target_update_interval,
    'd': dataset_type,
}
group_name = ''.join([k_i + '_' + str(config[k_i]) + '_' for k_i in config.keys()])
def wandb_callback(algo, epoch, total_step, data_dict):
    for name in data_dict.keys():
        wandb.log(data={name: data_dict[name]}, step=total_step)


# set up random seeds
seed_list = [101, 222, 333, 444, 555, 666, 777, 888, 999, 123]
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# start training
train_episodes, test_episodes = train_test_split(dataset, test_size=0.1)
for t_i in range(0, len(seed_list)):
    # set random seed
    seed = seed_list[t_i]
    set_seed(seed)

    # init wandb
    name = group_name + 'seed_' + str(seed)
    run = wandb.init(project="bayesian-"+args.env_name, entity="rlresearch", group=group_name, name=name, reinit=True) 
    wandb.config.update(config)

    # model 
    model = Model(
        batch_size=batch_size,
        n_frames=n_frames,
        learning_rate=learning_rate,
        target_update_interval=target_update_interval,
        q_func_factory=q_func,
        encoder_factory=encoder,
        scaler=scaler,
        alpha=alpha,
        use_gpu=True,
    )

    # training
    if args.mode == 'offline':
        model.fit(
            train_episodes,
            eval_episodes=test_episodes,
            n_epochs=n_epochs,
            scorers=scorers,
            callback=wandb_callback,
            verbose=False,
            save_metrics=False,
            eval_step_interval=eval_step_interval,
        )
    else:
        # prepare replay buffer
        buffer = d3rlpy.online.buffers.ReplayBuffer(maxlen=buffer_maxlen, env=env)
        model.fit_online(env, buffer, n_steps=n_steps, eval_env=eval_env)

    run.finish()

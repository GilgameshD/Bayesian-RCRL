'''
Author: Wenhao Ding
Email: wenhaod@andrew.cmu.edu
Date: 2022-08-05 19:12:08
LastEditTime: 2022-10-16 13:14:18
Description: 
'''

import argparse
import random
import numpy as np
import torch 
import os
import wandb
from sklearn.model_selection import train_test_split

import d3rlpy
from d3rlpy.models.q_functions import (
    BayesianQFunctionFactory, 
    MeanQFunctionFactory, 
    QRQFunctionFactory,
    IQNQFunctionFactory,
    FQFQFunctionFactory,
    C51QFunctionFactory, 
)


parser = argparse.ArgumentParser()
parser.add_argument('-wd', '--wandb_dir', type=str, default='/mnt/data1/wenhao', help='directory for saving wandb metadata')
parser.add_argument('-dd', '--d4rl_dataset_dir', type=str, default='/mnt/data1/wenhao/.d4rl/datasets', help='directory for saving d4rl dataset')
parser.add_argument('--env_name', type=str, default='seaquest', help='[cartpole, breakout, pong, seaquest]')
parser.add_argument('--env_type', type=str, default='atari', help='atari or gym')

parser.add_argument('--model_name', type=str, default='bc', help='[dqn, cql, bayes, bc')
parser.add_argument('--qf_name', type=str, default='none', help='[mean, c51, qr, iqn, fqf, bayes, none')
parser.add_argument('-dt', '--dataset_type', type=str, default='medium', help='[mixed, medium, expert]')

# beta policy model
parser.add_argument('-bmb', '--beta_model_base', type=str, default='./model', help='directory for saving beta policy model')
parser.add_argument('-be', '--beta_epoch', type=float, default=2, help='number of epoch for training beta policy (using BC)')
parser.add_argument('-blr', '--beta_learning_rate', type=float, default=0.00025, help='learning rate of beta policy model')

# training parameters
parser.add_argument('-ne', '--n_epochs', type=int, default=15, help='number of training epoch')
parser.add_argument('-bs', '--batch_size', type=int, default=32, help='batch size')
parser.add_argument('-lr', '--learning_rate', type=float, default=0.00025, help='learning rate')

# test parameters
parser.add_argument('-te', '--test_epsilon', type=float, default=0.01, help='use epsilon greedy during testing stage')
parser.add_argument('-esi', '--eval_step_interval', type=int, default=10000, help='interval of step of calling evaluation')
parser.add_argument('-nt', '--n_trials', type=int, default=10, help='number of online evaluation trails')

# Bayesian Q parameters
parser.add_argument('-c', '--threshold_c', type=float, default=0.1, help='condition threshold used in testing')
parser.add_argument('-nq', '--n_quantiles', type=int, default=51, help='number of quantile for C51 q-value')
parser.add_argument('-vmin', '--vmin', type=float, default=-10, help='lower bound of value function')
parser.add_argument('-vmax', '--vmax', type=float, default=10, help='upper bound of value function')
parser.add_argument('-ga', '--gamma', type=float, default=0.95, help='reward discount')
parser.add_argument('-wp', '--weight_penalty', type=float, default=0.5, help='weight of action L2 penalty for loss A')
parser.add_argument('-wR', '--weight_R', type=float, default=20.0, help='weight of loss R')
parser.add_argument('-tui', '--target_update_interval', type=int, default=8000, help='target update interval for q learning')

# CQL parameters 
parser.add_argument('-a', '--alpha', type=float, default=4.0, help='weight of conservative loss in CQL')

args = parser.parse_args()


# process some parameters
project = 'bayesian-'+args.env_name+'-sweep'
beta_model_dir = os.path.join(args.beta_model_base, args.env_name, args.dataset_type)
os.environ['D4RL_DATASET_DIR'] = args.d4rl_dataset_dir
dataset, env = d3rlpy.datasets.get_atari(args.env_name+'-more-'+args.dataset_type+'-v0')
if args.env_type == 'atari':
    scaler = 'pixel'
    encoder = 'pixel'
    n_frames = 4
else:
    scaler = 'standard'
    encoder = 'default'
    n_frames = 1


# select model
model_list = {
    'bayes': d3rlpy.algos.BayesianDiscreteDQN,
    'cql': d3rlpy.algos.DiscreteCQL,
    'dqn': d3rlpy.algos.DQN,
    'bc': d3rlpy.algos.DiscreteBC,
}
Model = model_list[args.model_name]


# select q function
qf_list = {
    'mean': MeanQFunctionFactory(n_quantiles=args.n_quantiles),
    'bayes': BayesianQFunctionFactory(
        n_quantiles=args.n_quantiles, 
        Vmin=args.vmin,
        Vmax=args.vmax,
        weight_penalty=args.weight_penalty, 
        weight_R=args.weight_R),
    'qr': QRQFunctionFactory(n_quantiles=args.n_quantiles),
    'iqn': IQNQFunctionFactory(n_quantiles=args.n_quantiles),
    'fqf': FQFQFunctionFactory(n_quantiles=args.n_quantiles),
    'c51': C51QFunctionFactory(n_quantiles=args.n_quantiles),
    'none': None
}
q_func = qf_list[args.qf_name]


# select evaluation metrics
scorers = {'environment': d3rlpy.metrics.evaluate_on_environment(env, n_trials=args.n_trials, epsilon=args.test_epsilon)}
if args.model_name in ['bc']:
    scorers.update({'action_match': d3rlpy.metrics.scorer.discrete_action_match_scorer})
else:
    scorers.update({'td_error': d3rlpy.metrics.td_error_scorer})


# set up wandb
config = {
    'env': args.env_name,
    'm': args.model_name,
    'qf': args.qf_name,
    'bs': args.batch_size,
    'nc': args.n_quantiles,
    'c': args.threshold_c,
    'lr': args.learning_rate,
    'tui': args.target_update_interval,
    'd': args.dataset_type,
    'te': args.test_epsilon,
    'wp': args.weight_penalty,
    'wR': args.weight_R,
    'be': args.beta_epoch,
    'ne': args.n_epochs,
    'ga': args.gamma,
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
for t_i in range(len(seed_list)):
    # set random seed
    seed = seed_list[t_i]
    #set_seed(seed)

    # have to use test episode to run scores
    train_episodes, test_episodes = train_test_split(dataset, test_size=0.01, shuffle=True)

    # init wandb
    name = group_name + 'seed_' + str(seed)
    run = wandb.init(project=project, entity='rlresearch', group=group_name, name=name, reinit=True, dir=args.wandb_dir) 
    wandb.config.update(config)

    # model 
    model = Model(
        batch_size=args.batch_size,
        n_frames=n_frames,
        beta_learning_rate=args.beta_learning_rate,
        learning_rate=args.learning_rate,
        target_update_interval=args.target_update_interval,
        q_func_factory=q_func,
        encoder_factory=encoder,
        scaler=scaler,
        alpha=args.alpha,
        gamma=args.gamma,
        threshold_c=args.threshold_c,
        use_gpu=True,
    )

    # for Bayesian-DQN model, we need a pre-train model
    if args.model_name == 'bayes': 
        model.fit_beta_policy(
            model_dir=beta_model_dir, 
            dataset=dataset, 
            n_epochs=args.beta_epoch,
        )

    # fit model
    model.fit(
        dataset=train_episodes,
        eval_episodes=test_episodes,
        n_epochs=args.n_epochs,
        scorers=scorers,
        callback=wandb_callback,
        verbose=False,
        save_metrics=False,
        eval_step_interval=args.eval_step_interval,
    )
    run.finish()

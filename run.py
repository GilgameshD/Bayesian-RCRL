'''
Author: Wenhao Ding
Email: wenhaod@andrew.cmu.edu
Date: 2022-08-05 19:12:08
LastEditTime: 2023-01-11 10:52:44
Description: 
'''

import gym
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
from d3rlpy.models.encoders import (
    VectorEncoderFactory,
    EBMEncoderFactory
)


parser = argparse.ArgumentParser()
parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu or not')
parser.add_argument('--stage', type=str, default='train', help='training or testing')

parser.add_argument('-wd', '--wandb_dir', type=str, default='/data/wenhao', help='directory for saving wandb metadata')
parser.add_argument('-dd', '--d4rl_dataset_dir', type=str, default='/data/wenhao/.d4rl/datasets', help='directory for saving d4rl dataset')
parser.add_argument('--env_name', type=str, default='halfcheetah', help='[cartpole, breakout, pong, seaquest, qbert, asterix | halfcheetah, walker2d, hopper]')
parser.add_argument('--env_type', type=str, default='gym', help='atari or gym')

# model selection
parser.add_argument('--model_name', type=str, default='rcrl', help='atari - [dqn, cql, bayes, bc], gym - [cbayes, rcrl, cbc, ccql, td3bc, bcq, bear, awac]')
parser.add_argument('--qf_name', type=str, default='none', help='[mean, c51, qr, iqn, fqf, bayes, none')
parser.add_argument('-dt', '--dataset_type', type=str, default='medium-expert', help='[mixed, medium, expert] - [random, medium-replay, medium, medium-expert, expert]')

# beta policy model
parser.add_argument('-bmb', '--beta_model_base', type=str, default='./model', help='directory for saving beta policy model')
parser.add_argument('-be', '--beta_epoch', type=int, default=20, help='number of epoch for training beta policy (using BC)')
parser.add_argument('-blr', '--beta_learning_rate', type=float, default=0.0005, help='learning rate of beta policy model')
parser.add_argument('-bwp', '--beta_weight_penalty', type=float, default=0.5, help='weight of action L2 penalty for beta policy (beta of BC)')

# training parameters
parser.add_argument('-ne', '--n_epochs', type=int, default=40, help='number of training epoch')
parser.add_argument('-bs', '--batch_size', type=int, default=512, help='batch size')
parser.add_argument('-lr', '--learning_rate', type=float, default=0.0005, help='learning rate')
parser.add_argument('-hd', '--hidden_dim', type=int, default=256, help='size of hidden layer for encoder')
parser.add_argument('-hn', '--hidden_num', type=int, default=3, help='number of hidden layer for encoder')
parser.add_argument('-dr', '--dropout_rate', type=float, default=0.0, help='dropout rate of encoder')
parser.add_argument('-bn', '--use_batch_norm', type=bool, default=False, help='use batch normalization')

# test parameters
parser.add_argument('-te', '--test_epsilon', type=float, default=0.0, help='epsilon in testing stage')
parser.add_argument('-esi', '--eval_step_interval', type=int, default=5000, help='interval of step of calling evaluation')
parser.add_argument('-nt', '--n_trials', type=int, default=10, help='number of online evaluation trails')

# Bayesian Q parameters
parser.add_argument('-c', '--threshold_c', type=float, default=0.1, help='condition threshold used in testing')
parser.add_argument('-nq', '--n_quantiles', type=int, default=40, help='number of quantile of Q-value or RTG')
parser.add_argument('-vmin', '--vmin', type=float, default=0, help='lower bound of value function or RTG')
parser.add_argument('-vmax', '--vmax', type=float, default=400, help='upper bound of value function or RTG (depends on gamma, 0.99 - 400, 0.95 - 100)')
parser.add_argument('-ga', '--gamma', type=float, default=0.99, help='reward discount')
parser.add_argument('-wp', '--weight_penalty', type=float, default=0.0, help='weight of action L2 penalty for loss A')
parser.add_argument('-wR', '--weight_R', type=float, default=1.0, help='weight of loss R')
parser.add_argument('-wA', '--weight_A', type=float, default=1.0, help='weight of loss A')
parser.add_argument('-tui', '--target_update_interval', type=int, default=100, help='target update interval for q learning')
parser.add_argument('-ns', '--n_steps', type=int, default=1, help='number of step to calculate reward')

# Energy-based model
parser.add_argument('-si', '--sample_iters', type=int, default=5, help='number of iteration')
parser.add_argument('-nsc', '--noise_scale', type=float, default=0.33, help='scale of noise')
parser.add_argument('-nsh', '--noise_shrink', type=float, default=0.5, help='shrink of noise')
parser.add_argument('-is', '--inference_samples', type=int, default=2**16, help='number of testing samples for action')
parser.add_argument('-nss', '--n_neg_samples', type=int, default=256, help='number of negative samples used for infoNCE')

# CQL parameters 
parser.add_argument('-a', '--alpha', type=float, default=4.0, help='weight of conservative loss in CQL')
args = parser.parse_args()


# process some parameters
project = 'bayesian-'+args.env_name+'-sweep'
beta_model_dir = os.path.join(args.beta_model_base, args.env_name, args.dataset_type)
os.environ['D4RL_DATASET_DIR'] = args.d4rl_dataset_dir
os.environ['D4RL_SUPPRESS_IMPORT_ERROR'] = '1'
if args.env_type == 'atari':
    dataset, env = d3rlpy.datasets.get_atari(args.env_name+'-more-'+args.dataset_type+'-v0', args.d4rl_dataset_dir)
    scaler = 'pixel'
    encoder = 'pixel'
    n_frames = 4
elif args.env_type == 'gym':
    dataset, env = d3rlpy.datasets.get_d4rl(args.env_name+'-'+args.dataset_type+'-v2', args.d4rl_dataset_dir)
    scaler = 'standard'
    n_frames = 1
    hidden_units = [args.hidden_dim for _ in range(args.hidden_num)]

    # dataset is small
    if args.dataset_type == 'medium-replay':
        args.n_epochs *= 4

    # set vmax accoridng to env name
    vmax_dict = {
        0.99: {
            'hopper': {
                'random': 60,
                'medium-replay': 330,
                'medium': 350,
                'medium-expert': 400,
                'expert': 400,
            },
            'walker2d': {
                'random': 10,
                'medium-replay': 450,
                'medium': 450,
                'medium-expert': 550,
                'expert': 550,
            },
            'halfcheetah': {
                'random': 20,
                'medium-replay': 550,
                'medium': 550,
                'medium-expert': 1200,
                'expert': 1200,
            }
        },
        0.95: {
            'hopper': {
                'random': 30,
                'medium-replay': 90,
                'medium': 90,
                'medium-expert': 90,
                'expert': 90,
            },
            'walker2d': {
                'random': 10,
                'medium-replay': 100,
                'medium': 100,
                'medium-expert': 110,
                'expert': 110,
            },
            'halfcheetah': {
                'random': 10,
                'medium-replay': 110,
                'medium': 120,
                'medium-expert': 250,
                'expert': 250,
            }
        }
    }
    args.vmax = vmax_dict[args.gamma][args.env_name][args.dataset_type]

    # for continuous Bayesian RCRL, we use reward-to-go rather than reward
    if args.model_name == 'rcrl': 
        encoder = EBMEncoderFactory(
            hidden_units=hidden_units, 
            dropout_rate=args.dropout_rate,
            use_batch_norm=args.use_batch_norm,
        )
    else:
        encoder = VectorEncoderFactory(hidden_units=hidden_units)


# select model
model_list = {
    'rcrl': d3rlpy.algos.RCRL,
    'cbayes': d3rlpy.algos.BayesianDQN,
    'bayes': d3rlpy.algos.BayesianDiscreteDQN,
    'cql': d3rlpy.algos.DiscreteCQL,
    'ccql': d3rlpy.algos.CQL,
    'dqn': d3rlpy.algos.DQN,
    'bc': d3rlpy.algos.DiscreteBC,
    'cbc': d3rlpy.algos.BC, # continuous BC
    'td3bc': d3rlpy.algos.TD3PlusBC, 
    'bcq': d3rlpy.algos.BCQ, 
    'bear': d3rlpy.algos.BEAR, 
    'awac': d3rlpy.algos.AWAC, 
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
        weight_R=args.weight_R,
        weight_A=args.weight_A,
        n_neg_samples=args.n_neg_samples,
    ),
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
elif args.model_name in ['cbc']:
    scorers.update({'action_match': d3rlpy.metrics.scorer.continuous_action_diff_scorer})
elif args.model_name in ['dqn', 'cql']:
    scorers.update({'td_error': d3rlpy.metrics.td_error_scorer})
else:
    pass

# set up wandb
config = {
    '': args.model_name + '_' + args.dataset_type,
    'nc': args.n_quantiles,
    'c': args.threshold_c,
    'lr': args.learning_rate,
    'wp': args.weight_penalty,
    'wR': args.weight_R,
    'wA': args.weight_A,
    'va': args.vmax,
    'si': args.sample_iters,
    'nsc': args.noise_scale,
    'nsh': args.noise_shrink,
    'nss': args.n_neg_samples,
    'hd': args.hidden_dim,
    'hn': args.hidden_num,
    'is': args.inference_samples,
    'dr': args.dropout_rate,
    'bn': int(args.use_batch_norm),
    'penalty': '',
    'pr': '',
}

group_name = ''.join([k_i + str(config[k_i]) + '_' for k_i in config.keys()])[:-1]
def wandb_callback(algo, epoch, total_step, data_dict):
    for name in data_dict.keys():
        wandb.log(data={name: data_dict[name]}, step=total_step)


# set up random seeds
seed_list = [111, 222, 333, 444, 555, 666, 777, 888, 999, 123]
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# start training
for t_i in range(len(seed_list)):
    # set random seed
    seed = seed_list[t_i]
    set_seed(seed)

    # have to use test episode to run scores
    #train_episodes, test_episodes = train_test_split(dataset, test_size=0.0, shuffle=True)

    # init wandb
    name = group_name + '_seed' + str(seed)
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
        beta=args.weight_penalty,                      # used for BC
        beta_weight_penalty=args.beta_weight_penalty,  # used for beta policy
        n_steps=args.n_steps,                          # used for reward
        gamma=args.gamma,
        noise_shrink=args.noise_shrink,
        noise_scale=args.noise_scale,
        sample_iters=args.sample_iters,
        threshold_c=args.threshold_c,
        inference_samples=args.inference_samples,      # used for continuous BRCRL
        n_quantiles=args.n_quantiles,                  # only valid for continuous BRCRL
        Vmin=args.vmin,
        Vmax=args.vmax,
        weight_R=args.weight_R,
        weight_A=args.weight_A,
        n_neg_samples=args.n_neg_samples,
        use_gpu=args.use_gpu,
    )

    # for Bayesian-DQN model, we need a pre-train model
    if args.model_name in ['bayes', 'cbayes']: 
        model.fit_beta_policy(
            model_dir=beta_model_dir, 
            dataset=dataset, 
            n_epochs=args.beta_epoch,
        )

    # fit model
    model.fit(
        dataset=dataset,
        #eval_episodes=test_episodes,
        n_epochs=args.n_epochs,
        scorers=scorers,
        callback=wandb_callback,
        verbose=False,
        save_metrics=False,
        eval_step_interval=args.eval_step_interval,
    )
    run.finish()

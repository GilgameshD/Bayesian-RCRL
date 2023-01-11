'''
Author: Wenhao Ding
Email: wenhaod@andrew.cmu.edu
Date: 2022-08-05 19:12:08
LastEditTime: 2023-01-09 13:36:53
Description: 
'''

import matplotlib.pyplot as plt
import numpy as np
from collections import Counter

import argparse
import d3rlpy
from d3rlpy.iterators import RoundIterator


def _convert_rtg_to_idx(rtg, _Vmin, _Vmax, _n_quantiles):
    """ Run a mapping from rtg to discrete index. e.g., [0, ..., 10] -> 0 """
    # get bin width
    bin_width = (_Vmax - _Vmin) / _n_quantiles

    rtg_idx_list = np.zeros_like(rtg) # [B]
    for b_i in range(_n_quantiles):
        #low = b_i * bin_width
        high = (b_i+1) * bin_width
        rtg_idx_list += (rtg >= high)
    return rtg_idx_list


parser = argparse.ArgumentParser()
parser.add_argument('-dd', '--d4rl_dataset_dir', type=str, default='/data/wenhao/.d4rl/datasets', help='directory for saving d4rl dataset')
parser.add_argument('--env_name', type=str, default='halfcheetah', help='[cartpole, breakout, pong, seaquest, qbert, asterix | ]')
parser.add_argument('--env_type', type=str, default='gym', help='atari or gym')
parser.add_argument('-dt', '--dataset_type', type=str, default='expert', help='[mixed, medium, expert] - []')
args = parser.parse_args()


env_name = ['halfcheetah', 'walker2d', 'hopper']
dataset_type = ['random', 'medium-replay', 'medium', 'medium-expert', 'expert']
for e_i in env_name:
    for d_i in dataset_type:
        dataset, env = d3rlpy.datasets.get_d4rl(e_i+'-'+d_i+'-v2', args.d4rl_dataset_dir)
        transitions = []
        for episode in dataset.episodes:
            transitions += episode.transitions

        _batch_size = 1
        _n_steps = 1
        _gamma = 0.95
        _n_frames = 1
        _real_ratio = 1
        _generated_maxlen = 100000
        iterator = RoundIterator(
            transitions,
            batch_size=_batch_size,
            n_steps=_n_steps,
            gamma=_gamma,
            n_frames=_n_frames,
            real_ratio=_real_ratio,
            generated_maxlen=_generated_maxlen,
            shuffle=False,
        )

        rtg = []
        for t_i in iterator:
            rtg.append(t_i.rtgs[0][0])

        print(len(rtg))

        bins = 40
        plt.hist(rtg, bins=bins)
        plt.tight_layout()
        plt.savefig('./'+e_i+'-'+d_i+'-v2-'+'rtg-hist-'+str(_gamma)+'.png', dpi=200)
        plt.close('all')

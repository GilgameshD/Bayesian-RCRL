'''
Author: Wenhao Ding
Email: wenhaod@andrew.cmu.edu
Date: 2022-08-05 19:12:08
LastEditTime: 2022-10-20 10:20:22
Description: 
'''

import argparse
import os

import d3rlpy


parser = argparse.ArgumentParser()
parser.add_argument('-dd', '--d4rl_dataset_dir', type=str, default='/mnt/data1/wenhao/.d4rl/datasets', help='directory for saving d4rl dataset')
parser.add_argument('--env_name', type=str, default='breakout', help='[breakout, pong, seaquest, qbert, asterix]')
parser.add_argument('-dt', '--dataset_type', type=str, default='medium', help='[mixed, medium, expert]')
args = parser.parse_args()

os.environ['D4RL_DATASET_DIR'] = args.d4rl_dataset_dir
dataset, _ = d3rlpy.datasets.get_atari(args.env_name+'-more-'+args.dataset_type+'-v0', args.d4rl_dataset_dir)
print('finish writing dataset: {}-{}'.format(args.env_name, args.dataset_type))

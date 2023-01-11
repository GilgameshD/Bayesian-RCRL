'''
Author: Wenhao Ding
Email: wenhaod@andrew.cmu.edu
Date: 2021-12-21 11:57:44
LastEditTime: 2023-01-09 21:19:20
Description: 
'''

from d4rl.infos import REF_MIN_SCORE, REF_MAX_SCORE


rcrl_data = [
    ['halfcheetah', 'medium-replay', 3183],
    ['halfcheetah', 'medium', 4869],
    ['halfcheetah', 'medium-expert', 10634],
    ['halfcheetah', 'expert', 10688],

    ['hopper', 'medium-replay', 2884],
    ['hopper', 'medium', 1914],
    ['hopper', 'medium-expert', 3589],
    ['hopper', 'expert', 3631],

    ['walker2d', 'medium-replay', 2664],
    ['walker2d', 'medium', 3260],
    ['walker2d', 'medium-expert', 4979],
    ['walker2d', 'expert', 5017],
]

for d_i in rcrl_data:
    env_name = d_i[0]
    dataset_type = d_i[1]
    value = d_i[2]
    dataset = env_name + '-' + dataset_type + '-v2'
    min_value = REF_MIN_SCORE[dataset]
    max_value = REF_MAX_SCORE[dataset]
    print('---------------------------------')
    normalized_score = (value - min_value) / (max_value - min_value)
    print(dataset, '|', value, '->', normalized_score)
print('---------------------------------')

'''
Author: Wenhao Ding
Email: wenhaod@andrew.cmu.edu
Date: 2021-12-21 11:57:44
LastEditTime: 2022-08-14 11:38:47
Description: 
'''

import numpy as np
import matplotlib.pyplot as plt
import os


def smooth(data, box_pt):
    box = np.ones(box_pt)/box_pt
    data_smooth = np.convolve(data, box, mode='same')
    return data_smooth


def read_model_data(path, data_length, load_dict=True, use_accumulated=True):
    test_reward_list = []
    for i in range(50):
        filename = '../plots/'+path+'/reward.'+str(i)+'.npy'
        if not os.path.exists(filename):
            continue
        
        if load_dict:
            data = np.load(filename, allow_pickle=True).item()
            if not use_accumulated:
                test_reward = data['final_reward'][0:data_length]
            else:
                test_reward = data['acc_reward'][0:data_length]
        else:
            test_reward = np.load(filename, allow_pickle=True)

        if len(test_reward_list) == 0 or len(test_reward) == len(test_reward_list[-1]):
            test_reward_list.append(test_reward)
    
    assert len(test_reward_list) > 0, 'Test reward list is empty'
    test_reward_list_mean = np.mean(test_reward_list, axis=0)
    test_reward_list_std = np.std(test_reward_list, axis=0)

    return test_reward_list_mean, test_reward_list_std



def read_data_minimal(path):
    test_reward_list = []
    minimal_len = np.inf
    for i in range(50):
        filename = '../plots/'+path+'/reward.'+str(i)+'.npy'
        if not os.path.exists(filename):
            continue
        test_reward = np.load(filename, allow_pickle=True)
        test_reward_list.append(test_reward)
        if len(test_reward) < minimal_len:
            minimal_len = len(test_reward)

    new_reward_list = []
    for i in test_reward_list:
        new_reward_list.append(i[0:minimal_len])
    
    test_reward_list_mean = np.mean(new_reward_list, axis=0)
    test_reward_list_std = np.std(new_reward_list, axis=0)
    return test_reward_list_mean, test_reward_list_std



color_list = [
    'C0', 
    'C1',
    'C2',
    'C3', 
    'C4', 
    'royalblue',
    '#9CCB86',
    'gold',
    'C9',
    'black',
    'deeppink'
]


type = 'ai2thor_iid'

plt.figure(figsize=(5, 5))
ax = plt.subplot(111)

alpha = 0.1

data = [
    [0, 66, 49, 79, 124, 36], 
    [0, 32, 75, 46, 69, 41],
    [0, 41, 34, 34, 50, 34],
    [0, 35, 41, 33, 48, 35],
    [0, 35, 69, 43, 54, 45],
    [0, 46, 131, 66, 56, 57],
    [0, 58, 63, 55, 91, 41],
    [0, 78, 91, 53, 42, 96],
    [0, 20, 40, 58, 35, 46],
    [0, 38, 35, 36, 38, 35],
]
data_mean = np.mean(data, axis=0)
data_std = np.std(data, axis=0)
ax.plot(range(len(data_mean)), data_mean, c=color_list[0], label='DT-Official-Repo')
ax.fill_between(range(len(data_mean)), data_mean-data_std, data_mean+data_std, facecolor=color_list[0], alpha=alpha)


data_mean, data_std = read_model_data('dt_offline_breakout', 10000, load_dict=False)
data_mean = np.concatenate([np.zeros(1,), data_mean])
data_std = np.concatenate([np.zeros(1,), data_std])
ax.plot(range(len(data_mean)), data_mean, c=color_list[1], label='DT')
ax.fill_between(range(len(data_mean)), data_mean-data_std, data_mean+data_std, facecolor=color_list[1], alpha=alpha)


data_mean, data_std = read_model_data('ddt_offline_breakout', 10000, load_dict=False)
data_mean = np.concatenate([np.zeros(1,), data_mean])
data_std = np.concatenate([np.zeros(1,), data_std])
ax.plot(range(len(data_mean)), data_mean, c=color_list[2], label='DT-Decoupled')
ax.fill_between(range(len(data_mean)), data_mean-data_std, data_mean+data_std, facecolor=color_list[2], alpha=alpha)


data = [
    [0, 15, 49, 37, 67, 56], 
    [0, 32, 61, 60, 68, 69],
    [0, 21, 46, 76, 55, 78],
    [0, 49, 74, 60, 54, 74],
    [0, 36, 24, 48, 31, 47],
    [0, 24, 58, 50, 46, 53],
    [0, 48, 48, 38, 63, 85],
    [0, 52, 44, 49, 52, 46],
    [0, 36, 62, 53, 56, 74],
    [0, 39, 34, 37, 37, 42]
]
data_mean = np.mean(data, axis=0)
data_std = np.std(data, axis=0)
ax.plot(range(len(data_mean)), data_mean, c=color_list[3], label='DT-Decoupled-Ruixiang')
ax.fill_between(range(len(data_mean)), data_mean-data_std, data_mean+data_std, facecolor=color_list[3], alpha=alpha)

#plt.axvline(x=20, color='gray', linestyle='--', label='DT-Online Buffer Size')

# Shrink current axis
box = ax.get_position()
#ax.set_position([box.x0, box.y0, box.width * 0.7, box.height])
#ax.legend(fontsize=9, ncol=1, bbox_to_anchor=(1, 0.4))
ax.legend(fontsize=9)
plt.xlim([0, 5])
#plt.ylim([1.5, 8])
#plt.ylim([3.5, 8.1])
plt.xlabel('Epoch number')
plt.ylabel('Test reward')
#plt.show()






plt.close('all')
plt.figure(figsize=(5, 5))
ax = plt.subplot(111)
alpha = 0.2


data_mean, data_std = read_data_minimal('ddt_offline_breakout_debug')
x = np.array(list(range(len(data_mean)))) * 100
ax.plot(x, data_mean, c=color_list[0], label='DT-Decoupled-Transformer')
ax.fill_between(x, data_mean-data_std, data_mean+data_std, facecolor=color_list[0], alpha=alpha)


data_mean, data_std = read_data_minimal('ddt_offline_breakout_debug_linear')
x = np.array(list(range(len(data_mean)))) * 100
ax.plot(x, data_mean, c=color_list[1], label='DT-Decoupled-Linear')
ax.fill_between(x, data_mean-data_std, data_mean+data_std, facecolor=color_list[1], alpha=alpha)



ax.legend(fontsize=9)
plt.xlim([0, 12000])
plt.grid()
plt.xlabel('Training step')
plt.ylabel('Test reward (10 Average)')
plt.show()

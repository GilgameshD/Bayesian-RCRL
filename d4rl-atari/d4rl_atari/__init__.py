'''
Author: Wenhao Ding
Email: wenhaod@andrew.cmu.edu
Date: 2022-09-09 15:13:43
LastEditTime: 2023-01-18 00:23:50
Description: 
'''
from gym.envs.registration import register

# list from https://github.com/openai/gym/blob/master/gym/envs/__init__.py
for game in [
        'adventure', 'air-raid', 'alien', 'amidar', 'assault', 'asterix',
        'asteroids', 'atlantis', 'bank-heist', 'battle-zone', 'beam-rider',
        'berzerk', 'bowling', 'boxing', 'breakout', 'carnival', 'centipede',
        'chopper-command', 'crazy-climber', 'defender', 'demon-attack',
        'double-dunk', 'elevator-action', 'enduro', 'fishing-derby', 'freeway',
        'frostbite', 'gopher', 'gravitar', 'hero', 'ice-hockey', 'jamesbond',
        'journey-escape', 'kangaroo', 'krull', 'kung-fu-master',
        'montezuma-revenge', 'ms-pacman', 'name-this-game', 'phoenix',
        'pitfall', 'pong', 'pooyan', 'private-eye', 'qbert', 'riverraid',
        'road-runner', 'robotank', 'seaquest', 'skiing', 'solaris',
        'space-invaders', 'star-gunner', 'tennis', 'time-pilot', 'tutankham',
        'up-n-down', 'venture', 'video-pinball', 'wizard-of-wor',
        'yars-revenge', 'zaxxon'
]:

    max_episode_steps = 108000
    for index in range(5):
        register(
            id='{}-mixed-v{}'.format(game, index),
            entry_point='d4rl_atari.envs:OfflineAtariEnv',
            max_episode_steps=max_episode_steps,
            kwargs={'game': game, 'index': index + 1, 'start_epoch': 1, 'last_epoch': 1}
        )

        register(
            id='{}-medium-v{}'.format(game, index),
            entry_point='d4rl_atari.envs:OfflineAtariEnv',
            max_episode_steps=max_episode_steps,
            kwargs={'game': game, 'index': index + 1, 'start_epoch': 10, 'last_epoch': 10}
        )

        register(
            id='{}-expert-v{}'.format(game, index),
            entry_point='d4rl_atari.envs:OfflineAtariEnv',
            max_episode_steps=max_episode_steps,
            kwargs={'game': game, 'index': index + 1, 'start_epoch': 50, 'last_epoch': 50}
        )

        # 1, 2, 3, 4, 5
        register(
            id='{}-more-mixed-v{}'.format(game, index),
            entry_point='d4rl_atari.envs:OfflineAtariEnv',
            max_episode_steps=max_episode_steps,
            kwargs={'game': game, 'index': index + 1, 'start_epoch': 1, 'last_epoch': 5}
        )

        # 10, 20, 30, 40, 50
        register(
            id='{}-more-medium-v{}'.format(game, index),
            entry_point='d4rl_atari.envs:OfflineAtariEnv',
            max_episode_steps=max_episode_steps,
            kwargs={'game': game, 'index': index + 1, 'start_epoch': 10, 'step_size': 10, 'last_epoch': 50}
        )

        # 10, 20, 30, 40, 50 but only use 10% data in each epoch
        register(
            id='{}-more-small-v{}'.format(game, index),
            entry_point='d4rl_atari.envs:OfflineAtariEnv',
            max_episode_steps=max_episode_steps,
            kwargs={'game': game, 'index': index + 1, 'start_epoch': 10, 'step_size': 10, 'last_epoch': 50, 'percent_each_epoch': 0.1}
        )

        # 46, 47, 48, 49, 50
        register(
            id='{}-more-expert-v{}'.format(game, index),
            entry_point='d4rl_atari.envs:OfflineAtariEnv',
            max_episode_steps=max_episode_steps,
            kwargs={'game': game, 'index': index + 1, 'start_epoch': 46, 'last_epoch': 50}
        )

    '''
    for index in range(5):
        for epoch in range(50):
            register(id='{}-epoch-{}-v{}'.format(game, epoch + 1, index),
                        entry_point='d4rl_atari.envs:OfflineAtariEnv',
                        max_episode_steps=108000,
                        kwargs={
                            'game': game,
                            'index': index + 1,
                            'start_epoch': epoch + 1,
                            'last_epoch': epoch + 1,
                        })
    '''

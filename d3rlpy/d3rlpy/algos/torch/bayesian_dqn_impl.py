'''
Author: Wenhao Ding
Email: wenhaod@andrew.cmu.edu
Date: 2022-09-07 14:24:44
LastEditTime: 2022-09-13 14:35:22
Description: 
'''

from typing import Optional, Sequence

import torch

from ...gpu import Device
from ...models.encoders import EncoderFactory
from ...models.optimizers import OptimizerFactory
from ...models.q_functions import QFunctionFactory
from ...preprocessing import RewardScaler, Scaler
from ...torch_utility import TorchMiniBatch
from .dqn_impl import DoubleDQNImpl


class BayesianDiscreteDQNImpl(DoubleDQNImpl):
    _alpha: float

    """ Based on the Double DQN implementation. 
        We overwrite compute_loss() which requires a new compute_error().
        We leave _compute_conservative_loss() untouched since we use the original CQL.
    """
    def __init__(
        self,
        observation_shape: Sequence[int],
        action_size: int,
        learning_rate: float,
        optim_factory: OptimizerFactory,
        encoder_factory: EncoderFactory,
        q_func_factory: QFunctionFactory,
        gamma: float,
        alpha: float,
        n_critics: int,
        use_gpu: Optional[Device],
        scaler: Optional[Scaler],
        reward_scaler: Optional[RewardScaler],
    ):
        super().__init__(
            observation_shape=observation_shape,
            action_size=action_size,
            learning_rate=learning_rate,
            optim_factory=optim_factory,
            encoder_factory=encoder_factory,
            q_func_factory=q_func_factory,
            gamma=gamma,
            n_critics=n_critics,
            use_gpu=use_gpu,
            scaler=scaler,
            reward_scaler=reward_scaler,
        )

        # the q function should always be bayesian q function
        assert q_func_factory.get_type() in ['bayesian', 'c51'], 'BayesianDiscreteDQNImpl requires DiscreteBayesianQFunction or C51QFunctionFactory'

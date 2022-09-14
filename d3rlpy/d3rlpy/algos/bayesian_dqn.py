'''
Author: Wenhao Ding
Email: wenhaod@andrew.cmu.edu
Date: 2022-09-07 14:24:44
LastEditTime: 2022-09-08 22:46:12
Description: 
'''

from typing import Any, Optional, Sequence

from ..argument_utility import (
    EncoderArg,
    QFuncArg,
    RewardScalerArg,
    ScalerArg,
    UseGPUArg,
)
from ..models.optimizers import AdamFactory, OptimizerFactory
from .dqn import DoubleDQN
from .torch.bayesian_dqn_impl import BayesianDiscreteDQNImpl


class BayesianDiscreteDQN(DoubleDQN):
    r"""DQN with Bayesian Q function.
    """

    _alpha: float
    _impl: Optional[BayesianDiscreteDQNImpl]

    def __init__(
        self,
        *,
        learning_rate: float = 6.25e-5,
        optim_factory: OptimizerFactory = AdamFactory(),
        encoder_factory: EncoderArg = "default",
        q_func_factory: QFuncArg = "mean",
        batch_size: int = 32,
        n_frames: int = 1,
        n_steps: int = 1,
        gamma: float = 0.99,
        alpha: float = 1.0,
        n_critics: int = 1,
        target_update_interval: int = 8000,
        use_gpu: UseGPUArg = False,
        scaler: ScalerArg = None,
        reward_scaler: RewardScalerArg = None,
        impl: Optional[BayesianDiscreteDQNImpl] = None,
        **kwargs: Any,
    ):
        super().__init__(
            learning_rate=learning_rate,
            optim_factory=optim_factory,
            encoder_factory=encoder_factory,
            q_func_factory=q_func_factory,
            batch_size=batch_size,
            n_frames=n_frames,
            n_steps=n_steps,
            gamma=gamma,
            n_critics=n_critics,
            target_update_interval=target_update_interval,
            use_gpu=use_gpu,
            scaler=scaler,
            reward_scaler=reward_scaler,
            impl=impl,
            **kwargs,
        )
        self._alpha = alpha

    def _create_impl(
        self, observation_shape: Sequence[int], action_size: int
    ) -> None:
        self._impl = BayesianDiscreteDQNImpl(
            observation_shape=observation_shape,
            action_size=action_size,
            learning_rate=self._learning_rate,
            optim_factory=self._optim_factory,
            encoder_factory=self._encoder_factory,
            q_func_factory=self._q_func_factory,
            gamma=self._gamma,
            alpha=self._alpha,
            n_critics=self._n_critics,
            use_gpu=self._use_gpu,
            scaler=self._scaler,
            reward_scaler=self._reward_scaler,
        )
        self._impl.build()

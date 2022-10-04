'''
Author: Wenhao Ding
Email: wenhaod@andrew.cmu.edu
Date: 2022-09-07 14:24:44
LastEditTime: 2022-10-03 20:20:03
Description: 
'''

import os
from typing import Any, Optional, Sequence

from ..argument_utility import (
    EncoderArg,
    QFuncArg,
    RewardScalerArg,
    ScalerArg,
    UseGPUArg,
)
from ..models.optimizers import AdamFactory, OptimizerFactory
from .dqn import DQN
from .torch.bayesian_dqn_impl import BayesianDiscreteDQNImpl
from ..algos.bc import DiscreteBC


class BayesianDiscreteDQN(DQN):
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
        n_critics: int = 1,
        target_update_interval: int = 8000,
        threshold_c: float = 1.0,
        penalty_w: float = 0.0,
        weight_R: float = 1.0,
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

        # condition c used for testing
        self.threshold_c = threshold_c
        self.penalty_w = penalty_w
        self.weight_R = weight_R

        # this is a pre-trained model used for \beta
        self.beta_model = DiscreteBC(
            batch_size=batch_size,
            n_frames=n_frames,
            learning_rate=learning_rate,
            q_func_factory=None,
            encoder_factory=encoder_factory,
            scaler=scaler,
            use_gpu=use_gpu,
        )

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
            n_critics=self._n_critics,
            use_gpu=self._use_gpu,
            scaler=self._scaler,
            reward_scaler=self._reward_scaler,
            beta_model=self.beta_model,
            threshold_c=self.threshold_c,
            penalty_w=self.penalty_w,
            weight_R=self.weight_R,
        )
        self._impl.build()

    def fit_beta_policy(self, model_dir, env, dataset, n_epochs) -> None:
        ''' Fit policy \beta(a|s) based on the entire dataset'''

        # create the folders
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        model_path = os.path.join(model_dir, 'beta_policy.pt')
        if os.path.exists(model_path):
            print('Beta model already exist, loading model...')
            self.beta_model.build_with_dataset(dataset)
            self.beta_model.load_model(model_path)
            return
        
        # fit beta policy model
        print('Training beta model...')
        self.beta_model.fit(
            dataset=dataset,
            n_epochs=n_epochs,
            verbose=False,
            save_metrics=False,
            eval_step_interval=10000000,
        )
        self.beta_model.save_model(model_path)

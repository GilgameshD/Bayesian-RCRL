'''
Author: Wenhao Ding
Email: wenhaod@andrew.cmu.edu
Date: 2022-09-07 14:24:44
LastEditTime: 2023-01-13 20:42:55
Description: 
'''

import os
from typing import Any, Dict, Optional, Sequence

from ..argument_utility import (
    EncoderArg,
    QFuncArg,
    RewardScalerArg,
    ActionScalerArg,
    ScalerArg,
    UseGPUArg,
    check_encoder,
    check_use_gpu
)
from ..models.optimizers import AdamFactory, OptimizerFactory
from .dqn import DQN
from .torch.bayesian_dqn_impl import BayesianDiscreteDQNImpl, BayesianDQNImpl, RCRLImpl
from ..algos.bc import DiscreteBC, BC
from .base import AlgoBase
from ..dataset import TransitionMiniBatch
from ..constants import ActionSpace, IMPL_NOT_INITIALIZED_ERROR
from ..models.encoders import VectorEncoderFactory


class BayesianDiscreteDQN(DQN):
    r"""Bayesian Distributional Agent with discrete action (based on DQN).
    """

    _alpha: float
    _impl: Optional[BayesianDiscreteDQNImpl]

    def __init__(
        self,
        *,
        beta_learning_rate: float = 0.00025,
        learning_rate: float = 0.00025,
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
        beta_weight_penalty: float = 0.5,
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

        # this is a pre-trained model used for \beta
        self.beta_model = DiscreteBC(
            batch_size=batch_size,
            n_frames=n_frames,
            learning_rate=beta_learning_rate,
            q_func_factory=None,
            beta=beta_weight_penalty,
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
        )
        self._impl.build()

    def fit_beta_policy(self, model_dir, dataset, n_epochs) -> None:
        ''' Fit policy \beta(a|s) based on the entire dataset'''

        # create the folders
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        model_path = os.path.join(model_dir, 'beta_policy.pt')
        if os.path.exists(model_path):
            print('Beta model already exist, loading model...')
            self.beta_model.build_with_dataset(dataset)
            self.beta_model.load_model(model_path)
        else:
            # fit beta policy model
            print('Training beta model...')
            self.beta_model.fit(
                dataset=dataset,
                n_epochs=n_epochs,
                verbose=False,
                save_metrics=False,
                eval_step_interval=10000000,
            )
            # skip saving if other process already saves the model
            if not os.path.exists(model_path):
                self.beta_model.save_model(model_path)


class BayesianDQN(DQN):
    r"""Bayesian Distributional Agent, using EBM for continuous action
    """

    _alpha: float
    _impl: Optional[BayesianDQNImpl]

    def __init__(
        self,
        *,
        beta_learning_rate: float = 0.00025,
        learning_rate: float = 0.00025,
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
        noise_shrink: float = 0.5,
        noise_scale: float = 0.33,
        sample_iters: int = 3,
        inference_samples: int = 2**14,
        beta_weight_penalty: float = 0.5,
        use_gpu: UseGPUArg = False,
        scaler: ScalerArg = None,
        reward_scaler: RewardScalerArg = None,
        impl: Optional[BayesianDQNImpl] = None,
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

        # for MCMC sampling
        self.sample_iters = sample_iters
        self.noise_shrink = noise_shrink
        self.noise_scale = noise_scale

        # action sample number
        self.inference_samples = inference_samples

        # this is a pre-trained model used for \beta
        encoder = VectorEncoderFactory()
        self.beta_model = BC(
            batch_size=batch_size,
            n_frames=n_frames,
            learning_rate=beta_learning_rate,
            q_func_factory=None,
            beta=beta_weight_penalty,
            encoder_factory=encoder,
            scaler='standard',
            use_gpu=use_gpu,
        )

    def get_action_type(self) -> ActionSpace:
        # although we inherit from DQN, we will use continuous action
        return ActionSpace.CONTINUOUS

    def _create_impl(
        self, observation_shape: Sequence[int], action_size: int
    ) -> None:
        self._impl = BayesianDQNImpl(
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
            noise_shrink=self.noise_shrink,
            noise_scale=self.noise_scale,
            sample_iters=self.sample_iters,
            inference_samples=self.inference_samples,
        )
        self._impl.build()

    def fit_beta_policy(self, model_dir, dataset, n_epochs) -> None:
        ''' Fit policy \beta(a|s) based on the entire dataset'''

        # create the folders
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        model_path = os.path.join(model_dir, 'beta_policy.pt')
        if os.path.exists(model_path):
            print('Beta model already exist, loading model...')
            self.beta_model.build_with_dataset(dataset)
            self.beta_model.load_model(model_path)
        else:
            # fit beta policy model
            print('Training beta model...')
            self.beta_model.fit(
                dataset=dataset,
                n_epochs=n_epochs,
                verbose=False,
                save_metrics=False,
                eval_step_interval=10000000,
            )
            # skip saving if other process already saves the model
            if not os.path.exists(model_path):
                self.beta_model.save_model(model_path)


class RCRL(AlgoBase):
    _alpha: float
    _impl: Optional[RCRLImpl]

    def __init__(
        self,
        *,
        learning_rate: float = 0.00025,
        optim_factory: OptimizerFactory = AdamFactory(),
        encoder_factory: EncoderArg = "default",
        batch_size: int = 32,
        n_frames: int = 1,
        n_steps: int = 1,
        gamma: float = 0.99,
        threshold_c: float = 1.0,
        noise_shrink: float = 0.5,
        noise_scale: float = 0.33,
        sample_iters: int = 3,
        inference_samples: int = 2**14,
        n_quantiles: int = 20,
        Vmin: float = 0,
        Vmax: float = 1000,
        weight_R: float = 1.0,
        weight_A: float = 1.0,
        n_neg_samples: int = 64,
        use_neg_rtg: bool = False,
        use_gpu: UseGPUArg = False,
        scaler: ScalerArg = None,
        action_scaler: ActionScalerArg = None,
        impl: Optional[RCRLImpl] = None,
        **kwargs: Any,
    ):
        super().__init__(
            batch_size=batch_size,
            n_frames=n_frames,
            n_steps=n_steps,
            gamma=gamma,
            scaler=scaler,
            action_scaler=action_scaler,
            kwargs=kwargs,
        )
        self._learning_rate = learning_rate
        self._optim_factory = optim_factory
        self._encoder_factory = check_encoder(encoder_factory)
        self._use_gpu = check_use_gpu(use_gpu)
        self._impl = impl

        # condition c used for testing
        self._threshold_c = threshold_c

        # for MCMC sampling
        self._sample_iters = sample_iters
        self._noise_shrink = noise_shrink
        self._noise_scale = noise_scale

        # action sample number
        self._inference_samples = inference_samples
        self._n_neg_samples = n_neg_samples

        # for rtg estimation
        self._n_quantiles = n_quantiles
        self._Vmin = Vmin
        self._Vmax = Vmax
        self._weight_R = weight_R
        self._weight_A = weight_A
        self._use_neg_rtg = use_neg_rtg
        
    def _create_impl(
        self, observation_shape: Sequence[int], action_size: int
    ) -> None:
        self._impl = RCRLImpl(
            observation_shape=observation_shape,
            action_size=action_size,
            learning_rate=self._learning_rate,
            optim_factory=self._optim_factory,
            encoder_factory=self._encoder_factory,
            use_gpu=self._use_gpu,
            scaler=self._scaler,
            action_scaler=self._action_scaler,
            threshold_c=self._threshold_c,
            noise_shrink=self._noise_shrink,
            noise_scale=self._noise_scale,
            sample_iters=self._sample_iters,
            inference_samples=self._inference_samples,
            n_quantiles=self._n_quantiles,
            Vmin=self._Vmin,
            Vmax=self._Vmax,
            weight_R=self._weight_R,
            weight_A=self._weight_A,
            n_neg_samples=self._n_neg_samples,
            use_neg_rtg=self._use_neg_rtg
        )
        self._impl.build()

    def get_action_type(self) -> ActionSpace:
        return ActionSpace.CONTINUOUS

    def _update(self, batch: TransitionMiniBatch) -> Dict[str, float]:
        assert self._impl is not None, IMPL_NOT_INITIALIZED_ERROR
        loss_infonce, loss_r = self._impl.update_imitator(batch.observations, batch.actions, batch.rtgs)
        return {"loss": loss_infonce, 'loss_r': loss_r}

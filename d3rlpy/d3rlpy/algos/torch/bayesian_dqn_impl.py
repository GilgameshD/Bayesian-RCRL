'''
Author: Wenhao Ding
Email: wenhaod@andrew.cmu.edu
Date: 2022-09-07 14:24:44
LastEditTime: 2022-10-14 11:07:42
Description: 
'''

import matplotlib.pyplot as plt
from typing import Optional, Sequence

import torch

from ...gpu import Device
from ...models.encoders import EncoderFactory
from ...models.optimizers import OptimizerFactory
from ...models.q_functions import QFunctionFactory
from ...preprocessing import RewardScaler, Scaler
from ...torch_utility import TorchMiniBatch
from .dqn_impl import DQNImpl


class BayesianDiscreteDQNImpl(DQNImpl):
    """ Based on the DQN implementation with the following modifications:
            (1) We use the same policy evaluation pipeline of C51 but adding a BC loss.
            (2) For target value calculation and action selection, we are different from DQN.
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
        n_critics: int,
        use_gpu: Optional[Device],
        scaler: Optional[Scaler],
        reward_scaler: Optional[RewardScaler],
        beta_model,
        threshold_c,
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
        assert q_func_factory.get_type() in ['bayesian'], 'BayesianDiscreteDQNImpl requires DiscreteBayesianQFunction'

        # beta policy learned from a BC model
        self.beta_model = beta_model
        self.threshold_c = threshold_c
        self.counter = 0

    def compute_loss(
        self,
        batch: TorchMiniBatch,
        q_tpn: torch.Tensor,
    ) -> torch.Tensor:
        assert self._q_func is not None
        return self._q_func.compute_error(
            observations=batch.observations,
            actions=batch.actions.long(),
            rewards=batch.rewards,
            target=q_tpn,
            terminals=batch.terminals,
            gamma=self._gamma**batch.n_steps,
        )

    def compute_target(self, batch: TorchMiniBatch) -> torch.Tensor:
        assert self._targ_q_func is not None
        with torch.no_grad():
            # we get log probability from a BC model
            log_probs_next_action = self.beta_model._impl._imitator(batch.next_observations)
            return self._targ_q_func.compute_target(
                batch.next_observations,
                log_probs_next_action,
                reduction="min",
            )

    def _predict_best_action(self, x: torch.Tensor) -> torch.Tensor:
        assert self._q_func is not None
        # NOTE: we change the forward() of _q_func to output p_R instead of q value
        logits_a_and_R = self._q_func.forward_bayesian(x)
        batch_size = logits_a_and_R.shape[0]
        action_dim = logits_a_and_R.shape[1]
        n_quantiles = logits_a_and_R.shape[2]
        p_a_and_R = torch.softmax(logits_a_and_R.view(batch_size, action_dim*n_quantiles), dim=1)
        p_a_and_R = p_a_and_R.view(batch_size, action_dim, n_quantiles)

        # calculate the threshold for R (assume B=1)
        # p(R) = \sum_{a} p(a, R)
        p_R = torch.sum(p_a_and_R, dim=1) # [B, _n_quantiles]
        sum = 0
        for idx in range(p_R.shape[1]-1, -1, -1):
            sum += p_R[0, idx]
            if sum >= self.threshold_c:
                break

        # calculate p(a, R|R > c)
        logits_a_and_R_cond_c = logits_a_and_R[:, :, idx:].contiguous()
        p_a_and_R_cond_c = torch.softmax(logits_a_and_R_cond_c.view(batch_size, action_dim*(n_quantiles-idx)), dim=1)
        p_a_and_R_cond_c = p_a_and_R_cond_c.view(batch_size, action_dim, n_quantiles-idx)

        # p(a |R > c) = \sum_{R} p(a, R|R > c)
        p_a_cond_c = p_a_and_R_cond_c.sum(dim=2)

        '''
        if self.counter % 10000 == 0:
            print(idx)
            max_value = torch.max(p_a_and_R).detach().cpu().numpy()
            plt.figure(figsize=(8, 8))
            plt.subplot(4, 1, 1)
            plt.ylim(0, max_value)
            plt.bar(range(len(p_a_and_R[0, 0])), p_a_and_R.detach().cpu().numpy()[0, 0])
            plt.subplot(4, 1, 2)
            plt.ylim(0, max_value)
            plt.bar(range(len(p_a_and_R[0, 1])), p_a_and_R.detach().cpu().numpy()[0, 1])
            plt.subplot(4, 1, 3)
            plt.bar(range(len(p_a_and_R[0, 2])), p_a_and_R.detach().cpu().numpy()[0, 2])
            plt.ylim(0, max_value)
            plt.subplot(4, 1, 4)
            plt.bar(range(len(p_a_and_R[0, 3])), p_a_and_R.detach().cpu().numpy()[0, 3])
            plt.ylim(0, max_value)
            plt.savefig('./plots/gamma_0_95/action_quantiles_'+str(self.counter)+'.png', dpi=200)
            plt.close('all')

            plt.figure(figsize=(3, 5))
            plt.imshow(p_a_and_R.detach().cpu().numpy()[0])
            plt.savefig('./plots/gamma_0_95/joint_distribution_'+str(self.counter)+'.png', dpi=200)
            plt.close('all')

            plt.figure(figsize=(5, 5))
            plt.imshow(p_a_and_R_cond_c.detach().cpu().numpy()[0])
            plt.savefig('./plots/gamma_0_95/p_a_and_R_cond_c_'+str(self.counter)+'.png', dpi=200)
            plt.close('all')
        self.counter += 1
        '''

        # sample from p(a|R > c)
        #action = torch.distributions.Categorical(p_a_cond_c).sample()

        # greedy
        action = p_a_cond_c.argmax(dim=-1)

        # use beta policy to select action
        #log_probs_next_action = self.beta_model._impl._imitator(x)
        #action = torch.exp(log_probs_next_action).argmax(dim=1)
        return action

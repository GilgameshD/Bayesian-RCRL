'''
Author: Wenhao Ding
Email: wenhaod@andrew.cmu.edu
Date: 2022-09-07 14:24:44
LastEditTime: 2023-01-11 02:18:53
Description: 
'''

import matplotlib.pyplot as plt
from typing import Optional, Sequence, cast

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Categorical
from torch.optim import Adam

from ...gpu import Device
from ...models.encoders import (
    EncoderFactory, 
    Encoder, 
    EncoderWithAction
)
from ...models.optimizers import OptimizerFactory
from ...models.q_functions import QFunctionFactory
from ...preprocessing import RewardScaler, Scaler
from ...torch_utility import TorchMiniBatch, torch_api, train_api
from .dqn_impl import DQNImpl
from .bc_impl import BCBaseImpl
from ...models.torch import EnsembleContinuousQFunction
from ...models.builders import create_continuous_q_function
from .base import TorchImplBase


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

        # calculate the threshold for R 
        # TODO: this method assumes B=1, so it is wrong when calculating TD error score in evaluation. should use torch.cumsum() instead
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
            plt.figure(figsize=(12, 12))
            action_dim = 18
            for i in range(action_dim):
                plt.subplot(action_dim, 1, i+1)
                plt.ylim(0, max_value)
                plt.bar(range(len(p_a_and_R[0, i])), p_a_and_R.detach().cpu().numpy()[0, i])

            plt.tight_layout()
            plt.savefig('./plots/gamma_0_95/action_quantiles_'+str(self.counter)+'.png', dpi=200)
            plt.close('all')

            plt.figure(figsize=(3, 5))
            plt.imshow(p_a_and_R.detach().cpu().numpy()[0])
            plt.tight_layout()
            plt.savefig('./plots/gamma_0_95/joint_distribution_'+str(self.counter)+'.png', dpi=200)
            plt.close('all')

            plt.figure(figsize=(5, 5))
            plt.imshow(p_a_and_R_cond_c.detach().cpu().numpy()[0])
            plt.tight_layout()
            plt.savefig('./plots/gamma_0_95/p_a_and_R_cond_c_'+str(self.counter)+'.png', dpi=200)
            plt.close('all')
        self.counter += 1
        '''

        # sample from p(a|R > c)
        #action = torch.distributions.Categorical(p_a_cond_c).sample()

        # select action with greedy
        action = p_a_cond_c.argmax(dim=-1)
        return action


class BayesianDQNImpl(DQNImpl):
    """ A continuous version of BayesianDQN
    """

    _q_func: Optional[EnsembleContinuousQFunction]
    _targ_q_func: Optional[EnsembleContinuousQFunction]

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
        noise_shrink,
        noise_scale,
        sample_iters,
        inference_samples,
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
        assert q_func_factory.get_type() in ['bayesian'], 'BayesianDQNImpl requires ContinuousBayesianQFunction'

        # beta policy learned from a BC model
        self.beta_model = beta_model
        self.threshold_c = threshold_c
        self.counter = 0

        # for MCMC sampler
        self.noise_scale = noise_scale
        self.noise_shrink = noise_shrink
        self.sample_iters = sample_iters
        self.inference_samples = inference_samples
        self.bounds = [-1.0, 1.0]

    def _build_network(self) -> None:
        self._q_func = create_continuous_q_function(
            self._observation_shape,
            self._action_size,
            self._encoder_factory,
            self._q_func_factory,
            n_ensembles=self._n_critics,
        )

    def compute_loss(
        self,
        batch: TorchMiniBatch,
        q_tpn: torch.Tensor,
    ) -> torch.Tensor:
        assert self._q_func is not None
        return self._q_func.compute_error(
            observations=batch.observations,
            actions=batch.actions,
            rewards=batch.rewards,
            target=q_tpn,
            terminals=batch.terminals,
            gamma=self._gamma**batch.n_steps,
        )

    def compute_target(self, batch: TorchMiniBatch) -> torch.Tensor:
        assert self._targ_q_func is not None
        with torch.no_grad():
            # we get action output from a continuous BC model
            action_next = self.beta_model._impl._imitator(batch.next_observations)
            return self._targ_q_func.compute_target(
                batch.next_observations,
                action_next,
                reduction="min",
            )

    def _predict_best_action(self, x: torch.Tensor) -> torch.Tensor:
        assert self._q_func is not None

        # augment x
        x_aug = x.unsqueeze(1).repeat(1, self.inference_samples, 1)
        batch_size = x_aug.size(0)
        state_size = x_aug.size(-1)

        # variables used in this step
        noise_scale = self.noise_scale

        # get random samples
        size = (batch_size, self.inference_samples, self._action_size)
        samples = torch.zeros(size, device=self.device).uniform_(self.bounds[0], self.bounds[1])

        # Derivative Free Optimizer (DFO)
        for _ in range(self.sample_iters):
            # compute energies
            energies = self._q_func.forward_energy(x_aug.reshape(-1, state_size), samples.reshape(-1, self._action_size), self.threshold_c)
            energies = energies.reshape(batch_size, self.inference_samples)
            probs = torch.softmax(-1.0 * energies, dim=-1)

            # resample with replacement
            idxs = torch.multinomial(probs, self.inference_samples, replacement=True)
            samples = samples[torch.arange(samples.size(0), device=self.device).unsqueeze(-1), idxs]

            # add noise and clip to target bounds
            samples = samples + torch.randn_like(samples) * noise_scale
            samples = samples.clamp(min=self.bounds[0], max=self.bounds[1])

            noise_scale *= self.noise_shrink

        # return target with highest probability
        energies = self._q_func.forward_energy(x_aug.reshape(-1, state_size), samples.reshape(-1, self._action_size), self.threshold_c)
        energies = energies.reshape(batch_size, self.inference_samples)
        probs = torch.softmax(-1.0 * energies, dim=-1)
        best_idxs = probs.argmax(dim=-1)
        action = samples[torch.arange(samples.size(0), device=self.device), best_idxs, :]

        '''
        if self.counter % 1000 == 0:
            # plot distribution of q value
            q_value_dist, _ = self._q_func.return_q_dist(x_aug.reshape(-1, state_size), samples.reshape(-1, self._action_size))
            plt.figure(figsize=(12, 12))
            plot_num = 10
            for i in range(plot_num):
                plt.subplot(plot_num, 1, i+1)
                plt.bar(range(len(q_value_dist[i])), q_value_dist[i].detach().cpu().numpy())

            plt.tight_layout()
            plt.savefig('./plots/weight_R_1/action_quantiles_'+str(self.counter)+'.png', dpi=200)
            plt.close('all')

        self.counter += 1
        '''

        return action


class EBM(nn.Module):
    def __init__(
        self,
        encoder: Encoder,
        action_size: int,
        n_quantiles: int,
        min_logstd: float = -4.0,
        max_logstd: float = 15.0,
    ):
        super().__init__()

        self._n_quantiles = n_quantiles
        self._min_logstd = min_logstd
        self._max_logstd = max_logstd
        self._encoder = encoder

        # for p(a|s)
        self._mu = nn.Linear(encoder.get_feature_size(), action_size)
        self._logstd = nn.Linear(encoder.get_feature_size(), action_size)

        # for p(r|a,s)
        self._rtg_c_a_s = nn.Linear(encoder.get_feature_size(), n_quantiles)

        # for p(r|s)
        self._rtg_c_s = nn.Linear(encoder.get_feature_size(), n_quantiles)

    def action_predictor(self, x: torch.Tensor) -> Normal:
        h_x = self._encoder.action_branch(x)
        mu = self._mu(h_x)
        logstd = self._logstd(h_x)
        clipped_logstd = logstd.clamp(self._min_logstd, self._max_logstd)
        return Normal(mu, clipped_logstd.exp())

    def value_predictor(self, x: torch.Tensor) -> Categorical:
        h_x = self._encoder.action_branch(x)
        logits = self._rtg_c_s(h_x)
        return Categorical(logits=logits)

    def rtg_predictor(self, x: torch.Tensor, action: torch.Tensor) -> Categorical:
        h_x_a = self._encoder.rtg_branch(x, action)
        logits = self._rtg_c_a_s(h_x_a)
        return Categorical(logits=logits)

    def forward(self, x: torch.Tensor, action: torch.Tensor, rtg_t_idx: torch.Tensor) -> torch.Tensor:
        pass

    def infer_energy(self, x: torch.Tensor, action: torch.Tensor, idx: int) -> torch.Tensor:
        beta_a_c_s = self.action_predictor(x)
        beta_r_c_s_a = self.rtg_predictor(x, action)

        # calculate energy log p(a|s)
        energy_a = -1.0 * beta_a_c_s.log_prob(action)               # [B*(N+1), dim_a]
        energy_a = energy_a.sum(-1)                                 # [B*(N+1)]

        # calculate energy log p(R>c|s, a)
        prob_sum = beta_r_c_s_a.probs[:, idx:].sum(-1) # [B*(N+1)]
        energy_r = -1.0 * torch.log(prob_sum)
        return energy_a, energy_r

    def compute_energy(self, x: torch.Tensor, action: torch.Tensor, rtg_t_idx: torch.Tensor) -> torch.Tensor:
        beta_a_c_s = self.action_predictor(x)
        beta_r_c_s_a = self.rtg_predictor(x, action)

        #print(beta_r_c_s_a.probs[0])

        # calculate energy
        energy_a = -1.0 * beta_a_c_s.log_prob(action)           # [B*(N+1), dim_a]
        energy_a = energy_a.sum(-1)                             # [B*(N+1)]

        use_log_prob = True
        if use_log_prob:
            # use log p(R=c)
            energy_r = -1.0 * beta_r_c_s_a.log_prob(rtg_t_idx)  # [B*(N+1)]
        else:
            # use log p(R>c)
            batch_size = energy_a.shape[0]
            index_mask = torch.arange(self._n_quantiles, device='cuda')[None]    # [1, _n_quantiles]
            index_mask = index_mask.repeat(batch_size, 1)                        # [B*(N+1), _n_quantiles]
            rtg_t_idx_extend = torch.clip(rtg_t_idx - 1, 0, self._n_quantiles)
            rtg_t_idx_extend = rtg_t_idx_extend[:, None].repeat(1, self._n_quantiles)
            index_mask = index_mask > rtg_t_idx_extend
            energy_r = torch.sum(beta_r_c_s_a.probs * index_mask, dim=1)
        return energy_a, energy_r


class RCRLImpl(BCBaseImpl):
    def __init__(
        self,
        observation_shape: Sequence[int],
        action_size: int,
        learning_rate: float,
        optim_factory: OptimizerFactory,
        encoder_factory: EncoderFactory,
        use_gpu: Optional[Device],
        scaler: Optional[Scaler],
        action_scaler: Optional[RewardScaler],
        threshold_c,
        noise_shrink,
        noise_scale,
        sample_iters,
        inference_samples,
        n_quantiles,
        Vmin,
        Vmax,
        weight_R,
        weight_A,
        n_neg_samples,
    ):
        super().__init__(
            observation_shape=observation_shape,
            action_size=action_size,
            learning_rate=learning_rate,
            optim_factory=optim_factory,
            encoder_factory=encoder_factory,
            use_gpu=use_gpu,
            scaler=scaler,
            action_scaler=action_scaler,
        )
        
        # for rtg estimation
        self._n_quantiles = n_quantiles
        self._Vmin = Vmin
        self._Vmax = Vmax
        self._weight_R = weight_R
        self._weight_A = weight_A

        # for DFO sampler
        self._noise_scale = noise_scale
        self._noise_shrink = noise_shrink
        self._sample_iters = sample_iters
        self._n_neg_samples = n_neg_samples
        self._bounds = [-1.0, 1.0]
        self._threshold_c = threshold_c
        self._inference_samples = inference_samples

    def _generate_negative_targets(self, target):
        # uniformly sample negative actions
        size = (target.shape[0], self._n_neg_samples, self._action_size)
        negatives = torch.zeros(size, device=self._device).uniform_(self._bounds[0], self._bounds[1])

        # Merge target and negatives: [B, N+1, D]
        targets = torch.cat([target.unsqueeze(dim=1), negatives], dim=1)

        # Generate a random permutation of the positives and negatives.
        permutation = torch.rand(targets.size(0), targets.size(1), device=self._device).argsort(dim=1)
        targets = targets[torch.arange(targets.size(0), device=self._device).unsqueeze(-1), permutation, :] # [B, N+1, D2]

        # Get the original index of the positive. This will serve as the class label for the loss.
        # Positive - 1, negative - 0
        ground_truth = (permutation == 0).nonzero()[:, 1].to(self._device)
        return targets, ground_truth

    def _convert_rtg_to_idx(self, rtg: torch.Tensor) -> torch.Tensor:
        """ Run a mapping from rtg to discrete index. e.g., [0, ..., 10] -> 0 """
        # get bin width
        bin_width = (self._Vmax - self._Vmin) / self._n_quantiles

        rtg_idx_list = torch.zeros_like(rtg) # [B]
        for b_i in range(self._n_quantiles):
            low = b_i * bin_width
            high = (b_i+1) * bin_width
            if b_i == self._n_quantiles - 1:
                rtg_idx_list += b_i * (rtg >= low)
            else:
                rtg_idx_list += b_i * ((rtg >= low) & (rtg < high))
        return rtg_idx_list.long()

    def _build_optim(self) -> None:
        assert self._imitator is not None
        assert self._value_network is not None
        self._optim = self._optim_factory.create(self._imitator.parameters(), lr=self._learning_rate)
        self._optim_r = Adam(self._value_network.parameters(), lr=self._learning_rate)

    def _build_network(self) -> None:
        # build encoder
        encoder = self._encoder_factory.create_ebm(self._observation_shape, self._action_size)

        # build energy model
        self._imitator = EBM(encoder, self._action_size, self._n_quantiles)

        # build value network
        self._value_network = self._encoder_factory.create_value_network(self._observation_shape, self._n_quantiles)

    @train_api
    @torch_api(scaler_targets=["obs_t"], action_scaler_targets=["act_t"])
    def update_imitator(self, obs_t: torch.Tensor, act_t: torch.Tensor, rtg_t: torch.Tensor) -> np.ndarray:
        assert self._optim is not None
        assert self._optim_r is not None

        # optimize joint distribution
        self._optim.zero_grad()
        loss_infonce = self.compute_infonce_loss(obs_t, act_t, rtg_t)
        loss_infonce.backward()
        self._optim.step()
        loss_infonce = loss_infonce.cpu().detach().numpy()

        # optimize value network
        self._optim_r.zero_grad()
        loss_r = self.compute_r_loss(obs_t, rtg_t) 
        loss_r.backward()      
        self._optim_r.step()
        loss_r = loss_r.cpu().detach().numpy()

        return loss_infonce, loss_r

    def compute_r_loss(self, obs_t: torch.Tensor, rtg_t: torch.Tensor) -> torch.Tensor:
        # log p(R|s)
        p_r_c_s = self._value_network(obs_t)
        rtg_t_pos_idx = self._convert_rtg_to_idx(rtg_t)[:, 0]
        likelihood_r = p_r_c_s.log_prob(rtg_t_pos_idx)   # [B*(N+1)]
        loss = -1.0 * likelihood_r.mean(dim=0)
        return loss

    def compute_infonce_loss(self, obs_t: torch.Tensor, act_t: torch.Tensor, rtg_t: torch.Tensor) -> torch.Tensor:
        # obs_t: [B, dim_s]
        # act_t: [B, dim_a]
        # rtg_t: [B, 1]
        assert self._imitator is not None

        # if we dont use negative RTG, some negative samples will never be explored and will be treated as positive
        use_neg_rtg = False

        # generate negative samples
        act_t_neg, act_t_gt = self._generate_negative_targets(act_t)  # [B, (N+1), dim_a], [B]
        batch_size = act_t_neg.size(0)
        sample_size = act_t_neg.size(1)
        act_t_neg = act_t_neg.reshape(batch_size * sample_size, -1)    # [B*(N+1), dim_a]

        # re-organize observation and action to form a new batch
        obs_t_aug = obs_t.unsqueeze(1).repeat((1, sample_size, 1))     # [B, N+1, dim_s]
        obs_t_aug = obs_t_aug.reshape(batch_size * sample_size, -1)    # [B*(N+1), dim_s]

        if use_neg_rtg:
            # generate negative RTG samples
            rtg_t_aug_idx = torch.randint(0, self._n_quantiles, size=(batch_size, sample_size, 1), device=self._device)
            batch_idx = torch.arange(batch_size, device=self._device)
            # convert continuous reward to index 
            rtg_t_pos_idx = self._convert_rtg_to_idx(rtg_t)
            rtg_t_aug_idx[batch_idx, act_t_gt, :] = rtg_t_pos_idx            # assign true rtg to positive samples
            rtg_t_aug_idx = rtg_t_aug_idx.reshape(batch_size * sample_size)  # [B*(N+1)]
        else:
            rtg_t_aug = rtg_t.unsqueeze(1).repeat((1, sample_size, 1))       # [B, N+1, 1]
            rtg_t_aug = rtg_t_aug.reshape(batch_size * sample_size)          # [B*(N+1)]
            # convert continuous reward to index 
            rtg_t_aug_idx = self._convert_rtg_to_idx(rtg_t_aug)

        # calculate energy
        energy_a, energy_r = self._imitator.compute_energy(obs_t_aug, act_t_neg, rtg_t_aug_idx)
        energy = energy_a + energy_r
        energy = energy.reshape(batch_size, sample_size)

        # infoNCE loss with negative action
        logits = -1.0 * energy
        loss_infonce = F.cross_entropy(logits, act_t_gt, reduction='mean')

        # loss to enforce beta(r|s, a)
        rtg_t_pos_idx = self._convert_rtg_to_idx(rtg_t)[:, 0] # [B]
        beta_r_c_s_a = self._imitator.rtg_predictor(obs_t, act_t)
        pos_energy_r = -1.0 * beta_r_c_s_a.log_prob(rtg_t_pos_idx)
        loss_action_value = pos_energy_r.mean(dim=0)

        '''
        # calculate loss rtg
        rtg_t_idx = self._convert_rtg_to_idx(rtg_t)[:, 0] # [B]
        likelihood_r = self._imitator.value_predictor(obs_t).log_prob(rtg_t_idx)   # [B]
        loss_r = -1.0 * likelihood_r.mean(dim=0)
        loss_infonce = loss_infonce + loss_r
        '''

        loss = self._weight_A * loss_infonce + self._weight_R * loss_action_value
        return loss

    def _predict_best_action(self, x: torch.Tensor) -> torch.Tensor:
        """ use DFO for inference """
        assert self._imitator is not None
        batch_size = x.size(0)

        rtg_from = 'state' # 'state', 'fix', 'action'
        rtg_select = 'greater'  # 'max', 'boundary', 'greater'

        if rtg_from in ['action', 'state']:
            if rtg_from == 'action':
                # calculate the threshold for rtg, use p(a|s) as prior instead of uniform samples
                num_action = 1000
                x_aug = x.unsqueeze(1).repeat(1, num_action, 1)                # [B, num_action, dim_s]

                action_dist = self._imitator.action_predictor(x)               # [B, dim_a]
                actions = action_dist.sample((num_action,)).transpose(1, 0)    # [B, num_action, dim_a]
                logits_rtg_c_as = self._imitator.rtg_predictor(x_aug.reshape(batch_size*num_action, -1), actions.reshape(batch_size*num_action, -1)).logits # [B*N, _n_quantiles]
                logits_rtg_c_as = logits_rtg_c_as.view(batch_size, num_action, self._n_quantiles)

                # p(R|s) = 1/N * sum_{a} p(R|a, s)
                logits_rtg_c_as = logits_rtg_c_as.mean(dim=1)   # [B, _n_quantiles]
                p_rtg_c_s = torch.softmax(logits_rtg_c_as, dim=1)
            elif rtg_from == 'state':
                rtg_dist = self._value_network(x)  
                #rtg_dist = self._imitator.value_predictor(x)
                p_rtg_c_s = rtg_dist.probs # [B, _n_quantiles]

            if rtg_select in ['boundary', 'greater']:
                # TODO: this method assumes B=1, so it is wrong when calculating TD error score in evaluation. should use torch.cumsum() instead
                sum = 0
                for idx in range(p_rtg_c_s.shape[1]-1, -1, -1):
                    sum += p_rtg_c_s[0, idx]
                    if sum >= self._threshold_c:
                        break
                rtg_aug = torch.ones((batch_size, self._inference_samples, 1), device=self._device) * idx
            elif rtg_select == 'max':
                threshold = torch.mean(p_rtg_c_s, dim=1)[:, None] * self._threshold_c  # [B, 1]
                threshold = threshold.repeat(1, self._n_quantiles)
                mask = p_rtg_c_s > threshold

                # TODO: this method assumes B=1, so it is wrong when calculating TD error score in evaluation. should use torch.cumsum() instead
                for idx in range(mask.shape[1]-1, -1, -1):
                    if mask[0, idx] > 0:
                        break
                rtg_aug = torch.ones((batch_size, self._inference_samples, 1), device=self._device) * idx
            #print(p_rtg_c_s.cpu().numpy())
            print(idx)
        elif rtg_from == 'fix':
            rtg_aug = torch.ones((batch_size, self._inference_samples, 1), device=self._device) * (self._n_quantiles - 1)
        else:
            raise ValueError('unknown rtg predictor')

        # augment x
        x_aug = x.unsqueeze(1).repeat(1, self._inference_samples, 1)
        new_batch_size = batch_size * self._inference_samples

        # variables used in this step
        noise_scale = self._noise_scale

        # get random actions
        #size = (batch_size, self._inference_samples, self._action_size)
        #actions = torch.zeros(size, device=self._device).uniform_(self._bounds[0], self._bounds[1])
        
        # get action from prior distirbution p(a|s)
        action_dist = self._imitator.action_predictor(x)                            # [B, dim_a]
        actions = action_dist.sample((self._inference_samples,)).transpose(1, 0)    # [B, _inference_samples, dim_a]
        actions = actions.clamp(min=self._bounds[0], max=self._bounds[1])

        # Derivative Free Optimizer (DFO)
        for _ in range(self._sample_iters):
            # compute energies
            if rtg_select == 'greater':
                energy_a, energy_r = self._imitator.infer_energy(x_aug.reshape(new_batch_size, -1), actions.reshape(new_batch_size, -1), idx)
            else:
                energy_a, energy_r = self._imitator.compute_energy(x_aug.reshape(new_batch_size, -1), actions.reshape(new_batch_size, -1), rtg_aug.reshape(new_batch_size))
            energy = energy_a + energy_r
            probs = torch.softmax(-1.0 * energy.reshape(batch_size, self._inference_samples), dim=-1)

            # resample with replacement
            idxs = torch.multinomial(probs, self._inference_samples, replacement=True)
            actions = actions[torch.arange(actions.size(0), device=self._device).unsqueeze(-1), idxs]

            # add noise and clip to target bounds
            actions = actions + torch.randn_like(actions) * noise_scale
            actions = actions.clamp(min=self._bounds[0], max=self._bounds[1])

            noise_scale *= self._noise_shrink

        # return target with highest probability
        if rtg_select == 'greater':
            energy_a, energy_r = self._imitator.infer_energy(x_aug.reshape(new_batch_size, -1), actions.reshape(new_batch_size, -1), idx)
        else:
            energy_a, energy_r = self._imitator.compute_energy(x_aug.reshape(new_batch_size, -1), actions.reshape(new_batch_size, -1), rtg_aug.reshape(new_batch_size))
        energy = energy_a + energy_r
        probs = torch.softmax(-1.0 * energy.reshape(batch_size, self._inference_samples), dim=-1)
        best_idxs = probs.argmax(dim=-1)
        action = actions[torch.arange(actions.size(0), device=self._device), best_idxs, :]
        return action

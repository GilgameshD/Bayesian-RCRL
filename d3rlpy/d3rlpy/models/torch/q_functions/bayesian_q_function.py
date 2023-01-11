'''
Author: Wenhao Ding
Email: wenhaod@andrew.cmu.edu
Date: 2022-09-07 14:24:44
LastEditTime: 2022-12-11 20:48:55
Description: 
'''

from typing import Optional, cast

import torch
from torch import nn
import torch.nn.functional as F

from ..encoders import Encoder, EncoderWithAction
from .base import ContinuousQFunction, DiscreteQFunction
from .utility import compute_reduce


class DiscreteBayesianQFunction(DiscreteQFunction, nn.Module):  # type: ignore
    _action_size: int
    _encoder: Encoder
    _n_quantiles: int
    _fc: nn.Linear

    def __init__(self, encoder: Encoder, action_size: int, n_quantiles: int, Vmin: float, Vmax: float, weight_penalty: float, weight_R: float, weight_A: float):
        super().__init__()
        self._encoder = encoder
        self._action_size = action_size
        self._n_quantiles = n_quantiles
        self._fc = nn.Linear(
            encoder.get_feature_size(), action_size * n_quantiles
        )

        self.device = 'cuda'
        self.eps = 1e-5

        # weights for loss
        self.weight_penalty = weight_penalty
        self.weight_R = weight_R
        self.weight_A = weight_A

        # for C51
        self.Vmin = Vmin
        self.Vmax = Vmax
        self.atoms = torch.linspace(self.Vmin, self.Vmax, self._n_quantiles, device=self.device)
        self.delta_atom = float(self.Vmax - self.Vmin) / float(self._n_quantiles - 1)

    def _compute_joint_logits(self, x: torch.Tensor) -> torch.Tensor:
        h = self._encoder(x)
        h = cast(torch.Tensor, self._fc(h))
        return h.view(-1, self._action_size, self._n_quantiles)

    def _compute_R_dist(self, logits_a_and_R: torch.Tensor) -> list:
        # log p(a|s) = log sum_{R} exp [log p(a, R|s)]
        logits_a = torch.logsumexp(logits_a_and_R, dim=2)

        # log p(a|s)= log sum_{R} p(a, R|s) DONT USE THIS UNSTABLE IMPLEMENTATION
        #p_a_and_R = torch.softmax(logits_a_and_R.view(-1, self.action_size*self._n_quantiles), dim=1)
        #p_a_and_R = p_a_and_R.view(-1, self.action_size, self._n_quantiles)
        #logits_a = torch.sum(p_a_and_R, dim=2).log()

        # p(R|s, a) = p(a, R|s) / p(a | s)
        p_R_given_a = F.softmax(logits_a_and_R, dim=2)
        log_p_R_given_a = F.log_softmax(logits_a_and_R, dim=2)
        return p_R_given_a, log_p_R_given_a, logits_a

    def _compute_values(self, x: torch.Tensor) -> torch.Tensor:
        logits_a_and_R = self._compute_joint_logits(x)
        p_R_given_a, log_p_R_given_a, logits_a = self._compute_R_dist(logits_a_and_R)
        return p_R_given_a, log_p_R_given_a, logits_a

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ''' Evaluate the Q-value of state x '''
        p_R_given_a, _, _ = self._compute_values(x)

        # multiply value by probability
        q_value = (p_R_given_a * self.atoms.view(1, 1, -1)).sum(dim=2) # [B, _action_size]
        return q_value

    def compute_error(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        target: torch.Tensor,
        terminals: torch.Tensor,
        gamma: float = 0.99,
        reduction: str = "mean",
    ) -> torch.Tensor:

        target_p_R = target.unsqueeze(1)
        assert target_p_R.shape == (observations.shape[0], 1, self._n_quantiles)

        # calculate bellman operator TZ
        TZ = rewards + gamma * (1 - terminals) * self.atoms.view(1, -1) # [B, _n_quantiles]
        TZ = TZ.clamp(self.Vmin, self.Vmax).unsqueeze(1) # [B, 1, _n_quantiles]

        # calculate quotient
        quotient = 1 - (TZ - self.atoms.view(1, -1, 1)).abs() / self.delta_atom  # [B, _n_quantiles, _n_quantiles]
        quotient = quotient.clamp(0, 1)
        
        # projected bellman operator \Phi TZ
        # no gradient for target distribution
        projected_TZ = quotient * target_p_R.detach() # [B, _n_quantiles, _n_quantiles]
        projected_TZ = projected_TZ.sum(dim=2) # [B, _n_quantiles]

        # get p(R|s, a) and p(a|s)
        _, log_p_R_given_a, logits_a = self._compute_values(observations) # _, [B, _action_size, _n_quantiles], [B, _action_size]
        batch_idx = torch.arange(observations.shape[0], device=self.device).long()
        log_p_R_given_a = log_p_R_given_a[batch_idx, actions, :]

        # TD loss (KL)
        loss_R = (projected_TZ * projected_TZ.add(self.eps).log() - projected_TZ * log_p_R_given_a).sum(dim=1)

        # BC loss
        action_penalty = self.weight_penalty * (logits_a**2).mean(dim=1)
        #loss_A = F.nll_loss(logits_a, actions.reshape(-1), reduction='none') + action_penalty
        loss_A = F.cross_entropy(logits_a, actions.reshape(-1), reduction='none') + action_penalty

        loss = self.weight_A * loss_A + self.weight_R * loss_R 
        return compute_reduce(loss, reduction)

    def compute_target(self, observations_next: torch.Tensor, log_probs_next_action: torch.Tensor) -> torch.Tensor:
        p_R_given_a_next, _, _ = self._compute_values(observations_next) # [B, _action_size, _n_quantiles], _

        if log_probs_next_action is None:
            return p_R_given_a_next

        # p(R|s') = \sum_{a'} p(a'| s') * p(R | a', s')
        p_R = (torch.exp(log_probs_next_action).unsqueeze(-1) * p_R_given_a_next).sum(dim=1)
        return p_R

    @property
    def action_size(self) -> int:
        return self._action_size

    @property
    def encoder(self) -> Encoder:
        return self._encoder


class ContinuousBayesianQFunction(ContinuousQFunction, nn.Module):  # type: ignore
    _action_size: int
    _encoder: EncoderWithAction
    _n_quantiles: int
    _fc: nn.Linear

    def __init__(
        self, 
        encoder: EncoderWithAction,
        n_quantiles: int, 
        Vmin: float, 
        Vmax: float, 
        weight_penalty: float, 
        weight_R: float, 
        weight_A: float,
        n_neg_samples: int,
    ):
        super().__init__()
        self._encoder = encoder
        self._action_size = encoder.action_size
        self._n_quantiles = n_quantiles
        self._fc = nn.Linear(encoder.get_feature_size(), self._n_quantiles)

        self.device = 'cuda'
        self.eps = 1e-10

        # weights for loss
        self.weight_penalty = weight_penalty
        self.weight_R = weight_R
        self.weight_A = weight_A

        # for C51
        self.Vmin = Vmin
        self.Vmax = Vmax
        self.atoms = torch.linspace(self.Vmin, self.Vmax, self._n_quantiles, device=self.device)
        self.delta_atom = float(self.Vmax - self.Vmin) / float(self._n_quantiles - 1)

        # for infoNCE loss
        self.n_neg_samples = n_neg_samples
        self.bounds = [-1.0, 1.0]

    def _generate_negative_targets(self, target):
        # uniformly sample negative actions
        size = (target.shape[0], self.n_neg_samples, self._action_size)
        negatives = torch.zeros(size, device=self.device).uniform_(self.bounds[0], self.bounds[1])

        # Merge target and negatives: [B, N+1, D].
        targets = torch.cat([target.unsqueeze(dim=1), negatives], dim=1)

        # Generate a random permutation of the positives and negatives.
        permutation = torch.rand(targets.size(0), targets.size(1), device=self.device).argsort(dim=1)
        targets = targets[torch.arange(targets.size(0), device=self.device).unsqueeze(-1), permutation, :] # [B, N+1, D2]

        # Get the original index of the positive. This will serve as the class label for the loss.
        # Positive - 1, negative - 0
        ground_truth = (permutation == 0).nonzero()[:, 1].to(self.device)
        return targets, ground_truth

    def _compute_distributional_q_logits(self, x: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        # get feature through the encoder
        h = self._encoder(x, action)
        h = cast(torch.Tensor, self._fc(h))
        return h.view(-1, self._n_quantiles)

    def _get_energy(self, x: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        ''' get energy E(s, a) and label '''
        q_logits = self._compute_distributional_q_logits(x, action)
        energy = -1.0 * torch.logsumexp(q_logits, dim=1)
        return energy

    def _get_energy_with_threshold_todo(self, x: torch.Tensor, action: torch.Tensor, threshold_c: float) -> torch.Tensor:
        ''' this should only be used in testing stage '''
        assert threshold_c <= 1.0 and threshold_c > 0, 'threshold c should be in (0, 1]'
        q_logits = self._compute_distributional_q_logits(x, action)
        q_value_dist = torch.softmax(q_logits, dim=1)          # [B, _n_quantiles]

        # normalize q logits and select buckets with threshold_c
        q_value_cdf = 1.0 - torch.cumsum(q_value_dist, dim=1)  # [B, _n_quantiles]
        q_value_mask = q_value_cdf <= threshold_c              # [B, _n_quantiles]

        # select high-reward buckets for each sample
        q_logits_masked = q_logits * q_value_mask

        energy = -1.0 * torch.logsumexp(q_logits_masked, dim=1)
        return energy

    def _get_energy_with_threshold(self, x: torch.Tensor, action: torch.Tensor, threshold_c: float) -> torch.Tensor:
        assert threshold_c <= 1.0 and threshold_c > 0, 'threshold c should be in (0, 1]'
        bucket_num = int(threshold_c * self._n_quantiles)

        q_logits = self._compute_distributional_q_logits(x, action)
        q_logits = q_logits[:, -bucket_num:] # [B, bucket_num]
        energy = -1.0 * torch.logsumexp(q_logits, dim=1)
        return energy

    def _get_q_value_dist(self, x: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        ''' get log q(s, a) '''
        q_logits = self._compute_distributional_q_logits(x, action)
        q_value_dist = torch.softmax(q_logits, dim=1)
        log_q_value_dist = torch.log_softmax(q_value_dist, dim=1)
        return q_value_dist, log_q_value_dist

    def forward(self, x: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        q_value_dist = self._get_q_value_dist(x, action)
        q_value = (self.atoms * q_value_dist).sum(dim=-1)
        return q_value

    def compute_error(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        target: torch.Tensor,
        terminals: torch.Tensor,
        gamma: float = 0.99,
        reduction: str = "mean",
    ) -> torch.Tensor:
        # augment (x, action) pairs with negative pairs
        action_aug, ground_truth  = self._generate_negative_targets(actions)
        batch_size = action_aug.size(0)
        sample_size = action_aug.size(1)

        # re-organize observation and action to form a new batch
        observations_aug = observations.unsqueeze(1).repeat((1, sample_size, 1))  # [B, N+1, D1]
        observations_aug = observations_aug.reshape(batch_size * sample_size, -1) # [B*(N+1), D1]
        action_aug = action_aug.reshape(batch_size * sample_size, -1)             # [B*(N+1), D2]

        # For every element in the mini-batch, there is 1 positive for which the EBM should output a low energy value, 
        # and N negatives for which the EBM should output high energy values.
        energy = self._get_energy(observations_aug, action_aug)
        energy = energy.reshape(batch_size, sample_size)
        logits = -1.0 * energy
        action_penalty = self.weight_penalty * (logits**2).mean(dim=1)
        loss_A = F.cross_entropy(logits, ground_truth, reduction='none') + action_penalty

        # calculate bellman operator TZ
        TZ = rewards + gamma * (1 - terminals) * self.atoms.view(1, -1) # [B, _n_quantiles]
        TZ = TZ.clamp(self.Vmin, self.Vmax).unsqueeze(1) # [B, 1, _n_quantiles]

        # calculate quotient
        quotient = 1 - (TZ - self.atoms.view(1, -1, 1)).abs() / self.delta_atom  # [B, _n_quantiles, _n_quantiles]
        quotient = quotient.clamp(0, 1)

        # projected bellman operator \Phi TZ. no gradient for target distribution
        target = target.unsqueeze(1)              # [B, 1, _n_quantiles]
        projected_TZ = quotient * target.detach() # [B, _n_quantiles, _n_quantiles]
        projected_TZ = projected_TZ.sum(dim=2)    # [B, _n_quantiles]

        # TD loss (critic loss)
        _, log_q_value_dist = self._get_q_value_dist(observations, actions)
        loss_R = - (projected_TZ * log_q_value_dist).sum(dim=1)

        # combine two losses
        loss = self.weight_A * loss_A + self.weight_R * loss_R 
        return compute_reduce(loss, reduction)

    def compute_target(self, observations_next: torch.Tensor, action_next: torch.Tensor) -> torch.Tensor:
        target_q_value_dist, _ = self._get_q_value_dist(observations_next, action_next) # [B, _n_quantiles]
        return target_q_value_dist

    @property
    def action_size(self) -> int:
        return self._action_size

    @property
    def encoder(self) -> Encoder:
        return self._encoder

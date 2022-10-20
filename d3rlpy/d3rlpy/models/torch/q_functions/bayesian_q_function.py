'''
Author: Wenhao Ding
Email: wenhaod@andrew.cmu.edu
Date: 2022-09-07 14:24:44
LastEditTime: 2022-10-18 21:45:04
Description: 
'''

from typing import Optional, cast

import torch
from torch import nn
import torch.nn.functional as F

from ..encoders import Encoder, EncoderWithAction
from .base import ContinuousQFunction, DiscreteQFunction
from .utility import (
    compute_quantile_loss,
    compute_reduce,
    pick_quantile_value_by_action,
)


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
        _, log_p_R_given_a, logits_a = self._compute_values(observations) # [B, _action_size, _n_quantiles], [B, _action_size]
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

    def compute_target(self, observations_next: torch.Tensor, log_probs_next_action: Optional[torch.Tensor] = None) -> torch.Tensor:
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
    def __init__(self, encoder: EncoderWithAction, n_quantiles: int):
        raise NotImplementedError('Continuous vesion Bayesian Q Function is not implemented')

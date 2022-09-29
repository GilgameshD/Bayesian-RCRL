'''
Author: Wenhao Ding
Email: wenhaod@andrew.cmu.edu
Date: 2022-09-07 14:24:44
LastEditTime: 2022-09-24 15:55:51
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


class DiscreteC51QFunction(DiscreteQFunction, nn.Module):  # type: ignore
    _action_size: int
    _encoder: Encoder
    _n_quantiles: int
    _fc: nn.Linear

    def __init__(self, encoder: Encoder, action_size: int, n_quantiles: int):
        super().__init__()
        self._encoder = encoder
        self._action_size = action_size
        self._n_quantiles = n_quantiles
        self._fc = nn.Linear(
            encoder.get_feature_size(), action_size * n_quantiles
        )

        self.device = 'cuda'

        # for C51
        self.Vmin = -10
        self.Vmax = 10
        self.atoms = torch.linspace(self.Vmin, self.Vmax, self._n_quantiles, device=self.device)
        self.delta_atom = float(self.Vmax - self.Vmin) / float(self._n_quantiles - 1)
        self.n_step = 1

    def _compute_distribution(self, h: torch.Tensor) -> torch.Tensor:
        h = cast(torch.Tensor, self._fc(h))
        h = h.view(-1, self._action_size, self._n_quantiles)
        prob = F.softmax(h, dim=2)
        log_prob = F.log_softmax(h, dim=2)
        return prob, log_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ''' Evaluate the Q-value of the state x'''
        h = self._encoder(x)
        prob, _ = self._compute_distribution(h)

        # multiply value by probability
        q_value = (prob * self.atoms).sum(dim=2) # [B, _action_size]
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
        assert target.shape == (observations.shape[0], self._n_quantiles)
        
        # calculate C51 TD error
        atoms_target = rewards + gamma ** self.n_step * (1 - terminals) * self.atoms.view(1, -1) # [B, _n_quantiles]
        atoms_target.clamp_(self.Vmin, self.Vmax)
        atoms_target = atoms_target.unsqueeze(1) # [B, 1, _n_quantiles]
        target_prob = (1 - (atoms_target - self.atoms.view(1, -1, 1)).abs() / self.delta_atom).clamp(0, 1) * target.unsqueeze(1) # [B, _n_quantiles, _n_quantiles]
        target_prob = target_prob.sum(-1) # [B, _n_quantiles]

        # calculate distribution over all actions
        h = self._encoder(observations)
        _, log_prob = self._compute_distribution(h)
        batch_idx = torch.arange(observations.shape[0], device=self.device).long()
        log_prob = log_prob[batch_idx, actions, :]

        # KL divergence
        loss = (target_prob * target_prob.add(1e-5).log() - target_prob * log_prob).sum(dim=1)
        return compute_reduce(loss, reduction)

    def compute_target(self, x: torch.Tensor, action: Optional[torch.Tensor] = None) -> torch.Tensor:
        h = self._encoder(x)
        next_prob, _ = self._compute_distribution(h)

        if action is None:
            return next_prob

        batch_idx = torch.arange(x.shape[0], device=self.device).long()
        return next_prob[batch_idx, action, :]

    @property
    def action_size(self) -> int:
        return self._action_size

    @property
    def encoder(self) -> Encoder:
        return self._encoder

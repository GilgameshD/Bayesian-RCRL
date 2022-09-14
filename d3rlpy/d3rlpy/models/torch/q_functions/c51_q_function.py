'''
Author: Wenhao Ding
Email: wenhaod@andrew.cmu.edu
Date: 2022-09-07 14:24:44
LastEditTime: 2022-09-13 16:43:23
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

        # for C51
        self.Vmin = -10
        self.Vmax = 10
        self.atoms = torch.linspace(self.Vmin, self.Vmax, self._n_quantiles, device='cuda')
        self.delta_atom = float(self.Vmax - self.Vmin) / float(self._n_quantiles - 1)
        self.n_step = 1

    def _compute_distribution(self, h: torch.Tensor) -> torch.Tensor:
        h = cast(torch.Tensor, self._fc(h))
        h = h.view(-1, self._action_size, self._n_quantiles)
        return F.softmax(h, dim=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ''' Evaluate the Q-value of the state x'''
        h = self._encoder(x)
        dists = self._compute_distribution(h)

        # multiply value by probability
        q_value = (dists * self.atoms).sum(dim=2) # [B, _action_size]
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
        target_prob = (1 - (atoms_target - self.atoms.view(1, -1, 1)).abs() / self.delta_atom).clamp(0, 1) * target.unsqueeze(1) # [B, 1, _n_quantiles]
        target_prob = target_prob.sum(-1) # [B, 1]

        # calculate distribution over all actions
        h = self._encoder(observations)
        dists = self._compute_distribution(h)
        action_idx = actions[:, None, None].repeat(1, 1, dists.shape[2])
        dists = torch.gather(dists, dim=1, index=action_idx)    # [B, 1]

        # KL divergence
        loss = (target_prob*target_prob.add(1e-5).log() - target_prob*dists.log()).sum(-1)
        return compute_reduce(loss, reduction)

    def compute_target(
        self, x: torch.Tensor, action: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        h = self._encoder(x)
        next_dist = self._compute_distribution(h)

        if action is None:
            return next_dist

        action_idx = action[:, None, None].repeat(1, 1, self._n_quantiles)
        return next_dist.gather(dim=1, index=action_idx)

    @property
    def action_size(self) -> int:
        return self._action_size

    @property
    def encoder(self) -> Encoder:
        return self._encoder

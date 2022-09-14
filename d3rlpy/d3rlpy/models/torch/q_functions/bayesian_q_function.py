'''
Author: Wenhao Ding
Email: wenhaod@andrew.cmu.edu
Date: 2022-09-07 14:24:44
LastEditTime: 2022-09-13 13:29:52
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

    def _compute_joint_logits(self, h: torch.Tensor) -> torch.Tensor:
        h = cast(torch.Tensor, self._fc(h))
        return h.view(-1, self._action_size, self._n_quantiles)

    def _compute_R_dist(self, logits_a_and_R: torch.Tensor) -> list:
        # from logits to probability
        p_a_and_R = F.softmax(logits_a_and_R.view(-1, self._action_size * self._n_quantiles), dim=1)
        p_a_and_R = p_a_and_R.view(-1, self._action_size, self._n_quantiles)

        # p(a|s) = sum_{R} p(a, R|s)
        logits_a = p_a_and_R.sum(dim=2) # [B, _action_size]

        # p(R|s, a) = p(a, R|s) / p(a | s)
        #p_R = p_a_and_R / logits_a[:, :, None] # [B, _action_size, _n_quantiles]
        # use softmax and log-softmax
        p_R = F.softmax(p_a_and_R, dim=2)
        return p_R, logits_a

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ''' Evaluate the Q-value of the state x'''
        h = self._encoder(x)
        logits_a_and_R = self._compute_joint_logits(h)
        p_R, _ = self._compute_R_dist(logits_a_and_R)

        # multiply value by probability
        q_value = (p_R * self.atoms).sum(dim=2) # [B, _action_size]
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

        # p(a, R|s)
        h = self._encoder(observations)
        logits_a_and_R = self._compute_joint_logits(h)
        p_R, logits_a = self._compute_R_dist(logits_a_and_R)

        action_idx = actions[:, None, None].repeat(1, 1, p_R.shape[2])
        p_R = torch.gather(p_R, dim=1, index=action_idx)    # [B, 1]
        #p_R = pick_quantile_value_by_action(p_R, actions)  # [B, 1]

        # log p(R|s, a)
        loss_R = (target_prob * target_prob.add(1e-5).log() - target_prob * p_R.log()).sum(-1)

        # log p(a|s)
        loss_A = F.cross_entropy(logits_a, actions.reshape(-1))
        loss = loss_R
        return compute_reduce(loss, reduction)

    def compute_target(
        self, x: torch.Tensor, action: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        h = self._encoder(x)
        logits_a_and_R = self._compute_joint_logits(h)
        p_R_next, _ = self._compute_R_dist(logits_a_and_R)

        if action is None:
            return p_R_next

        action_idx = action[:, None, None].repeat(1, 1, p_R_next.shape[2])
        return torch.gather(p_R_next, dim=1, index=action_idx)
        #return pick_quantile_value_by_action(p_R_next, action)

    @property
    def action_size(self) -> int:
        return self._action_size

    @property
    def encoder(self) -> Encoder:
        return self._encoder


class ContinuousBayesianQFunction(ContinuousQFunction, nn.Module):  # type: ignore
    def __init__(self, encoder: EncoderWithAction, n_quantiles: int):
        raise NotImplementedError('Continuous vesion Bayesian Q Function is not implemented')

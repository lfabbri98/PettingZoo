"""Memoria temporanea delle esperienze raccolte dalla policy corrente."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor

from model import ACTION_SIZE, OBSERVATION_SIZE


@dataclass(frozen=True)
class RolloutBatch:
    """Versione in tensor di tutte le transizioni nel buffer."""

    states: Tensor
    raw_actions: Tensor
    rewards: Tensor
    dones: Tensor
    values: Tensor
    log_probs: Tensor


class RolloutBuffer:
    """Conserva le esperienze fino al successivo aggiornamento PPO.

    Il buffer è *on-policy*: dopo aver usato le sue transizioni per aggiornare
    il modello, viene svuotato e riutilizzato.
    """

    def __init__(self) -> None:
        self.states: list[Tensor] = []
        self.raw_actions: list[Tensor] = []
        self.rewards: list[float] = []
        self.dones: list[bool] = []
        self.values: list[float] = []
        self.log_probs: list[float] = []

    def __len__(self) -> int:
        return len(self.rewards)

    def add(
        self,
        *,
        state: Tensor,
        raw_action: Tensor,
        reward: float,
        done: bool,
        value: Tensor,
        log_prob: Tensor,
    ) -> None:
        """Aggiunge una transizione, rimuovendola dal grafo dei gradienti."""
        if state.shape != (OBSERVATION_SIZE,):
            raise ValueError(f"Lo stato deve avere forma ({OBSERVATION_SIZE},).")
        if raw_action.shape != (ACTION_SIZE,):
            raise ValueError(
                f"L'azione deve avere forma ({ACTION_SIZE},), non un batch."
            )
        if not math.isfinite(reward):
            raise ValueError("Il reward deve essere finito.")
        if value.numel() != 1 or log_prob.numel() != 1:
            raise ValueError("Value e log_prob devono essere scalari.")

        self.states.append(state.detach().cpu().clone())
        self.raw_actions.append(raw_action.detach().cpu().clone())
        self.rewards.append(float(reward))
        self.dones.append(bool(done))
        self.values.append(float(value.detach().cpu().item()))
        self.log_probs.append(float(log_prob.detach().cpu().item()))

    def as_batch(self) -> RolloutBatch:
        """Restituisce tutte le esperienze come tensor PyTorch."""
        if not self:
            raise RuntimeError("Il buffer è vuoto.")
        return RolloutBatch(
            states=torch.stack(self.states),
            raw_actions=torch.stack(self.raw_actions),
            rewards=torch.tensor(self.rewards, dtype=torch.float32),
            dones=torch.tensor(self.dones, dtype=torch.bool),
            values=torch.tensor(self.values, dtype=torch.float32),
            log_probs=torch.tensor(self.log_probs, dtype=torch.float32),
        )

    def clear(self) -> None:
        """Rimuove le esperienze dopo l'aggiornamento della policy."""
        self.states.clear()
        self.raw_actions.clear()
        self.rewards.clear()
        self.dones.clear()
        self.values.clear()
        self.log_probs.clear()

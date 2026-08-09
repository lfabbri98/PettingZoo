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
    terminated: Tensor
    truncated: Tensor
    values: Tensor
    next_values: Tensor
    log_probs: Tensor


@dataclass(frozen=True)
class GaeTargets:
    """Target del critic e vantaggio della policy per ogni transizione."""

    advantages: Tensor
    returns: Tensor


class RolloutBuffer:
    """Conserva le esperienze fino al successivo aggiornamento PPO.

    Il buffer è *on-policy*: dopo aver usato le sue transizioni per aggiornare
    il modello, viene svuotato e riutilizzato.
    """

    def __init__(self) -> None:
        self.states: list[Tensor] = []
        self.raw_actions: list[Tensor] = []
        self.rewards: list[float] = []
        self.terminated: list[bool] = []
        self.truncated: list[bool] = []
        self.values: list[float] = []
        self.next_values: list[float] = []
        self.log_probs: list[float] = []

    def __len__(self) -> int:
        return len(self.rewards)

    def add(
        self,
        *,
        state: Tensor,
        raw_action: Tensor,
        reward: float,
        terminated: bool,
        truncated: bool,
        value: Tensor,
        next_value: Tensor,
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
        if value.numel() != 1 or next_value.numel() != 1 or log_prob.numel() != 1:
            raise ValueError("Value, next_value e log_prob devono essere scalari.")
        for name, tensor in {
            "stato": state,
            "azione": raw_action,
            "value": value,
            "next_value": next_value,
            "log_prob": log_prob,
        }.items():
            if not bool(torch.isfinite(tensor).all().item()):
                raise ValueError(f"{name} deve contenere solo valori finiti.")

        self.states.append(state.detach().cpu().clone())
        self.raw_actions.append(raw_action.detach().cpu().clone())
        self.rewards.append(float(reward))
        self.terminated.append(bool(terminated))
        self.truncated.append(bool(truncated))
        self.values.append(float(value.detach().cpu().item()))
        self.next_values.append(float(next_value.detach().cpu().item()))
        self.log_probs.append(float(log_prob.detach().cpu().item()))

    def as_batch(self) -> RolloutBatch:
        """Restituisce tutte le esperienze come tensor PyTorch."""
        if not self:
            raise RuntimeError("Il buffer è vuoto.")
        return RolloutBatch(
            states=torch.stack(self.states),
            raw_actions=torch.stack(self.raw_actions),
            rewards=torch.tensor(self.rewards, dtype=torch.float32),
            terminated=torch.tensor(self.terminated, dtype=torch.bool),
            truncated=torch.tensor(self.truncated, dtype=torch.bool),
            values=torch.tensor(self.values, dtype=torch.float32),
            next_values=torch.tensor(self.next_values, dtype=torch.float32),
            log_probs=torch.tensor(self.log_probs, dtype=torch.float32),
        )

    def compute_gae(
        self, *, gamma: float = 0.99, gae_lambda: float = 0.95
    ) -> GaeTargets:
        """Calcola vantaggi GAE e target di valore per tutte le transizioni.

        Una partita terminata non ha futuro, quindi usa ``next_value = 0``.
        Un timeout è invece un'interruzione artificiale: usa la stima del
        critic per il passo successivo, ma non propaga GAE nell'episodio
        eventualmente successivo presente nel buffer.
        """
        if not 0.0 <= gamma <= 1.0 or not 0.0 <= gae_lambda <= 1.0:
            raise ValueError("gamma e gae_lambda devono essere compresi fra 0 e 1.")

        batch = self.as_batch()
        advantages = torch.zeros_like(batch.rewards)
        gae = 0.0
        for index in range(len(self) - 1, -1, -1):
            terminal = float(batch.terminated[index])
            episode_end = float(batch.terminated[index] or batch.truncated[index])
            delta = (
                batch.rewards[index]
                + gamma * (1.0 - terminal) * batch.next_values[index]
                - batch.values[index]
            )
            gae = delta + gamma * gae_lambda * (1.0 - episode_end) * gae
            advantages[index] = gae

        return GaeTargets(advantages=advantages, returns=advantages + batch.values)

    def clear(self) -> None:
        """Rimuove le esperienze dopo l'aggiornamento della policy."""
        self.states.clear()
        self.raw_actions.clear()
        self.rewards.clear()
        self.terminated.clear()
        self.truncated.clear()
        self.values.clear()
        self.next_values.clear()
        self.log_probs.clear()

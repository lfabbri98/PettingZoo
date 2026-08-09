#Rete neurale di base per l'agente di tennis.

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.distributions import Normal


OBSERVATION_SIZE = 15
ACTION_SIZE = 4


@dataclass(frozen=True)
class ActionSample:
    """
    Classe per rappresentare un esito della scelta della policy.
    """

    action: Tensor
    raw_action: Tensor
    log_prob: Tensor
    mean: Tensor
    value: Tensor


class ActorCritic(nn.Module):
    """
    Modello continuo per allenare l'agente
    """

    def __init__(
        self,
        observation_size: int = OBSERVATION_SIZE,
        action_size: int = ACTION_SIZE,
        hidden_size: int = 128,
    ) -> None:
        super().__init__()
        if observation_size <= 0 or action_size != ACTION_SIZE or hidden_size <= 0:
            raise ValueError("Dimensioni della rete non valide.")

        self.encoder = nn.Sequential(
            nn.Linear(observation_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
        )
        self.actor_mean = nn.Linear(hidden_size, action_size)
        self.critic = nn.Linear(hidden_size, 1)
        # Un solo parametro per componente dell'azione, condiviso tra gli stati.
        self.log_std = nn.Parameter(torch.full((action_size,), -0.5))
        #Un oggetto di tipo Parameter viene riconosciuto da Pytorch come allennabile dalla rete

    def forward(self, observations: Tensor) -> tuple[Tensor, Tensor]:
        """Calcola media della policy e valore del critic."""
        features = self.encoder(observations)
        mean = self.actor_mean(features)
        value = self.critic(features).squeeze(-1)
        return mean, value

    def distribution(self, observations: Tensor) -> tuple[Normal, Tensor]:
        """Costruisce la distribuzione della policy e restituisce anche ``V(s)``."""
        mean, value = self.forward(observations)
        std = self.log_std.exp().expand_as(mean)
        return Normal(mean, std), value

    @staticmethod
    def _to_environment_action(raw_action: Tensor) -> Tensor:
        """Converte un'azione gaussiana nei limiti del simulatore."""
        squashed = torch.tanh(raw_action)
        force = (squashed[..., 2:3] + 1.0) / 2.0
        return torch.cat((squashed[..., :2], force, squashed[..., 3:]), dim=-1)

    @torch.no_grad()
    def act(self, observation: Tensor, *, deterministic: bool = False) -> ActionSample:
        """Sceglie un'azione valida per una singola osservazione o un batch.

        In training si campiona dalla gaussiana per esplorare. In valutazione
        ``deterministic=True`` usa la media della policy, senza rumore.
        """
        distribution, value = self.distribution(observation)
        raw_action = distribution.mean if deterministic else distribution.sample()
        log_prob = distribution.log_prob(raw_action).sum(dim=-1)
        return ActionSample(
            action=self._to_environment_action(raw_action),
            raw_action=raw_action,
            log_prob=log_prob,
            mean=distribution.mean,
            value=value,
        )

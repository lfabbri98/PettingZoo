"""Collegamento minimale tra il modello PyTorch e il simulatore di tennis.

Non esegue alcun training: usa il modello corrente per giocare un episodio e
restituisce il suo esito. Questo file sarà la base per la raccolta delle
traiettorie nel passo successivo.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import Tensor

from model import ACTION_SIZE, OBSERVATION_SIZE, ActorCritic
from tennis_env import TennisEnv


@dataclass(frozen=True)
class EpisodeResult:
    """Riassunto di un episodio giocato dal modello."""

    total_reward: float
    steps: int
    terminated: bool
    truncated: bool
    winner: str | None
    scores: tuple[int, int]


def observation_to_tensor(observation: Sequence[float]) -> Tensor:
    """Converte l'osservazione del simulatore in input PyTorch ``float32``."""
    if len(observation) != OBSERVATION_SIZE:
        raise ValueError(f"L'osservazione deve contenere {OBSERVATION_SIZE} valori.")
    return torch.tensor(observation, dtype=torch.float32)


def action_to_tuple(action: Tensor) -> tuple[float, float, float, float]:
    """Converte l'azione del modello nella sequenza richiesta da ``env.step``."""
    if action.shape != (ACTION_SIZE,):
        raise ValueError(
            "Questa funzione accetta l'azione di un singolo stato, non un batch."
        )
    values = action.detach().cpu().tolist()
    return values[0], values[1], values[2], values[3]


def run_episode(
    environment: TennisEnv,
    model: ActorCritic,
    *,
    seed: int | None = None,
    deterministic: bool = False,
) -> EpisodeResult:
    """Fa giocare al modello un episodio completo nel simulatore.

    ``deterministic=False`` mantiene l'esplorazione della policy e sarà usato
    durante il training. Per valutare il modello si usa ``True``.
    """
    observation, _ = environment.reset(seed=seed)
    total_reward = 0.0

    while True:
        observation_tensor = observation_to_tensor(observation)
        sample = model.act(observation_tensor, deterministic=deterministic)
        action = action_to_tuple(sample.action)
        observation, reward, terminated, truncated, info = environment.step(action)
        total_reward += reward

        if terminated or truncated:
            return EpisodeResult(
                total_reward=total_reward,
                steps=info["steps"],
                terminated=terminated,
                truncated=truncated,
                winner=info["winner"],
                scores=info["scores"],
            )


def main() -> None:
    """Esegue una breve partita di prova con una policy non ancora addestrata."""
    environment = TennisEnv(points_to_win=11, max_steps_per_episode=20_000)
    model = ActorCritic()
    result = run_episode(environment, model, seed=42)
    print(
        "Episodio concluso: "
        f"reward={result.total_reward}, passi={result.steps}, "
        f"punteggio alto-basso={result.scores}, vincitore={result.winner}, "
        f"timeout={result.truncated}"
    )


if __name__ == "__main__":
    main()

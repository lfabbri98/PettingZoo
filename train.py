"""Training PPO per il simulatore di tennis."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import Tensor

from model import ActorCritic
from play import EpisodeResult, run_episode
from rollout import RolloutBuffer
from tennis_env import TennisEnv


@dataclass(frozen=True)
class PPOConfig:
    """Iperparametri del training PPO."""

    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    value_coefficient: float = 0.5
    entropy_coefficient: float = 0.01
    max_grad_norm: float = 0.5
    update_epochs: int = 10
    minibatch_size: int = 256
    target_kl: float | None = 0.02

    #Controlli integrità iperparametri
    def __post_init__(self) -> None:
        if self.learning_rate <= 0 or self.minibatch_size <= 0:
            raise ValueError("learning_rate e minibatch_size devono essere positivi.")
        if not 0 <= self.gamma <= 1 or not 0 <= self.gae_lambda <= 1:
            raise ValueError("gamma e gae_lambda devono essere compresi fra 0 e 1.")
        if self.clip_ratio <= 0 or self.value_coefficient < 0:
            raise ValueError("I coefficienti PPO non sono validi.")
        if self.entropy_coefficient < 0 or self.max_grad_norm <= 0:
            raise ValueError("I coefficienti PPO non sono validi.")
        if self.update_epochs <= 0 or (self.target_kl is not None and self.target_kl <= 0):
            raise ValueError("update_epochs e target_kl non sono validi.")


@dataclass(frozen=True)
class UpdateMetrics:
    policy_loss: float
    value_loss: float
    entropy: float
    approximate_kl: float
    clip_fraction: float
    optimization_steps: int


@dataclass(frozen=True)
class TrainingResult:
    episodes: tuple[EpisodeResult, ...]
    updates: tuple[UpdateMetrics, ...]


class PPOTrainer:
    """
    Trainer del modello
    Prende in input un oggetto di classe ActorCritic (modello) e un oggetto PPOConfig che
    contiene gli iperparametri del modello.
    """

    def __init__(
        self, model: ActorCritic, config: PPOConfig | None = None
    ) -> None:
        self.model = model
        self.config = config or PPOConfig()
        self.optimizer = torch.optim.Adam(model.parameters(), lr=self.config.learning_rate)

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    def update(self, buffer: RolloutBuffer) -> UpdateMetrics:
        """
        Trainer della rete, prende le esperienze raccolte nel buffer e aggiorna
        i pesi della rete.
        """
        #Check buffer non empty
        if not buffer:
            raise RuntimeError("Impossibile aggiornare PPO con un buffer vuoto.")

        #Conversione buffer a tensori per pytorch
        batch = buffer.as_batch()
        #Calcolo dei target tramite GAE
        targets = buffer.compute_gae(
            gamma=self.config.gamma, gae_lambda=self.config.gae_lambda
        )
        device = self.device
        states = batch.states.to(device)
        raw_actions = batch.raw_actions.to(device)
        #Storage delle vecchie log_probs contenute nel buffer
        old_log_probs = batch.log_probs.to(device)
        returns = targets.returns.to(device)
        #calcolo e normalizzazione dei vantaggi per aumentare la stabilità
        advantages = targets.advantages.to(device)
        advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)

        #Inizializzazione metriche
        policy_losses: list[float] = []
        value_losses: list[float] = []
        entropies: list[float] = []
        kls: list[float] = []
        clip_fractions: list[float] = []
        optimization_steps = 0
        self.model.train()

        for _ in range(self.config.update_epochs):
            #Divisione in batches
            indices = torch.randperm(len(buffer), device=device)
            stop_early = False
            for start in range(0, len(buffer), self.config.minibatch_size):
                index = indices[start : start + self.config.minibatch_size]
                #Rivaluta le azioni presenti nel buffer con il modello allo stato attuale
                #per ogni batch
                new_log_probs, entropy, values = self.model.evaluate_actions(
                    states[index], raw_actions[index]
                )
                #Calcolo differenza rispetto alle probabilità precendenti
                log_ratio = new_log_probs - old_log_probs[index]
                ratio = log_ratio.exp()
                #Calcolo della PPO loss
                unclipped = ratio * advantages[index]
                #Clip per aumentare stabilità                
                clipped = ratio.clamp(
                    1 - self.config.clip_ratio, 1 + self.config.clip_ratio
                ) * advantages[index]
                policy_loss = -torch.minimum(unclipped, clipped).mean()
                value_loss = torch.nn.functional.mse_loss(values, returns[index])
                entropy_mean = entropy.mean()
                loss = (
                    policy_loss
                    + self.config.value_coefficient * value_loss
                    - self.config.entropy_coefficient * entropy_mean
                )
                #Ciclo di ottimizzazione
                #zero_grad a ogni ciclo perchè in pytorch i gradienti sono cumulati
                self.optimizer.zero_grad()
                loss.backward()
                #Limito la grandezza complessiva dei gradienti del modello
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.max_grad_norm
                )
                self.optimizer.step()

                approximate_kl = (old_log_probs[index] - new_log_probs).mean()
                clip_fraction = (
                    (ratio - 1.0).abs() > self.config.clip_ratio
                ).float().mean()
                policy_losses.append(float(policy_loss.detach().cpu()))
                value_losses.append(float(value_loss.detach().cpu()))
                entropies.append(float(entropy_mean.detach().cpu()))
                kls.append(float(approximate_kl.detach().cpu()))
                clip_fractions.append(float(clip_fraction.detach().cpu()))
                optimization_steps += 1
                if (
                    self.config.target_kl is not None
                    and approximate_kl > self.config.target_kl
                ):
                    stop_early = True
                    break
            if stop_early:
                break

        buffer.clear()
        return UpdateMetrics(
            policy_loss=sum(policy_losses) / len(policy_losses),
            value_loss=sum(value_losses) / len(value_losses),
            entropy=sum(entropies) / len(entropies),
            approximate_kl=sum(kls) / len(kls),
            clip_fraction=sum(clip_fractions) / len(clip_fractions),
            optimization_steps=optimization_steps,
        )


def train(
    environment: TennisEnv,
    trainer: PPOTrainer,
    *,
    total_episodes: int,
    rollout_episodes: int = 8,
    seed: int | None = None,
    checkpoint_path: str | Path | None = None,
) -> TrainingResult:
    """Allena la policy e, facoltativamente, salva un checkpoint finale."""
    if total_episodes <= 0 or rollout_episodes <= 0:
        raise ValueError("total_episodes e rollout_episodes devono essere positivi.")

    episodes: list[EpisodeResult] = []
    updates: list[UpdateMetrics] = []
    buffer = RolloutBuffer()
    while len(episodes) < total_episodes:
        episodes.append(
            run_episode(
                environment,
                trainer.model,
                seed=None if seed is None else seed + len(episodes),
                buffer=buffer,
            )
        )
        if len(episodes) % rollout_episodes == 0 or len(episodes) == total_episodes:
            rollout_start = max(0, len(episodes) - rollout_episodes)
            rollout = episodes[rollout_start:]
            metrics = trainer.update(buffer)
            updates.append(metrics)
            mean_reward = sum(episode.total_reward for episode in rollout) / len(rollout)
            mean_steps = sum(episode.steps for episode in rollout) / len(rollout)
            print(
                f"[update {len(updates):03d} | episodi {rollout_start + 1}-"
                f"{len(episodes)}/{total_episodes}] "
                f"reward medio={mean_reward:.2f}, passi medi={mean_steps:.1f}, "
                f"policy_loss={metrics.policy_loss:.4f}, "
                f"value_loss={metrics.value_loss:.4f}, "
                f"entropia={metrics.entropy:.4f}, "
                f"KL={metrics.approximate_kl:.5f}, "
                f"clip={metrics.clip_fraction:.1%}, "
                f"passi_ottimizzazione={metrics.optimization_steps}",
                flush=True,
            )

    if checkpoint_path is not None:
        checkpoint = Path(checkpoint_path)
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"model_state_dict": trainer.model.state_dict(), "ppo_config": asdict(trainer.config)},
            checkpoint,
        )
    return TrainingResult(episodes=tuple(episodes), updates=tuple(updates))


def main() -> None:
    parser = argparse.ArgumentParser(description="Allena un agente PPO nel tennis simulator.")
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--rollout-episodes", type=int, default=8)
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/tennis_ppo.pt"))
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    result = train(
        TennisEnv(),
        PPOTrainer(ActorCritic()),
        total_episodes=args.episodes,
        rollout_episodes=args.rollout_episodes,
        seed=args.seed,
        checkpoint_path=args.checkpoint,
    )
    last = result.episodes[-1]
    print(
        f"Training completato: {len(result.episodes)} episodi, "
        f"{len(result.updates)} aggiornamenti, checkpoint={args.checkpoint}. "
        f"Ultimo reward={last.total_reward}, punteggio={last.scores}."
    )


if __name__ == "__main__":
    main()

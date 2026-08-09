"""Training PPO per il simulatore di tennis."""

from __future__ import annotations

import argparse
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import Tensor

from model import ActorCritic
from play import EpisodeResult, run_episode
from rl import classic_policy, easy_policy, medium_policy
from rollout import RolloutBuffer
from tennis_env import TennisEnv


@dataclass(frozen=True)
class PPOConfig:
    """Iperparametri del training PPO."""

    learning_rate: float = 5e-4
    gamma: float = 0.95
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    value_coefficient: float = 0.5
    entropy_coefficient: float = 0.001
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


@dataclass(frozen=True)
class EvaluationMetrics:
    wins: int
    losses: int
    timeouts: int
    mean_reward: float
    mean_agent_score: float
    mean_opponent_score: float

    @property
    def win_rate(self) -> float:
        return self.wins / (self.wins + self.losses + self.timeouts)


@dataclass(frozen=True)
class CurriculumStage:
    name: str
    environment_factory: Callable[[], TennisEnv]
    target_win_rate: float | None


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


def _save_checkpoint(
    model: ActorCritic, config: PPOConfig, checkpoint_path: str | Path
) -> None:
    """Salva uno stato riutilizzabile del modello e della sua configurazione."""
    checkpoint = Path(checkpoint_path)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"model_state_dict": model.state_dict(), "ppo_config": asdict(config)},
        checkpoint,
    )


def evaluate(
    environment_factory: Callable[[], TennisEnv],
    model: ActorCritic,
    *,
    episodes: int = 40,
    seed: int | None = None,
) -> EvaluationMetrics:
    """Valuta una policy senza esplorazione su episodi separati dal training."""
    if episodes <= 0:
        raise ValueError("episodes deve essere positivo.")

    environment = environment_factory()
    was_training = model.training
    model.eval()
    try:
        results = [
            run_episode(
                environment,
                model,
                seed=None if seed is None else seed + index,
                deterministic=True,
            )
            for index in range(episodes)
        ]
    finally:
        model.train(was_training)
    return EvaluationMetrics(
        wins=sum(result.winner == "bottom" for result in results),
        losses=sum(result.winner == "top" for result in results),
        timeouts=sum(result.truncated for result in results),
        mean_reward=sum(result.total_reward for result in results) / len(results),
        mean_agent_score=sum(result.scores[1] for result in results) / len(results),
        mean_opponent_score=sum(result.scores[0] for result in results) / len(results),
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
            wins = sum(episode.winner == "bottom" for episode in rollout)
            losses = sum(episode.winner == "top" for episode in rollout)
            timeouts = sum(episode.truncated for episode in rollout)
            mean_opponent_score = sum(episode.scores[0] for episode in rollout) / len(rollout)
            mean_agent_score = sum(episode.scores[1] for episode in rollout) / len(rollout)
            print(
                f"[update {len(updates):03d} | episodi {rollout_start + 1}-"
                f"{len(episodes)}/{total_episodes}] "
                f"reward medio={mean_reward:.2f}, "
                f"risultati V-S-T={wins}-{losses}-{timeouts}, "
                f"punti medi agente-avversario={mean_agent_score:.2f}-"
                f"{mean_opponent_score:.2f}, passi medi={mean_steps:.1f}, "
                f"policy_loss={metrics.policy_loss:.4f}, "
                f"value_loss={metrics.value_loss:.4f}, "
                f"entropia={metrics.entropy:.4f}, "
                f"KL={metrics.approximate_kl:.5f}, "
                f"clip={metrics.clip_fraction:.1%}, "
                f"passi_ottimizzazione={metrics.optimization_steps}",
                flush=True,
            )
            if checkpoint_path is not None:
                _save_checkpoint(trainer.model, trainer.config, checkpoint_path)

    return TrainingResult(episodes=tuple(episodes), updates=tuple(updates))


def main() -> None:
    parser = argparse.ArgumentParser(description="Allena un agente PPO nel tennis simulator.")
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--rollout-episodes", type=int, default=8)
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/tennis_ppo.pt"))
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--evaluation-episodes", type=int, default=40)
    parser.add_argument("--evaluation-interval", type=int, default=40)
    args = parser.parse_args()

    if args.episodes <= 0:
        parser.error("episodes deve essere positivo.")
    if args.evaluation_episodes <= 0 or args.evaluation_interval <= 0:
        parser.error("I parametri di valutazione devono essere positivi.")

    stages = (
        CurriculumStage(
            name="facile: imparare a colpire la pallina",
            environment_factory=lambda: TennisEnv(
                opponent_policy=easy_policy,
                points_to_win=1,
                max_steps_per_episode=1_000,
            ),
            target_win_rate=0.70,
        ),
        CurriculumStage(
            name="intermedia: sostenere gli scambi",
            environment_factory=lambda: TennisEnv(
                opponent_policy=medium_policy,
                points_to_win=3,
                max_steps_per_episode=2_000,
            ),
            target_win_rate=0.60,
        ),
        CurriculumStage(
            name="finale: vincere contro la policy classica",
            environment_factory=lambda: TennisEnv(opponent_policy=classic_policy),
            target_win_rate=None,
        ),
    )

    trainer = PPOTrainer(
        ActorCritic(), PPOConfig(entropy_coefficient=0.001)
    )
    episodes: list[EpisodeResult] = []
    updates: list[UpdateMetrics] = []
    seed_offset = 0
    stage_index = 0
    remaining_episodes = args.episodes
    final_stage_started = False
    while remaining_episodes > 0:
        stage = stages[stage_index]
        final_stage_started = final_stage_started or stage.target_win_rate is None
        episode_count = min(args.evaluation_interval, remaining_episodes)
        print(
            f"\n=== Fase {stage_index + 1}/3: {stage.name} "
            f"({episode_count} episodi prima della valutazione) ===",
            flush=True,
        )
        stage_result = train(
            stage.environment_factory(),
            trainer,
            total_episodes=episode_count,
            rollout_episodes=args.rollout_episodes,
            seed=None if args.seed is None else args.seed + seed_offset,
            checkpoint_path=args.checkpoint,
        )
        episodes.extend(stage_result.episodes)
        updates.extend(stage_result.updates)
        seed_offset += episode_count
        remaining_episodes -= episode_count

        evaluation = evaluate(
            stage.environment_factory,
            trainer.model,
            episodes=args.evaluation_episodes,
            seed=None if args.seed is None else args.seed + 100_000 + seed_offset,
        )
        print(
            f"[valutazione deterministica] V-S-T="
            f"{evaluation.wins}-{evaluation.losses}-{evaluation.timeouts}, "
            f"win rate={evaluation.win_rate:.1%}, "
            f"punti medi agente-avversario={evaluation.mean_agent_score:.2f}-"
            f"{evaluation.mean_opponent_score:.2f}, "
            f"reward medio={evaluation.mean_reward:.2f}",
            flush=True,
        )
        if (
            stage.target_win_rate is not None
            and evaluation.win_rate >= stage.target_win_rate
        ):
            print(
                f"Promozione: raggiunta soglia del {stage.target_win_rate:.0%} "
                "di vittorie.",
                flush=True,
            )
            stage_index += 1

    result = TrainingResult(episodes=tuple(episodes), updates=tuple(updates))
    last = result.episodes[-1]
    curriculum_status = "completato" if final_stage_started else "incompleto"
    print(
        f"Training completato: {len(result.episodes)} episodi, "
        f"{len(result.updates)} aggiornamenti, curriculum={curriculum_status}, "
        f"checkpoint={args.checkpoint}. "
        f"Ultimo reward={last.total_reward}, punteggio={last.scores}."
    )


if __name__ == "__main__":
    main()

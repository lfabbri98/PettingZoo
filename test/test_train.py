import torch

from model import ActorCritic
from tennis_env import TennisEnv
from train import PPOConfig, PPOTrainer, evaluate, train


def test_ppo_update_changes_model_and_clears_on_policy_buffer() -> None:
    environment = TennisEnv(max_steps_per_episode=1)
    trainer = PPOTrainer(
        ActorCritic(hidden_size=8),
        PPOConfig(update_epochs=2, minibatch_size=1, target_kl=None),
    )
    initial_parameters = [parameter.detach().clone() for parameter in trainer.model.parameters()]

    result = train(environment, trainer, total_episodes=1, rollout_episodes=1, seed=3)

    assert len(result.episodes) == 1
    assert result.updates[0].optimization_steps == 2
    assert any(
        not torch.equal(before, after)
        for before, after in zip(initial_parameters, trainer.model.parameters(), strict=True)
    )


def test_training_saves_a_reusable_checkpoint(tmp_path) -> None:
    checkpoint = tmp_path / "model.pt"
    trainer = PPOTrainer(ActorCritic(hidden_size=8), PPOConfig(update_epochs=1, minibatch_size=1))

    train(
        TennisEnv(max_steps_per_episode=1),
        trainer,
        total_episodes=1,
        rollout_episodes=1,
        checkpoint_path=checkpoint,
    )

    saved = torch.load(checkpoint, weights_only=True)
    assert set(saved) == {"model_state_dict", "ppo_config"}


def test_evaluate_uses_deterministic_episodes() -> None:
    result = evaluate(
        lambda: TennisEnv(max_steps_per_episode=1),
        ActorCritic(hidden_size=8),
        episodes=3,
        seed=5,
    )

    assert result.wins == 0
    assert result.losses == 0
    assert result.timeouts == 3
    assert result.win_rate == 0
    assert result.mean_agent_score == 0
    assert result.mean_opponent_score == 0

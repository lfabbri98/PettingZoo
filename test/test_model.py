import pytest

torch = pytest.importorskip("torch")

from model import ACTION_SIZE, OBSERVATION_SIZE, ActorCritic


def test_actor_critic_returns_valid_single_environment_action() -> None:
    model = ActorCritic()
    observation = torch.zeros(OBSERVATION_SIZE)

    sample = model.act(observation, deterministic=True)

    assert sample.action.shape == (ACTION_SIZE,)
    assert sample.value.shape == ()
    assert torch.all(sample.action[:2] >= -1)
    assert torch.all(sample.action[:2] <= 1)
    assert 0 <= sample.action[2] <= 1
    assert -1 <= sample.action[3] <= 1


def test_actor_critic_supports_observation_batches() -> None:
    model = ActorCritic()
    observations = torch.zeros((5, OBSERVATION_SIZE))

    mean, values = model(observations)
    sample = model.act(observations, deterministic=False)

    assert mean.shape == (5, ACTION_SIZE)
    assert values.shape == (5,)
    assert sample.action.shape == (5, ACTION_SIZE)
    assert torch.all(sample.action[:, 2] >= 0)
    assert torch.all(sample.action[:, 2] <= 1)

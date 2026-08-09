import pytest

torch = pytest.importorskip("torch")

from model import ACTION_SIZE, OBSERVATION_SIZE, ActorCritic
from play import action_to_tuple, observation_to_tensor, run_episode
from tennis_env import TennisEnv


def test_observation_to_tensor_keeps_the_expected_shape_and_dtype() -> None:
    observation = (0.0,) * OBSERVATION_SIZE

    tensor = observation_to_tensor(observation)

    assert tensor.shape == (OBSERVATION_SIZE,)
    assert tensor.dtype == torch.float32


def test_action_to_tuple_rejects_a_batch() -> None:
    with pytest.raises(ValueError, match="batch"):
        action_to_tuple(torch.zeros((2, ACTION_SIZE)))


def test_run_episode_connects_model_to_environment() -> None:
    environment = TennisEnv(max_steps_per_episode=1)
    model = ActorCritic()

    result = run_episode(environment, model, seed=42)

    assert result.steps == 1
    assert result.terminated is False
    assert result.truncated is True
    assert result.winner is None
    assert result.scores == (0, 0)

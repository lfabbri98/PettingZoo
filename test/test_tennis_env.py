import pytest

from tennis_env import TennisEnv


def test_reset_returns_a_12_value_observation_and_is_reproducible() -> None:
    first_env = TennisEnv()
    second_env = TennisEnv()

    first_observation, first_info = first_env.reset(seed=42)
    second_observation, second_info = second_env.reset(seed=42)

    assert len(first_observation) == 12
    assert first_observation == second_observation
    assert first_info == second_info


def test_step_returns_observation_reward_end_flags_and_info() -> None:
    env = TennisEnv(max_steps_per_episode=1)
    env.reset(seed=42)

    observation, reward, terminated, truncated, info = env.step((0, 0, 0.5, 0))

    assert len(observation) == 12
    assert reward == 0
    assert terminated is False
    assert truncated is True
    assert info["steps"] == 1


def test_step_requires_reset_and_valid_action() -> None:
    env = TennisEnv()

    with pytest.raises(RuntimeError, match="reset"):
        env.step((0, 0, 0, 0))

    env.reset()
    with pytest.raises(ValueError, match="limiti"):
        env.step((2, 0, 0, 0))


def test_episode_ends_only_when_a_player_reaches_eleven_points() -> None:
    env = TennisEnv()
    env.reset(seed=42)
    env.court.bottom_score = 10
    env.ball.y = env.court.bounds.top - 1

    _, reward, terminated, truncated, info = env.step((0, 0, 0, 0))

    assert reward == 1
    assert terminated is True
    assert truncated is False
    assert info["scorer"] == "bottom"
    assert info["scores"] == (0, 11)

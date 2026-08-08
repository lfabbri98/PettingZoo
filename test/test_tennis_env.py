import random
import subprocess
import sys
from pathlib import Path

import pytest

from environment import Ball, BoundingBox, Player
from rl import PlayerAction, classic_policy
from tennis_env import TennisEnv


Action = tuple[float, float, float, float]


def classic_agent_action(env: TennisEnv) -> Action:
    action = classic_policy(
        env.bottom_player,
        env.ball,
        env.court.bounds,
        "bottom",
        is_active=env.active_player == "bottom",
        is_serving=env._is_serving("bottom"),
        rng=env._rng,
    )
    max_speed = env.ball.max_speed
    assert max_speed is not None
    min_speed = env.config["ball"]["min_shot_speed"]
    return (
        action.direction[0],
        action.direction[1],
        (action.shot_force - min_speed) / (max_speed - min_speed),
        action.shot_angle / env.bottom_player.max_shot_angle,
    )


def force_point(env: TennisEnv, scorer: str) -> None:
    env.ball.y = (
        env.court.bounds.top - 1
        if scorer == "bottom"
        else env.court.bounds.bottom + 1
    )


def test_reset_returns_a_15_value_observation_and_is_reproducible() -> None:
    first_env = TennisEnv()
    second_env = TennisEnv()

    first_observation, first_info = first_env.reset(seed=42)
    second_observation, second_info = second_env.reset(seed=42)

    assert len(first_observation) == 15
    assert first_observation == second_observation
    assert first_info == second_info
    assert first_info["scorer"] is None
    assert first_info["winner"] is None


def test_step_returns_observation_reward_end_flags_and_info() -> None:
    env = TennisEnv(max_steps_per_episode=1, frame_skip=3)
    env.reset(seed=42)

    observation, reward, terminated, truncated, info = env.step((0, 0, 0.5, 0))

    assert len(observation) == 15
    assert reward == 0
    assert terminated is False
    assert truncated is True
    assert info["steps"] == 1
    assert info["physics_steps"] == 3
    assert info["winner"] is None


def test_step_requires_reset_and_valid_action() -> None:
    env = TennisEnv()

    with pytest.raises(RuntimeError, match="reset"):
        env.step((0, 0, 0, 0))

    env.reset()
    with pytest.raises(ValueError, match="limiti"):
        env.step((2, 0, 0, 0))


@pytest.mark.parametrize("winner", ["top", "bottom"])
def test_episode_ends_when_either_player_reaches_eleven_points(winner: str) -> None:
    env = TennisEnv()
    env.reset(seed=42)
    if winner == "top":
        env.court.top_score = 10
    else:
        env.court.bottom_score = 10
    force_point(env, winner)

    _, reward, terminated, truncated, info = env.step((0, 0, 0, 0))

    assert reward == (1 if winner == "bottom" else -1)
    assert terminated is True
    assert truncated is False
    assert info["scorer"] == winner
    assert info["winner"] == winner
    assert max(info["scores"]) == 11


def test_non_terminal_point_returns_consistent_new_rally_state() -> None:
    env = TennisEnv()
    env.reset(seed=3)
    force_point(env, "bottom")

    observation, reward, terminated, truncated, info = env.step((0, 0, 0, 0))

    assert reward == 1
    assert terminated is False
    assert truncated is False
    assert info["scorer"] == "bottom"
    assert info["scores"] == (0, 1)
    assert info["active_player"] == env.active_player
    assert observation == env._get_observation()
    assert observation[-3:-1] == pytest.approx((1 / 11, 0))
    expected_active = 1 if env.active_player == "bottom" else -1
    assert observation[-1] == expected_active


def test_normalized_force_selects_desired_output_speed() -> None:
    env = TennisEnv()
    env.reset(seed=42)

    minimum = env._agent_action((0, 0, 0, 0)).shot_force
    maximum = env._agent_action((0, 0, 1, 0)).shot_force

    assert minimum == env.config["ball"]["min_shot_speed"]
    assert maximum == env.config["ball"]["max_speed"]


def test_same_seed_and_actions_produce_the_same_trajectory() -> None:
    first_env = TennisEnv()
    second_env = TennisEnv()
    first_env.reset(seed=7)
    second_env.reset(seed=7)

    actions = [(0.2, -0.3, 0.75, 0.1)] * 20
    first_trajectory = [first_env.step(action) for action in actions]
    second_trajectory = [second_env.step(action) for action in actions]

    assert first_trajectory == second_trajectory


def test_custom_opponent_policy_is_used_for_each_physics_tick() -> None:
    calls = 0

    def opponent_policy(
        player: Player,
        ball: Ball,
        court_bounds: BoundingBox,
        side: str,
        *,
        is_active: bool,
        is_serving: bool,
        rng: random.Random,
    ) -> PlayerAction:
        nonlocal calls
        calls += 1
        return PlayerAction((0, 0), 150, 0)

    env = TennisEnv(frame_skip=3, opponent_policy=opponent_policy)
    env.reset(seed=42)

    env.step((0, 0, 0.5, 0))

    assert calls == 3


def test_default_config_is_independent_from_current_directory(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)

    env = TennisEnv()
    observation, _ = env.reset(seed=42)

    assert len(observation) == 15


def test_core_wrapper_does_not_import_pygame() -> None:
    project_root = Path(__file__).parents[1]
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import tennis_env; print('pygame' in sys.modules)",
        ],
        cwd=project_root,
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.stdout.strip() == "False"


def test_complete_classic_match_reaches_eleven_before_default_timeout() -> None:
    env = TennisEnv()
    env.reset(seed=0)

    while True:
        _, _, terminated, truncated, info = env.step(classic_agent_action(env))
        if terminated or truncated:
            break

    assert terminated is True
    assert truncated is False
    assert max(info["scores"]) == 11
    assert info["winner"] in ("top", "bottom")

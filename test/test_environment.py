import math

import pygame
import pytest
import rl

from environment import Ball, Player, TennisCourt, player_is_behind_ball
from rl import classic_policy, observation


PLAYER_CONFIG = {"width": 20, "height": 10, "max_speed": 50}


def make_player(x: float = 50, y: float = 50) -> Player:
    return Player(x, y, (0, 0, 0), PLAYER_CONFIG)


def test_player_move_updates_position_and_velocity() -> None:
    player = make_player()

    player.move((0.5, 0), 0.2)

    assert (player.x, player.y) == (55, 50)
    assert (player.vx, player.vy) == (25, 0)


def test_player_cannot_exceed_configured_max_speed() -> None:
    with pytest.raises(ValueError):
        make_player().move((1, 1), 0.1)


def test_player_accepts_a_normalized_direction_with_rounding_error() -> None:
    player = make_player()

    player.move((-0.22857751264011125, 0.9735257165145973), 0.1)

    assert math.hypot(player.vx, player.vy) == pytest.approx(player.max_speed)


def test_player_reset_restores_initial_position_and_stops_movement() -> None:
    player = make_player(x=30, y=70)
    player.move((0.5, -0.5), 0.2)

    player.reset()

    assert (player.x, player.y) == (30, 70)
    assert (player.vx, player.vy) == (0, 0)


def test_player_stays_inside_the_court() -> None:
    player = make_player(x=-10, y=200)
    bounds = pygame.Rect(0, 0, 100, 100)

    player.keep_inside(bounds)

    assert (player.x, player.y) == (10, 95)


@pytest.mark.parametrize(
    ("side", "initial_y", "expected_y"),
    [
        ("top", 80, 45),
        ("bottom", 20, 55),
    ],
)
def test_player_cannot_cross_the_net(
    side: str, initial_y: float, expected_y: float
) -> None:
    player = make_player(y=initial_y)

    player.keep_in_half(pygame.Rect(0, 0, 100, 100), side)

    assert player.y == expected_y


def test_player_with_unknown_side_stays_inside_the_court() -> None:
    player = make_player(y=200)

    player.keep_in_half(pygame.Rect(0, 0, 100, 100), "unknown")

    assert player.y == 95


def test_player_uses_configured_maximum_shot_angle() -> None:
    player = Player(
        50,
        50,
        (0, 0, 0),
        {**PLAYER_CONFIG, "max_shot_angle": 35},
    )

    player.choose_shot(force=100, angle=35)

    assert player.shot_angle == 35
    with pytest.raises(ValueError):
        player.choose_shot(force=100, angle=36)


def test_ball_move_updates_position() -> None:
    ball = Ball(10, 20, 30, -40)

    ball.move(0.5)

    assert (ball.x, ball.y) == (25, 0)


@pytest.mark.parametrize(
    ("ball", "expected"),
    [
        (Ball(-1, 50, -3, 0), (0, 50, 3, 0)),
        (Ball(101, 50, 3, 0), (100, 50, -3, 0)),
    ],
)
def test_ball_bounces_on_lateral_court_edges(ball: Ball, expected: tuple[float, ...]) -> None:
    ball.keep_inside(pygame.Rect(0, 0, 100, 100))

    assert (ball.x, ball.y, ball.vx, ball.vy) == expected


@pytest.mark.parametrize(
    ("ball_y", "scorer"),
    [
        (-1, "bottom"),
        (101, "top"),
        (0, None),
        (100, None),
    ],
)
def test_ball_check_point_detects_the_opponent_of_the_exit_side(
    ball_y: float, scorer: str | None
) -> None:
    ball = Ball(50, ball_y, 0, 0)

    assert ball.check_point(pygame.Rect(0, 0, 100, 100)) == scorer


def test_classic_policy_slows_down_when_approaching_the_ball() -> None:
    bottom_player = make_player(x=50, y=80)
    ball = Ball(x=50, y=85, vx=0, vy=-50)

    ball.move(0.14)

    assert ball.y < bottom_player.y
    action = classic_policy(bottom_player, ball, pygame.Rect(0, 0, 100, 100), "bottom")
    assert action.direction == pytest.approx((0, -2 / 15))
    assert player_is_behind_ball(bottom_player, ball, "bottom") is True


def test_classic_policy_uses_maximum_speed_when_ball_is_far_away() -> None:
    player = make_player(x=50, y=80)
    ball = Ball(50, 60, 0, 0)

    action = classic_policy(player, ball, pygame.Rect(0, 0, 100, 100), "bottom")

    assert action.direction == (0, -1)


@pytest.mark.parametrize(
    ("side", "player_position", "ball_position", "ball_vy", "expected_direction"),
    [
        ("top", (50, 40), (50, 30), -50, (0, -1)),
        ("bottom", (50, 60), (50, 70), 50, (0, 1)),
    ],
)
def test_classic_policy_does_not_slow_for_ball_escaping_to_baseline(
    side: str,
    player_position: tuple[float, float],
    ball_position: tuple[float, float],
    ball_vy: float,
    expected_direction: tuple[float, float],
) -> None:
    player = make_player(*player_position)
    ball = Ball(*ball_position, 0, ball_vy)

    action = classic_policy(player, ball, pygame.Rect(0, 0, 100, 100), side)

    assert action.direction == expected_direction
    assert action.shot_angle == 0


def test_baseline_chase_returns_ball_straight_to_the_other_half() -> None:
    player = make_player()
    player.move((0, 1), 0)
    player.choose_shot(force=80, angle=0)
    ball = Ball(player.x, player.y, 30, 40)

    assert ball.hit_by(player, "bottom") is True
    assert (ball.vx, ball.vy) == (0, -130)


def test_ball_reset_places_ball_on_the_randomly_selected_server() -> None:
    ball = Ball(0, 0, 0, 0)
    top_player = make_player(x=20, y=10)
    bottom_player = make_player(x=80, y=70)

    server_side = ball.reset(top_player, bottom_player)

    expected_server = top_player if server_side == "top" else bottom_player
    assert server_side in ("top", "bottom")
    assert (ball.x, ball.y) == (expected_server.x, expected_server.y)
    assert (ball.vx, ball.vy) == (0, 0)


def test_ball_speed_is_limited_after_a_hit() -> None:
    player = make_player()
    player.choose_shot(force=100, angle=0)
    ball = Ball(player.x, player.y, 0, 250, max_speed=300)

    assert ball.hit_by(player) is True

    assert ball.vx == 0
    assert ball.vy == -300


@pytest.mark.parametrize(
    ("side", "expected_vy"),
    [("top", 100), ("bottom", -100)],
)
def test_serve_is_hit_towards_the_opponent(side: str, expected_vy: float) -> None:
    player = make_player()
    player.choose_shot(force=100, angle=0)
    ball = Ball(player.x, player.y, 0, 0)

    assert ball.hit_by(player, side) is True
    assert ball.vy == expected_vy


def test_court_add_point_updates_the_correct_score() -> None:
    court = TennisCourt({"width": 100, "height": 80})

    court.add_point("top")
    court.add_point("bottom")
    court.add_point("bottom")

    assert (court.top_score, court.bottom_score) == (1, 2)


def test_ball_hit_replaces_horizontal_velocity_with_shot_component() -> None:
    player = make_player()
    player.move((1, 0), 0.1)
    ball = Ball(player.x, player.y, 12, 25)

    hit = ball.hit_by(player)

    assert hit is True
    assert (ball.vx, ball.vy) == pytest.approx((0, -(12**2 + 25**2) ** 0.5))


def test_player_shot_force_and_angle_deviate_ball_trajectory() -> None:
    player = make_player()
    player.move((1, 0), 0.1)
    player.choose_shot(force=100, angle=30)
    ball = Ball(player.x, player.y, 12, 25)

    hit = ball.hit_by(player)

    assert hit is True
    outgoing_speed = (12**2 + 25**2) ** 0.5 + 100
    assert ball.vx == pytest.approx(outgoing_speed / 2)
    assert ball.vy == pytest.approx(-outgoing_speed * 3**0.5 / 2)


@pytest.mark.parametrize(
    ("force", "angle"),
    [(-1, 0), (10, -90), (10, 90)],
)
def test_player_rejects_invalid_shot_choices(force: float, angle: float) -> None:
    with pytest.raises(ValueError):
        make_player().choose_shot(force, angle)


def test_ball_is_hit_only_once_while_inside_the_same_player() -> None:
    player = make_player()
    player.move((1, 0), 0.1)
    ball = Ball(player.x, player.y, 12, 25)

    ball.hit_by(player)
    second_hit = ball.hit_by(player)

    assert second_hit is False
    assert (ball.vx, ball.vy) == pytest.approx((0, -(12**2 + 25**2) ** 0.5))


def test_ball_can_be_hit_again_after_leaving_player_box() -> None:
    player = make_player()
    player.move((1, 0), 0.1)
    ball = Ball(player.x, player.y, 12, 25)

    assert ball.hit_by(player) is True
    ball.x = 0
    assert ball.hit_by(player) is False
    ball.x = player.x
    assert ball.hit_by(player) is True
    assert (ball.vx, ball.vy) == pytest.approx((0, (12**2 + 25**2) ** 0.5))


def test_observation_is_normalized_from_bottom_agent_perspective() -> None:
    agent = make_player(x=25, y=80)
    agent.move((0.5, -0.5), 0)
    opponent = make_player(x=75, y=20)
    opponent.move((-0.5, 0.5), 0)
    ball = Ball(40, 60, 25, -50, max_speed=100)

    state = observation(agent, opponent, ball, pygame.Rect(0, 0, 100, 100), "bottom")

    assert state == pytest.approx(
        (0.25, 0.8, 0.5, -0.5, 0.75, 0.2, -0.5, 0.5, 0.4, 0.6, 0.25, -0.5)
    )


def test_observation_mirrors_vertical_axis_for_top_agent() -> None:
    agent = make_player(x=25, y=20)
    agent.move((0.5, 0.5), 0)
    opponent = make_player(x=75, y=80)
    opponent.move((-0.5, -0.5), 0)
    ball = Ball(40, 40, 25, 50, max_speed=100)

    state = observation(agent, opponent, ball, pygame.Rect(0, 0, 100, 100), "top")

    assert state == pytest.approx(
        (0.25, 0.8, 0.5, -0.5, 0.75, 0.2, -0.5, 0.5, 0.4, 0.6, 0.25, -0.5)
    )


def test_classic_policy_returns_a_valid_move_and_shot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    player = make_player(x=20, y=80)
    ball = Ball(80, 70, 0, 0)
    monkeypatch.setattr(rl.random, "uniform", lambda start, end: 0)

    action = classic_policy(player, ball, pygame.Rect(0, 0, 100, 100), "bottom")

    assert action.direction == pytest.approx((60 / (3700**0.5), -10 / (3700**0.5)))
    assert action.shot_force == 80
    assert -player.max_shot_angle <= action.shot_angle <= player.max_shot_angle


def test_classic_policy_varies_the_angle_of_normal_returns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    player = make_player(x=20, y=80)
    ball = Ball(80, 70, 0, 0)
    monkeypatch.setattr(rl.random, "uniform", lambda start, end: 8)

    action = classic_policy(player, ball, pygame.Rect(0, 0, 100, 100), "bottom")

    assert action.shot_angle == 8


def test_random_policy_angle_changes_the_ball_trajectory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    player = make_player()
    ball = Ball(player.x, player.y, 0, 25)
    monkeypatch.setattr(rl.random, "uniform", lambda start, end: 30)

    action = classic_policy(player, ball, pygame.Rect(0, 0, 100, 100), "bottom")
    player.choose_shot(action.shot_force, action.shot_angle)

    assert ball.hit_by(player, "bottom") is True
    assert ball.vx == pytest.approx(52.5)
    assert ball.vy == pytest.approx(-105 * 3**0.5 / 2)


def test_classic_policy_varies_the_serve_choice(monkeypatch: pytest.MonkeyPatch) -> None:
    player = make_player(x=50, y=80)
    ball = Ball(50, 80, 0, 0)

    monkeypatch.setattr(
        rl.random,
        "choice",
        lambda choices: 90,
    )
    monkeypatch.setattr(rl.random, "uniform", lambda start, end: 20)
    action = classic_policy(
        player, ball, pygame.Rect(0, 0, 100, 100), "bottom", is_serving=True
    )

    assert action.shot_force == 90
    assert action.shot_angle == 20


@pytest.mark.parametrize(
    ("side", "player_position", "expected_direction"),
    [
        ("top", (20, 40), (2**-0.5, -2**-0.5)),
        ("bottom", (80, 60), (-2**-0.5, 2**-0.5)),
    ],
)
def test_inactive_classic_player_returns_to_defensive_position(
    side: str,
    player_position: tuple[float, float],
    expected_direction: tuple[float, float],
) -> None:
    player = make_player(*player_position)
    ball = Ball(50, 50, 0, 0)

    action = classic_policy(
        player, ball, pygame.Rect(0, 0, 100, 100), side, is_active=False
    )

    assert action.direction == pytest.approx(expected_direction)


def test_active_classic_player_waits_defensively_for_ball_in_other_half() -> None:
    player = make_player(x=80, y=60)
    ball = Ball(50, 20, 0, 0)

    action = classic_policy(player, ball, pygame.Rect(0, 0, 100, 100), "bottom")

    assert action.direction == pytest.approx((-2**-0.5, 2**-0.5))


def test_waiting_classic_player_aligns_with_ball_instead_of_court_center() -> None:
    player = make_player(x=55, y=60)
    ball = Ball(60, 20, 0, 0)

    action = classic_policy(player, ball, pygame.Rect(0, 0, 100, 100), "bottom")

    assert action.direction == pytest.approx((2**-0.5, 2**-0.5))

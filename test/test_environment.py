import pygame
import pytest

from environment import Ball, Player, TennisCourt


PLAYER_CONFIG = {"width": 20, "height": 10, "speed": 50}


def make_player(x: float = 50, y: float = 50) -> Player:
    return Player(x, y, (0, 0, 0), PLAYER_CONFIG)


def test_player_move_updates_position_and_velocity() -> None:
    player = make_player()

    player.move((1, -1), 0.2)

    assert (player.x, player.y) == (60, 40)
    assert (player.vx, player.vy) == (50, -50)


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


def test_ball_reset_centers_ball_and_gives_it_a_new_velocity() -> None:
    ball = Ball(0, 0, 0, 0)
    ball.reset(pygame.Rect(0, 0, 100, 80))

    assert (ball.x, ball.y) == (50, 40)
    assert 80 <= abs(ball.vx) <= 160
    assert 80 <= abs(ball.vy) <= 160


def test_court_add_point_updates_the_correct_score() -> None:
    court = TennisCourt({"width": 100, "height": 80})

    court.add_point("top")
    court.add_point("bottom")
    court.add_point("bottom")

    assert (court.top_score, court.bottom_score) == (1, 2)


def test_ball_hit_adds_horizontal_player_velocity_and_reverses_vertical_velocity() -> None:
    player = make_player()
    player.move((1, 0), 0.1)
    ball = Ball(player.x, player.y, 12, 25)

    hit = ball.hit_by(player)

    assert hit is True
    assert (ball.vx, ball.vy) == (62, -25)


def test_ball_is_hit_only_once_while_inside_the_same_player() -> None:
    player = make_player()
    player.move((1, 0), 0.1)
    ball = Ball(player.x, player.y, 12, 25)

    ball.hit_by(player)
    second_hit = ball.hit_by(player)

    assert second_hit is False
    assert (ball.vx, ball.vy) == (62, -25)


def test_ball_can_be_hit_again_after_leaving_player_box() -> None:
    player = make_player()
    player.move((1, 0), 0.1)
    ball = Ball(player.x, player.y, 12, 25)

    assert ball.hit_by(player) is True
    ball.x = 0
    assert ball.hit_by(player) is False
    ball.x = player.x
    assert ball.hit_by(player) is True
    assert (ball.vx, ball.vy) == (112, 25)

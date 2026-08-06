import pygame
import pytest

from environment import Ball, Player


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


def test_ball_move_updates_position() -> None:
    ball = Ball(10, 20, 30, -40)

    ball.move(0.5)

    assert (ball.x, ball.y) == (25, 0)


@pytest.mark.parametrize(
    ("ball", "expected"),
    [
        (Ball(-1, 50, -3, 0), (0, 50, 3, 0)),
        (Ball(101, 50, 3, 0), (100, 50, -3, 0)),
        (Ball(50, -1, 0, -4), (50, 0, 0, 4)),
        (Ball(50, 101, 0, 4), (50, 100, 0, -4)),
    ],
)
def test_ball_bounces_on_each_court_edge(ball: Ball, expected: tuple[float, ...]) -> None:
    ball.keep_inside(pygame.Rect(0, 0, 100, 100))

    assert (ball.x, ball.y, ball.vx, ball.vy) == expected


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

import pygame
from environment import Ball, Player, TennisCourt


def draw_court(
    screen: pygame.Surface,
    court: TennisCourt,
    color: tuple[int, int, int],
    line_color: tuple[int, int, int],
) -> None:
    court_rect = pygame.Rect(
        round(court.x),
        round(court.y),
        round(court.width),
        round(court.length),
    )
    pygame.draw.rect(screen, color, court_rect)

    line_width = 3
    center_y = court_rect.centery
    top_service_y = court_rect.top + court.length // 4
    bottom_service_y = court_rect.bottom - court.length // 4

    # Perimetro, rete, linee di servizio e linea centrale di servizio.
    pygame.draw.rect(screen, line_color, court_rect, line_width)
    pygame.draw.line(
        screen,
        line_color,
        (court_rect.left, center_y),
        (court_rect.right, center_y),
        line_width,
    )
    pygame.draw.line(
        screen,
        line_color,
        (court_rect.left, top_service_y),
        (court_rect.right, top_service_y),
        line_width,
    )
    pygame.draw.line(
        screen,
        line_color,
        (court_rect.left, bottom_service_y),
        (court_rect.right, bottom_service_y),
        line_width,
    )
    pygame.draw.line(
        screen,
        line_color,
        (court_rect.centerx, top_service_y),
        (court_rect.centerx, bottom_service_y),
        line_width,
    )


def draw_player(screen: pygame.Surface, player: Player) -> None:
    rect = pygame.Rect(
        round(player.rect.left),
        round(player.rect.top),
        round(player.width),
        round(player.height),
    )
    pygame.draw.rect(screen, player.color, rect, border_radius=5)


def draw_ball(screen: pygame.Surface, ball: Ball, color: tuple[int, int, int]) -> None:
    pygame.draw.circle(screen, color, (round(ball.x), round(ball.y)), 7)


def draw_score(
    screen: pygame.Surface,
    font: pygame.font.Font,
    court: TennisCourt,
    color: tuple[int, int, int],
) -> None:
    score_surface = font.render(
        f"Alto {court.top_score} - {court.bottom_score} Basso",
        True,
        color,
    )
    score_position = ((screen.get_width() - score_surface.get_width()) // 2, 20)
    screen.blit(score_surface, score_position)

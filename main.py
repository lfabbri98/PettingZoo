import math

import pygame
import environment as env

def create_screen(window_config) -> None:
    pygame.init()
    screen = pygame.display.set_mode((window_config["width"], window_config["height"]))
    pygame.display.set_caption(window_config["title"])
    return screen 


def direction_towards_ball(player: env.Player, ball: env.Ball) -> tuple[int, int]:
    """Controller temporaneo: muove il giocatore verso la pallina in due dimensioni."""
    direction_x = (ball.x > player.x) - (ball.x < player.x)
    direction_y = (ball.y > player.y) - (ball.y < player.y)
    return (direction_x, direction_y)


def player_is_behind_ball(player: env.Player, ball: env.Ball, side: str) -> bool:
    """Restituisce se il giocatore è nella posizione corretta per colpire."""
    if side == "top":
        return player.y <= ball.y
    if side == "bottom":
        return player.y >= ball.y
    raise ValueError(f"Lato giocatore non valido: {side}")


def demo_shot_choice(
    player: env.Player,
    ball: env.Ball,
    court_rect: pygame.Rect,
    side: str,
) -> tuple[float, float]:
    """Policy dimostrativa: tira verso l'angolo più lontano dall'avversario."""
    horizontal_margin = court_rect.width * 0.15
    target_x = (
        court_rect.right - horizontal_margin
        if player.x < court_rect.centerx
        else court_rect.left + horizontal_margin
    )
    target_y = (
        court_rect.bottom - horizontal_margin
        if side == "top"
        else court_rect.top + horizontal_margin
    )
    horizontal_distance = target_x - ball.x
    vertical_distance = abs(target_y - ball.y)
    angle = math.degrees(math.atan2(horizontal_distance, vertical_distance))
    angle = max(-player.max_shot_angle, min(player.max_shot_angle, angle))

    # I valori vengono scelti a runtime: qui potranno essere sostituiti dalla policy.
    force = 80
    return force, angle


def main() -> None:
    config = env.parse_parameters("parameters.yml")

    screen = create_screen(config["window"])
    court = env.TennisCourt(config["court"])

    player_config = config["player"]
    colors = config["colors"]

    top_player = env.Player(500, 103, colors["player_top"], player_config)
    bottom_player = env.Player(500, 547, colors["player_bottom"], player_config)
    # Demo: la pallina viene ricevuta automaticamente dal giocatore di turno.
    ball = env.Ball(500, 460, 50, 120, config["ball"]["max_speed"])
    active_player = "bottom"
    score_font = pygame.font.Font(None, 36)

    clock = pygame.time.Clock()
    running = True
    while running:
        delta_time = clock.tick(config["window"]["fps"]) / 1000
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        screen.fill(config["colors"]["background"])
        court_rect = court.draw_court(
            screen,
            colors["court"],
            colors["lines"],
        )

        if active_player == "top":
            top_player_direction = direction_towards_ball(top_player, ball)
            bottom_player_direction = (0, 0)
        else:
            top_player_direction = (0, 0)
            bottom_player_direction = direction_towards_ball(bottom_player, ball)

        top_player.move(top_player_direction, delta_time)
        bottom_player.move(bottom_player_direction, delta_time)
        top_player.keep_in_half(court_rect, "top")
        bottom_player.keep_in_half(court_rect, "bottom")

        ball.move(delta_time)
        scorer = ball.check_point(court_rect)

        if scorer is not None:
            court.add_point(scorer)
            print(
                f"Punto {scorer}! "
                f"Alto: {court.top_score} - Basso: {court.bottom_score}"
            )
            ball.reset(court_rect)
            top_player.reset()
            bottom_player.reset()
            active_player = "bottom" if ball.vy > 0 else "top"
        else:
            ball.keep_inside(court_rect)
            top_player.choose_shot(
                *demo_shot_choice(top_player, ball, court_rect, "top")
            )
            bottom_player.choose_shot(
                *demo_shot_choice(bottom_player, ball, court_rect, "bottom")
            )
            top_player_hit = False
            bottom_player_hit = False
            if active_player == "top" and player_is_behind_ball(
                top_player, ball, "top"
            ):
                top_player_hit = ball.hit_by(top_player)
            elif active_player == "bottom" and player_is_behind_ball(
                bottom_player, ball, "bottom"
            ):
                bottom_player_hit = ball.hit_by(bottom_player)

            if top_player_hit:
                active_player = "bottom"
            elif bottom_player_hit:
                active_player = "top"

        top_player.draw(screen)
        bottom_player.draw(screen)
        pygame.draw.circle(screen, colors["lines"], (round(ball.x), round(ball.y)), 7)
        score_surface = score_font.render(
            f"Alto {court.top_score} - {court.bottom_score} Basso",
            True,
            colors["lines"],
        )
        score_position = ((screen.get_width() - score_surface.get_width()) // 2, 20)
        screen.blit(score_surface, score_position)

        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    main()

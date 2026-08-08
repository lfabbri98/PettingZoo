import pygame

import render
from rl import classic_policy
from tennis_env import TennisEnv


def create_screen(window_config: dict) -> pygame.Surface:
    pygame.init()
    screen = pygame.display.set_mode(
        (window_config["width"], window_config["height"])
    )
    pygame.display.set_caption(window_config["title"])
    return screen


def classic_agent_action(environment: TennisEnv) -> tuple[float, float, float, float]:
    """Converte la policy demo del giocatore basso nell'azione del wrapper."""
    action = classic_policy(
        environment.bottom_player,
        environment.ball,
        environment.court.bounds,
        "bottom",
        is_active=environment.active_player == "bottom",
        is_serving=environment.active_player == "bottom"
        and environment.ball.vx == 0
        and environment.ball.vy == 0,
    )
    max_ball_speed = environment.ball.max_speed
    assert max_ball_speed is not None
    min_ball_speed = environment.config["ball"]["min_shot_speed"]
    return (
        action.direction[0],
        action.direction[1],
        (action.shot_force - min_ball_speed) / (max_ball_speed - min_ball_speed),
        action.shot_angle / environment.bottom_player.max_shot_angle,
    )


def main() -> None:
    environment = TennisEnv(max_steps_per_episode=20_000, frame_skip=1)
    config = environment.config
    colors = config["colors"]
    screen = create_screen(config["window"])
    environment.reset()
    score_font = pygame.font.Font(None, 36)

    clock = pygame.time.Clock()
    running = True
    while running:
        clock.tick(config["window"]["fps"])
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        screen.fill(config["colors"]["background"])
        _, _, terminated, truncated, info = environment.step(
            classic_agent_action(environment)
        )

        if info["scorer"] is not None:
            print(
                f"Punto {info['scorer']}! "
                f"Alto: {info['scores'][0]} - Basso: {info['scores'][1]}"
            )
        if terminated:
            print("Partita conclusa: si inizia una nuova partita.")
            environment.reset()
        elif truncated:
            print("Partita interrotta per limite di frame: si ricomincia.")
            environment.reset()

        render.draw_court(
            screen, environment.court, colors["court"], colors["lines"]
        )
        render.draw_player(screen, environment.top_player)
        render.draw_player(screen, environment.bottom_player)
        render.draw_ball(screen, environment.ball, colors["lines"])
        render.draw_score(screen, score_font, environment.court, colors["lines"])

        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    main()

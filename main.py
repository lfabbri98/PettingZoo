import pygame
import environment as env

def create_screen(window_config) -> None:
    pygame.init()
    screen = pygame.display.set_mode((window_config["width"], window_config["height"]))
    pygame.display.set_caption(window_config["title"])
    return screen 


def main() -> None:
    config = env.parse_parameters("parameters.yml")

    screen = create_screen(config["window"])
    court = env.TennisCourt(config["court"])

    player_config = config["player"]
    colors = config["colors"]

    p1 = env.Player(500, 103, colors["player_bottom"], player_config)
    p2 = env.Player(500, 547, colors["player_top"], player_config)

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        screen.fill(config["colors"]["background"])
        court.draw_court(screen, config["colors"]["court"])

        p1.draw(screen)
        p2.draw(screen)

        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    main()

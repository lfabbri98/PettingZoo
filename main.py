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

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        screen.fill(config["colors"]["background"])
        court.draw_court(screen, config["colors"]["court"])
        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    main()

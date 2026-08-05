"""
Definizione di ambiente e regole del simulatore
"""

from dataclasses import dataclass
import pygame
import yaml
from pathlib import Path

def parse_parameters(config_name: str) -> dict:
    #Parsing parametri necessari. Ritorna un dizionario con dentro config
    with open(config_name, "r") as f:
        config = yaml.safe_load(f)

    return config

class Player:
    """Definizione del giocatore"""

    x: float
    y: float
    color: tuple[int, int, int]
    width: int #px
    height: int #px
    speed: int #px/s

    @property 
    def rect(self) -> pygame.Rect:
        return pygame.Rect(
            round(self.x - self.width / 2),
            round(self.y - self.height / 2),
            self.width,
            self.height,
        )

    def move(self, direction: tuple, delta_time: float) -> None:

        if direction[0] not in (-1, 0, 1) or direction[1] not in (-1, 0, 1):
            raise ValueError(f"Azione non valida: {direction}. Usa -1, 0 oppure 1.")

        self.x += direction[0] * self.speed * delta_time
        self.y += direction[1] * self.speed * delta_time

    def draw(self, screen: pygame.Surface) -> None:
        pygame.draw.rect(screen, self.color, self.rect, border_radius=5)

    def __init__(self, x, y, color, player_config):
        """
        Nel costruttore inizializzo i parametri propri del giocatore
        """

        self.width = player_config["width"]
        self.height = player_config["height"]
        self.speed = player_config["speed"]
        self.x = x
        self.y = y
        self.color = color


class TennisCourt:

    """Definizione del campo da gioco"""

    width: int
    length: int

    def __init__(self, court_config):
        self.width = court_config["width"]
        self.length = court_config["height"]

    def draw_court(self, screen: pygame.Surface, color: tuple[int, int, int] = (52, 122, 76)) -> pygame.Rect:
        """Disegna il campo centrato, con il lato lungo orientato verticalmente."""
        court_rect = pygame.Rect(
            (screen.get_width() - self.width) // 2,
            (screen.get_height() - self.length) // 2,
            self.width,
            self.length,
        )
        pygame.draw.rect(screen, color, court_rect)
        return court_rect

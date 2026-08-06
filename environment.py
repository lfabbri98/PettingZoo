"""
Definizione di ambiente e regole del simulatore
"""

from dataclasses import dataclass
import pygame
import yaml
from pathlib import Path
import random

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
    vx: float
    vy: float

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

        self.vx = direction[0] * self.speed
        self.vy = direction[1] * self.speed
        self.x += self.vx * delta_time
        self.y += self.vy * delta_time

    def draw(self, screen: pygame.Surface) -> None:
        pygame.draw.rect(screen, self.color, self.rect, border_radius=5)

    def keep_inside(self, bounds: pygame.Rect) -> None:
        self.x = max(bounds.left + self.width/2, min(self.x, bounds.right-self.width/2))
        self.y = max(bounds.top + self.height / 2, min(self.y, bounds.bottom - self.height / 2))

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
        self.vx = 0
        self.vy = 0


class Ball:
    """Pallina, descritta dalla posizione e dalle componenti della velocità."""

    x: float
    y: float
    vx: float
    vy: float

    def __init__(self, x: float, y: float, vx: float, vy: float):
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self._players_in_contact: set[int] = set()

    def move(self, delta_time: float) -> None:
        """Aggiorna la posizione della pallina in base alla sua velocità."""
        self.x += self.vx * delta_time
        self.y += self.vy * delta_time

    def keep_inside(self, bounds: pygame.Rect) -> None:
        """Fa rimbalzare la pallina sui bordi laterali del campo."""
        if self.x <= bounds.left:
            self.x = bounds.left
            self.vx = abs(self.vx)
        elif self.x >= bounds.right:
            self.x = bounds.right
            self.vx = -abs(self.vx)

    def check_point(self, bounds: pygame.Rect) -> str | None:
        """Restituisce il giocatore che segna se la pallina supera il fondo campo."""
        if self.y < bounds.top:
            return "bottom"
        if self.y > bounds.bottom:
            return "top"
        return None

    def reset(self, bounds: pygame.Rect) -> None:
        """Rimette la pallina al centro con una nuova velocità casuale."""
        self.x = bounds.centerx
        self.y = bounds.centery
        self.vx = random.choice((-1, 1)) * random.randint(80, 160)
        self.vy = random.choice((-1, 1)) * random.randint(80, 160)
        self._players_in_contact.clear()

    def hit_by(self, player: Player) -> bool:
        """Applica il colpo del giocatore se la pallina entra nel suo rettangolo.

        La componente orizzontale della pallina riceve la velocità orizzontale
        del giocatore, mentre quella verticale cambia sempre verso.
        """
        player_id = id(player)
        if not player.rect.collidepoint(round(self.x), round(self.y)):
            self._players_in_contact.discard(player_id)
            return False

        if player_id in self._players_in_contact:
            return False

        self.vx += player.vx
        self.vy = -self.vy
        self._players_in_contact.add(player_id)
        return True


class TennisCourt:

    """Definizione del campo da gioco"""

    width: int
    length: int
    top_score: int
    bottom_score: int

    def __init__(self, court_config):
        self.width = court_config["width"]
        self.length = court_config["height"]
        self.top_score = 0
        self.bottom_score = 0

    def add_point(self, scorer: str) -> None:
        """Assegna un punto al giocatore indicato."""
        if scorer == "top":
            self.top_score += 1
        elif scorer == "bottom":
            self.bottom_score += 1
        else:
            raise ValueError(f"Giocatore non valido: {scorer}")

    def draw_court(
        self,
        screen: pygame.Surface,
        color: tuple[int, int, int] = (52, 122, 76),
        line_color: tuple[int, int, int] = (245, 245, 240),
    ) -> pygame.Rect:
        """Disegna il campo centrato, con il lato lungo orientato verticalmente."""
        court_rect = pygame.Rect(
            (screen.get_width() - self.width) // 2,
            (screen.get_height() - self.length) // 2,
            self.width,
            self.length,
        )
        pygame.draw.rect(screen, color, court_rect)

        line_width = 3
        center_y = court_rect.centery
        top_service_y = court_rect.top + self.length // 4
        bottom_service_y = court_rect.bottom - self.length // 4

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
        return court_rect

"""
Definizione di ambiente e regole del simulatore
"""

from dataclasses import dataclass
import math
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
    max_speed: int #px/s
    max_shot_angle: float
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

    def move(self, direction: tuple[float, float], delta_time: float) -> None:
        """Muove il giocatore secondo una direzione normalizzata.

        Ogni componente della direzione può variare tra -1 e 1. Il suo modulo
        non può superare 1, così la velocità risultante non supera
        ``max_speed``.
        """
        if len(direction) != 2 or math.hypot(*direction) > 1:
            raise ValueError(
                f"Azione non valida: {direction}. Il modulo deve essere al massimo 1."
            )

        self.vx = direction[0] * self.max_speed
        self.vy = direction[1] * self.max_speed
        self.x += self.vx * delta_time
        self.y += self.vy * delta_time

    def reset(self) -> None:
        """Riporta il giocatore alla posizione iniziale e ne azzera il movimento."""
        self.x = self.initial_x
        self.y = self.initial_y
        self.vx = 0
        self.vy = 0

    def choose_shot(self, force: float, angle: float) -> None:
        """Imposta il prossimo colpo.

        ``force`` è espressa in pixel al secondo; ``angle`` è espresso in gradi,
        con 0° per un colpo dritto e valori positivi verso destra.
        """
        if force < 0:
            raise ValueError("La forza del colpo non può essere negativa.")
        if not -self.max_shot_angle <= angle <= self.max_shot_angle:
            raise ValueError(
                "L'angolo del colpo deve essere compreso tra "
                f"-{self.max_shot_angle} e {self.max_shot_angle} gradi."
            )

        self.shot_force = force
        self.shot_angle = angle

    def draw(self, screen: pygame.Surface) -> None:
        pygame.draw.rect(screen, self.color, self.rect, border_radius=5)

    def keep_inside(self, bounds: pygame.Rect) -> None:
        self.x = max(bounds.left + self.width/2, min(self.x, bounds.right-self.width/2))
        self.y = max(bounds.top + self.height / 2, min(self.y, bounds.bottom - self.height / 2))

    def keep_in_half(self, bounds: pygame.Rect, side: str) -> None:
        """Mantiene il giocatore nella metà alta o bassa del campo."""
        self.keep_inside(bounds)

        if side == "top":
            self.y = min(self.y, bounds.centery - self.height / 2)
        elif side == "bottom":
            self.y = max(self.y, bounds.centery + self.height / 2)

    def __init__(self, x, y, color, player_config):
        """
        Nel costruttore inizializzo i parametri propri del giocatore
        """

        self.width = player_config["width"]
        self.height = player_config["height"]
        self.max_speed = player_config["max_speed"]
        self.max_shot_angle = player_config.get("max_shot_angle", 89)
        if not 0 < self.max_shot_angle <= 89:
            raise ValueError("max_shot_angle deve essere compreso tra 0 e 89 gradi.")
        self.x = x
        self.y = y
        self.initial_x = x
        self.initial_y = y
        self.color = color
        self.vx = 0
        self.vy = 0
        self.shot_force = 0
        self.shot_angle = 0


class Ball:
    """Pallina, descritta dalla posizione e dalle componenti della velocità."""

    x: float
    y: float
    vx: float
    vy: float
    max_speed: float | None

    def __init__(
        self,
        x: float,
        y: float,
        vx: float,
        vy: float,
        max_speed: float | None = None,
    ):
        if max_speed is not None and max_speed <= 0:
            raise ValueError("La velocità massima della pallina deve essere positiva.")
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.max_speed = max_speed
        self._players_in_contact: set[int] = set()

    def limit_speed(self) -> None:
        """Limita il modulo della velocità della pallina, se configurato."""
        if self.max_speed is None:
            return

        speed = math.hypot(self.vx, self.vy)
        if speed > self.max_speed:
            scale = self.max_speed / speed
            self.vx *= scale
            self.vy *= scale

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
        self.limit_speed()
        self._players_in_contact.clear()

    def hit_by(self, player: Player) -> bool:
        """Applica il colpo del giocatore se la pallina entra nel suo rettangolo.

        La forza scelta dal giocatore viene trasformata in un vettore secondo
        l'angolo scelto e sommata alla sua velocità. La componente verticale
        della pallina viene sempre rivolta verso l'altra metà campo.
        """
        player_id = id(player)
        if not player.rect.collidepoint(round(self.x), round(self.y)):
            self._players_in_contact.discard(player_id)
            return False

        if player_id in self._players_in_contact:
            return False

        angle_radians = math.radians(player.shot_angle)
        shot_vx = player.shot_force * math.sin(angle_radians)
        shot_vy = player.shot_force * math.cos(angle_radians)
        outgoing_direction = -1 if self.vy > 0 else 1

        self.vx += player.vx + shot_vx
        self.vy = outgoing_direction * (abs(self.vy) + shot_vy) + player.vy
        self.limit_speed()
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

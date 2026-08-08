"""
Definizione di ambiente e regole del simulatore
"""

import math
import yaml
from pathlib import Path
import random
from dataclasses import dataclass

@dataclass
class BoundingBox:
    left: float
    top: float
    width: float
    height: float

    @property
    def right(self) -> float:
        return self.left + self.width

    @property
    def bottom(self) -> float:
        return self.top + self.height

    @property
    def centerx(self) -> float:
        return self.left + self.width / 2

    @property
    def centery(self) -> float:
        return self.top + self.height / 2

    def collidepoint(self, px: float, py: float) -> bool:
        return self.left <= px <= self.right and self.top <= py <= self.bottom


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
    def rect(self) -> BoundingBox:
        return BoundingBox(
            self.x - self.width / 2,
            self.y - self.height / 2,
            self.width,
            self.height,
        )

    def move(self, direction: tuple[float, float], delta_time: float) -> None:
        """Muove il giocatore secondo una direzione normalizzata.

        Ogni componente della direzione può variare tra -1 e 1. Il suo modulo
        non può superare 1, così la velocità risultante non supera
        ``max_speed``.
        """
        if len(direction) != 2:
            raise ValueError(
                f"Azione non valida: {direction}. Il modulo deve essere al massimo 1."
            )
        direction_length = math.hypot(*direction)
        if direction_length > 1 + 1e-9:
            raise ValueError(
                f"Azione non valida: {direction}. Il modulo deve essere al massimo 1."
            )
        if direction_length > 1:
            direction = (
                direction[0] / direction_length,
                direction[1] / direction_length,
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

    def keep_inside(self, bounds: BoundingBox) -> None:
        self.x = max(bounds.left + self.width/2, min(self.x, bounds.right-self.width/2))
        self.y = max(bounds.top + self.height / 2, min(self.y, bounds.bottom - self.height / 2))

    def keep_in_half(self, bounds: BoundingBox, side: str) -> None:
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

    def keep_inside(self, bounds: BoundingBox) -> None:
        """Fa rimbalzare la pallina sui bordi laterali del campo."""
        if self.x <= bounds.left:
            self.x = bounds.left
            self.vx = abs(self.vx)
        elif self.x >= bounds.right:
            self.x = bounds.right
            self.vx = -abs(self.vx)

    def check_point(self, bounds: BoundingBox) -> str | None:
        """Restituisce il giocatore che segna se la pallina supera il fondo campo."""
        if self.y < bounds.top:
            return "bottom"
        if self.y > bounds.bottom:
            return "top"
        return None

    def reset(self, top_player: Player, bottom_player: Player) -> str:
        """Prepara un nuovo punto assegnando casualmente il servizio.

        La pallina viene posizionata sul giocatore che serve e resta ferma:
        sarà il suo primo colpo ad avviare lo scambio. Restituisce ``"top"``
        oppure ``"bottom"``, cioè il lato del servitore.
        """
        server_side = random.choice(("top", "bottom"))
        server = top_player if server_side == "top" else bottom_player
        self.x = server.x
        self.y = server.y
        self.vx = 0
        self.vy = 0
        self._players_in_contact.clear()
        return server_side

    def hit_by(self, player: Player, side: str | None = None) -> bool:
        """Applica il colpo del giocatore se la pallina entra nel suo rettangolo.

        La velocità in arrivo e la forza scelta dal giocatore determinano il
        modulo della nuova velocità. L'angolo scelto ne determina entrambe le
        componenti, mentre il lato del giocatore orienta la pallina verso
        l'altra metà campo.
        """
        player_id = id(player)
        if not player.rect.collidepoint(self.x, self.y):
            self._players_in_contact.discard(player_id)
            return False

        if player_id in self._players_in_contact:
            return False

        angle_radians = math.radians(player.shot_angle)
        if side is not None and side not in ("top", "bottom"):
            raise ValueError(f"Lato giocatore non valido: {side}")
        outgoing_direction = (
            1 if side == "top" else -1 if side == "bottom" else -1 if self.vy > 0 else 1
        )
        outgoing_speed = math.hypot(self.vx, self.vy) + player.shot_force
        self.vx = outgoing_speed * math.sin(angle_radians)
        self.vy = outgoing_direction * outgoing_speed * math.cos(angle_radians)
        self.limit_speed()
        self._players_in_contact.add(player_id)
        return True


def player_is_behind_ball(player: Player, ball: Ball, side: str) -> bool:
    """Restituisce se il giocatore è nella posizione corretta per colpire."""
    if side == "top":
        return player.y <= ball.y
    if side == "bottom":
        return player.y >= ball.y
    raise ValueError(f"Lato giocatore non valido: {side}")


class TennisCourt:

    """Definizione del campo da gioco"""

    width: float
    length: float
    x: float
    y: float
    top_score: int
    bottom_score: int

    def __init__(self, court_config, x: float, y: float):
        self.width = court_config["width"]
        self.length = court_config["height"]
        self.x = x
        self.y = y
        self.top_score = 0
        self.bottom_score = 0

    @property
    def bounds(self) -> BoundingBox:
        return BoundingBox(self.x, self.y, self.width, self.length)

    def add_point(self, scorer: str) -> None:
        """Assegna un punto al giocatore indicato."""
        if scorer == "top":
            self.top_score += 1
        elif scorer == "bottom":
            self.bottom_score += 1
        else:
            raise ValueError(f"Giocatore non valido: {scorer}")

    def reset_score(self) -> None:
        """Azzera il punteggio per iniziare una nuova partita."""
        self.top_score = 0
        self.bottom_score = 0

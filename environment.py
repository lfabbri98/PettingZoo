"""
Definizione di ambiente e regole del simulatore
"""

import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


Config = dict[str, Any]


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


def _positive_number(config: Config, section: str, key: str) -> float:
    try:
        value = config[section][key]
    except (KeyError, TypeError) as error:
        raise ValueError(f"Parametro mancante: {section}.{key}") from error
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        raise ValueError(f"{section}.{key} deve essere un numero positivo.")
    return float(value)


def validate_parameters(config: object) -> Config:
    """Valida la configurazione prima di costruire il simulatore."""
    if not isinstance(config, dict):
        raise ValueError("La configurazione deve essere un dizionario YAML.")

    for section in ("window", "court", "ball", "player", "colors"):
        if not isinstance(config.get(section), dict):
            raise ValueError(
                f"Sezione di configurazione mancante o non valida: {section}"
            )

    window_width = _positive_number(config, "window", "width")
    window_height = _positive_number(config, "window", "height")
    _positive_number(config, "window", "fps")
    if not isinstance(config["window"].get("title"), str):
        raise ValueError("window.title deve essere una stringa.")

    court_width = _positive_number(config, "court", "width")
    court_height = _positive_number(config, "court", "height")
    if court_width > window_width or court_height > window_height:
        raise ValueError("Il campo deve essere contenuto nella finestra.")

    max_ball_speed = _positive_number(config, "ball", "max_speed")
    min_shot_speed = _positive_number(config, "ball", "min_shot_speed")
    if min_shot_speed >= max_ball_speed:
        raise ValueError("ball.min_shot_speed deve essere inferiore a ball.max_speed.")

    player_width = _positive_number(config, "player", "width")
    player_height = _positive_number(config, "player", "height")
    _positive_number(config, "player", "max_speed")
    max_shot_angle = _positive_number(config, "player", "max_shot_angle")
    if max_shot_angle > 89:
        raise ValueError("player.max_shot_angle deve essere al massimo 89 gradi.")
    if player_width > court_width or player_height * 2 > court_height:
        raise ValueError(
            "Le dimensioni dei giocatori non sono compatibili con il campo."
        )

    for name in ("court", "lines", "player_top", "player_bottom", "background"):
        color = config["colors"].get(name)
        if (
            not isinstance(color, list)
            or len(color) != 3
            or any(
                isinstance(channel, bool)
                or not isinstance(channel, int)
                or not 0 <= channel <= 255
                for channel in color
            )
        ):
            raise ValueError(
                f"colors.{name} deve contenere tre interi compresi tra 0 e 255."
            )

    return config


def parse_parameters(config_name: str | Path) -> Config:
    """Carica e valida i parametri YAML del simulatore."""
    with Path(config_name).open("r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)
    return validate_parameters(config)


class Player:
    """Definizione del giocatore"""

    x: float
    y: float
    color: tuple[int, int, int]
    width: int  # px
    height: int  # px
    max_speed: int  # px/s
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

        ``force`` è la velocità d'uscita desiderata, espressa in pixel al
        secondo. ``angle`` è espresso in gradi, con 0° per un colpo dritto e
        valori positivi verso destra.
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
        self.x = max(
            bounds.left + self.width / 2,
            min(self.x, bounds.right - self.width / 2),
        )
        self.y = max(
            bounds.top + self.height / 2,
            min(self.y, bounds.bottom - self.height / 2),
        )

    def keep_in_half(self, bounds: BoundingBox, side: str) -> None:
        """Mantiene il giocatore nella metà alta o bassa del campo."""
        if side not in ("top", "bottom"):
            raise ValueError(f"Lato giocatore non valido: {side}")
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
        if self.width <= 0 or self.height <= 0 or self.max_speed <= 0:
            raise ValueError(
                "Dimensioni e velocità del giocatore devono essere positive."
            )
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
        # Posizione precedente: serve per non perdere collisioni quando la
        # pallina attraversa un giocatore in un singolo tick fisico.
        self.previous_x = x
        self.previous_y = y
        self._moved_since_last_collision = False
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
        self.previous_x = self.x
        self.previous_y = self.y
        self._moved_since_last_collision = True
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

    def reset(
        self,
        top_player: Player,
        bottom_player: Player,
        rng: random.Random | None = None,
    ) -> str:
        """Prepara un nuovo punto assegnando casualmente il servizio.

        La pallina viene posizionata sul giocatore che serve e resta ferma:
        sarà il suo primo colpo ad avviare lo scambio. Restituisce ``"top"``
        oppure ``"bottom"``, cioè il lato del servitore.
        """
        random_source = rng if rng is not None else random
        server_side = random_source.choice(("top", "bottom"))
        server = top_player if server_side == "top" else bottom_player
        self.x = server.x
        self.y = server.y
        self.previous_x = self.x
        self.previous_y = self.y
        self._moved_since_last_collision = False
        self.vx = 0
        self.vy = 0
        self._players_in_contact.clear()
        return server_side

    def hit_by(self, player: Player, side: str | None = None) -> bool:
        """Applica il colpo del giocatore se la pallina entra nel suo rettangolo.

        La forza scelta dal giocatore è la velocità d'uscita desiderata.
        L'angolo ne determina le componenti, mentre il lato del giocatore
        orienta la pallina verso l'altra metà campo.
        """
        player_id = id(player)
        if not self._crossed_player(player.rect):
            self._players_in_contact.discard(player_id)
            self._moved_since_last_collision = False
            return False

        if player_id in self._players_in_contact:
            self._moved_since_last_collision = False
            return False

        angle_radians = math.radians(player.shot_angle)
        if side is not None and side not in ("top", "bottom"):
            raise ValueError(f"Lato giocatore non valido: {side}")
        if side == "top":
            outgoing_direction = 1
        elif side == "bottom":
            outgoing_direction = -1
        else:
            outgoing_direction = -1 if self.vy > 0 else 1
        outgoing_speed = player.shot_force
        self.vx = outgoing_speed * math.sin(angle_radians)
        self.vy = outgoing_direction * outgoing_speed * math.cos(angle_radians)
        self.limit_speed()
        self._players_in_contact.add(player_id)
        self._moved_since_last_collision = False
        return True

    def _crossed_player(self, player_rect: BoundingBox) -> bool:
        """Restituisce se l'ultimo spostamento interseca ``player_rect``.

        Il controllo puntuale della sola posizione finale fa attraversare la
        pallina un giocatore quando il tick è più lungo dell'altezza del suo
        rettangolo. L'algoritmo di Liang-Barsky verifica invece il segmento
        compreso fra la posizione precedente e quella corrente.
        """
        if (
            not self._moved_since_last_collision
            or player_rect.collidepoint(self.x, self.y)
        ):
            return player_rect.collidepoint(self.x, self.y)

        delta_x = self.x - self.previous_x
        delta_y = self.y - self.previous_y
        parameters = (
            (-delta_x, self.previous_x - player_rect.left),
            (delta_x, player_rect.right - self.previous_x),
            (-delta_y, self.previous_y - player_rect.top),
            (delta_y, player_rect.bottom - self.previous_y),
        )
        entry_time = 0.0
        exit_time = 1.0
        for direction, distance in parameters:
            if direction == 0:
                if distance < 0:
                    return False
                continue
            boundary_time = distance / direction
            if direction < 0:
                entry_time = max(entry_time, boundary_time)
            else:
                exit_time = min(exit_time, boundary_time)
            if entry_time > exit_time:
                return False
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

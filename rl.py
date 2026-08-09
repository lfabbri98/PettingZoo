"""Interfacce e policy di supporto per il reinforcement learning.

Questo modulo traduce lo stato del simulatore in input per il trainer e
contiene le policy di supporto.
"""

from dataclasses import dataclass
import math
import random

from environment import Ball, BoundingBox, Player


@dataclass(frozen=True)
class PlayerAction:
    """
    Azione prodotta da una policy per un singolo frame.
    Viene tipizzata come classe per migliorare la leggibilità
    """

    direction: tuple[float, float]
    shot_force: float
    shot_angle: float


def observation(
    agent: Player,
    opponent: Player,
    ball: Ball,
    court_bounds: BoundingBox,
    agent_side: str,
    *,
    agent_score: int = 0,
    opponent_score: int = 0,
    points_to_win: int = 11,
    active_side: str | None = None,
) -> tuple[float, ...]:
    """Restituisce lo stato numerico visto dal punto di vista dell'agente.

    Il campo viene ribaltato verticalmente per l'agente in alto: per la policy
    l'agente gioca sempre nella metà bassa. Le coordinate sono normalizzate in
    ``[0, 1]`` e le velocità rispetto alle rispettive velocità massime. Gli
    ultimi tre valori rappresentano punteggio dell'agente, punteggio
    dell'avversario e turno attivo dal punto di vista dell'agente.
    """
    if agent_side not in ("top", "bottom"):
        raise ValueError(f"Lato giocatore non valido: {agent_side}")
    if court_bounds.width <= 0 or court_bounds.height <= 0:
        raise ValueError("Le dimensioni del campo devono essere positive.")
    if ball.max_speed is None:
        raise ValueError("Per l'osservazione RL, ball.max_speed deve essere configurata.")
    if points_to_win <= 0 or agent_score < 0 or opponent_score < 0:
        raise ValueError("Punteggi e soglia di vittoria non validi.")
    if active_side is not None and active_side not in ("top", "bottom"):
        raise ValueError(f"Lato giocatore attivo non valido: {active_side}")

    def clamp(value: float, lower: float, upper: float) -> float:
        return max(lower, min(value, upper))

    #Normalizziamo le coordinate
    def normalized_x(x: float) -> float:
        return clamp((x - court_bounds.left) / court_bounds.width, 0.0, 1.0)

    def normalized_y(y: float) -> float:
        value = (y - court_bounds.top) / court_bounds.height
        #Rovescio il campo se l'agente è il giocatore top. In questo modo non ho due
        #situazioni separate da fare imparare all'agente
        value = 1 - value if agent_side == "top" else value
        return clamp(value, 0.0, 1.0)

    vertical_sign = -1 if agent_side == "top" else 1
    if active_side is None:
        active_value = 0.0
    else:
        active_value = 1.0 if active_side == agent_side else -1.0
    return (
        normalized_x(agent.x),
        normalized_y(agent.y),
        agent.vx / agent.max_speed,
        vertical_sign * agent.vy / agent.max_speed,
        normalized_x(opponent.x),
        normalized_y(opponent.y),
        opponent.vx / opponent.max_speed,
        vertical_sign * opponent.vy / opponent.max_speed,
        normalized_x(ball.x),
        normalized_y(ball.y),
        ball.vx / ball.max_speed,
        vertical_sign * ball.vy / ball.max_speed,
        min(agent_score / points_to_win, 1.0),
        min(opponent_score / points_to_win, 1.0),
        active_value,
    )


def classic_policy(
    player: Player,
    ball: Ball,
    court_bounds: BoundingBox,
    side: str,
    is_active: bool = True,
    is_serving: bool = False,
    rng: random.Random | None = None,
) -> PlayerAction:
    """Avversario classico semplice per le prime fasi di training.

    Il giocatore attivo rincorre la pallina solo quando questa è nella sua metà
    campo. Negli altri casi si posiziona dietro la pallina: ne segue la
    coordinata orizzontale, restando però lontano dalla rete.
    Durante il servizio, l'angolo laterale e la forza variano casualmente.
    """
    if side not in ("top", "bottom"):
        raise ValueError(f"Lato giocatore non valido: {side}")

    ball_is_in_own_half = (
        ball.y <= court_bounds.centery
        if side == "top"
        else ball.y >= court_bounds.centery
    )
    ball_is_escaping_towards_baseline = False
    if is_active and ball_is_in_own_half:
        target_x, target_y = ball.x, ball.y
        distance_x = target_x - player.x
        distance_y = target_y - player.y
        distance = math.hypot(distance_x, distance_y)
        slowdown_distance = min(court_bounds.width, court_bounds.height) * 0.15
        ball_is_escaping_towards_baseline = (
            ball.vy < 0 and distance_y < 0
            if side == "top"
            else ball.vy > 0 and distance_y > 0
        )
        speed_factor = (
            1.0
            if ball_is_escaping_towards_baseline
            else min(1.0, distance / slowdown_distance)
        )
        direction = (
            (0.0, 0.0)
            if distance == 0
            else (
                speed_factor * distance_x / distance,
                speed_factor * distance_y / distance,
            )
        )
    else:
        recovery_margin = court_bounds.height * 0.15
        target_x = ball.x
        target_y = (
            court_bounds.top + recovery_margin
            if side == "top"
            else court_bounds.bottom - recovery_margin
        )

        direction_x = (target_x > player.x) - (target_x < player.x)
        direction_y = (target_y > player.y) - (target_y < player.y)
        direction_length = math.hypot(direction_x, direction_y)
        direction = (
            (0.0, 0.0)
            if direction_length == 0
            else (direction_x / direction_length, direction_y / direction_length)
        )

    random_source = rng if rng is not None else random

    if ball.max_speed is None:
        normal_speed = 80.0
        escaping_speed = 80.0
        serve_speeds = (110.0, 120.0, 130.0)
    else:
        normal_speed = ball.max_speed * 0.8
        escaping_speed = ball.max_speed
        serve_speeds = tuple(ball.max_speed * ratio for ratio in (0.7, 0.85, 1.0))

    if is_serving:
        return PlayerAction(
            direction=direction,
            shot_force=random_source.choice(serve_speeds),
            shot_angle=random_source.uniform(-player.max_shot_angle, player.max_shot_angle),
        )

    if ball_is_escaping_towards_baseline:
        return PlayerAction(direction=direction, shot_force=escaping_speed, shot_angle=0)

    return PlayerAction(
        direction=direction,
        shot_force=normal_speed,
        shot_angle=random_source.uniform(-player.max_shot_angle, player.max_shot_angle),
    )


def _scaled_classic_policy(
    player: Player,
    ball: Ball,
    court_bounds: BoundingBox,
    side: str,
    *,
    is_active: bool,
    is_serving: bool,
    rng: random.Random,
    movement_scale: float,
    force_scale: float,
    angle_scale: float,
) -> PlayerAction:
    """Riduce le capacità della policy classica per il curriculum PPO."""
    action = classic_policy(
        player,
        ball,
        court_bounds,
        side,
        is_active=is_active,
        is_serving=is_serving,
        rng=rng,
    )
    return PlayerAction(
        direction=(
            action.direction[0] * movement_scale,
            action.direction[1] * movement_scale,
        ),
        shot_force=action.shot_force * force_scale,
        shot_angle=action.shot_angle * angle_scale,
    )


def easy_policy(
    player: Player,
    ball: Ball,
    court_bounds: BoundingBox,
    side: str,
    *,
    is_active: bool,
    is_serving: bool,
    rng: random.Random,
) -> PlayerAction:
    """Avversario lento e prevedibile per imparare i primi colpi."""
    return _scaled_classic_policy(
        player,
        ball,
        court_bounds,
        side,
        is_active=is_active,
        is_serving=is_serving,
        rng=rng,
        movement_scale=0.35,
        force_scale=0.55,
        angle_scale=0.35,
    )


def medium_policy(
    player: Player,
    ball: Ball,
    court_bounds: BoundingBox,
    side: str,
    *,
    is_active: bool,
    is_serving: bool,
    rng: random.Random,
) -> PlayerAction:
    """Avversario intermedio prima del passaggio alla policy classica."""
    return _scaled_classic_policy(
        player,
        ball,
        court_bounds,
        side,
        is_active=is_active,
        is_serving=is_serving,
        rng=rng,
        movement_scale=0.65,
        force_scale=0.75,
        angle_scale=0.65,
    )

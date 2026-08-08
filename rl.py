"""Interfacce e policy di supporto per il reinforcement learning.

Questo modulo traduce lo stato del simulatore in input per il trainer e
contiene gli avversari deterministici.
"""

from dataclasses import dataclass
import math
import random

import pygame

from environment import Ball, Player


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
    court_bounds: pygame.Rect,
    agent_side: str,
) -> tuple[float, ...]:
    """Restituisce lo stato numerico visto dal punto di vista dell'agente.

    Il campo viene ribaltato verticalmente per l'agente in alto: per la policy
    l'agente gioca sempre nella metà bassa. Le coordinate sono normalizzate in
    ``[0, 1]`` e le velocità rispetto alle rispettive velocità massime.
    """
    if agent_side not in ("top", "bottom"):
        raise ValueError(f"Lato giocatore non valido: {agent_side}")
    if court_bounds.width <= 0 or court_bounds.height <= 0:
        raise ValueError("Le dimensioni del campo devono essere positive.")
    if ball.max_speed is None:
        raise ValueError("Per l'osservazione RL, ball.max_speed deve essere configurata.")

    #Normalizziamo le coordinate
    def normalized_x(x: float) -> float:
        return (x - court_bounds.left) / court_bounds.width

    def normalized_y(y: float) -> float:
        value = (y - court_bounds.top) / court_bounds.height
        #Rovescio il campo se l'agente è il giocatore top. In questo modo non ho due
        #situazioni separate da fare imparare all'agente
        return 1 - value if agent_side == "top" else value

    vertical_sign = -1 if agent_side == "top" else 1
    #Ritorna una tupla di 12 valori che contengono la situazione attuale del gioco vista da parte dell'agente
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
    )


def classic_policy(
    player: Player,
    ball: Ball,
    court_bounds: pygame.Rect,
    side: str,
    is_active: bool = True,
    is_serving: bool = False,
) -> PlayerAction:
    """Avversario deterministico semplice per le prime fasi di training.

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

    if is_serving:
        return PlayerAction(
            direction=direction,
            shot_force=random.choice((70, 80, 90)),
            shot_angle=random.uniform(-player.max_shot_angle, player.max_shot_angle),
        )

    if ball_is_escaping_towards_baseline:
        return PlayerAction(direction=direction, shot_force=80, shot_angle=0)

    return PlayerAction(
        direction=direction,
        shot_force=80,
        shot_angle=random.uniform(-player.max_shot_angle, player.max_shot_angle),
    )

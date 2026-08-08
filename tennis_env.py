"""Wrapper headless per addestrare un giocatore contro ``classic_policy``."""

from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Any, Sequence

from environment import Ball, Player, TennisCourt, parse_parameters, player_is_behind_ball
from rl import PlayerAction, classic_policy, observation


Observation = tuple[float, ...]
StepResult = tuple[Observation, float, bool, bool, dict[str, Any]]


class TennisEnv:
    """Espone il simulatore tramite ``reset`` e ``step``, senza dipendenze RL.

    Un episodio e' una partita al meglio di 11 punti: ogni punto avvia un nuovo
    scambio e l'episodio termina quando un giocatore raggiunge 11 punti, oppure
    viene troncato al raggiungimento del limite di passi. Il giocatore basso e'
    controllato dall'agente; quello alto usa ``classic_policy``.

    L'azione e' una sequenza di quattro valori:
    ``[move_x, move_y, force, angle]``. ``move_x`` e ``move_y`` appartengono a
    ``[-1, 1]`` e sono normalizzati internamente se formano una diagonale.
    ``force`` e' in ``[0, 1]`` rispetto alla velocita' massima della pallina;
    ``angle`` e' in ``[-1, 1]`` rispetto al massimo angolo configurato.
    """

    action_bounds = ((-1.0, 1.0), (-1.0, 1.0), (0.0, 1.0), (-1.0, 1.0))

    def __init__(
        self,
        config_path: str | Path = "parameters.yml",
        *,
        max_steps_per_episode: int = 2_000,
        points_to_win: int = 11,
    ) -> None:
        if max_steps_per_episode <= 0:
            raise ValueError("max_steps_per_episode deve essere positivo.")
        if points_to_win <= 0:
            raise ValueError("points_to_win deve essere positivo.")

        self.config = parse_parameters(str(config_path))
        self.max_steps_per_episode = max_steps_per_episode
        self.points_to_win = points_to_win
        self.delta_time = 1 / self.config["window"]["fps"]
        self._rng = random.Random()
        self._steps = 0
        self._is_reset = False
        self._episode_finished = False
        self.court: TennisCourt
        self.top_player: Player
        self.bottom_player: Player
        self.ball: Ball
        self.active_player: str

    def reset(self, *, seed: int | None = None) -> tuple[Observation, dict[str, Any]]:
        """Inizia un nuovo episodio e restituisce osservazione e metadati."""
        if seed is not None:
            self._rng.seed(seed)

        window = self.config["window"]
        court_config = self.config["court"]
        court_x = (window["width"] - court_config["width"]) / 2
        court_y = (window["height"] - court_config["height"]) / 2
        self.court = TennisCourt(court_config, court_x, court_y)

        colors = self.config["colors"]
        player_config = self.config["player"]
        player_x = self.court.bounds.centerx
        margin = self.court.length * 0.06
        self.top_player = Player(
            player_x, self.court.y + margin, colors["player_top"], player_config
        )
        self.bottom_player = Player(
            player_x,
            self.court.y + self.court.length - margin,
            colors["player_bottom"],
            player_config,
        )
        self.ball = Ball(0, 0, 0, 0, self.config["ball"]["max_speed"])
        self.active_player = self.ball.reset(
            self.top_player, self.bottom_player, rng=self._rng
        )
        self._steps = 0
        self._is_reset = True
        self._episode_finished = False
        return self._get_observation(), self._get_info()

    def step(self, action: Sequence[float]) -> StepResult:
        """Applica un'azione e avanza il simulatore di un frame."""
        if not self._is_reset:
            raise RuntimeError("Chiama reset() prima di step().")
        if self._episode_finished:
            raise RuntimeError("L'episodio e' terminato: chiama reset().")
        action_values = self._validate_action(action)

        agent_action = self._agent_action(action_values)
        opponent_action = classic_policy(
            self.top_player,
            self.ball,
            self.court.bounds,
            "top",
            is_active=self.active_player == "top",
            is_serving=self._is_serving("top"),
            rng=self._rng,
        )
        self._advance(agent_action, opponent_action)
        self._steps += 1

        scorer = self.ball.check_point(self.court.bounds)
        reward = 0.0
        if scorer is not None:
            self.court.add_point(scorer)
            reward = 1.0 if scorer == "bottom" else -1.0
        terminated = max(self.court.top_score, self.court.bottom_score) >= self.points_to_win
        truncated = not terminated and self._steps >= self.max_steps_per_episode

        info = self._get_info()
        info["scorer"] = scorer
        if scorer is not None and not terminated:
            self.top_player.reset()
            self.bottom_player.reset()
            self.active_player = self.ball.reset(
                self.top_player, self.bottom_player, rng=self._rng
            )
        self._episode_finished = terminated or truncated
        return self._get_observation(), reward, terminated, truncated, info

    def _validate_action(self, action: Sequence[float]) -> tuple[float, float, float, float]:
        if len(action) != 4:
            raise ValueError("Un'azione deve contenere esattamente quattro valori.")
        try:
            values = tuple(float(value) for value in action)
        except (TypeError, ValueError) as error:
            raise ValueError("L'azione deve contenere solo valori numerici.") from error

        for value, (lower, upper) in zip(values, self.action_bounds, strict=True):
            if not math.isfinite(value) or not lower <= value <= upper:
                raise ValueError(f"Azione fuori dai limiti consentiti: {values}")
        return values  # type: ignore[return-value]

    def _agent_action(
        self, action: tuple[float, float, float, float]
    ) -> PlayerAction:
        direction_x, direction_y, force, angle = action
        length = math.hypot(direction_x, direction_y)
        if length > 1:
            direction_x /= length
            direction_y /= length

        max_speed = self.ball.max_speed
        assert max_speed is not None
        return PlayerAction(
            direction=(direction_x, direction_y),
            shot_force=force * max_speed,
            shot_angle=angle * self.bottom_player.max_shot_angle,
        )

    def _advance(self, agent_action: PlayerAction, opponent_action: PlayerAction) -> None:
        #Fa eseguire le azioni ai giocatori
        
        bounds = self.court.bounds
        self.top_player.move(opponent_action.direction, self.delta_time)
        self.bottom_player.move(agent_action.direction, self.delta_time)
        self.top_player.keep_in_half(bounds, "top")
        self.bottom_player.keep_in_half(bounds, "bottom")
        self.ball.move(self.delta_time)

        if self.ball.check_point(bounds) is not None:
            return

        self.ball.keep_inside(bounds)
        self.top_player.choose_shot(
            opponent_action.shot_force, opponent_action.shot_angle
        )
        self.bottom_player.choose_shot(agent_action.shot_force, agent_action.shot_angle)
        if self.active_player == "top" and player_is_behind_ball(
            self.top_player, self.ball, "top"
        ):
            if self.ball.hit_by(self.top_player, "top"):
                self.active_player = "bottom"
        elif self.active_player == "bottom" and player_is_behind_ball(
            self.bottom_player, self.ball, "bottom"
        ):
            if self.ball.hit_by(self.bottom_player, "bottom"):
                self.active_player = "top"

    def _is_serving(self, side: str) -> bool:
        return (
            self.active_player == side and self.ball.vx == 0 and self.ball.vy == 0
        )

    def _get_observation(self) -> Observation:
        return observation(
            self.bottom_player,
            self.top_player,
            self.ball,
            self.court.bounds,
            "bottom",
        )

    def _get_info(self) -> dict[str, Any]:
        return {
            "active_player": self.active_player,
            "scores": (self.court.top_score, self.court.bottom_score),
            "steps": self._steps,
        }

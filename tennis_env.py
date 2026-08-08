"""Wrapper headless per addestrare un giocatore contro ``classic_policy``."""

from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Any, Protocol, Sequence

from environment import (
    Ball,
    BoundingBox,
    Player,
    TennisCourt,
    parse_parameters,
    player_is_behind_ball,
)
from rl import PlayerAction, classic_policy, observation


Observation = tuple[float, ...]
StepResult = tuple[Observation, float, bool, bool, dict[str, Any]]
DEFAULT_CONFIG_PATH = Path(__file__).with_name("parameters.yml")


class OpponentPolicy(Protocol):
    """Firma richiesta a una policy usata come avversario."""

    def __call__(
        self,
        player: Player,
        ball: Ball,
        court_bounds: BoundingBox,
        side: str,
        *,
        is_active: bool,
        is_serving: bool,
        rng: random.Random,
    ) -> PlayerAction: ...


class TennisEnv:
    """Espone il simulatore tramite ``reset`` e ``step``, senza dipendenze RL.

    Un episodio è una partita vinta dal primo giocatore che raggiunge la soglia
    configurata. Ogni punto avvia un nuovo scambio. Il giocatore basso è
    controllato dall'agente; la policy del giocatore alto è iniettabile.

    L'azione è una sequenza di quattro valori:
    ``[move_x, move_y, force, angle]``. ``move_x`` e ``move_y`` appartengono a
    ``[-1, 1]`` e sono normalizzati internamente se formano una diagonale.
    ``force`` è in ``[0, 1]`` e rappresenta la velocità d'uscita desiderata tra
    il minimo e il massimo configurati; ``angle`` è in ``[-1, 1]`` rispetto al
    massimo angolo configurato.
    """

    action_bounds = ((-1.0, 1.0), (-1.0, 1.0), (0.0, 1.0), (-1.0, 1.0))

    def __init__(
        self,
        config_path: str | Path | None = None,
        *,
        max_steps_per_episode: int = 5_000,
        points_to_win: int = 11,
        frame_skip: int = 4,
        opponent_policy: OpponentPolicy = classic_policy,
    ) -> None:
        if max_steps_per_episode <= 0:
            raise ValueError("max_steps_per_episode deve essere positivo.")
        if points_to_win <= 0:
            raise ValueError("points_to_win deve essere positivo.")
        if frame_skip <= 0:
            raise ValueError("frame_skip deve essere positivo.")

        self.config_path = (
            Path(config_path) if config_path is not None else DEFAULT_CONFIG_PATH
        )
        self.config = parse_parameters(self.config_path)
        self.max_steps_per_episode = max_steps_per_episode
        self.points_to_win = points_to_win
        self.frame_skip = frame_skip
        self.opponent_policy = opponent_policy
        self.delta_time = 1 / self.config["window"]["fps"]
        self._rng = random.Random()
        self._steps = 0
        self._physics_steps = 0
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
        self._physics_steps = 0
        self._is_reset = True
        self._episode_finished = False
        return self._get_observation(), self._get_info(scorer=None, winner=None)

    def step(self, action: Sequence[float]) -> StepResult:
        """Applica una decisione per ``frame_skip`` tick fisici."""
        if not self._is_reset:
            raise RuntimeError("Chiama reset() prima di step().")
        if self._episode_finished:
            raise RuntimeError("L'episodio è terminato: chiama reset().")
        action_values = self._validate_action(action)

        agent_action = self._agent_action(action_values)
        scorer = None
        reward = 0.0
        terminated = False
        for _ in range(self.frame_skip):
            self._advance(agent_action, self._opponent_action())
            self._physics_steps += 1
            scorer = self.ball.check_point(self.court.bounds)
            if scorer is not None:
                self.court.add_point(scorer)
                reward = 1.0 if scorer == "bottom" else -1.0
                terminated = (
                    max(self.court.top_score, self.court.bottom_score)
                    >= self.points_to_win
                )
                if not terminated:
                    self._reset_rally()
                break

        self._steps += 1
        truncated = not terminated and self._steps >= self.max_steps_per_episode
        self._episode_finished = terminated or truncated
        winner = scorer if terminated else None
        info = self._get_info(scorer=scorer, winner=winner)
        return self._get_observation(), reward, terminated, truncated, info

    def _opponent_action(self) -> PlayerAction:
        return self.opponent_policy(
            self.top_player,
            self.ball,
            self.court.bounds,
            "top",
            is_active=self.active_player == "top",
            is_serving=self._is_serving("top"),
            rng=self._rng,
        )

    def _reset_rally(self) -> None:
        self.top_player.reset()
        self.bottom_player.reset()
        self.active_player = self.ball.reset(
            self.top_player, self.bottom_player, rng=self._rng
        )

    def _validate_action(
        self, action: Sequence[float]
    ) -> tuple[float, float, float, float]:
        if len(action) != 4:
            raise ValueError("Un'azione deve contenere esattamente quattro valori.")
        try:
            values = tuple(float(value) for value in action)
        except (TypeError, ValueError) as error:
            raise ValueError("L'azione deve contenere solo valori numerici.") from error

        for value, (lower, upper) in zip(values, self.action_bounds, strict=True):
            if not math.isfinite(value) or not lower <= value <= upper:
                raise ValueError(f"Azione fuori dai limiti consentiti: {values}")
        return values[0], values[1], values[2], values[3]

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
        min_speed = self.config["ball"]["min_shot_speed"]
        desired_speed = min_speed + force * (max_speed - min_speed)
        return PlayerAction(
            direction=(direction_x, direction_y),
            shot_force=desired_speed,
            shot_angle=angle * self.bottom_player.max_shot_angle,
        )

    def _advance(
        self, agent_action: PlayerAction, opponent_action: PlayerAction
    ) -> None:
        """Esegue un singolo tick fisico per entrambi i giocatori."""
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
        self.bottom_player.choose_shot(
            agent_action.shot_force, agent_action.shot_angle
        )
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
            agent_score=self.court.bottom_score,
            opponent_score=self.court.top_score,
            points_to_win=self.points_to_win,
            active_side=self.active_player,
        )

    def _get_info(
        self, *, scorer: str | None, winner: str | None
    ) -> dict[str, Any]:
        return {
            "active_player": self.active_player,
            "scores": (self.court.top_score, self.court.bottom_score),
            "steps": self._steps,
            "physics_steps": self._physics_steps,
            "scorer": scorer,
            "winner": winner,
        }

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

    Il reward premia i punti e i colpi riusciti del giocatore basso, e
    penalizza quelli subiti. Aggiunge un bonus alla vittoria, un piccolo costo
    a ogni decisione e, in caso di timeout, un termine proporzionale alla
    differenza punti. Questi ultimi due termini impediscono che un pareggio
    0-0 al timeout sia una strategia conveniente.

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
        points_to_win: int = 5,
        frame_skip: int = 4,
        opponent_policy: OpponentPolicy = classic_policy,
        point_reward: float = 1.0,
        successful_hit_reward: float = 0.002,
        win_reward: float = 5.0,
        step_penalty: float = 0.0002,
        timeout_score_coefficient: float = 0.5,
    ) -> None:
        if max_steps_per_episode <= 0:
            raise ValueError("max_steps_per_episode deve essere positivo.")
        if points_to_win <= 0:
            raise ValueError("points_to_win deve essere positivo.")
        if frame_skip <= 0:
            raise ValueError("frame_skip deve essere positivo.")
        reward_parameters = {
            "point_reward": point_reward,
            "successful_hit_reward": successful_hit_reward,
            "win_reward": win_reward,
            "step_penalty": step_penalty,
            "timeout_score_coefficient": timeout_score_coefficient,
        }
        if not all(math.isfinite(value) and value >= 0 for value in reward_parameters.values()):
            raise ValueError("I parametri del reward devono essere finiti e non negativi.")
        if point_reward == 0:
            raise ValueError("point_reward deve essere positivo.")

        self.config_path = (
            Path(config_path) if config_path is not None else DEFAULT_CONFIG_PATH
        )
        self.config = parse_parameters(self.config_path)
        self.max_steps_per_episode = max_steps_per_episode
        self.points_to_win = points_to_win
        self.frame_skip = frame_skip
        self.opponent_policy = opponent_policy
        self.point_reward = point_reward
        self.successful_hit_reward = successful_hit_reward
        self.win_reward = win_reward
        self.step_penalty = step_penalty
        self.timeout_score_coefficient = timeout_score_coefficient
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
        reward = -self.step_penalty
        terminated = False
        for _ in range(self.frame_skip):
            hitter = self._advance(agent_action, self._opponent_action())
            self._physics_steps += 1
            if hitter == "bottom":
                reward += self.successful_hit_reward
            scorer = self.ball.check_point(self.court.bounds)
            if scorer is not None:
                self.court.add_point(scorer)
                reward += self.point_reward if scorer == "bottom" else -self.point_reward
                terminated = (
                    max(self.court.top_score, self.court.bottom_score)
                    >= self.points_to_win
                )
                if terminated:
                    reward += self.win_reward if scorer == "bottom" else -self.win_reward
                if not terminated:
                    self._reset_rally()
                break

        self._steps += 1
        truncated = not terminated and self._steps >= self.max_steps_per_episode
        if truncated:
            score_difference = self.court.bottom_score - self.court.top_score
            reward += self.timeout_score_coefficient * score_difference
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
    ) -> str | None:
        """Esegue un tick fisico e restituisce chi ha colpito la pallina."""
        bounds = self.court.bounds
        self.top_player.move(opponent_action.direction, self.delta_time)
        self.bottom_player.move(agent_action.direction, self.delta_time)
        self.top_player.keep_in_half(bounds, "top")
        self.bottom_player.keep_in_half(bounds, "bottom")
        self.ball.move(self.delta_time)

        if self.ball.check_point(bounds) is not None:
            return None

        self.ball.keep_inside(bounds)
        self.top_player.choose_shot(
            opponent_action.shot_force, opponent_action.shot_angle
        )
        self.bottom_player.choose_shot(
            agent_action.shot_force, agent_action.shot_angle
        )
        top_is_behind_ball = (
            player_is_behind_ball(self.top_player, self.ball, "top")
            or self.top_player.y <= self.ball.previous_y
        )
        bottom_is_behind_ball = (
            player_is_behind_ball(self.bottom_player, self.ball, "bottom")
            or self.bottom_player.y >= self.ball.previous_y
        )
        if self.active_player == "top" and top_is_behind_ball:
            if self.ball.hit_by(self.top_player, "top"):
                self.active_player = "bottom"
                return "top"
        elif self.active_player == "bottom" and bottom_is_behind_ball:
            if self.ball.hit_by(self.bottom_player, "bottom"):
                self.active_player = "top"
                return "bottom"
        return None

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

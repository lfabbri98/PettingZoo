import pytest
import torch

from model import ACTION_SIZE, OBSERVATION_SIZE, ActorCritic
from rollout import RolloutBuffer


def test_buffer_stores_a_transition_as_plain_tensors() -> None:
    model = ActorCritic()
    state = torch.zeros(OBSERVATION_SIZE)
    sample = model.act(state)
    buffer = RolloutBuffer()

    buffer.add(
        state=state,
        raw_action=sample.raw_action,
        reward=1.0,
        terminated=True,
        truncated=False,
        value=sample.value,
        next_value=torch.tensor(0.0),
        log_prob=sample.log_prob,
    )
    batch = buffer.as_batch()

    assert len(buffer) == 1
    assert batch.states.shape == (1, OBSERVATION_SIZE)
    assert batch.raw_actions.shape == (1, ACTION_SIZE)
    assert batch.rewards.tolist() == [1.0]
    assert batch.terminated.tolist() == [True]
    assert batch.truncated.tolist() == [False]
    assert batch.values.shape == (1,)
    assert batch.next_values.shape == (1,)
    assert batch.log_probs.shape == (1,)


def test_buffer_rejects_wrong_state_shape() -> None:
    buffer = RolloutBuffer()

    with pytest.raises(ValueError, match="stato"):
        buffer.add(
            state=torch.zeros(2),
            raw_action=torch.zeros(ACTION_SIZE),
            reward=0.0,
            terminated=False,
            truncated=False,
            value=torch.tensor(0.0),
            next_value=torch.tensor(0.0),
            log_prob=torch.tensor(0.0),
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("state", torch.full((OBSERVATION_SIZE,), float("nan"))),
        ("raw_action", torch.full((ACTION_SIZE,), float("inf"))),
        ("value", torch.tensor(float("nan"))),
        ("next_value", torch.tensor(float("inf"))),
        ("log_prob", torch.tensor(float("nan"))),
    ],
)
def test_buffer_rejects_non_finite_tensors(field: str, value: torch.Tensor) -> None:
    transition = {
        "state": torch.zeros(OBSERVATION_SIZE),
        "raw_action": torch.zeros(ACTION_SIZE),
        "reward": 0.0,
        "terminated": False,
        "truncated": False,
        "value": torch.tensor(0.0),
        "next_value": torch.tensor(0.0),
        "log_prob": torch.tensor(0.0),
    }
    transition[field] = value

    with pytest.raises(ValueError, match="finiti"):
        RolloutBuffer().add(**transition)


def test_compute_gae_uses_terminal_reward_and_resets_the_episode() -> None:
    buffer = RolloutBuffer()
    state = torch.zeros(OBSERVATION_SIZE)
    action = torch.zeros(ACTION_SIZE)

    buffer.add(
        state=state,
        raw_action=action,
        reward=0.0,
        terminated=False,
        truncated=False,
        value=torch.tensor(0.2),
        next_value=torch.tensor(0.4),
        log_prob=torch.tensor(0.0),
    )
    buffer.add(
        state=state,
        raw_action=action,
        reward=1.0,
        terminated=True,
        truncated=False,
        value=torch.tensor(0.4),
        next_value=torch.tensor(0.0),
        log_prob=torch.tensor(0.0),
    )

    targets = buffer.compute_gae(gamma=1.0, gae_lambda=1.0)

    assert targets.advantages.tolist() == pytest.approx([0.8, 0.6])
    assert targets.returns.tolist() == pytest.approx([1.0, 1.0])


def test_compute_gae_bootstraps_at_a_timeout_without_crossing_episodes() -> None:
    buffer = RolloutBuffer()
    buffer.add(
        state=torch.zeros(OBSERVATION_SIZE),
        raw_action=torch.zeros(ACTION_SIZE),
        reward=0.0,
        terminated=False,
        truncated=True,
        value=torch.tensor(0.2),
        next_value=torch.tensor(0.7),
        log_prob=torch.tensor(0.0),
    )

    targets = buffer.compute_gae(gamma=0.99, gae_lambda=0.95)

    assert targets.advantages.tolist() == pytest.approx([0.493])
    assert targets.returns.tolist() == pytest.approx([0.693])

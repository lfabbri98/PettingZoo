import pytest

torch = pytest.importorskip("torch")

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
        done=True,
        value=sample.value,
        log_prob=sample.log_prob,
    )
    batch = buffer.as_batch()

    assert len(buffer) == 1
    assert batch.states.shape == (1, OBSERVATION_SIZE)
    assert batch.raw_actions.shape == (1, ACTION_SIZE)
    assert batch.rewards.tolist() == [1.0]
    assert batch.dones.tolist() == [True]
    assert batch.values.shape == (1,)
    assert batch.log_probs.shape == (1,)


def test_buffer_rejects_wrong_state_shape() -> None:
    buffer = RolloutBuffer()

    with pytest.raises(ValueError, match="stato"):
        buffer.add(
            state=torch.zeros(2),
            raw_action=torch.zeros(ACTION_SIZE),
            reward=0.0,
            done=False,
            value=torch.tensor(0.0),
            log_prob=torch.tensor(0.0),
        )

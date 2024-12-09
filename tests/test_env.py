import numpy as np
import pytest
from env.racing_env import RacingEnv


@pytest.fixture
def env():
    e = RacingEnv()
    yield e
    e.close()


def test_reset_obs_shape(env):
    obs, info = env.reset()
    assert obs.shape == (11,)  # 2 + 9 ray sensors


def test_step_returns_correct_types(env):
    env.reset()
    obs, reward, terminated, truncated, info = env.step(np.array([0.0, 1.0]))
    assert obs.shape == (11,)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info, dict)


def test_off_track_terminates(env):
    obs, _ = env.reset()
    # Drive straight off the track
    for _ in range(200):
        obs, reward, terminated, truncated, info = env.step(np.array([0.0, 1.0]))
        if terminated:
            break
    # Eventually should terminate
    assert terminated or truncated

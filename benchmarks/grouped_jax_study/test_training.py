"""Checks for the SAC update and the replay buffer used by both engines."""

import numpy as np


def test_sac_changes_policy_and_keeps_actions_in_range():
  import jax
  import jax.numpy as jnp

  from benchmarks.grouped_jax_study import sac

  key = jax.random.PRNGKey(7)
  state = sac.initialize(5, 4, key)
  observation = jnp.ones((12, 5), dtype=jnp.float32)
  key, draw, learn = jax.random.split(key, 3)
  action = sac.act(state.actor, observation, jax.random.split(draw, len(observation)))
  new, losses = sac.update(state, sac.Transition(
      observation, action, jnp.full(12, -1.0, dtype=jnp.float32),
      observation, jnp.zeros(12, dtype=jnp.float32),
  ), learn)
  assert action.shape == (12, 4)
  assert np.all(np.abs(action) <= 1)
  assert not np.array_equal(action[0], action[1])
  assert np.isfinite(np.asarray(losses)).all()
  assert not np.array_equal(new.actor[-1][0], state.actor[-1][0])
  assert not np.array_equal(new.critic[-1][0], state.critic[-1][0])


def test_replay_wrap_retains_the_latest_complete_transitions():
  from benchmarks.grouped_jax_study import sac
  from benchmarks.grouped_jax_study.train import Replay

  replay = Replay(5, 2, 1)
  for offset in (0, 3):
    x = np.arange(offset, offset + 3, dtype=np.float32)
    obs = np.stack((x, -x), axis=1)
    replay.add(sac.Transition(obs, x[:, None], x, obs + 1, x == 5))
  obs, _, reward, next_obs, done = replay.sample(80, np.random.default_rng(10))
  assert set(reward) == {1, 2, 3, 4, 5}
  np.testing.assert_equal(obs[:, 0], reward)
  np.testing.assert_equal(next_obs, obs + 1)
  np.testing.assert_equal(done, reward == 5)

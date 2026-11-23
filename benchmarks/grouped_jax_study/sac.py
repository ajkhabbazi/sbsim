"""JAX soft actor-critic with twin critics and a tuned entropy temperature."""

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax.example_libraries import optimizers


_init_optimizer, _optimize, _parameters = optimizers.adam(3e-4)


class Transition(NamedTuple):
  observation: jax.Array
  action: jax.Array
  reward: jax.Array
  next_observation: jax.Array
  terminal: jax.Array


class State(NamedTuple):
  actor_optimizer: optimizers.OptimizerState
  critic_optimizer: optimizers.OptimizerState
  entropy_optimizer: optimizers.OptimizerState
  target: tuple
  updates: jax.Array

  @property
  def actor(self):
    return _parameters(self.actor_optimizer)

  @property
  def critic(self):
    return _parameters(self.critic_optimizer)


def _layers(key, widths):
  result = []
  for index, (source, target) in enumerate(zip(widths, widths[1:])):
    key, draw = jax.random.split(key)
    scale = (2 / source)**.5 if index < len(widths) - 2 else .003
    weights = jax.random.normal(draw, (source, target), jnp.float32) * scale
    result.append((weights, jnp.zeros(target, jnp.float32)))
  return tuple(result)


def initialize(observation_size, action_size, key):
  actor_key, first_critic, second_critic = jax.random.split(key, 3)
  critic_widths = (observation_size + action_size, 256, 256, 1)
  double_critic = jax.tree.map(
      lambda first, second: jnp.stack((first, second)),
      _layers(first_critic, critic_widths), _layers(second_critic, critic_widths),
  )
  return State(
      _init_optimizer(_layers(actor_key, (observation_size, 256, 256, 2 * action_size))),
      _init_optimizer(double_critic),
      _init_optimizer(jnp.log(jnp.asarray(.2, dtype=jnp.float32))),
      double_critic, jnp.asarray(0, dtype=jnp.int32),
  )


def _network(parameters, inputs):
  for weights, bias in parameters[:-1]:
    inputs = jax.nn.relu(inputs @ weights + bias)
  weights, bias = parameters[-1]
  return inputs @ weights + bias


def _draw_policy(actor, observation, keys):
  mean, log_std = jnp.split(_network(actor, observation), 2, axis=-1)
  log_std = jnp.clip(log_std, -5, 2)
  noise = jax.vmap(lambda k: jax.random.normal(k, (mean.shape[-1],), mean.dtype))(keys)
  latent = mean + jnp.exp(log_std) * noise
  action = jnp.tanh(latent)
  # Stable log(1 - tanh(x)**2), including near policy saturation.
  jacobian = 2 * (jnp.log(2.) - latent - jax.nn.softplus(-2 * latent))
  log_probability = jnp.sum(
      -.5 * (noise**2 + jnp.log(2 * jnp.pi)) - log_std - jacobian, axis=-1
  )
  return action, log_probability


@jax.jit
def act(actor, observation, keys):
  action, _ = _draw_policy(actor, observation, keys)
  return action


def _values(critic, observation, action):
  inputs = jnp.concatenate((observation, action), axis=-1)
  return jax.vmap(_network, in_axes=(0, None))(critic, inputs)[..., 0]


@jax.jit
def update(state, batch, key):
  target_key, policy_key = jax.random.split(key)
  row_count, action_size = batch.action.shape
  target_keys = jax.random.split(target_key, row_count)
  policy_keys = jax.random.split(policy_key, row_count)
  log_alpha = _parameters(state.entropy_optimizer)
  alpha = jnp.exp(log_alpha)
  successor, successor_logp = _draw_policy(state.actor, batch.next_observation, target_keys)
  future = _values(state.target, batch.next_observation, successor).min(axis=0)
  target = jax.lax.stop_gradient(
      batch.reward + .99 * (1 - batch.terminal) * (future - alpha * successor_logp)
  )

  def critic_loss(parameters):
    estimate = _values(parameters, batch.observation, batch.action)
    return jnp.mean(jnp.square(estimate - target))

  critic_error, gradient = jax.value_and_grad(critic_loss)(state.critic)
  critic_optimizer = _optimize(state.updates, gradient, state.critic_optimizer)
  critic = _parameters(critic_optimizer)

  def actor_loss(parameters):
    action, logp = _draw_policy(parameters, batch.observation, policy_keys)
    value = _values(critic, batch.observation, action).min(axis=0)
    return jnp.mean(alpha * logp - value), jnp.mean(logp)

  (actor_error, mean_logp), gradient = jax.value_and_grad(actor_loss, has_aux=True)(state.actor)
  actor_optimizer = _optimize(state.updates, gradient, state.actor_optimizer)
  entropy_gradient = action_size - mean_logp
  entropy_optimizer = _optimize(state.updates, entropy_gradient, state.entropy_optimizer)
  target = jax.tree.map(lambda old, new: .995 * old + .005 * new, state.target, critic)
  return (
      State(actor_optimizer, critic_optimizer, entropy_optimizer, target, state.updates + 1),
      (critic_error, actor_error, alpha),
  )

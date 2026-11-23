"""Run the same JAX SAC learner over single or parallel building environments."""

import argparse
from collections import defaultdict
from dataclasses import asdict
from datetime import datetime
from hashlib import sha256
import json
import os
from pathlib import Path
from time import perf_counter

import numpy as np


REPLAY_CAPACITY = 50_000
MODELS = Path(__file__).resolve().parents[1] / 'eplus_rl_speed' / 'matched_vav' / 'models'


class Replay:
  """Host-side ring buffer for individual building transitions."""

  def __init__(self, capacity, observation_size, action_size):
    self._capacity = capacity
    shapes = ((observation_size,), (action_size,), (), (observation_size,), ())
    self._data = [np.empty((capacity, *shape), dtype=np.float32) for shape in shapes]
    self._inserted = 0

  def add(self, transitions):
    count = len(transitions.reward)
    if count > self._capacity:
      raise ValueError('Each set of parallel transitions must fit in replay')
    indices = np.arange(self._inserted, self._inserted + count) % self._capacity
    for buffer, values in zip(self._data, transitions, strict=True):
      buffer[indices] = values
    self._inserted += count

  def sample(self, count, rng):
    from .sac import Transition

    indices = rng.integers(min(self._inserted, self._capacity), size=count)
    return Transition(*(values[indices] for values in self._data))


def _on_host(array):
  return np.asarray(array, dtype=np.float32)


def _log_device_memory(log, device, stage):
  if device.platform != 'gpu':
    return
  timestamp = datetime.now().astimezone().isoformat(timespec='milliseconds')
  try:
    stats = device.memory_stats()
  except RuntimeError as error:
    print(f'{timestamp} {stage}: JAX allocator statistics unavailable ({error})', file=log)
    return
  print(f'{timestamp} {stage}: JAX allocator bytes {stats}', file=log)


def train(env, args, device, timing, *, log_progress=False, debug_log=None):
  import jax

  from . import sac

  parallel = env.num_envs
  target = args.transitions or parallel * env.episode_steps
  if target % parallel:
    raise ValueError('--transitions must be divisible by --num-envs')
  if target < args.collect_every:
    raise ValueError('--transitions must reach the first collection interval')

  @jax.jit
  def choose(actor, observation, key):
    return sac.act(actor, observation, jax.random.split(key, parallel))

  if log_progress:
    print(f'  Initializing SAC learner on {device}', flush=True)
  start = perf_counter()
  key = jax.random.PRNGKey(args.seed)
  key, initial_key = jax.random.split(key)
  state = jax.block_until_ready(sac.initialize(env.observation_size, env.action_size, initial_key))
  rng = np.random.default_rng(args.seed)
  replay = Replay(REPLAY_CAPACITY, env.observation_size, env.action_size)
  timing['learner_setup_s'] = perf_counter() - start
  if log_progress:
    print(f"  SAC learner ready in {timing['learner_setup_s']:.2f}s", flush=True)
  if debug_log is not None:
    _log_device_memory(debug_log, device, 'After learner initialization')

  obs = None
  total = episode_step = updates = 0
  next_update = args.collect_every
  started_episodes = episodes = 0
  reset_s, step_s = [], []
  cumulative_reward = np.zeros(parallel, dtype=np.float64)
  returns, first_action_spread = [], None
  info = None
  saw_policy = False
  while total < target:
    if episode_step == 0:
      start = perf_counter()
      obs = jax.block_until_ready(env.reset(args.seed + started_episodes))
      reset_s.append(perf_counter() - start)
      started_episodes += parallel

    start = perf_counter()
    if total < args.collect_every:
      actions = rng.uniform(-1, 1, (parallel, env.action_size)).astype(np.float32)
      timing['random_exploration_s'] += perf_counter() - start
    else:
      if args.engine == 'eplus':
        obs_device = jax.block_until_ready(jax.device_put(obs, device))
        timing['observation_transfer_s'] += perf_counter() - start
        start = perf_counter()
      else:
        obs_device = obs
      key, draw = jax.random.split(key)
      actions = jax.block_until_ready(choose(state.actor, obs_device, draw))
      timing['first_policy_s' if not saw_policy else 'policy_s'] += perf_counter() - start
      saw_policy = True

    start = perf_counter()
    if args.engine == 'eplus':
      env_actions = _on_host(actions)
    else:
      env_actions = jax.block_until_ready(jax.device_put(actions, device))
    timing['action_transfer_s'] += perf_counter() - start

    start = perf_counter()
    successor, reward = jax.block_until_ready(env.step(env_actions))
    step_s.append(perf_counter() - start)
    if log_progress and total == 0:
      print(f'  First environment step finished in {step_s[-1]:.2f}s', flush=True)

    start = perf_counter()
    terminal = episode_step + 1 == env.episode_steps
    reward = _on_host(reward)
    action_host = _on_host(env_actions)
    next_host = _on_host(successor)
    replay.add(sac.Transition(
        _on_host(obs), action_host, reward, next_host,
        np.full(parallel, terminal, dtype=np.float32),
    ))
    if first_action_spread is None and saw_policy:
      first_action_spread = float(action_host.std(axis=0).mean())
    if not (np.isfinite(next_host).all() and np.isfinite(reward).all()):
      raise RuntimeError('The building produced nonfinite temperatures or rewards')
    cumulative_reward += reward
    timing['replay_write_and_transfer_s'] += perf_counter() - start
    total += parallel
    obs = successor
    episode_step += 1

    while total >= next_update:
      if log_progress and updates == 0:
        print(f'  {total}/{target} transitions; starting first SAC update', flush=True)
      for _ in range(args.updates):
        trace_update = debug_log is not None and updates < args.updates
        if trace_update:
          print(f'Update {updates + 1}: sampling and transferring replay', file=debug_log)
        start = perf_counter()
        samples = jax.block_until_ready(jax.device_put(replay.sample(args.batch_size, rng), device))
        timing['replay_sample_and_transfer_s'] += perf_counter() - start
        start = perf_counter()
        if trace_update:
          print(f'Update {updates + 1}: splitting random key', file=debug_log)
        key, learn = jax.random.split(key)
        if trace_update:
          print(f'Update {updates + 1}: reading pre-dispatch allocator stats', file=debug_log)
          _log_device_memory(debug_log, device, f'Update {updates + 1} before dispatch')
          print(f'Update {updates + 1}: calling JAX update', file=debug_log)
        pending_update = sac.update(state, samples, learn)
        if trace_update:
          print(f'Update {updates + 1}: waiting for the device', file=debug_log)
        state, losses = jax.block_until_ready(pending_update)
        if trace_update:
          print(f'Update {updates + 1}: reading losses on host', file=debug_log)
        info = [float(item) for item in losses]
        timing['first_update_s' if updates == 0 else 'learner_update_s'] += perf_counter() - start
        if not np.isfinite(info).all():
          raise RuntimeError('SAC produced a nonfinite update')
        if log_progress and updates == 0:
          print(f"  First SAC update finished in {timing['first_update_s']:.2f}s", flush=True)
        updates += 1
        if trace_update:
          print(f'Update {updates}: finished', file=debug_log)
          _log_device_memory(debug_log, device, f'Update {updates} after completion')
      next_update += args.collect_every
      if log_progress and not terminal:
        print(f'  {total}/{target} transitions, {updates} updates', flush=True)

    if terminal:
      returns.extend(cumulative_reward.tolist())
      cumulative_reward.fill(0)
      episode_step = 0
      episodes += parallel
      print(f'{total}/{target} transitions, {updates} updates, {episodes} complete episodes', flush=True)

  timing['reset_s'] = sum(reset_s)
  timing['simulator_first_step_s'] = step_s[0]
  timing['simulator_warm_s'] = sum(step_s[1:])
  warm = step_s[1:] or step_s[:1]
  return state, dict(
      transitions=total, vector_steps=len(step_s), gradient_updates=updates,
      started_episodes=started_episodes, completed_episodes=episodes,
      reset_per_wave_s=reset_s,
      simulator_mean_ms_per_vector_step=1000 * float(np.mean(warm)),
      simulator_ms_per_transition=1000 * sum(warm) / (len(warm) * parallel),
      episode_comfort_returns=returns, first_policy_cross_building_action_std=first_action_spread,
      final_critic_loss=info[0], final_actor_loss=info[1], final_alpha=info[2],
  )


def _parser():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--engine', choices=('sbsim', 'eplus'), required=True)
  parser.add_argument('--platform', choices=('cpu', 'gpu'), default='cpu', help='JAX device')
  parser.add_argument('--interior-mass', action='store_true', help='Add one effective thermal mass node per SBSim zone')
  parser.add_argument('--num-envs', type=int, default=1)
  parser.add_argument('--episode-days', type=int, default=7)
  parser.add_argument('--transitions', type=int, help='Default is one complete episode per environment')
  parser.add_argument('--collect-every', type=int, default=96, help='Transitions between learner blocks')
  parser.add_argument('--updates', type=int, default=8, help='Learner updates per block')
  parser.add_argument('--batch-size', type=int, default=256)
  parser.add_argument('--seed', type=int, default=0)
  parser.add_argument('--epw', required=True, type=Path)
  parser.add_argument('--model-dir', type=Path)
  parser.add_argument('--energyplus-dir', type=Path)
  parser.add_argument('--output', required=True, type=Path)
  return parser


def main():
  parser = _parser()
  args = parser.parse_args()
  if min(args.num_envs, args.episode_days, args.collect_every, args.updates, args.batch_size) < 1:
    parser.error('Environment and training counts must be positive')
  if args.transitions is not None and (args.transitions < 1 or args.transitions % args.num_envs):
    parser.error('--transitions must be a positive multiple of --num-envs')
  if not 0 <= args.seed < 2**32:
    parser.error('--seed must be an unsigned 32-bit integer')
  if args.num_envs > REPLAY_CAPACITY:
    parser.error('--num-envs must fit in replay')
  if args.engine == 'eplus' and args.energyplus_dir is None:
    parser.error('--energyplus-dir is required for EnergyPlus')
  if args.engine == 'eplus' and args.interior_mass:
    parser.error('--interior-mass applies to the SBSim engine')
  model = args.model_dir or MODELS / ('zone100_office' if args.engine == 'sbsim' else 'zone100_detailed_office')
  os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')

  overall_start = perf_counter()
  import jax

  try:
    device = jax.devices(args.platform)[0]
  except (IndexError, RuntimeError) as error:
    parser.error(f'JAX {args.platform} device unavailable. {error}')
  timing = defaultdict(float)
  timing['device_setup_s'] = perf_counter() - overall_start
  args.output.mkdir(parents=True, exist_ok=False)
  mass_config = None
  start = perf_counter()
  with jax.default_device(device):
    if args.engine == 'sbsim':
      from .interior_mass import InteriorMass
      from .jax_env import JaxOffice
      mass_config = InteriorMass() if args.interior_mass else None
      env = JaxOffice(
          model, args.epw, args.num_envs, args.episode_days,
          platform=args.platform, mass_config=mass_config,
      )
    else:
      from .eplus_env import EnergyPlusOffice
      env = EnergyPlusOffice(
          model, args.epw, args.energyplus_dir, args.output, args.num_envs, args.episode_days
      )
    timing['environment_setup_s'] = perf_counter() - start
    try:
      state, results = train(env, args, device, timing)
    finally:
      start = perf_counter()
      env.close()
      timing['shutdown_s'] = perf_counter() - start
  total_s = perf_counter() - overall_start
  timing['other_s'] = total_s - sum(timing.values())
  actor = {}
  for layer, (weights, bias) in enumerate(state.actor):
    actor[f'weights_{layer}'] = np.asarray(weights)
    actor[f'bias_{layer}'] = np.asarray(bias)
  np.savez_compressed(args.output / 'actor.npz', **actor)

  environment_info = {}
  for field in ('thermal_nodes', 'interior_mass_nodes', 'initialization_s', 'compilation_s'):
    if hasattr(env, field):
      environment_info[field] = getattr(env, field)
  if mass_config is not None:
    environment_info['interior_mass'] = asdict(mass_config)
  logs = sorted(args.output.glob('worker-*/episode-*/eplusout.end'))
  if logs:
    environment_info['energyplus_completion'] = [f.read_text().strip() for f in logs]

  doc = dict(
      engine=args.engine, platform=device.platform, device=device.device_kind, jax=jax.__version__,
      cpu_affinity=sorted(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else None,
      num_envs=args.num_envs, actions_per_env=env.action_size, observations_per_env=env.observation_size,
      episode_days=args.episode_days, steps_per_episode=env.episode_steps, seed=args.seed,
      collect_every=args.collect_every, updates_per_collection=args.updates, batch_size=args.batch_size,
      floorplan_sha256=sha256((model / 'floor_plan.npy').read_bytes()).hexdigest(),
      weather_sha256=sha256(args.epw.read_bytes()).hexdigest(),
      environment_info=environment_info, elapsed_s=total_s,
      transitions_per_s=results['transitions'] / total_s, timing_s=dict(timing), **results,
  )
  rendered = json.dumps(doc, indent=2, allow_nan=False) + '\n'
  (args.output / 'summary.json').write_text(rendered, encoding='utf-8')
  print(rendered)


if __name__ == '__main__':
  main()

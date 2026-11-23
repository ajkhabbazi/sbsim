"""Time one matched-model fixed rollout or SAC training run on CPU."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import time

# Configure the accelerator and native thread pools before ML imports.
os.environ.setdefault('WRAPT_DISABLE_EXTENSIONS', 'true')
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['JAX_PLATFORMS'] = 'cpu'

import numpy as np
import tensorflow as tf
import tf_agents
from tf_agents.policies import py_tf_eager_policy
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.train import actor
from tf_agents.train import learner
from tf_agents.train.utils import spec_utils
from tf_agents.utils import nest_utils

from benchmarks.eplus_rl_speed.train_bench import ACTOR_FC_LAYERS
from benchmarks.eplus_rl_speed.train_bench import REPLAY_CAPACITY
from benchmarks.eplus_rl_speed.train_bench import create_actor_network
from smart_control.reinforcement_learning.agents.sac_agent import create_sac_agent

from . import rl_env


def run_rollout(env, steps, seed):
  actions = (
      np.random.default_rng(seed).uniform(
          -1, 1, (steps, *env.action_spec().shape)
      ).astype(np.float32)
  )
  env.reset()
  start = time.perf_counter()
  for action in actions:
    if env.current_time_step().is_last():
      env.reset()
    env.step(action)
  return dict(
      collect_s=time.perf_counter() - start,
      action_sha256=hashlib.sha256(actions.tobytes()).hexdigest(),
      gradient_updates=0,
  )


def run_sac(env, args):
  setup = time.perf_counter()
  _, action_spec, ts_spec = spec_utils.get_tensor_specs(env)
  counter = tf.Variable(0, trainable=False, dtype=tf.int64)
  agent = create_sac_agent(
      time_step_spec=ts_spec,
      action_spec=action_spec,
      actor_network=create_actor_network(ACTOR_FC_LAYERS, action_spec),
      train_step_counter=counter,
  )
  replay = tf_uniform_replay_buffer.TFUniformReplayBuffer(
      data_spec=agent.collect_data_spec,
      batch_size=1,
      max_length=REPLAY_CAPACITY,
  )

  @tf.function
  def add_batch(transition):
    replay.add_batch(transition)

  def observe(transition):
    add_batch(nest_utils.batch_nested_array(transition))

  collector = actor.Actor(
      env,
      py_tf_eager_policy.PyTFEagerPolicy(
          agent.collect_policy, use_tf_function=True
      ),
      counter,
      steps_per_run=args.collect_steps,
      observers=[observe],
  )
  agent_setup_s = time.perf_counter() - setup - sum(env.reset_times)
  trainer = None
  loops = []
  first_update_s = 0.0
  learner_setup_s = 0.0
  for _ in range(args.steps // args.collect_steps):
    begin = time.perf_counter()
    collector.run()
    collect_s = time.perf_counter() - begin
    updates = args.updates
    if trainer is None:
      setup = time.perf_counter()
      dataset = replay.as_dataset(
          sample_batch_size=args.batch_size, num_steps=2
      )
      # Boundary frames retain the terminal observation for the preceding
      # transition, but cannot start a training transition themselves.
      dataset = (
          dataset.unbatch()
          .filter(
              lambda experience, _: tf.logical_not(experience.is_boundary()[0])
          )
          .batch(args.batch_size, drop_remainder=True)
      )
      options = tf.data.Options()
      options.experimental_optimization.inject_prefetch = False
      dataset = dataset.with_options(options)
      trainer = learner.Learner(
          root_dir=str(args.output / 'learner'),
          train_step=counter,
          agent=agent,
          experience_dataset_fn=lambda: dataset,
          summary_interval=0,
          checkpoint_interval=0,
      )
      learner_setup_s = time.perf_counter() - setup
      begin = time.perf_counter()
      first_loss = float(trainer.run(iterations=1).loss.numpy())
      first_update_s = time.perf_counter() - begin
      if not np.isfinite(first_loss):
        raise RuntimeError('SAC first update returned a nonfinite loss')
      updates -= 1
    begin = time.perf_counter()
    loss = float(trainer.run(iterations=updates).loss.numpy())
    train_s = time.perf_counter() - begin
    if not np.isfinite(loss):
      raise RuntimeError('SAC returned a nonfinite loss')
    loops.append(
        dict(
            steps=env.total_steps,
            collect_s=collect_s,
            warm_update_s=train_s,
            updates=updates,
        )
    )

  count = int(counter.numpy())
  if count != args.steps // args.collect_steps * args.updates:
    raise RuntimeError(f'SAC completed {count} updates with the wrong count')
  warm_s = sum(loop['warm_update_s'] for loop in loops)
  return dict(
      collect_s=sum(loop['collect_s'] for loop in loops),
      agent_setup_s=agent_setup_s,
      learner_setup_s=learner_setup_s,
      first_update_s=first_update_s,
      warm_update_s=warm_s,
      warm_ms_per_update=1000 * warm_s / (count - 1),
      gradient_updates=count,
      loops=loops,
  )


def _engine_info(env, engine):
  if engine == 'sbsim':
    solver = env.backend.solver
    info = dict(
        version=env.backend.version,
        precision='float64',
        backend='JAX CPU' if solver == 'jax' else 'SciPy CPU',
        solver=solver,
    )
    sim = env.backend.sim
    coupled = getattr(sim, 'has_interior_longwave', False)
    info['interior_radiation'] = coupled or sim.building.include_radiative_heat_transfer
    info['interior_mass'] = sim.building.include_interior_mass
    info['exterior_radiation'] = env.backend.exterior_radiation
    active_cells = int(np.count_nonzero(sim.building.len_neighbors >= 2))
    info['active_heat_nodes'] = getattr(sim, 'active_heat_nodes', active_cells)
    if solver in ('reduced-air', 'reduced-envelope', 'condensed-envelope'):
      info['air_block_cells'] = env.backend.air_block_cells
    if solver == 'condensed-envelope':
      info['iterated_surface_nodes'] = sim.iterated_surface_nodes
    if coupled:
      info.update(
          radiosity_nodes=sim.radiosity_nodes,
          radiosity_unknowns_per_solve=sim.radiosity_unknowns_per_solve,
          air_block_cells=env.backend.air_block_cells,
          radiation_method='iterated' if sim.nonlinear else 'linear',
      )
    return info
  info = {}
  errors = []
  for log_dir in env.backend.run_dirs:
    err = (log_dir / 'eplusout.err').read_text(encoding='utf-8')
    info['version'] = err.splitlines()[0].split(',')[2].strip()
    status = re.search(
        r'(\d+) Warning;\s*(\d+) Severe Errors',
        (log_dir / 'eplusout.end').read_text(encoding='utf-8'),
    )
    if not status or int(status[2]):
      raise RuntimeError(
          f'EnergyPlus logged severe errors or omitted status: {log_dir}'
      )
    errors.append(int(status[1]))
  return dict(
      **info, backend='EnergyPlus Python API', warnings_per_episode=errors
  )


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--engine', choices=['eplus', 'sbsim'], required=True)
  parser.add_argument('--control', choices=['shared', 'zone-energy'], default='shared')
  parser.add_argument('--mode', choices=['rollout', 'sac'], required=True)
  parser.add_argument(
      '--sbsim-solver',
      choices=[
          'jax', 'cached-lu', 'fast-lu', 'optimized-lu', 'tuned-lu', 'reduced-air',
          'reduced-envelope', 'condensed-envelope',
          'coupled-radiation', 'linear-radiation', 'linear-radiation-envelope',
      ],
      default='jax',
  )
  parser.add_argument(
      '--air-block-cells',
      type=int,
      help=(
          'Maximum side length in air CVs per group; defaults to one group per'
          ' connected zone'
      ),
  )
  parser.add_argument(
      '--sbsim-exterior-radiation',
      choices=['full', 'linear-longwave', 'solar-only', 'off'],
      default='full', help='Exterior radiation ablation for SBSim fast-lu or tuned-lu',
  )
  parser.add_argument('--model-dir', type=Path, required=True)
  parser.add_argument('--epw', type=Path, required=True)
  parser.add_argument('--energyplus-dir', type=Path, required=True)
  parser.add_argument('--output', type=Path, required=True)
  parser.add_argument(
      '--episode-days', type=int, default=1,
      help='Consecutive weather days before a simulator reset',
  )
  parser.add_argument(
      '--steps', type=int,
      help='Total steps; defaults to 576 in daily mode and one episode otherwise',
  )
  parser.add_argument('--collect-steps', type=int, default=96)
  parser.add_argument(
      '--updates', type=int, default=24, help='SAC updates per loop'
  )
  parser.add_argument('--batch-size', type=int, default=256)
  parser.add_argument('--seed', type=int, default=0)
  args = parser.parse_args()
  if args.control != 'shared' and args.engine != 'eplus':
    parser.error('--control zone-energy requires EnergyPlus')
  if args.episode_days < 1:
    parser.error('--episode-days must be positive')
  if args.steps is None:
    args.steps = (
        576 if args.episode_days == 1
        else args.episode_days * rl_env.EPISODE_STEPS
    )
  if args.engine == 'eplus' and args.sbsim_solver != 'jax':
    parser.error('--sbsim-solver is for SBSim')
  if args.sbsim_exterior_radiation != 'full' and (
      args.engine != 'sbsim' or args.sbsim_solver not in ('fast-lu', 'tuned-lu')
  ):
    parser.error('Exterior radiation ablations require SBSim fast-lu or tuned-lu')
  if args.air_block_cells is not None and (
      args.sbsim_solver not in (
          'reduced-air', 'reduced-envelope', 'condensed-envelope',
          'coupled-radiation', 'linear-radiation', 'linear-radiation-envelope',
      ) or args.air_block_cells < 1
  ):
    parser.error(
        '--air-block-cells needs a positive size and a grouped solver'
    )
  if not args.epw.is_file():
    parser.error(f'EPW file does not exist: {args.epw}')
  if args.steps < 2:
    parser.error('--steps must be at least 2')
  if args.mode == 'sac' and (
      args.collect_steps < 2
      or args.steps % args.collect_steps
      or args.updates < 2
      or args.batch_size < 1
  ):
    parser.error(
        'SAC requires steps divisible by collect-steps, at least two collection'
        ' steps and updates per loop, and a positive batch size'
    )
  args.output.mkdir(parents=True, exist_ok=False)

  tf.config.threading.set_intra_op_parallelism_threads(1)
  tf.config.threading.set_inter_op_parallelism_threads(1)
  tf.keras.utils.set_random_seed(args.seed)
  start = time.perf_counter()
  env = rl_env.make_env(
      args.engine,
      args.model_dir,
      args.epw,
      args.energyplus_dir,
      args.output,
      sbsim_solver=args.sbsim_solver,
      air_block_cells=args.air_block_cells,
      episode_days=args.episode_days,
      exterior_radiation=args.sbsim_exterior_radiation,
      control=args.control,
  )
  model_setup_s = time.perf_counter() - start
  try:
    if args.mode == 'sac':
      result = run_sac(env, args)
    else:
      result = run_rollout(env, args.steps, args.seed)
  finally:
    close_start = time.perf_counter()
    env.close()
    close_s = time.perf_counter() - close_start
  total_s = time.perf_counter() - start
  expected_episodes, remaining = divmod(args.steps, env.steps_per_episode)
  if (
      env.total_steps != args.steps
      or env.completed_episodes != expected_episodes
      or len(env.reset_times) != expected_episodes + bool(remaining)
  ):
    raise RuntimeError(
        'Simulator step or episode counts differ from the workload'
    )
  warm = np.array(env.step_times[1:]) * 1000
  result.update(
      engine=args.engine,
      control=args.control,
      mode=args.mode,
      seed=args.seed,
      real_steps=env.total_steps,
      episodes=env.completed_episodes,
      episode_days=args.episode_days,
      steps_per_episode=env.steps_per_episode,
      observation_size=env.observation_spec().shape[0],
      action_size=env.action_spec().shape[0],
      engine_info=_engine_info(env, args.engine),
      tensorflow=tf.__version__,
      tf_agents=tf_agents.__version__,
      python=platform.python_version(),
      model_setup_s=model_setup_s,
      reset_s=env.reset_times,
      first_step_s=env.step_times[0],
      backend_step_s=sum(env.step_times),
      warm_step_mean_ms=float(warm.mean()),
      warm_step_p50_ms=float(np.median(warm)),
      warm_step_p90_ms=float(np.percentile(warm, 90)),
      close_s=close_s,
      total_s=total_s,
  )
  if hasattr(os, 'sched_getaffinity'):
    result['cpu_affinity'] = sorted(os.sched_getaffinity(0))
  if args.control == 'zone-energy':
    result['last_episode_hvac_kwh'] = sum(env.backend.hvac_kwh)
    result['last_episode_energy_samples'] = len(env.backend.hvac_kwh)
  report = json.dumps(result, indent=2, allow_nan=False)
  (args.output / 'summary.json').write_text(report + '\n', encoding='utf-8')
  print(report)


if __name__ == '__main__':
  main()

"""Exercise the SAC learner without a building simulator or weather files."""

import argparse
import faulthandler
import os
from time import perf_counter

import jax
import numpy as np

from . import sac
from .train import Replay


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--platform', choices=('cpu', 'gpu'), default='gpu')
  parser.add_argument('--zones', type=int, default=16)
  parser.add_argument('--batch-size', type=int, default=256)
  parser.add_argument('--updates', type=int, default=16)
  parser.add_argument('--timeout', type=float, default=60, help='Overall timeout in seconds')
  args = parser.parse_args()
  if min(args.zones, args.batch_size, args.updates, args.timeout) <= 0:
    parser.error('Zones, batch size, updates and timeout must be positive')

  # The building solver enables x64 before the learner is started.
  jax.config.update('jax_enable_x64', True)
  faulthandler.dump_traceback_later(args.timeout, exit=True)
  try:
    device = jax.devices(args.platform)[0]
    print(f'JAX {jax.__version__} on {device.device_kind}', flush=True)
    print(f"CUDA_LAUNCH_BLOCKING={os.environ.get('CUDA_LAUNCH_BLOCKING', '<default>')}", flush=True)
    obs_size, action_size = args.zones + 1, 2 * args.zones
    rng = np.random.default_rng(0)
    initial_actions = rng.uniform(-1, 1, (96, action_size)).astype(np.float32)
    obs = np.random.default_rng(1).normal(0, .4, (97, obs_size)).astype(np.float32)
    replay = Replay(96, obs_size, action_size)
    replay.add(sac.Transition(
        obs[:-1], initial_actions, -np.abs(obs[1:, :-1]).mean(axis=1),
        obs[1:], np.zeros(96, dtype=np.float32),
    ))
    with jax.default_device(device):
      key, initial_key = jax.random.split(jax.random.PRNGKey(0))
      state = jax.block_until_ready(sac.initialize(obs_size, action_size, initial_key))
      for index in range(1, args.updates + 1):
        print(f'{index}/{args.updates}: preparing replay batch', flush=True)
        batch = jax.block_until_ready(jax.device_put(replay.sample(args.batch_size, rng), device))
        key, learn = jax.random.split(key)
        start = perf_counter()
        print(f'{index}/{args.updates}: calling SAC update', flush=True)
        pending = sac.update(state, batch, learn)
        print(f'{index}/{args.updates}: waiting for device', flush=True)
        state, losses = jax.block_until_ready(pending)
        losses = tuple(float(loss) for loss in losses)
        if not np.isfinite(losses).all():
          raise RuntimeError(f'Nonfinite losses at update {index}: {losses}')
        print(f'{index}/{args.updates}: finished in {perf_counter() - start:.3f}s', flush=True)
  finally:
    faulthandler.cancel_dump_traceback_later()


if __name__ == '__main__':
  main()

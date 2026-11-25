"""Process-isolated, per-zone EnergyPlus environments for the SAC study."""

from dataclasses import dataclass
from datetime import date
from datetime import timedelta
import json
from multiprocessing.connection import Connection
from numbers import Integral
import os
from pathlib import Path
import socket
import subprocess
import sys
import traceback

import numpy as np

from .matched_vav import generate
from .matched_vav import settings


_WORKER_MODULE = 'benchmarks.grouped_jax_study.eplus_env'
_TIMEOUT = 120


@dataclass
class _Worker:
  process: subprocess.Popen
  connection: Connection
  log: Path

  def receive(self):
    if not self.connection.poll(_TIMEOUT):
      raise RuntimeError(f'EnergyPlus study worker timed out. See {self.log}')
    try:
      status, value = self.connection.recv()
    except (EOFError, OSError) as error:
      raise RuntimeError(
          f'EnergyPlus study worker exited. See {self.log}'
      ) from error
    if status != 'ok':
      raise RuntimeError(
          f'EnergyPlus study worker failed: {value}. See {self.log}'
      )
    return value

  def dispose(self):
    self.connection.close()
    if self.process.poll() is None:
      self.process.terminate()
      try:
        self.process.wait(timeout=5)
      except subprocess.TimeoutExpired:
        self.process.kill()
        self.process.wait()


def _launch_worker(model_path, epw, energyplus_dir, output, run_day, days):
  output.mkdir()
  log_path = output / 'worker.log'
  parent, child = socket.socketpair()
  with parent, child, log_path.open('wb') as log:
    process = subprocess.Popen(
        [
            sys.executable,
            '-m',
            _WORKER_MODULE,
            str(child.fileno()),
            str(model_path),
            str(epw),
            str(energyplus_dir),
            str(output),
            run_day.isoformat(),
            str(days),
        ],
        cwd=Path(__file__).resolve().parents[2],
        pass_fds=(child.fileno(),),
        stdout=log,
        stderr=log,
        env=dict(
            os.environ,
            CUDA_VISIBLE_DEVICES='-1',
            OMP_NUM_THREADS='1',
            OPENBLAS_NUM_THREADS='1',
            MKL_NUM_THREADS='1',
            TF_NUM_INTEROP_THREADS='1',
            TF_NUM_INTRAOP_THREADS='1',
        ),
    )
    return _Worker(process, Connection(parent.detach()), log_path)


class EnergyPlusOffice:
  """Advance B detailed office models together with fixed comfort rewards.

  Each world owns a persistent subprocess and one EnergyPlus API instance.
  Actions alternate heating and cooling for each zone in floorplan order.
  Worlds replay the same EPW and calendar; reset seeds keep physics fixed.
  """

  def __init__(
      self, model_dir, epw, energyplus_dir, output, num_envs, episode_days
  ):
    if any(
        isinstance(value, bool) or not isinstance(value, Integral) or value < 1
        for value in (num_envs, episode_days)
    ):
      raise ValueError('num_envs and episode_days must be positive integers')
    self.num_envs = int(num_envs)
    self.episode_steps = int(episode_days) * 24 * settings.STEPS_PER_HOUR
    self._steps = None
    self._workers = []
    self._closed = False

    model_dir = Path(model_dir).resolve()
    epw = Path(epw).resolve()
    energyplus_dir = Path(energyplus_dir).resolve()
    output = Path(output).resolve()
    if (
        not epw.is_file()
        or not (energyplus_dir / 'pyenergyplus/api.py').is_file()
    ):
      raise ValueError('Provide an EPW and an installation with pyenergyplus')
    manifest = generate.read_manifest(model_dir)
    plan, run_day = generate.load_plan(model_dir)
    if not manifest.get('detailed_office'):
      raise ValueError('EnergyPlusOffice requires a detailed office model')
    self.zone_count = len(plan.rooms)
    self.observation_size = self.zone_count + 1
    self.action_size = self.zone_count * 2
    model_path = model_dir / 'model.epJSON'
    if json.loads(model_path.read_text(encoding='utf-8')) != (
        generate.energyplus_from_manifest(plan, run_day, manifest)
    ):
      raise ValueError('model.epJSON does not match floorplan.json')

    # Benchmark callers may already have created an enclosing output folder.
    output.mkdir(parents=True, exist_ok=True)
    try:
      for index in range(self.num_envs):
        self._workers.append(
            _launch_worker(
                model_path,
                epw,
                energyplus_dir,
                output / f'worker-{index:02d}',
                run_day,
                int(episode_days),
            )
        )
      for worker in self._workers:
        worker.receive()
    except BaseException:
      self._dispose()
      raise

  def _request(self, command, payloads):
    if self._closed:
      raise RuntimeError('EnergyPlusOffice is closed')
    # Send every action first so the native simulations run concurrently.
    for worker, payload in zip(self._workers, payloads):
      try:
        worker.connection.send((command, payload))
      except (EOFError, OSError) as error:
        raise RuntimeError(
            f'EnergyPlus study worker exited. See {worker.log}'
        ) from error
    return [worker.receive() for worker in self._workers]

  def _temperatures(self, command, payloads):
    try:
      values = np.asarray(self._request(command, payloads), dtype=np.float32)
      if (
          values.shape != (self.num_envs, self.observation_size)
          or not np.isfinite(values).all()
      ):
        raise RuntimeError(
            'EnergyPlus returned invalid temperature observations'
        )
      return values
    except BaseException:
      self._dispose()
      raise

  def reset(self, seed: int) -> np.ndarray:
    del seed  # Each episode starts at the saved model's deterministic phase.
    values = self._temperatures('reset', [None] * self.num_envs)
    self._steps = 0
    return (values - 22.5) / 10.0

  def step(self, actions) -> tuple[np.ndarray, np.ndarray]:
    if self._closed or self._steps is None or self._steps >= self.episode_steps:
      raise RuntimeError('EnergyPlusOffice needs a reset before stepping')
    actions = np.asarray(actions, dtype=np.float64)
    if (
        actions.shape != (self.num_envs, self.action_size)
        or not np.isfinite(actions).all()
    ):
      raise ValueError(
          f'Expected finite actions with shape ({self.num_envs},'
          f' {self.action_size})'
      )
    normalized = np.clip(actions, -1.0, 1.0).reshape(self.num_envs, self.zone_count, 2)
    pairs = ([20.5, 24.5] + 1.5 * normalized).reshape(
        self.num_envs, self.action_size
    )
    values = self._temperatures('step', pairs)
    self._steps += 1
    zones = values[:, :-1]
    penalty = np.maximum(settings.HEATING_SETPOINT_C - zones, 0) + np.maximum(
        zones - settings.COOLING_SETPOINT_C, 0
    )
    return (values - 22.5) / 10.0, -np.mean(penalty, axis=1)

  def close(self):
    if self._closed:
      return
    try:
      self._request('close', [None] * self.num_envs)
      for worker in self._workers:
        if worker.process.wait(timeout=_TIMEOUT):
          raise RuntimeError(
              f'EnergyPlus study worker exited. See {worker.log}'
          )
    finally:
      self._dispose()

  def _dispose(self):
    self._closed = True
    workers, self._workers = self._workers, []
    for worker in workers:
      worker.dispose()


def _make_backend(model_path, epw, energyplus_dir, output, run_day, days):
  from .matched_vav import energyplus_zone_control
  from .matched_vav import rl_env

  model = json.loads(model_path.read_text(encoding='utf-8'))
  # Keep actuators aligned with the floorplan's numeric room order.
  names = [f'room_{index}' for index in range(1, len(model['Zone']) + 1)]
  last_day = run_day + timedelta(days=days - 1)
  period = model['RunPeriod']['Smoke Day']
  period.update(
      end_year=last_day.year,
      end_month=last_day.month,
      end_day_of_month=last_day.day,
  )
  if last_day.year == run_day.year + 1 and (last_day.month, last_day.day) < (
      run_day.month,
      run_day.day,
  ):
    # EnergyPlus 24.2 TMY date wrapping matches the existing RL adapter.
    del period['begin_year'], period['end_year']
    period['day_of_week_for_start_day'] = rl_env.WEEKDAYS[run_day.weekday()]
  schedule_names = energyplus_zone_control._separate_setpoints(model, names)
  backend = rl_env._EnergyPlus(
      model, epw, energyplus_dir, output, names, days * rl_env.EPISODE_STEPS
  )
  backend.setpoint_schedule_names = schedule_names
  return backend


def _serve_worker():
  descriptor, model, epw, energyplus_dir, output, run_day, days = sys.argv[1:]
  with Connection(int(descriptor)) as connection:
    backend = None
    try:
      backend = _make_backend(
          Path(model),
          Path(epw),
          Path(energyplus_dir),
          Path(output),
          date.fromisoformat(run_day),
          int(days),
      )
      connection.send(('ok', None))
      while True:
        command, payload = connection.recv()
        if command == 'reset':
          values = backend.reset()
          expected = 2 * len(backend.names)
          if len(set(backend.actuator_handles)) != expected:
            raise RuntimeError(f'Expected {expected} independent thermostat actuators')
        elif command == 'step':
          values = backend.step(payload)
        elif command == 'close':
          backend.close()
          connection.send(('ok', None))
          return 0
        else:
          raise RuntimeError(f'Unexpected simulator request {command!r}')
        connection.send(('ok', values))
    except EOFError:
      if backend is not None:
        backend.close()
      return 0
    except Exception as error:
      traceback.print_exc()
      try:
        connection.send(('error', str(error)))
      except (EOFError, OSError):
        pass
      return 1


if __name__ == '__main__':
  sys.exit(_serve_worker())

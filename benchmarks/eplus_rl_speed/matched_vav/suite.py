"""Run repeated matched rollout and SAC benchmarks with one Linux CPU."""

import argparse
from dataclasses import asdict
from datetime import date
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import statistics
import subprocess
import sys
import time

from . import settings
from .floorplan import PlanSpec
from .generate import write_models

CASES = {
    'zone01': PlanSpec(1, window_width_m=2),
    'zone05': PlanSpec(5, window_width_m=2),
    'zone05_fine': PlanSpec(5, window_width_m=2, cv_size_m=0.25),
    'zone16': PlanSpec(16, window_width_m=2),
}
ENGINES = ('eplus', 'sbsim')
MODES = ('rollout', 'sac')


def _run(args, case, mode, engine, seed, env):
  folder = args.output / 'runs' / case / mode / str(seed)
  folder.mkdir(parents=True, exist_ok=True)
  model = args.output / 'models' / case
  output = folder / engine
  log = folder / f'{engine}.log'
  solver = ['--sbsim-solver', args.sbsim_solver] if engine == 'sbsim' else []
  begin = time.perf_counter()
  with log.open('w', encoding='utf-8') as stream:
    proc = subprocess.run(
        [
            'taskset',
            '-c',
            str(args.cpu),
            sys.executable,
            '-m',
            'benchmarks.eplus_rl_speed.matched_vav.benchmark',
            '--engine',
            engine,
            '--mode',
            mode,
            '--model-dir',
            str(model),
            '--epw',
            str(args.epw),
            '--energyplus-dir',
            str(args.energyplus_dir),
            '--output',
            str(output),
            '--steps',
            str(args.steps),
            '--episode-days',
            str(args.episode_days),
            '--collect-steps',
            str(args.collect_steps),
            '--updates',
            str(args.updates),
            '--batch-size',
            str(args.batch_size),
            '--seed',
            str(seed),
            *solver,
        ],
        env=env,
        stdout=stream,
        stderr=subprocess.STDOUT,
        timeout=300,
        check=False,
    )
  wall_s = time.perf_counter() - begin
  if proc.returncode:
    raise RuntimeError(f'{case} {mode} {engine} seed {seed} failed. See {log}')
  result = json.loads((output / 'summary.json').read_text(encoding='utf-8'))
  if result['cpu_affinity'] != [args.cpu]:
    raise RuntimeError(
        f'{case} {mode} {engine} did not run on the selected CPU'
    )
  print(
      f'{case} {mode} seed={seed} {engine}:'
      f' {result["warm_step_mean_ms"]:.3f} ms/step, {result["total_s"]:.2f} s'
      ' total',
      flush=True,
  )
  return dict(case=case, process_wall_s=wall_s, **result)


def _medians(runs, cases, repeats, modes):
  results = []
  for case in cases:
    for mode in modes:
      by_engine = {}
      for engine in ENGINES:
        selected = [
            r
            for r in runs
            if (r['case'], r['mode'], r['engine']) == (case, mode, engine)
        ]
        if len(selected) != repeats:
          raise RuntimeError(f'Missing repeats for {case} {mode} {engine}')
        fields = (
            'model_setup_s',
            'first_step_s',
            'warm_step_mean_ms',
            'warm_step_p50_ms',
            'warm_step_p90_ms',
            'backend_step_s',
            'collect_s',
            'total_s',
            'process_wall_s',
        )
        if mode == 'sac':
          fields += (
              'first_update_s',
              'warm_update_s',
              'warm_ms_per_update',
              'agent_setup_s',
              'learner_setup_s',
          )
        metrics = {
            key: statistics.median(r[key] for r in selected) for key in fields
        }
        metrics['first_reset_s'] = statistics.median(
            r['reset_s'][0] for r in selected
        )
        metrics['reset_and_backend_step_s'] = statistics.median(
            sum(r['reset_s']) + r['backend_step_s'] for r in selected
        )
        metrics['later_reset_s'] = statistics.median(
            [duration for r in selected for duration in r['reset_s'][1:]]
            or [0.0]
        )
        by_engine[engine] = metrics
      ratios = {
          field: by_engine['sbsim'][field] / by_engine['eplus'][field]
          for field in ('warm_step_mean_ms', 'total_s')
      }
      results.append(
          dict(case=case, mode=mode, **by_engine, sbsim_div_eplus=ratios)
      )
  return results


def _check_pair(pair):
  first, second = pair
  keys = (
      'real_steps',
      'episodes',
      'steps_per_episode',
      'observation_size',
      'action_size',
      'gradient_updates',
      'tensorflow',
      'tf_agents',
      'action_sha256',
  )
  if any(first.get(key) != second.get(key) for key in keys):
    raise RuntimeError('Paired runs did not complete matching workloads')


def _cpu_description(cpu):
  for block in Path('/proc/cpuinfo').read_text().split('\n\n'):
    fields = dict(
        (k.strip(), v.strip())
        for k, _, v in (line.partition(':') for line in block.splitlines())
    )
    if fields.get('processor') == str(cpu):
      return fields.get('model name', platform.processor())
  return platform.processor()


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--energyplus-dir', type=Path, required=True)
  parser.add_argument('--epw', type=Path, required=True)
  parser.add_argument('--output', type=Path, required=True)
  parser.add_argument('--cpu', type=int, required=True)
  parser.add_argument('--repeats', type=int, default=3)
  parser.add_argument('--case', action='append', choices=CASES)
  parser.add_argument(
      '--mode',
      action='append',
      choices=MODES,
      help='Workload to run (default: both)',
  )
  parser.add_argument(
      '--zones',
      type=int,
      nargs='+',
      help=(
          'Generate these zone counts with the standard room, window, and CV'
          ' sizes'
      ),
  )
  parser.add_argument('--episode-days', type=int, default=1)
  parser.add_argument('--steps', type=int)
  parser.add_argument(
      '--sbsim-solver',
      choices=[
          'jax', 'cached-lu', 'fast-lu', 'optimized-lu', 'tuned-lu', 'reduced-air',
          'reduced-envelope', 'condensed-envelope',
      ],
      default='jax',
  )
  parser.add_argument('--collect-steps', type=int, default=96)
  parser.add_argument('--updates', type=int, default=24)
  parser.add_argument('--batch-size', type=int, default=256)
  parser.add_argument('--energyplus-version', default='24.2')
  configuration = parser.add_mutually_exclusive_group()
  configuration.add_argument(
      '--office',
      action='store_true',
      help='Use scheduled internal loads and outdoor air',
  )
  configuration.add_argument(
      '--compare-office',
      action='store_true',
      help='Run both baseline and office models',
  )
  args = parser.parse_args()
  if args.episode_days < 1:
    parser.error('--episode-days must be positive')
  if args.steps is None:
    args.steps = (
        576 if args.episode_days == 1
        else args.episode_days * 24 * settings.STEPS_PER_HOUR
    )
  if not hasattr(os, 'sched_getaffinity') or shutil.which('taskset') is None:
    parser.error('This suite needs Linux CPU affinity and taskset')
  if args.cpu not in os.sched_getaffinity(0):
    parser.error('The requested CPU is outside the process affinity')
  if (
      not args.epw.is_file()
      or not (args.energyplus_dir / 'pyenergyplus' / 'api.py').is_file()
  ):
    parser.error(
        'Provide an existing EPW and an installation with pyenergyplus'
    )
  modes = tuple(dict.fromkeys(args.mode or MODES))
  if args.repeats < 1 or args.steps < 2:
    parser.error('Specify positive repeats and at least two steps')
  if 'sac' in modes and (
      args.collect_steps < 2
      or args.steps % args.collect_steps
      or args.updates < 2
      or args.batch_size < 1
  ):
    parser.error(
        'Invalid SAC batch size, update count, or steps per collection loop'
    )
  if args.zones and any(count < 1 for count in args.zones):
    parser.error('--zones must contain positive zone counts')
  cases = {case: CASES[case] for case in args.case or ()}
  for count in args.zones or ():
    cases[f'zone{count:02d}'] = PlanSpec(count, window_width_m=2)
  if not cases:
    cases = dict(CASES)
  office_cases = set(cases) if args.office else set()
  if args.compare_office:
    variations = {f'{name}_office': spec for name, spec in cases.items()}
    office_cases.update(variations)
    cases.update(variations)
  args.output.mkdir(parents=True, exist_ok=False)
  run_day = date(2024, 7, 15)
  for case, spec in cases.items():
    write_models(
        args.output / 'models' / case,
        spec,
        run_day,
        args.energyplus_version,
        office=case in office_cases,
    )

  controls = dict(
      JAX_ENABLE_COMPILATION_CACHE='false',
      JAX_PLATFORMS='cpu',
      CUDA_VISIBLE_DEVICES='-1',
      OMP_NUM_THREADS='1',
      OPENBLAS_NUM_THREADS='1',
      MKL_NUM_THREADS='1',
      TF_CPP_MIN_LOG_LEVEL='3',
  )
  env = dict(os.environ, **controls)
  runs = []
  for seed in range(args.repeats):
    case_order = list(cases)
    if args.compare_office and seed % 2:
      case_order.reverse()
    for index, case in enumerate(case_order):
      for mode in modes:
        order = ENGINES if (seed + index) % 2 == 0 else ENGINES[::-1]
        pair = [_run(args, case, mode, engine, seed, env) for engine in order]
        _check_pair(pair)
        runs.extend(pair)
  report = dict(
      cpu=_cpu_description(args.cpu),
      logical_cpu=args.cpu,
      platform=platform.platform(),
      thread_environment=controls,
      steps=args.steps,
      episode_days=args.episode_days,
      sbsim_solver=args.sbsim_solver,
      repeats=args.repeats,
      epw=args.epw.name,
      epw_sha256=hashlib.sha256(args.epw.read_bytes()).hexdigest(),
      run_day=run_day.isoformat(),
      energyplus_input_version=args.energyplus_version,
      office_cases=sorted(office_cases),
      modes=list(modes),
      collect_steps=args.collect_steps,
      updates_per_loop=args.updates,
      batch_size=args.batch_size,
      specs={case: asdict(spec) for case, spec in cases.items()},
      medians=_medians(runs, cases, args.repeats, modes),
      runs=runs,
  )
  report_path = args.output / 'benchmark.json'
  report_path.write_text(
      json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8'
  )
  print(report_path.resolve())


if __name__ == '__main__':
  main()

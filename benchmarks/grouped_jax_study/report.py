"""Write benchmark data, figures, and English and Korean Markdown reports."""

import argparse
import csv
import json
import os
from pathlib import Path
import shutil

import matplotlib

matplotlib.use('Agg')
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np


_LABELS = {
    'full_jax': 'Full JAX',
    'full_linear': 'Full linear',
    'grouped_jax': 'Grouped JAX',
    'energyplus': 'EnergyPlus',
}
_COLORS = {
    'full_jax': '#374785',
    'full_linear': '#25a18e',
    'grouped_jax': '#d35126',
    'energyplus': '#586e75',
}
_ARTIFACTS = {
    'timing': ('timing.csv', 'timing.png'),
    'physics': (
        'physics.csv',
        'physics_errors.csv',
        'physics.npz',
        'physics.png',
    ),
    'sac': ('sac.csv', 'sac.png'),
}


def _csv(path, rows):
  if not rows:
    return
  fields = list(dict.fromkeys(k for row in rows for k in row))
  with path.open('w', encoding='utf-8', newline='') as stream:
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    for row in rows:
      writer.writerow({
          key: json.dumps(value) if isinstance(value, (dict, list)) else value
          for key, value in row.items()
      })


def _style(ax):
  ax.grid(axis='y', alpha=0.2)
  ax.spines[['top', 'right']].set_visible(False)


def _by_zone(root, path, rows, key, ylabel):
  fig, ax = plt.subplots(figsize=(7.8, 4.2), layout='constrained')
  for engine in ('full_jax', 'grouped_jax', 'energyplus'):
    values = sorted(
        (row for row in rows if row['engine'] == engine),
        key=lambda row: row['zones'],
    )
    ax.plot(
        [row['zones'] for row in values],
        [row[key] for row in values],
        marker='o',
        linewidth=2,
        label=_LABELS[engine],
        color=_COLORS[engine],
    )
  ax.set_xlabel('Thermal zones')
  ax.set_ylabel(ylabel)
  ax.set_xticks(sorted({row['zones'] for row in rows}))
  ax.legend(frameon=False)
  _style(ax)
  fig.savefig(root / path, dpi=160)
  plt.close(fig)


def timing(root, rows):
  _csv(root / 'timing.csv', rows)
  _by_zone(root, 'timing.png', rows, 'warm_ms', 'Warm step wall time [ms]')


def sac(root, rows):
  flattened = [dict(row, **row['timing_s']) for row in rows]
  for row in flattened:
    del row['timing_s']
  _csv(root / 'sac.csv', flattened)
  _by_zone(root, 'sac.png', rows, 'elapsed_s', 'SAC wall time [s]')


def temperature(root, traces, metrics, times, names, forcing):
  full = traces['full_jax']
  linear_bias = traces['full_linear'] - full
  grouped_bias = traces['grouped_jax'] - full
  grouping_effect = traces['grouped_jax'] - traces['full_linear']
  with (root / 'physics.csv').open('w', encoding='utf-8', newline='') as stream:
    writer = csv.writer(stream)
    writer.writerow([
        'interval_end',
        'zone',
        'full_nonlinear_c',
        'full_linear_c',
        'grouped_linear_c',
        'linearization_error_k',
        'grouped_total_error_k',
        'grouping_effect_k',
    ])
    for row, stamp in enumerate(times):
      writer.writerows(
          zip(
              [stamp.isoformat()] * len(names),
              names,
              full[row],
              traces['full_linear'][row],
              traces['grouped_jax'][row],
              linear_bias[row],
              grouped_bias[row],
              grouping_effect[row],
              strict=True,
          )
      )
  np.savez_compressed(
      root / 'physics.npz',
      interval_end=np.array([d.isoformat() for d in times]),
      zones=np.array(names),
      **forcing,
      **traces,
  )

  comparisons = []
  for label, delta in (
      ('full_linear_minus_full_nonlinear', linear_bias),
      ('grouped_linear_minus_full_nonlinear', grouped_bias),
      ('grouped_linear_minus_full_linear', grouping_effect),
  ):
    absolute = np.abs(delta)
    step, zone = np.unravel_index(absolute.argmax(), absolute.shape)
    comparisons.append(
        dict(
            comparison=label,
            bias_k=float(delta.mean()),
            mae_k=float(absolute.mean()),
            rmse_k=float(np.sqrt(np.mean(delta**2))),
            p95_absolute_k=float(np.percentile(absolute, 95)),
            max_absolute_k=float(absolute[step, zone]),
            maximum_zone=names[zone],
            maximum_interval_end=times[step].isoformat(),
        )
    )
  _csv(root / 'physics_errors.csv', comparisons)

  room = int(np.abs(grouped_bias).mean(axis=0).argmax())
  fig, (top, bottom) = plt.subplots(
      2, 1, figsize=(10.5, 6.3), sharex=True, layout='constrained'
  )
  for variant, values in traces.items():
    top.plot(
        times,
        values[:, room],
        label=_LABELS[variant],
        color=_COLORS[variant],
        lw=1.5,
        linestyle='--' if variant == 'full_linear' else '-',
        alpha=0.9,
    )
  top.set_title(f'Zone temperature in {names[room]}')
  top.set_ylabel('Temperature [°C]')
  top.legend(frameon=False, loc='upper left', ncol=3)
  for variant, delta in (
      ('full_linear', linear_bias),
      ('grouped_jax', grouped_bias),
  ):
    absolute = np.abs(delta)
    bottom.plot(
        times,
        np.percentile(absolute, 95, axis=1),
        label=_LABELS[variant],
        color=_COLORS[variant],
        lw=1.5,
    )
  bottom.set_ylabel('95th percentile absolute\nzone error [K]')
  bottom.set_xlabel('Local standard time')
  locator = mdates.AutoDateLocator(tz=times.tz, minticks=5, maxticks=9)
  bottom.xaxis.set_major_locator(locator)
  bottom.xaxis.set_major_formatter(
      mdates.ConciseDateFormatter(locator, tz=times.tz)
  )
  bottom.legend(frameon=False, ncol=2)
  for ax in (top, bottom):
    _style(ax)
  fig.savefig(root / 'physics.png', dpi=160)
  plt.close(fig)
  return dict(
      zones=len(names),
      steps=len(times),
      metrics=metrics,
      comparisons=comparisons,
  )


_REPORT_TEXT = {
    'en': {
        'language_link': '[한국어](REPORT_ko.md)',
        'title': 'Full and Grouped SBSim Compared with Detailed EnergyPlus',
        'weather': (
            'The same weather from {epw} was used for {days} days starting on '
            '{start_day}. Zones measure 5 × 4 m, the grid spacing is '
            '{cv_size_m} m, and the exterior-window width is {window_width_m} '
            'm. The time interval is 5 minutes, with {episode_steps:,} steps '
            'per episode.'
        ),
        'jax': (
            'JAX {jax} ran on {jax_device}. The detailed EnergyPlus input '
            'version is `{energyplus_input_version}`.'
        ),
        'model_comparison': (
            'The full-node and grouped-node SBSim models use the same envelope, '
            'air, and thermal-mass properties and VAV system. Detailed '
            'EnergyPlus has separate envelope, load, and HVAC implementations, '
            'so temperature differences between engines also include these '
            'modeling differences.'
        ),
        'section_affinity': (
            'The logical CPUs used for the separate runs were {values}.'
        ),
        'run_affinity': 'The logical CPUs used for the run were `{values}`.',
        'grouped_methods': (
            'The grouped solver setting for each section was {values}.'
        ),
        'source_directories': (
            'The raw-data directories for the individual runs were {values}.'
        ),
        'timing_heading': 'Step Time at Fixed Setpoints',
        'timing_description': (
            'A 21–24 °C setpoint range was applied on weekdays from 07:00–19:00 '
            'and 19–26 °C at all other times. The average excluding the first '
            'step includes setpoint application, simulation, observation, and '
            'EnergyPlus process communication.'
        ),
        'timing_header': (
            '| Zones | Engine | Initialization s | Reset s | First step s | '
            'Subsequent ms/step | States |'
        ),
        'timing_comparison': (
            'At {largest} zones, the full JAX and EnergyPlus step times were '
            '{full_ratio:.2f} and {energyplus_ratio:.2f} times the grouped JAX '
            'step time, respectively.'
        ),
        'timing_alt': 'Step time by grid and engine',
        'timing_artifact': 'Measurements: `timing.csv`',
        'physics_heading': (
            'Temperature Differences from Longwave Linearization and Grid '
            'Grouping'
        ),
        'physics_description': (
            'HVAC heat by diffuser was recorded from the {zones}-zone full '
            'nonlinear SBSim run. The same heat was replayed in full linear and '
            'grouped linear SBSim to compare the effects of longwave '
            'linearization and grid grouping. All three variants use indoor '
            'and outdoor radiation and interior thermal mass.'
        ),
        'physics_error_header': (
            '| Comparison | MAE K | RMSE K | 95th percentile K | Maximum K |'
        ),
        'comparison_labels': {
            'full_linear_minus_full_nonlinear': (
                'Full linear vs. full nonlinear'
            ),
            'grouped_linear_minus_full_nonlinear': (
                'Grouped linear vs. full nonlinear'
            ),
            'grouped_linear_minus_full_linear': (
                'Grouped linear vs. full linear'
            ),
        },
        'physics_metrics_header': (
            '| Heat balance | Solver | States | Steps by LU/linear-solve count '
            '| Subsequent ms/step |'
        ),
        'solve_count': '{count} {solve} × {steps:,} steps',
        'physics_timing_note': (
            'The full nonlinear baseline timing includes HVAC feedback and heat '
            'recording; timing for the two linear models includes fixed-heat '
            'replay and zone-temperature reads.'
        ),
        'physics_alt': (
            'Representative-zone temperature and temperature error over time'
        ),
        'physics_artifacts': (
            'Five-minute temperature time series: `physics.csv`; error '
            'statistics: `physics_errors.csv`; temperatures and HVAC heat: '
            '`physics.npz`'
        ),
        'sac_heading': 'Actual SAC Training',
        'sac_description': (
            'JAX SAC was trained for one episode per environment with seed '
            '{seed}. {updates} updates were performed every {collect_every} '
            'transitions, with batch size {batch_size}. Heating and cooling '
            'setpoints were controlled independently in each zone. The reward '
            'was the negative zone-average amount by which temperatures fell '
            'outside 21–24 °C.'
        ),
        'sac_header': (
            '| Zones | Engine | Simulator device | Learner device | Transitions '
            '| Updates | Total s | Simulator ms/transition |'
        ),
        'sac_alt': 'JAX SAC runtime by engine and zone count',
        'sac_artifact': 'SAC timing and training statistics: `sac.csv`',
        'summary_artifact': 'Environment and aggregate values: `summary.json`',
    },
    'ko': {
        'language_link': '[English](REPORT.md)',
        'title': '전체·그룹 SBSim과 상세 EnergyPlus 비교',
        'weather': (
            '{start_day}부터 {days}일간 {epw}의 같은 날씨를 사용했다. 구역은 '
            '5 × 4 m, 격자는 {cv_size_m} m, 외벽 창 폭은 {window_width_m} m다. '
            '시간 간격은 5분이며, 에피소드당 {episode_steps:,}스텝이다.'
        ),
        'jax': (
            'JAX {jax}에서 {jax_device}를 사용했다. 상세 EnergyPlus 입력 '
            '버전은 `{energyplus_input_version}`이다.'
        ),
        'model_comparison': (
            'SBSim 전체 노드와 그룹 노드는 같은 외피·공기·열질량 물성과 VAV '
            '계통을 사용한다. 상세 EnergyPlus에는 별도의 외피·부하·공조 '
            '구현이 있으므로 엔진 간 온도 차이에는 이런 모델 차이도 들어간다.'
        ),
        'section_affinity': '분리 실행에 사용한 논리 CPU는 {values}이다.',
        'run_affinity': '실행에 사용한 논리 CPU는 `{values}`이다.',
        'grouped_methods': '부분별 그룹 풀이 설정은 {values}이다.',
        'source_directories': '개별 실행 원자료 디렉터리는 {values}이다.',
        'timing_heading': '고정 설정온도 스텝 시간',
        'timing_description': (
            '주중 07–19시는 21–24 °C, 그 밖의 시간은 19–26 °C 설정을 '
            '적용했다. 첫 스텝을 제외한 평균에는 설정온도 적용, 시뮬레이션, '
            '관측과 EnergyPlus 프로세스 통신이 포함된다.'
        ),
        'timing_header': (
            '| 존 수 | 엔진 | 초기화 초 | 리셋 초 | 첫 스텝 초 | 이후 '
            'ms/스텝 | 상태 수 |'
        ),
        'timing_comparison': (
            '{largest}존의 전체 JAX와 EnergyPlus 스텝 시간은 그룹 JAX의 각각 '
            '{full_ratio:.2f}배와 {energyplus_ratio:.2f}배였다.'
        ),
        'timing_alt': '격자 및 엔진별 스텝 시간',
        'timing_artifact': '측정값 `timing.csv`',
        'physics_heading': '장파 선형화와 격자 그룹화에 따른 온도 차이',
        'physics_description': (
            '{zones}존 전체 비선형 SBSim에서 적용한 디퓨저별 HVAC 열을 '
            '기록했다. 전체 선형 및 그룹 선형 SBSim에 같은 열을 재생해 장파 '
            '선형화와 격자 그룹화의 효과를 비교했다. 세 변형 모두 실내·외 '
            '복사와 내부 열질량을 사용한다.'
        ),
        'physics_error_header': (
            '| 비교 | 평균 절대 오차 K | RMSE K | 95백분위 K | 최대 K |'
        ),
        'comparison_labels': {
            'full_linear_minus_full_nonlinear': '전체 선형 대 전체 비선형',
            'grouped_linear_minus_full_nonlinear': (
                '그룹 선형 대 전체 비선형'
            ),
            'grouped_linear_minus_full_linear': '그룹 선형 대 전체 선형',
        },
        'physics_metrics_header': (
            '| 열수지 | 풀이 | 상태 수 | LU·선형 풀이 횟수별 스텝 수 | 이후 '
            'ms/스텝 |'
        ),
        'solve_count': '{count}회 풀이 × {steps:,}스텝',
        'physics_timing_note': (
            '전체 비선형 기준 시간에는 HVAC 피드백과 열량 기록이, 두 선형 '
            '모델의 시간에는 고정 열량 재생과 존 온도 읽기가 들어간다.'
        ),
        'physics_alt': '대표 존의 온도와 시간별 온도 오차',
        'physics_artifacts': (
            '5분 온도 시계열 `physics.csv`, 오차 통계 `physics_errors.csv`, '
            '온도와 HVAC 열 `physics.npz`'
        ),
        'sac_heading': '실제 SAC 학습',
        'sac_description': (
            '환경마다 한 에피소드, 시드 {seed}로 JAX SAC를 학습했다. '
            '{collect_every}개 전이마다 {updates}회 업데이트하고 배치 크기는 '
            '{batch_size}다. 각 존의 난방·냉방 설정을 독립 제어하고, 21–24 '
            '°C 범위를 벗어난 온도의 존 평균에 음의 부호를 붙여 보상으로 '
            '사용했다.'
        ),
        'sac_header': (
            '| 존 수 | 엔진 | 열수지 장치 | 학습 장치 | 전이 | 업데이트 | '
            '전체 초 | 시뮬레이터 ms/전이 |'
        ),
        'sac_alt': '엔진과 존 수에 따른 JAX SAC 실행 시간',
        'sac_artifact': 'SAC 시간 및 학습 통계 `sac.csv`',
        'summary_artifact': '환경과 집계값 `summary.json`',
    },
}


def _markdown_lines(summary, language):
  text = _REPORT_TEXT[language]
  setup = summary['setup']
  lines = [
      text['language_link'],
      '',
      f"# {text['title']}",
      '',
      text['weather'].format(
          start_day=setup['start_day'],
          days=setup['days'],
          epw=setup['epw'],
          cv_size_m=setup['cv_size_m'],
          window_width_m=setup['window_width_m'],
          episode_steps=setup['episode_steps'],
      ),
      '',
      text['jax'].format(
          jax=setup['jax'],
          jax_device=setup['jax_device'],
          energyplus_input_version=setup['energyplus_input_version'],
      ),
      '',
      text['model_comparison'],
      '',
  ]
  affinity = summary.get('section_cpu_affinity')
  if affinity:
    values = ', '.join(
        f'{section} `{cores}`' for section, cores in affinity.items()
    )
    lines.extend([text['section_affinity'].format(values=values), ''])
  else:
    lines.extend([
        text['run_affinity'].format(values=setup['cpu_affinity']),
        '',
    ])
  methods = summary.get('section_grouped_method', {})
  if methods:
    values = ', '.join(
        f'{section} `{method}`' for section, method in methods.items()
    )
    lines.extend([text['grouped_methods'].format(values=values), ''])
  sources = summary.get('section_artifact_dir', {})
  if sources:
    values = ', '.join(
        f'{section} `{path}`' for section, path in sources.items()
    )
    lines.extend([text['source_directories'].format(values=values), ''])
  if 'timing' in summary:
    lines.extend([
        f"## {text['timing_heading']}",
        '',
        text['timing_description'],
        '',
        text['timing_header'],
        '| ---: | --- | ---: | ---: | ---: | ---: | ---: |',
    ])
    for row in summary['timing']:
      lines.append(
          f"| {row['zones']} | {_LABELS[row['engine']]} |"
          f" {row['setup_s']:.3f} | {row['reset_s']:.3f} |"
          f" {row['first_step_s']:.3f} | {row['warm_ms']:.3f} |"
          f" {row.get('thermal_nodes', '—')} |"
      )
    largest = max(row['zones'] for row in summary['timing'])
    results = {
        r['engine']: r['warm_ms']
        for r in summary['timing']
        if r['zones'] == largest
    }
    if {'full_jax', 'grouped_jax', 'energyplus'} <= results.keys():
      grouped = results['grouped_jax']
      lines.extend([
          '',
          text['timing_comparison'].format(
              largest=largest,
              full_ratio=results['full_jax'] / grouped,
              energyplus_ratio=results['energyplus'] / grouped,
          ),
      ])
    lines.extend([
        '',
        f"![{text['timing_alt']}](timing.png)",
        '',
        text['timing_artifact'],
        '',
    ])
  if 'physics' in summary:
    result = summary['physics']
    lines.extend([
        f"## {text['physics_heading']}",
        '',
        text['physics_description'].format(zones=result['zones']),
        '',
        text['physics_error_header'],
        '| --- | ---: | ---: | ---: | ---: |',
    ])
    labels = text['comparison_labels']
    for row in result['comparisons']:
      lines.append(
          f"| {labels[row['comparison']]} | {row['mae_k']:.4f} |"
          f" {row['rmse_k']:.4f} | {row['p95_absolute_k']:.4f} |"
          f" {row['max_absolute_k']:.4f} |"
      )
    if result.get('metrics'):
      lines.extend([
          '',
          text['physics_metrics_header'],
          '| --- | --- | ---: | --- | ---: |',
      ])
    for row in result.get('metrics', []):
      solves = ', '.join(
          text['solve_count'].format(
              count=count,
              steps=steps,
              solve='solve' if str(count) == '1' else 'solves',
          )
          for count, steps in row['linear_solves_per_step'].items()
      )
      lines.append(
          f"| {_LABELS[row['variant']]} | `{row['method']}` |"
          f" {row['thermal_nodes']:,} | {solves} | {row['mean_step_ms']:.3f} |"
      )
    if result.get('metrics'):
      lines.extend(['', text['physics_timing_note']])
    lines.extend([
        '',
        f"![{text['physics_alt']}](physics.png)",
        '',
        text['physics_artifacts'],
        '',
    ])
  if 'sac' in summary:
    lines.extend([
        f"## {text['sac_heading']}",
        '',
        text['sac_description'].format(
            seed=setup['seed'],
            collect_every=setup['collect_every'],
            updates=setup['updates'],
            batch_size=setup['batch_size'],
        ),
        '',
        text['sac_header'],
        '| ---: | --- | --- | --- | ---: | ---: | ---: | ---: |',
    ])
    for row in summary['sac']:
      simulator_platform = (
          'cpu' if row['engine'] == 'energyplus' else setup['platform']
      )
      learner_platform = row.get('learner_platform', setup['platform'])
      lines.append(
          f"| {row['zones']} | {_LABELS[row['engine']]} |"
          f" {simulator_platform.upper()} | {learner_platform.upper()} |"
          f" {row['transitions']:,} | {row['gradient_updates']:,} |"
          f" {row['elapsed_s']:.2f} |"
          f" {row['simulator_ms_per_transition']:.3f} |"
      )
    lines.extend([
        '',
        f"![{text['sac_alt']}](sac.png)",
        '',
        text['sac_artifact'],
        '',
    ])
  lines.append(text['summary_artifact'])
  return lines


def markdown(root, summary):
  """Write matching English and Korean Markdown reports."""
  for language, filename in (('en', 'REPORT.md'), ('ko', 'REPORT_ko.md')):
    lines = _markdown_lines(summary, language)
    (root / filename).write_text('\n'.join(lines) + '\n', encoding='utf-8')


def merge(root, inputs):
  """Publish one report from independently executed sections."""
  summaries = [
      (path, json.loads((path / 'summary.json').read_text(encoding='utf-8')))
      for path in inputs
  ]
  setup = summaries[0][1]['setup']
  section_keys = {'cpu_affinity', 'grouped_method'}
  common_setup = {k: v for k, v in setup.items() if k not in section_keys}
  measurements = {}
  affinity = {}
  methods = {}
  sources = {}
  for path, summary in summaries:
    common = {k: v for k, v in summary['setup'].items() if k not in section_keys}
    if common != common_setup:
      raise ValueError(f'Benchmark configuration differs in {path}')
    for section in summary['sections']:
      if section in measurements:
        raise ValueError(f'Duplicate {section} results in {path}')
      measurements[section] = summary[section]
      affinity[section] = summary['setup']['cpu_affinity']
      methods[section] = summary.get('section_grouped_method', {}).get(
          section, summary['setup']['grouped_method']
      )
      sources[section] = path
  root.mkdir(parents=True, exist_ok=False)
  for section, path in sources.items():
    for name in _ARTIFACTS[section]:
      shutil.copyfile(path / name, root / name)
  summary = dict(
      sections=list(measurements),
      setup=setup,
      section_cpu_affinity=affinity,
      section_grouped_method=methods,
      section_artifact_dir={
          section: os.path.relpath(path.resolve(), root.resolve())
          for section, path in sources.items()
      },
      **measurements,
  )
  payload = json.dumps(summary, indent=2, allow_nan=False) + '\n'
  (root / 'summary.json').write_text(payload, encoding='utf-8')
  markdown(root, summary)


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description=merge.__doc__)
  parser.add_argument('--output', type=Path, required=True)
  parser.add_argument('inputs', type=Path, nargs='+')
  args = parser.parse_args()
  merge(args.output, args.inputs)

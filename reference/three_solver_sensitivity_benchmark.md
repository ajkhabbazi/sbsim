# Heat-balance three-solver benchmark와 sensitivity analysis

작성일: 2026년 9월 17일

이 benchmark는 동일한 유한체적 열평형식을 다음 세 backend로 비교한다.

1. `NumpyScipySimulator`: NumPy coefficient assembly와 SciPy `spsolve`
2. `TFSimulator(solver_method="cg")`: TensorFlow matrix-free Jacobi-PCG
3. `TFSimulator(solver_method="fdm")`: TensorFlow Jacobi relaxation

`TFSimulator`는 SciPy를 import하지 않으며 linear solve를 위해 Tensor를 NumPy로 변환하지 않는다.
TensorFlow PCG는 `float64` inner solve를 사용하고, physics coefficient와 최종 building
temperature는 기존 TensorFlow 경로와 같이 `float32`다.

## 실행 방법

```bash
cache_dir=$(mktemp -d /tmp/sbsim-pycache-three-solver.XXXXXX)
MPLCONFIGDIR=/tmp/matplotlib-cache \
PYTHONPYCACHEPREFIX="$cache_dir" \
poetry run python \
  reference/three_solver_sensitivity_benchmark.py \
  --hours 1 \
  --time-step-seconds 300 \
  --repeats 3 \
  --sensitivity-sizes 8,16,24,32,48,64 \
  --sensitivity-repeats 2 \
  --cg-sensitivity-tolerances 1e-6,1e-8,1e-10
```

TF-PCG의 사전 warm-up 시간은 CSV의 `cg_warmup_ms`에 별도로 기록하고, timestep boxplot과 size
sensitivity의 `elapsed_ms`에서는 제외한다. 첫 호출이면 이 값에 `tf.function` tracing도 포함될 수 있지만,
이미 같은 signature가 trace되었다면 단순 warm-up 시간만 포함하므로 compile-only time으로 해석하지 않는다.

## 1시간 multistep 결과

환경은 AMD Ryzen 7 5800X CPU, TensorFlow 2.15.1, SciPy 1.15.3이다. 300초 timestep 12개를
repeat 3회 실행해 case와 solver마다 36개 timing sample을 얻었다.

| case               | NumPy/SciPy ms | TF-PCG ms | TF-FDM ms | FDM/PCG | FDM/direct | PCG peak error K | FDM peak error K |
| ------------------ | -------------: | --------: | --------: | ------: | ---------: | ---------------: | ---------------: |
| base               |          1.798 |    17.333 |   250.976 |   14.5x |     139.6x |         0.000181 |         0.000915 |
| interior mass      |          3.809 |    33.191 |   267.871 |    8.1x |      70.3x |         0.000173 |         0.000902 |
| radiation          |          3.354 |    33.509 |   290.934 |    8.7x |      86.7x |         0.000177 |         0.000991 |
| mass and radiation |          7.311 |    67.764 |   485.457 |    7.2x |      66.4x |         0.000689 |         0.000916 |

현재 test fixture 크기에서는 NumPy/SciPy direct가 가장 빠르다. TF-PCG는 direct보다 약 8.7--10.0배
느리지만 TF-FDM보다 약 7.2--14.5배 빠르다. 모든 case에서 direct reference와 TF-PCG의 최대 공기 온도 차이는
0.689 mK 이하이고 TF-FDM은 0.991 mK 이하다. 내부 질량 온도의 최대 차이는 두 TensorFlow method 모두
0.239 mK 이하다.

TF-PCG의 inner iteration 중앙값은 base 14회, interior mass 2회, radiation 4회, mass와
radiation을 함께 켠 경우 3회였다. Interior mass와 radiation case의 outer fixed-point
iteration마다 새 linearized system을 푼다는 점에서 이 값은 outer iteration 수가 아니라 마지막 inner
PCG iteration 수다.

## Grid-size sensitivity

복사와 내부 질량이 없는 정사각형 room에서 active CV 수를 키워 한 timestep의 warm 실행 시간을 비교했다.

| active CVs | NumPy/SciPy ms | TF-PCG ms | TF-FDM ms |
| ---------: | -------------: | --------: | --------: |
|        100 |          1.547 |    20.631 |   477.258 |
|        324 |          4.905 |    23.766 |   600.903 |
|        676 |          9.871 |    20.389 |   629.301 |
|      1,156 |         16.181 |    20.209 |   632.545 |
|      2,500 |         35.052 |    25.470 |   687.119 |
|      4,356 |         68.871 |    25.376 |   696.875 |

이 CPU 환경에서는 TF-PCG와 현재 NumPy/SciPy 구현의 crossover가 약 1,156--2,500 active CV 사이에
나타났다. 이는 SciPy solver 자체만의 비교가 아니라 NumPy class의 Python row assembly까지 포함한
end-to-end timestep 비교다. Sparse matrix assembly를 더 vectorize하면 direct crossover가
더 큰 grid로 이동할 수 있다.

TF-FDM은 grid가 커져도 iteration count와 Python outer-loop overhead 때문에 세 방법 중 가장 느렸다.
GPU와 여러 building batch를 포함하면 crossover는 달라질 수 있으므로 이 표를 GPU 결론으로 일반화하지 않는다.

## CG tolerance sensitivity

Base case 한 timestep에서 NumPy/SciPy direct solution을 reference로 비교했다.

| CG relative residual tolerance | median inner iterations | median ms | peak air difference K |
| -----------------------------: | ----------------------: | --------: | --------------------: |
|                      $10^{-6}$ |                      10 |    14.979 |              0.105008 |
|                      $10^{-8}$ |                      18 |    15.878 |              0.000320 |
|                     $10^{-10}$ |                      25 |    16.240 |              0.000164 |

$10^{-6}$은 약 1.3 ms를 절약하지만 0.105 K 오차를 남겨 heat-balance reference comparison에는 너무
느슨하다. 기본값 $10^{-10}$은 accuracy-first 설정이다. $10^{-8}$도 이 fixture에서는 sub-mK이므로
throughput 우선 설정의 후보지만, 다른 conductivity contrast와 grid size에서도 별도 확인해야 한다.

## 생성 파일

기본 출력 경로는 `reference/benchmark_outputs/three_solver_sensitivity/`다.

- `multistep_timing.csv`, `multistep_errors.csv`, `multistep_summary.csv`
- `size_sensitivity.csv`, `cg_tolerance_sensitivity.csv`
- `timing_boxplots.png`, `size_sensitivity.png`
- `cg_tolerance_sensitivity.png`
- 네 물리 조합의 `heatmap_*.png`

Heatmap은 NumPy/SciPy direct, TF-PCG, TF-FDM의 온도장과 두 TensorFlow 결과의 direct
reference 대비 절대 차이를 같은 그림에 표시한다.

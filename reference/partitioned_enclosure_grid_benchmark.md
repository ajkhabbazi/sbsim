# Partitioned Enclosure Grid Benchmark

이 벤치마크는 방 하나의 내부 격자 크기는 고정하고, 정사각형 방의 개수를 늘려 전체 격자와 radiative enclosure 수가 함께
증가하도록 구성한다.

각 방은 주변 방과 벽으로 분리되며 건물 생성 시 `partition_radiative_enclosures=True`를 명시한다. 따라서 복사
view-factor와 IFA 행렬은 모든 복사 노드를 포함하는 단일 dense 행렬이 아니라 enclosure별 block으로 생성된다.

기본 설정은 다음과 같다.

- 방 하나의 실제 내부 크기: 5 m × 5 m
- 방 하나의 실제 내부 면적: 25 m², 약 269.1 ft²
- CV 크기: 20 cm × 20 cm
- 방 하나의 내부 격자: 25×25 CV
- 한 변의 방 개수: 1, 2, 3, 4, 5, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28
- 전체 enclosure 수: `rooms_per_side²`
- 각 repeat의 simulation 길이: 12 steps × 300 seconds = 1 hour
- Picard 및 CG 최대 반복 횟수: 5,000
- SciPy 수렴 기준: 1e-4 K
- FP32 온도장을 사용하는 TensorFlow solver 수렴 기준: 1.5e-4 K
- applications: base, base + radiation, base + radiation + interior mass
- CPU solvers: NumPy/SciPy direct, TensorFlow PCG, resident FP32 PCG, FDM
- GPU solvers: TensorFlow PCG, resident FP32 PCG, FDM

최대 기본 형상은 28×28, 총 784개 방과 758×758 전체 grid다. 5 m × 5 m 방의 순수 실내 면적 합은 약 211,000
ft²이며, 벽 CV까지 포함하여 그래프에 표시되는 active footprint는 약 246,100 ft²다.

## 실행

### 전체 격자와 grouped JAX 비교

저장소 루트에서 다음 스크립트를 실행한다. `jax_cg`와 `grouped_jax`를 각각 CPU와 GPU에서 측정한다. 네 경우 모두 실내
장파, 외부 장파·일사, 내부 열질량이 켜진 `base_rad_interior_mass_exterior`를 사용한다. 두 모델에 같은 창과 대기
조건을 주고, 창을 통과한 일사는 건물의 내부 열질량에 넣는다.

```bash
bash reference/run_partitioned_enclosure_grouped_jax.sh
```

`jax_cg`는 전체 격자에서 비선형 복사를 반복 반영한다. `grouped_jax`는 실내·외부 장파를 선형화하고 그룹 공기·벽·창·내부
열질량을 JAX CG로 함께 푼다. CPU와 GPU의 `grouped_jax`는 같은 배정밀도 식과 CG 수렴 기준을 사용한다. 그룹 솔버의
`thermal_node_count`는 질량을 포함한 총 상태 수이고, `mass_node_count`는 그중 질량 상태 수다.
`warmup_ms`에 열수지 조립과 JAX 컴파일을 기록한다. `elapsed_ms`에는 그 이후의 열수지 스텝과 필요한 호스트·장치 사이
데이터 이동을 포함한다.

정확도 기준은 전체 격자의 배정밀도 비선형 희소 직접 풀이다. 그룹 모델은 방 안의 공기 온도를 하나로 묶고 면적비 실내 복사를 간소화하므로
전체 모델과의 오차에 그룹화와 복사 근사가 함께 반영된다. 격자 최대 오차는 고정 열량 50 W가 들어가는 첫 디퓨저 셀의 국소 온도차를
포함한다. 존 평균을 비교하려면 저장된 온도장을 각 방의 공기 셀에서 평균한다.

기본은 5 m 정사각형 방, 20 cm 격자, 방 수 1·4·16·36·64·100, 5분 스텝 12회, 규모당 3회 반복이다.
`ROOMS_PER_SIDE`는 전체 방 수의 제곱근이다. CPU만 빠르게 확인하려면 별도 결과 경로에서 다음과 같이 실행한다.

```bash
RUN_GPU=0 ROOMS_PER_SIDE=1,2 REPEATS=1 \
OUTPUT_DIR=/tmp/sbsim-partitioned-grouped-cpu \
bash reference/run_partitioned_enclosure_grouped_jax.sh
```

GPU는 기본으로 `.venv/jax_cuda/bin/python`을 사용한다. 다른 Python 환경에 CUDA용 JAX를 설치했다면
`JAX_GPU_PYTHON`에 그 실행 파일의 절대 경로를 지정한다. 해당 Python에서
`import jax; print(jax.devices("gpu"))`로 장치를 확인할 수 있다. `RUN_CPU=0`으로 GPU만 실행할 수
있다.

기본 결과 위치는
`reference/benchmark_outputs/partitioned_enclosure_sensitivity/grouped_jax_full_physics/`다.
`cpu/`와 `gpu/`의 `partitioned_enclosure_grid_time.csv`에 측정값을 기록한다.
`all_solvers/all_solvers_time.png`와 `all_solvers/all_solvers_accuracy.png`에는 시간과
비선형 희소 직접 풀이 기준 오차를, `prediction_fields/`에는 마지막 스텝의 격자온도 및 질량온도 오차를 저장한다. GPU
시작·실행 실패는 `benchmark_failures.csv`에 기록한다.

### 여러 솔버와 물리 옵션 비교

```bash
bash reference/run_partitioned_enclosure_all_solvers.sh
```

작은 CPU 검증 실행 예시는 다음과 같다.

```bash
RUN_GPU=0 \
ROOMS_PER_SIDE=1,2,3 \
ROOM_SIZE_METERS=5 \
CV_SIZE_CM=20 \
SIMULATION_STEPS=12 \
REPEATS=1 \
CASE_TIMEOUT=10m \
bash reference/run_partitioned_enclosure_all_solvers.sh
```

기본 결과 경로는 기존 benchmark와 분리된 다음 디렉터리다.

```text
reference/benchmark_outputs/partitioned_enclosure_sensitivity/all_solvers_uniform/
```

주요 결과는 다음과 같다.

- `cpu/partitioned_enclosure_grid_time.csv`
- `gpu/partitioned_enclosure_grid_time.csv`
- `cpu_gpu_partitioned_enclosure_grid_time.csv`
- `applications_by_solver_<device>_<solver>.png`
- `partitioned_enclosure_memory_scaling.png`
- `cpu_gpu_grid_time_<application>.png`
- `cpu_gpu_grid_time_linear_<application>.png`
- `cpu_gpu_boxline_<application>.png`
- `cpu_gpu_boxline_linear_<application>.png`
- `cpu_gpu_speedup_<application>.png`
- `benchmark.log`
- `benchmark_failures.csv`
- `benchmark_timeouts.csv`

CSV는 timestep 시간뿐 아니라 다음 setup 및 메모리 지표도 기록한다.

- `building_setup_ms`: 건물 및 view-factor block 생성 시간
- `solver_build_ms`: solver 객체 생성 시간
- `warmup_ms`: TensorFlow tracing/warmup 시간
- `elapsed_ms`: 모든 simulation step을 포함한 전체 시간
- `mean_step_ms`: step당 평균 시간
- `median_step_ms`: step당 중앙값 시간
- `maximum_step_ms`: 가장 느린 step 시간
- `simulation_steps`: 측정한 연속 timestep 수
- `simulated_duration_seconds`: 물리적으로 시뮬레이션한 시간
- `radiative_enclosures`: 실제 생성된 enclosure block 수
- `max_enclosure_nodes`: 가장 큰 enclosure의 복사 노드 수
- `radiation_matrix_entries`: 저장된 view-factor 및 IFA block 원소 수
- `radiation_matrix_mib`: 저장된 view-factor 및 IFA block 메모리

`radiation_matrix_mib`는 최종적으로 보관되는 행렬 block만 계산하며, 행렬 생성 중 발생하는 임시 배열의 peak
memory는 포함하지 않는다.

Interior mass application도 기본 실행에 포함한다. 이 case는 기존 solver의 air solve와 mass
closed-form 갱신이 radiation Picard loop 안에서 함께 수렴하는 비용을 측정한다. Air와 mass를 같은 block
system으로 푸는 새 coupled solver는 별도의 coupled partitioned-enclosure benchmark에서
비교한다.

모든 grid scaling 그래프에는 active control volume 축과 함께 실제 active floor area를 ft² 단위로
표시한다. `elapsed_ms` 그래프는 기본 설정에서 단일 step이 아니라 12-step, 1-hour simulation 전체 시간을
나타낸다.

# SBSim 전체·그룹 격자 열수지

**한국어** | [English](README.md)

`GroupedJaxSimulator`는 기존 SBSim 평면, HVAC, 기상과 `step_sim()`을 사용한다. 전체 격자와 그룹 격자에
같은 유한체적 열수지와 존별 실내 장파식을 적용한다. 실내 장파, 외부 복사, 내부 열질량, 장파 선형화와 풀이 방식을 지정해 계산량과 온도
차이를 비교할 수 있다.

이 폴더에서는 [`run.py`](run.py)가 실행·측정하고 [`report.py`](report.py)가 표·CSV·그림을 만든다.
`results/`는 이 API로 생성한 원자료와 보고서다. 공통 사무실 평면·기상 입력은 상위 벤치마크의 재사용 모듈을,
EnergyPlus·SAC 어댑터는 [`eplus_env.py`](eplus_env.py), [`physics.py`](physics.py),
[`train.py`](train.py), [`sac.py`](sac.py)를 사용한다.

## 기존 SBSim 모델에서 사용

`FloorPlanBasedBuilding`, `FloorPlanBasedHvac` 및 기상 컨트롤러를 기존 코드에서 만들고 시뮬레이터 클래스만
바꾼다. 아래 `floorplan_building`, `hvac`, `weather`, `start_timestamp`는 그 코드의 객체와 시작
시각이다.

```python
from smart_control.simulator.grouped_heat_balance import InteriorMass
from smart_control.simulator.grouped_jax_simulator import GroupedJaxSimulator

sim = GroupedJaxSimulator(
    building=floorplan_building,
    hvac=hvac,
    weather_controller=weather,
    start_timestamp=start_timestamp,
    time_step_sec=300,
    convergence_threshold=1e-6,
    iteration_limit=100,
    iteration_warning=100,
    grouped=True,
    include_interior_radiative_heat_transfer=True,
    include_exterior_radiative_heat_transfer=True,
    include_interior_mass=True,
    interior_mass=InteriorMass(),
    linearized_radiative_heat_transfer=True,
    platform='cpu',
    solver_method='sparse_lu',
)
sim.reset()
h = weather.get_air_convection_coefficient(start_timestamp)
sim.prepare_thermal_system(h)
sim.step_sim()
zone_temperatures_k = sim.building.get_zone_average_temps()
```

`prepare_thermal_system(h)`는 선택 사항이다. 생략하면 첫 열수지 스텝에서 준비한다. 외기 대류 계수 $h$가 같으면
조립과 준비 결과를 재사용한다. 시작 시각과 `time_step_sec`는 기상 데이터가 지원하는 범위로 맞춘다.

`TFSimulator`와 `GroupedJaxSimulator`는 `SimulatorFlexibleGeometries`의 공통 호출 규약을
사용한다. 기존 평면 기반 RL 코드에서는 `SimulatorBuilding(simulator=sim, occupancy=occupancy)`에
위 `sim`을 넘긴다. 기존 관측·행동과 VAV 호출을 그대로 쓸 수 있다. 두 모델을 나란히 실행할 때는 가변 상태가 섞이지 않도록
Building과 HVAC를 솔버별로 생성한다. `solver_method`는 각 솔버가 지원하는 값으로 설정한다.

| 옵션                                       | 기본값                  | 의미                                                                                                                                                                |
| ------------------------------------------ | ----------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `grouped`                                  | `True`                  | 연결된 공기와 일정한 경계를 가진 벽·창 구간을 묶는다. `False`는 활성 셀마다 온도 상태를 둔다.                                                                       |
| `include_interior_radiative_heat_transfer` | `None`                  | 실내 장파다. `None`은 `Building.include_radiative_heat_transfer`를 따른다.                                                                                          |
| `include_exterior_radiative_heat_transfer` | `None`                  | 외부 장파와 흡수·투과 일사다. `None`은 `Building.include_exterior_radiative_heat_transfer`를 따른다.                                                                |
| `include_interior_mass`                    | `None`                  | 내부 열질량이다. `None`이면 `InteriorMass` 명시 여부 또는 `Building.include_interior_mass`로 정한다. `False`는 질량을 끈다.                                         |
| `interior_mass`                            | `None`                  | 명시하면 바닥면적당 유효 물성을 사용한다. 생략 시 활성화된 기존 Building 질량 물성을 사용한다. 질량을 켰고 Building 물성이 없으면 기본 `InteriorMass()`를 사용한다. |
| `linearized_radiative_heat_transfer`       | `True`                  | 22.5 °C에서의 고정 기울기와 직전 온도 보정을 사용한다. `False`는 새 온도의 장파를 반복 반영한다.                                                                    |
| `platform` / `solver_method`               | `'cpu'` / `'sparse_lu'` | 계산 장치와 선형 시스템 풀이를 지정한다.                                                                                                                            |

`wall_depth_cells`는 벽 두께를 셀 단위로 받으며 기본은 1이다. 예를 들어 두께 0.50 m를 0.25 m 셀로 표현하면 층 두
개를 지정한다. 그룹 모드에서 `air_block_cells`를 지정하면 연결 공기를 더 작은 격자 블록으로 제한할 수 있다.

`InteriorMass()`의 기본 유효 물성은 열용량 30 kJ/(m²·K), 공기와의 교환 3 W/(m²·K), 복사에 노출된 면적은
바닥면적의 1배, 방사율은 0.90이다. `capacity_per_m2`, `air_conductance_per_m2`,
`exposed_area_per_m2`, `emissivity`로 바꾼다. 이 값은 가구와 내부 물체의 효과를 나타내며 실측 보정이 필요할 때
모델 입력으로 다룬다. 외부 복사를 켜면 기상 컨트롤러의 복사 입력과 Building의 외벽·창 광학 물성을 사용한다.

창이 있는 입력은 Building에 `include_exterior_radiative_heat_transfer=True`를 지정해 창을
경계면으로 먼저 구성한다. 이후 솔버의 외부 복사 스위치로 계산을 켜고 끌 수 있다. Building의 기존 체적 질량 물성을 쓸 때에는 평면의
외곽을 외기 셀로 패딩한다. 솔버 옵션으로 지정하는 `InteriorMass()`는 자체 바닥면적당 물성을 사용한다.

준비 뒤 `thermal_node_count`와 `mass_node_count`에서 상태 수를, `group_map`에서 셀별 온도 상태 번호를
확인할 수 있다. `group_map`의 `-1`은 외기 경계다. 스텝 뒤 `state_temperatures`는 전체 상태의 켈빈 온도이고,
`mass_temperature`는 질량이 활성화된 위치의 온도를 원래 격자에 표시한다. 기존 시각화에는 `sim.building.temp`를
사용할 수 있다. `last_solve_count`는 한 스텝에서 선형 시스템을 푼 횟수다. CG 자체의 반복 횟수와는 구분한다.

## 풀이 방식

그룹 모드는 같은 온도를 쓰는 셀의 열수지를 합해 `Pᵀ A P`를 만든다. 실내 장파의 면적비 식과 외부 장파를 `R`로 모으면, 선형 스텝은
`A₀ + sR`의 희소 열수지 하나로 공기·벽·창·질량을 함께 푼다. `s = 4 × 295.65³`은 고정 기울기다. 직전 온도 및 기상,
단파와 HVAC 열량을 우변에 넣는다.

| 풀이        | CPU              | GPU      | 처리                                                                      |
| ----------- | ---------------- | -------- | ------------------------------------------------------------------------- |
| `sparse_lu` | SciPy            |          | 희소 LU를 재사용한다. 선형이면 스텝당 한 번 직접 푼다.                    |
| `cg`        | JAX              | JAX      | 희소 켤레기울기다. 선형이면 장파 외부 반복 없이 내부 CG 반복을 수행한다.  |
| `dense`     | JAX              | JAX      | 호스트에서 희소 LU로 밀집 응답을 준비한 뒤 장치에서 행렬 곱으로 적용한다. |
| `spsolve`   | JAX의 SciPy 경로 | JAX CUDA | 희소 직접 풀이다. CUDA는 희소 QR을 사용한다.                              |

GPU 실행은 JAX가 지원 장치를 인식하는 환경에서 `platform='gpu'`와 JAX 풀이를 지정한다. 단일 모델의
`GroupedJaxSimulator`는 기존 호스트의 VAV를 실행하고 스텝 뒤 결과를 호스트의 Building 격자에 기록한다. SAC와
여러 건물 배치를 장치에서 실행한 이전 `JaxOffice`의 결과는 별도 구현의 측정치다.

행렬 유도와 예제는
[`benchmarks/grouped_jax_study/HEAT_BALANCE_EQUATIONS_ko.md`](HEAT_BALANCE_EQUATIONS_ko.md),
기존 평면의 그룹 지도 및 과거 SAC 결과는
[`benchmarks/grouped_jax_study/LUMPED_MODEL_ko.md`](LUMPED_MODEL_ko.md)에 있다.

## 작은 합성 평면에서 실행

저장소 루트에서 다음 명령은 기상 파일 없이 네 풀이를 작은 합성 평면에 적용한다. 첫 스텝 시간에는 지연된 행렬 준비와 필요한 JAX 컴파일이
들어간다. 후기 스텝 시간과 첫 풀이 대비 최대 절대 격자 온도 차이도 함께 출력한다.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python \
  -m smart_control.simulator.benchmark_solvers --tiles 1 2 --steps 3 \
  --solvers grouped_jax grouped_jax_cg grouped_jax_dense grouped_jax_spsolve
```

GPU에서 실행할 때는 JAX 풀이만 지정하고 `--grouped-platform gpu`를 추가한다.

## 사무실 벤치마크 재현

주간 비교는 공통 사무실 평면과 Chicago O’Hare TMY3 EPW를 사용한다. 기본은 2024년 7월 15일부터 7일, 5분 간격,
SBSim 격자 0.50 m다. SBSim 세 조건은 같은 면적비 실내 장파, 외부 복사, 유효 내부 열질량과 VAV 계통을 사용한다.
EnergyPlus는 공통 존의 배치에 상세 사무실 공조 및 외피 모델을 적용한다. `EPW`에는 기상 파일 경로를, `EPLUS_DIR`에는
`pyenergyplus`가 있는 EnergyPlus 설치 경로를 지정한다.

CPU와 GPU 명령을 실행할 터미널에서 설치 경로를 먼저 설정한다. 아래는 이 머신에 복원한 EnergyPlus 24.2의 경로이며 CPU
보고서와 동일한 EPW를 포함한다. 다른 머신에서는 설치 경로와 EPW 경로를 해당 환경에 맞게 지정한다.

```bash
EPLUS_DIR="$HOME/.local/share/sbsim/energyplus-24.2"
EPW="$EPLUS_DIR/WeatherData/USA_IL_Chicago-OHare.Intl.AP.725300_TMY3.epw"
test -r "$EPW" && test -r "$EPLUS_DIR/pyenergyplus/api.py"
```

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  taskset -c 8 .venv/bin/python -m benchmarks.grouped_jax_study.run \
  --epw "$EPW" --energyplus-dir "$EPLUS_DIR" \
  --zones 16 36 100 --physics-zones 100 --days 7 \
  --platform cpu --grouped-method dense --physics-grouped-method sparse_lu \
  --output /tmp/sbsim-grouped-jax-study
```

`taskset -c 8`은 Linux에서 실행을 논리 CPU 하나에 고정하는 예다. 현재 프로세스에 허용된 코어 번호로 바꾼다. 새 실행에는
새 `--output` 경로를 지정한다. `--section timing`, `--section physics`, `--section sac`를
지정해 필요한 부분만 실행할 수 있다. 여러 부분을 실행하려면 옵션을 반복한다. 생략하면 세 부분을 모두 실행한다. `physics`만 실행할
때는 `--energyplus-dir`를 생략할 수 있다. 연구 실행의 기본 그룹 솔버는 `dense`다. 위 명령의
`--physics-grouped-method sparse_lu`는 물리 재생에서만 CPU 희소 직접 풀이를 사용한다.

중단한 실행은 이전 프로세스를 종료한 뒤 같은 설정과 `--output`에 `--resume`을 추가한다. 완료된 측정 부분과 저장된 개별
timing·SAC 케이스를 재사용한다. `--section sac --resume`은 기존 시간·온도 결과를 남긴 채 SAC만 실행한다. 학습
중간 상태는 저장하지 않으므로 중단된 에피소드는 처음부터 실행하며, 그때 남은 로그는 `interrupted/`에 보관한다. SAC는 환경 준비
완료, 학습기 초기화 시작·완료, 첫 업데이트 시작과 이후 수집 진행률을 출력한다.

SAC 학습이 장시간 진행되지 않을 때는 같은 명령에 `--sac-debug`를 추가한다.
`sac/<존 수>-<엔진>/sac_debug.log`에 첫 업데이트 묶음의 배치 전송과 JAX 호출, GPU 학습기의 JAX 메모리 할당
통계, Python 오류를 기록한다. 학습 중 45초마다 Python 호출 스택도 같은 파일에 남긴다. GPU를 사용하는 SAC 케이스에서
`--sac-debug`를 켜면 `sac/<존 수>-<엔진>/gpu_telemetry.csv`에 별도 `nvidia-smi` 프로세스가 2초마다
측정한 전체 VRAM, 사용량, 잔여량과 GPU 사용률을 기록한다. WSL에서 지원하지 않는 항목은 `N/A`로 기록될 수 있다. JAX의
할당량과 전체 장치 사용량은 각각 프로세스 할당기와 장치 전체를 나타낸다.

[`sac_probe.py`](sac_probe.py)는 16존 실행과 같은 17개 관측, 32개 행동, 배치 크기 256으로 SAC 업데이트를
16회 실행한다. 합성 궤적을 사용하며 시뮬레이터나 기상 파일을 열지 않는다. 60초 안에 끝나지 않으면 마지막 호출 스택을 출력하고 종료한다.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
.venv/jax_cuda/bin/python -m benchmarks.grouped_jax_study.sac_probe
```

`--platform gpu --sac-learner-platform cpu`는 SBSim 열수지에 GPU를 사용하고 SAC 정책과 업데이트를
CPU에서 수행한다. 같은 CPU 학습기를 전체 JAX, 그룹 JAX, EnergyPlus에 적용한다. 결과 표에 열수지 장치와 학습 장치를
함께 기록한다. GPU 학습 단계에서 정체된 실행은 `--section sac --resume`과 함께 이 설정으로 이어갈 수 있다. 완료된
SAC 결과의 학습 장치가 다르면 새 `--output`을 지정해 두 측정을 보관한다.

| 실행      | 비교                                                                                                                                                                                    |
| --------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `timing`  | 여러 존 규모에서 전체 격자 비선형 JAX CG, 그룹 격자 선형 및 상세 EnergyPlus를 실행한다. 같은 설정온도 시퀀스를 주고 준비, 리셋과 첫 스텝, 이후 스텝을 기록한다.                         |
| `physics` | 전체 비선형 SBSim의 온도와 실제 디퓨저별 HVAC 열량을 기록한다. 같은 열량으로 전체 선형과 그룹 선형을 재생해 온도 차이를 구한다. 전체 선형은 CPU에서 희소 LU, GPU에서 JAX CG를 사용한다. |
| `sac`     | 전체 비선형 SBSim, 그룹 선형 SBSim, 상세 EnergyPlus에 같은 JAX SAC 학습 구성을 적용한다. 환경 스텝, 정책 실행과 학습 업데이트를 구분해 기록한다.                                        |

기본 SAC 실행은 모델마다 에피소드 하나에 2,016개 전이를 수집하며, 96개 전이마다 배치 크기 256으로 8회씩 총 168회 학습
업데이트를 수행한다. 관측은 존별 공기와 외기온도, 행동은 존별 난방·냉방 설정온도, 보상은 고정 쾌적 범위 밖 온도 편차다. 실행이 끝날
때까지의 처리량을 측정하며 정책의 수렴이나 절감률은 이 실험의 범위 밖이다.

고정 열량 재생은 격자와 장파 근사의 효과를 분리한다. 기준 온도는 VAV가 반응하는 실행에서 기록하고, 두 후보는 그때 입력된 열량을 그대로
사용한다. 따라서 `physics`의 절대 실행 시간과 `timing`의 폐루프 스텝 시간은 각각의 측정 범위를 함께 봐야 한다. 폐루프에서는
작은 온도 차이도 VAV 제어 전환과 다음 스텝 열량을 바꿀 수 있다. 처리량은 사용한 시드, 존 규모, 솔버, 플랫폼, 에피소드 길이와 함께
읽는다.

## 결과 파일 읽기

이번 CPU 측정의 timing·physics·SAC 통합 보고서는
[`results/cpu/REPORT_ko.md`](results/cpu/REPORT_ko.md)다. 그림 세 장이 본문에 삽입돼 있다. 개별
실행의 원자료는 [`results/timing_cpu/`](results/timing_cpu/),
[`results/physics_cpu/`](results/physics_cpu/),
[`results/sac_cpu/`](results/sac_cpu/)에 있다. 통합 보고서는 이 세 디렉터리를
`python -m benchmarks.grouped_jax_study.report --output <새 디렉터리> <세 디렉터리>`로 다시
만들 수 있다.

다음 경로는 `--output`으로 넘긴 디렉터리 기준이다. 실행한 부분의 결과만 생성된다.

| 경로                                | 내용                                                                               |
| ----------------------------------- | ---------------------------------------------------------------------------------- |
| `REPORT.md`, `REPORT_ko.md`         | 영어·한국어 보고서의 실행 조건, 결과 표와 그림                                     |
| `summary.json`                      | EPW SHA-256, 장치·버전, 질량 물성, 시간과 제어 설정, 실행 부분별 결과              |
| `timing.csv`, `timing.png`          | 존 수와 엔진별 준비·리셋·스텝 시간. 스텝 평균과 95백분위수에서 첫 스텝을 제외한다. |
| `timing/<존 수>-<엔진>/rollout.npz` | 각 5분 스텝의 실행 시간과 존 온도                                                  |
| `physics.csv`, `physics.png`        | 5분 구간 종료 시각과 존별 온도, 대표 존과 시간별 오차 그림                         |
| `physics_errors.csv`, `physics.npz` | 비교별 온도 오차 요약, 원래 온도 시계열 및 재생용 디퓨저 열량                      |
| `sac.csv`, `sac.png`                | 전이·업데이트 수, 환경과 학습 단계별 시간 및 전체 실행 시간                        |
| `sac/<존 수>-<엔진>/actor.npz`      | 각 SAC 실행의 정책 신경망 가중치                                                   |

SBSim의 `setup_s`에는 미리 호출한 `prepare_thermal_system()`이 포함되며, `thermal_prepare_s`는
그 일부를 따로 잰 값이다. EnergyPlus의 초기화와 기상 워밍업은 리셋에 들어간다. EnergyPlus는 CPU의 네이티브 공조와
열수지를 실행하므로 `--platform gpu`를 선택해도 해당 엔진의 계산 장치는 CPU다. 이때 플랫폼 선택은 JAX 솔버와 SAC 학습에
적용한다.

### GPU 재현

`JAX_GPU_PYTHON`은 CUDA용 JAX가 설치된 Python이다. CUDA 오버레이가 있는 환경에서는 저장소 루트의
`.venv/jax_cuda/bin/python`을 사용한다. `EPW`와 `EPLUS_DIR`은 위 CPU 예제와 같은 입력이다.

```bash
unset JAX_PLATFORMS
JAX_GPU_PYTHON="${JAX_GPU_PYTHON:-$PWD/.venv/jax_cuda/bin/python}"
"$JAX_GPU_PYTHON" -c 'import jax; print(jax.devices("gpu"))' && \
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 "$JAX_GPU_PYTHON" \
  -m benchmarks.grouped_jax_study.run \
  --epw "$EPW" --energyplus-dir "$EPLUS_DIR" \
  --zones 16 36 100 --physics-zones 100 --days 7 --platform gpu \
  --grouped-method dense --physics-grouped-method cg \
  --output benchmarks/grouped_jax_study/results/gpu
```

각 섹션이 끝날 때마다 `summary.json`과 영어 `REPORT.md`, 한국어 `REPORT_ko.md`가 갱신된다. GPU에서 희소
직접 풀이를 측정하려면 CUDA `spsolve` 지원 환경에서 `--physics-grouped-method spsolve`를 지정한다. 현재
저장된 [`results/gpu/summary.json`](results/gpu/summary.json)과
[`results/gpu/REPORT_ko.md`](results/gpu/REPORT_ko.md)는 JAX 0.4.34가 NVIDIA
GeForce RTX 3080을 사용한 timing·physics 결과를 담고 있다. GPU SAC 결과는 아직 이 디렉터리의 완료 결과에
포함되지 않았다.

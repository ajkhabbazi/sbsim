# `TFDeviceResidentSimulator` 동작 및 TF-CG/PCG 해법

## 문서 범위와 용어

이 문서는 `smart_control/simulator/tf_device_resident_simulator.py`가 한 열 시뮬레이션 시간
단계를 어떻게 계산하는지 설명한다. 특히 다음 항목을 다룬다.

- 기본 열전도/대류 문제(baseline)
- 실내 장파 복사(radiative heat transfer)가 추가된 문제
- 실내 열질량(interior mass)이 추가된 문제
- 기존 Picard 반복이 장치 상주(device-resident) 구현에서 어떻게 바뀌었는지

코드에는 `THCG`라는 이름의 클래스나 함수가 없다. 따라서 이 문서에서는 요청의 **THCG**를 이 파일이 사용하는 **TensorFlow
기반 CG 경로**, 더 정확하게는 Jacobi 전처리 행렬-비구성 선형 켤레기울기법(matrix-free preconditioned
conjugate gradient, PCG)으로 해석한다.

## 한눈에 보는 역할

`TFDeviceResidentSimulator`는 `TFSimulator`를 상속하고 다음 세 가지를 한 장치 안에서 수행한다.

1. 외부 Picard 고정점 반복
2. 각 Picard 반복에서의 공기 온도 선형계 PCG 풀이
3. 공기 온도에 따른 실내 열질량 온도 갱신

전체 시간 단계는 `@tf.function`으로 컴파일된 `_device_timestep()` 안의 `tf.while_loop`에서 실행된다.
온도 상태는 `tf.Variable`인 `_device_temperature`와 `_device_mass_temperature`에 유지되므로,
선택적으로 호스트 동기화를 끄면 시간 단계 사이에 NumPy 왕복을 피할 수 있다.

주요 공개 동작은 다음과 같다.

- `finite_differences_timestep()`: 한 시간 단계를 실제로 진행한다.
- `warmup()`: TensorFlow 그래프를 추적하되, 계산 뒤 장치 상태를 원래 값으로 복원한다.
- `sync_state_from_building()`: NumPy 기반 `building` 상태를 장치 변수로 복사한다.
- `sync_state_to_building()`: 장치 변수를 NumPy 기반 `building` 상태로 복사한다.

## 공통 에너지 방정식

각 활성 공기 제어체적(CV)의 이산 에너지 평형은 개념적으로 다음 형태다.

$$
A(T^{(k)}, M^{(k)})\,T^{(k+1)} =
b(T^{(k)}, M^{(k)}, T^n, T_\infty, Q)
$$

여기서

- $T^n$: 이전 시간 단계의 공기 온도
- $T^{(k)}$: 현재 Picard 반복의 공기 온도 추정값
- $M^{(k)}$: 현재 Picard 반복의 실내 열질량(interior mass) 온도 추정값
- $T_\infty$: 외기 온도
- $Q$: 디퓨저 등 외부 열원
- $A$: 저장, 전도, 대류, 복사 자기항, 열질량 결합항으로 구성된 행렬
- $b$: 이전 단계 저장열, 외기, 열원, 복사 구동항, 열질량 온도항으로 구성된 우변

`_solve_air_on_device()`는 분모에 해당하는 대각 성분과 이웃 CV 결합 계수를 만들고, 이 값을 이용해 행렬을 명시적으로
조립하지 않은 채 PCG를 실행한다.

활성 CV에 대한 행렬-벡터 곱은 다음과 같은 5점 스텐실이다.

$$
(Ax)_P = d_P x_P
- c_{P,L}x_L-c_{P,R}x_R-c_{P,U}x_U-c_{P,D}x_D
$$

외부 CV는 항등행으로 고정하고 온도를 외기 온도로 설정한다. 활성 CV와 외부 CV 사이의 전도 기여는 미리 우변으로 이동한다.

## PCG 내부 해법

실제 PCG 구현은 `tf_simulator._solve_with_preconditioned_conjugate_gradient()`를
재사용한다.

1. 현재 온도장을 초기해로 사용한다.
2. 잔차 $r=b-Ax$를 계산한다.
3. 행렬 대각선으로 구성된 Jacobi 전처리기 $M=\operatorname{diag}(A)$를 사용하여 $z=M^{-1}r$를 계산한다.
4. 표준 CG의 탐색 방향, 스텝 크기, 잔차를 `tf.while_loop`에서 갱신한다.
5. 다음 조건을 만족하면 내부 반복을 끝낸다.

$$
\lVert r\rVert_2 \le
\text{cg\_tolerance}\,\max(\lVert b\rVert_2, 1)
$$

이 구현은 장치 상주 경로에서 FP32를 사용한다. 반복 제한까지 위 조건을 만족하지 못하면
`finite_differences_timestep()`이 `RuntimeError`를 발생시킨다.

## 1. Baseline 문제

Baseline은 실내 복사와 실내 열질량이 모두 꺼진 경우다. 이때 한 시간 단계의 계수는 현재 Picard 추정값에 의존하지 않는다.

대각 성분에는 다음 항이 들어간다.

$$
d_P =
\sum_f A_f\Gamma_f
+ \sum_{f\in B_P} A_f\Gamma_{\infty,f}
+ \frac{\rho c u v z}{\Delta t}
$$

우변에는 다음 항이 들어간다.

$$
b_P =
\frac{\rho c u v z}{\Delta t}T_P^n
+ Q_P
+ \sum_{f\in B_P}A_f\Gamma_{\infty,f}T_\infty
$$

활성 이웃 공기 온도와의 전도항은 $A$의 비대각 결합으로 들어간다. 따라서 문제 전체가 하나의 선형계이며 PCG 한 번으로 해당 시간 단계의
해를 구할 수 있다.

코드는 `complete_linear_solve`를 참으로 만들고 외부 반복 제한을 1로 설정한다. 기존 반복식처럼 작은 온도 변화가 나올
때까지 여러 번 공기장을 sweep할 필요가 없다.

## 2. 복사 문제

실내 장파 복사는 표면 온도에 비선형적으로 의존한다. 따라서 PCG 자체가 원래의 비선형 문제를 한 번에 푸는 것은 아니다. 대신 각 외부
Picard 반복에서 현재 표면 온도로 복사항을 다시 선형화한 뒤, 그 선형화된 공기 문제를 PCG로 정확도 있게 푼다.

`_radiative_surface_temperatures()`는 복사 enclosure에 참여하는 표면 온도를 모은다. 실내 열질량도 활성화된
경우 열질량 마스크 위치에는 공기 온도 대신 열질량 온도를 넣는다.

두 복사 표면 $i$, $j$ 사이의 복사 열교환은 다음과 같다.

$$
q_{ij} = \sigma\left(T_j^4-T_i^4\right)
$$

4제곱 차이를 인수분해하면 다음 항등식을 얻는다.

$$
T_j^4-T_i^4
= (T_j-T_i)(T_i^2+T_j^2)(T_i+T_j)
$$

따라서 복사 열교환을 온도 차와 복사 conductance의 곱으로 쓸 수 있다.

$$
q_{ij}=h_{ij}(T_j-T_i)
$$

$$
h_{ij}=\sigma(T_i^2+T_j^2)(T_i+T_j)
$$

이 인수분해 자체는 근사가 아니라 정확한 항등식이다. 두 표면 온도가 서로 가까우면 $h_{ij}\approx4\sigma T^3$가 되므로,
이를 직관적으로는 복사항의 "$T^3$ 부분을 고정한다"고 이해할 수 있다. 다만 실제 코드는 하나의 $T^3$ 값이 아니라 두 표면 온도로
계산한 $\sigma(T_i^2+T_j^2)(T_i+T_j)$를 사용한다.

Picard 반복 $k$에서는 현재 표면 온도로 복사 conductance를 계산한다.

$$
h_{ij}^{(k)}=
\sigma\left((T_i^{(k)})^2+(T_j^{(k)})^2\right)
\left(T_i^{(k)}+T_j^{(k)}\right)
$$

`_get_longwave_grid_terms()`는 이 값을 사용하여 전체 복사 교환을 두 부분으로 나눈다.

$$
q_i=g_i^{(k)}-c_i^{(k)}T_i^{(k+1)}
$$

여기서

$$
c_i^{(k)}=\sum_{j\neq i}M_{ij}h_{ij}^{(k)}
$$

$$
g_i^{(k)}=\sum_{j\neq i}M_{ij}h_{ij}^{(k)}T_j^{(k)}
$$

이며, 이 식의 $M_{ij}$는 표면 사이의 복사 교환 행렬 계수다. 이 문서의 interior mass 온도 기호 $M^{(k)}$와는 다른
기호이므로 구분해야 한다.

- 자기 온도에 비례하는 $c_i^{(k)}T_i^{(k+1)}$는 좌변으로 이동하여 공기 선형계의 대각 성분에 더한다.
- 다른 표면에서 오는 복사 구동항 $g_i^{(k)}$는 면적 가중 후 우변에 더한다.
- $h_{ij}^{(k)}$, $c_i^{(k)}$, $g_i^{(k)}$는 한 Picard 반복 동안 고정한다.
- 미지수인 자기 표면 온도 $T_i^{(k+1)}$는 전도 및 대류 온도와 함께 푼다.

따라서 "고정된 복사항이 포함된 선형계를 exact하게 푼다"는 말은 전체 $T^4$ 비선형 문제를 한 번에 푼다는 뜻이 아니다. 현재
Picard 반복에서 복사 계수와 구동항을 고정하여 만든 다음 선형계를 푼다는 뜻이다.

$$
A^{(k)}T^{(k+1)}=b^{(k)}
$$

- NumPy/SciPy direct solver는 이 선형계를 sparse direct solve로 푼다.
- CG/PCG는 같은 선형계를 지정된 residual tolerance까지 반복적으로 푼다.
- FDM은 같은 semi-implicit 복사항을 사용하지만 선형계 전체를 풀지 않고 CV별 relaxation sweep 한 번만 수행한다.

Picard가 수렴하면 $T^{(k+1)}=T^{(k)}$가 되므로, lagged coefficient가 현재 온도와 일치하고 원래의 $T^4$
복사 평형을 회복한다. 즉, 복사 인수분해는 exact하고 온도 의존 계수를 한 반복 늦게 사용하는 부분이 Picard 근사다.

실내 열질량 표면이 복사에 참여할 때 해당 표면 온도는 공기 온도 $T$가 아니라 열질량 온도 $M$이다. 현재 공기 선형계는 $M$을 동시에
미지수로 풀지 않으므로, 열질량 표면의 복사 자기항은 공기 행렬의 대각선으로 옮기지 않고 Picard 구동항으로 유지한다.

PCG 결과로 표면 온도가 바뀌면 다음 Picard 반복에서 복사 conductance와 구동항을 다시 계산한다. 즉, 구조는 다음과 같다.

```text
현재 표면 온도
  -> 복사 계수와 복사 구동항 재계산
  -> 선형화된 공기 온도장을 PCG로 풀이
  -> 온도 변화 확인
  -> 수렴하지 않았으면 반복
```

## 3. 실내 열질량 문제

실내 열질량 노드는 대응하는 공기 CV와만 열을 교환하는 단일 집중 열용량 노드로 취급된다.

공기 방정식에서 열질량 결합 컨덕턴스는 다음과 같다.

$$
G_{\mathrm{mass}} = \frac{k_{\mathrm{mass}}uv}{z}
$$

이 항은 공기 온도 $T$에 대한 손실항과 열질량 온도 $M$에 의한 구동항으로 나뉜다.

- PCG 대각선: $+G_{\mathrm{mass}}$
- PCG 우변: $+G_{\mathrm{mass}}M^{(k)}$

공기장을 푼 뒤 `_update_mass_on_device()`는 다음 식으로 새 열질량 추정값을 계산한다.

$$
M^{(k+1)} =
\frac{T^{(k+1)} + t_{0,\mathrm{mass}}M^n}
{1+t_{0,\mathrm{mass}}}
$$

$$
t_{0,\mathrm{mass}} =
\frac{\rho_{\mathrm{mass}}c_{\mathrm{mass}}z^2}
{k_{\mathrm{mass}}\Delta t}
$$

중요한 점은 식의 과거 열질량 값이 $M^{(k)}$가 아니라 \*\*시간 단계 시작 시점의 고정된 값 $M^n$\*\*이라는 것이다.
Picard 반복마다 현재 추정값을 다시 시간 적분의 시작값으로 쓰면 한 시간 단계 안에서 열질량이 여러 번 전진하는 오류가 생긴다. 이 구현은
매 외부 반복에서 항상 $M^n$으로부터 후보를 다시 계산하고, 연속한 후보 $M^{(k+1)}-M^{(k)}$만 수렴 판정에 사용한다.

따라서 열질량 문제의 반복 구조는 다음과 같다.

```text
M^(k)를 고정하고 공기 선형계를 PCG로 풀이 -> T^(k+1)
T^(k+1)와 고정된 M^n으로 열질량 후보 계산 -> M^(k+1)
공기 변화량과 열질량 변화량의 최댓값으로 수렴 확인
```

## 4. 복사와 열질량을 함께 사용하는 문제

두 기능이 모두 켜지면 같은 외부 Picard 반복 안에서 다음 순서를 사용한다.

1. $T^{(k)}$와 $M^{(k)}$를 이용해 복사 표면 온도를 구성한다.
2. 복사 자기항과 복사 구동항을 계산한다.
3. $M^{(k)}$를 공기 방정식의 열질량 구동 온도로 사용한다.
4. 선형화된 공기 문제를 PCG로 풀어 $T^{(k+1)}$를 얻는다.
5. $T^{(k+1)}$와 고정된 $M^n$으로 $M^{(k+1)}$를 계산한다.
6. 다음 값으로 외부 수렴을 판정한다.

$$
\Delta_{\max} = \max\left(
\lVert T^{(k+1)}-T^{(k)}\rVert_\infty,
\lVert M^{(k+1)}-M^{(k)}\rVert_\infty
\right)
$$

$\Delta_{\max}$가 `convergence_threshold` 이하가 되면 종료한다. 그렇지 않으면 새 공기 및 열질량 추정값으로
복사항과 결합항을 다시 계산한다.

## 기존 Picard 루프와 달라진 점

### 기존 구조

기존 `Simulator.finite_differences_timestep()`은 대체로 다음 구조였다.

1. `building.temp`의 NumPy 복사본으로 시작한다.
2. Python `for` 루프로 Picard 반복을 수행한다.
3. `update_temperature_estimates()`가 CV를 순회하며 온도를 한 번 sweep한다.
4. 실내 열질량이 있으면 공기 sweep 뒤 별도로 갱신한다.
5. NumPy 배열의 최대 변화량으로 수렴을 검사한다.
6. 종료 시 결과를 `building` 배열에 저장한다.

기본 CPU 구현에서는 CV가 순차 갱신되므로 전도 부분은 Gauss-Seidel 성격을 가진다. 복사량은 sweep 시작 시 한 번 계산되어
해당 sweep 동안 고정된다.

### 장치 상주 구조

`TFDeviceResidentSimulator`에서는 다음과 같이 바뀌었다. 아래 표에서 direct solve 또는 CG가
"linearized air system"을 푼다는 것은 해당 Picard 반복에서 복사 계수와 열질량 온도를 고정한 공기 선형계를 뜻한다.

| Calculation step                                   | FDM                                                                                           | NumPy/SciPy Direct                                  | TensorFlow CG                              | TF Device-Resident CG                      |
| -------------------------------------------------- | --------------------------------------------------------------------------------------------- | --------------------------------------------------- | ------------------------------------------ | ------------------------------------------ |
| Outer Picard loop                                  | Python `for` loop                                                                             | Python `for` loop                                   | Python `for` loop                          | TensorFlow `tf.while_loop`                 |
| Radiation coefficients                             | Evaluated from current $T^{(k)}$ and $M^{(k)}$                                                | Same                                                | Same                                       | Same, entirely on device                   |
| Radiative $T^3$-like factor                        | Exact pairwise factor $h_{ij}^{(k)}=\sigma((T_i^{(k)})^2+(T_j^{(k)})^2)(T_i^{(k)}+T_j^{(k)})$ | Same                                                | Same                                       | Same                                       |
| Air equation                                       | One point-by-point relaxation sweep                                                           | Full linear system                                  | Full linear system                         | Full linear system                         |
| Conduction and convection                          | Advanced by one FDM sweep                                                                     | Solved by sparse direct solver                      | Solved by PCG                              | Solved by FP32 PCG                         |
| Radiation treatment                                | Semi-implicit diagonal term plus lagged driving term                                          | Same linearization, solved directly                 | Same linearization, solved to CG tolerance | Same linearization, solved to CG tolerance |
| Interior-mass coupling in air equation             | Uses fixed $M^{(k)}$                                                                          | Uses fixed $M^{(k)}$                                | Uses fixed $M^{(k)}$                       | Uses fixed $M^{(k)}$                       |
| Air result per Picard iteration                    | One relaxation step toward $T^{(k+1)}$                                                        | Exact solution of the current linearized air system | Approximate solution within PCG tolerance  | Approximate solution within PCG tolerance  |
| Interior-mass update                               | Closed-form update using $T^{(k+1)}$ and fixed $M^n$                                          | Same                                                | Same                                       | Same, on device                            |
| Nonlinear radiation solution                       | Recovered through outer Picard convergence                                                    | Recovered through outer Picard convergence          | Recovered through outer Picard convergence | Recovered through outer Picard convergence |
| Inner linear iteration                             | FDM relaxation sweep                                                                          | None; sparse factorization/direct solve             | TensorFlow PCG `tf.while_loop`             | TensorFlow PCG `tf.while_loop`             |
| TensorFlow–NumPy transfers during Picard iteration | Not applicable                                                                                | Not applicable                                      | Transfers at the outer-loop boundary       | Avoided; state remains in `tf.Variable`    |
| Baseline without radiation or mass                 | Requires relaxation convergence                                                               | One direct solve                                    | One PCG solve                              | One device-resident PCG solve              |

핵심 변화는 Picard 반복을 제거한 것이 아니라 **역할을 축소한 것**이다.

- 전도·대류·저장항만 있는 선형 baseline에서는 외부 Picard 반복이 필요 없다.
- 복사 비선형성과 공기-열질량 결합이 있을 때만 외부 Picard 반복을 유지한다.
- 각 외부 반복 안의 공기 문제는 relaxation 한 번이 아니라 PCG로 충분히 수렴시킨다.

이를 중첩 반복으로 표현하면 다음과 같다.

```text
시간 단계 n -> n+1
  외부 Picard 반복: 복사/열질량 결합 수렴
    내부 PCG 반복: 현재 계수에서 공기 선형계 수렴
```

## 상태와 동기화 시 주의점

- `sync_to_host_each_step=True`가 기본값이므로 일반 사용에서는 매 시간 단계 뒤 장치 결과가
    `building.temp`와 `building.interior_mass_temp`에 복사된다.
- 이 옵션이 `False`이면 `building`의 NumPy 온도는 자동으로 바뀌지 않는다. 결과를 읽기 전에
    `sync_state_to_building()`을 호출해야 한다.
- 반대로 외부 코드가 NumPy 기반 `building` 온도를 직접 수정했다면 `sync_state_from_building()`으로 장치
    상태에 반영해야 한다.
- `building.input_q`는 각 시간 단계 호출 시 텐서로 변환되므로, 디퓨저 열원 변경은 다음 단계에 반영된다.
- `warmup()`은 그래프 추적 비용을 실제 시뮬레이션 전에 지불하기 위한 함수이며 온도 상태를 전진시키지 않는다.

## 수렴 및 진단 값

한 단계 실행 뒤 다음 값으로 동작을 확인할 수 있다.

- `last_outer_iteration_count`: 외부 Picard 반복 횟수
- `last_outer_max_delta`: 마지막 외부 반복의 최대 온도 변화량
- `last_cg_iteration_count`: 마지막 공기 선형계의 PCG 반복 횟수
- `last_cg_residual_norm`: 마지막 PCG 잔차의 2-노름

Baseline에서는 정상적으로 외부 반복 횟수가 1이어야 한다. 복사나 열질량이 활성화된 경우에는 모델 결합이 수렴할 때까지 1회보다 많아질
수 있다.

## 구현 흐름 요약

```text
finite_differences_timestep()
  -> 입력 외기 온도, 대류계수, input_q를 텐서화
  -> _device_timestep()
       -> 이전 시간 단계의 T^n, M^n 고정
       -> 외부 tf.while_loop
            -> 복사항 계산
            -> 공기 방정식의 대각선과 우변 계산
            -> 행렬-비구성 Jacobi-PCG
            -> M^n 기준 열질량 후보 계산
            -> 공기/질량 최대 변화량 계산
       -> 장치 tf.Variable에 최종 상태 저장
  -> PCG 실패 시 예외 발생
  -> 설정에 따라 장치 상태를 building NumPy 배열로 동기화
```

요약하면, 이 구현은 선형 열전도 문제는 PCG 한 번으로 끝내고, 복사 및 실내 열질량 때문에 생기는 비선형 또는 분할 결합만 외부
Picard 반복으로 해결한다. 그 결과 기존 해석의 고정점은 유지하면서도 Python 반복과 장치-호스트 데이터 이동을 줄인다.

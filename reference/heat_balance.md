# 열평형 모델과 정확해 솔버 업데이트 리뷰

작성일: 2026년 9월 17일

## 범위

이 문서는 `heatbalance_fix` 브랜치에서 수행한 열평형(heat balance) 변경 사항을 리뷰하기 위한 문서다.

대상 파일은 다음과 같다.

- `smart_control/simulator/simulator.py`
- `smart_control/simulator/tf_simulator.py`
- `smart_control/simulator/tf_simulator_heat_balance_test.py`
- 관련 flexible floor-plan 및 TensorFlow 테스트

주요 변경 사항은 다음과 같다.

1. 두 CV가 공유하는 면의 컨덕턴스를 양쪽 half-cell 열저항으로 계산한다.
2. 외기 경계면에서 대류막과 boundary CV의 half-cell을 직렬 저항으로 처리한다.
3. corner와 edge CV의 실제 체적 및 공유 면적을 형상에 맞게 반영한다.
4. 모든 열평형 항을 W 단위로 조립해 에너지 보존형 유한체적식을 사용한다.
5. 내부 질량을 이전 시간 단계 기준의 결합 방정식으로 갱신한다.
6. 실내 장파복사를 대각 계수와 구동 열원으로 분해해 반복 안정성을 확보한다.
7. NumPy/SciPy sparse direct solver와 TensorFlow matrix-free PCG를 분리한다.
8. `TFSimulator`에서 SciPy 및 Tensor-to-NumPy solve 경로를 제거한다.
9. 기본, 내부 질량, 복사, 내부 질량과 복사를 함께 켠 조건을 모두 테스트한다.

이 문서에서 **정확해 또는 직접해**는 연속 열방정식의 닫힌 형태 해를 뜻하지 않는다. 동일한 이산 연립방정식

$$
\mathbf A\mathbf T=\mathbf b
$$

를 반복 완화 없이 희소 행렬 분해로 직접 푼 결과를 뜻한다.

## 1. 기호와 단위

| 기호          | 의미                                           | 단위     |
| ------------- | ---------------------------------------------- | -------- |
| $P$           | 열평형을 계산하는 owner CV                     | --       |
| $N_f$         | 면 $f$ 반대편의 neighbor CV                    | --       |
| $T_P$         | 현재 시간 단계의 새 CV 온도                    | K        |
| $T_P^{(-)}$   | 이전 시간 단계의 CV 온도                       | K        |
| $T_{N_f}$     | neighbor CV 온도                               | K        |
| $T_{amb}$     | 외기 온도                                      | K        |
| $u_P,v_P$     | CV의 두 수평 방향 길이                         | m        |
| $z$           | floor height 또는 CV 높이                      | m        |
| $A_f$         | 두 CV가 공유하는 면적                          | m²       |
| $k_P$         | 열전도율                                       | W/(m K)  |
| $\rho_P$      | 밀도                                           | kg/m³    |
| $c_P$         | 비열                                           | J/(kg K) |
| $h$           | 외기 대류 열전달계수                           | W/(m² K) |
| $\Gamma_f$    | 내부 공유 면의 단위 면적당 컨덕턴스            | W/(m² K) |
| $A_f\Gamma_f$ | 면적을 반영한 내부 면 컨덕턴스                 | W/K      |
| $C_P$         | 시간 이산화된 저장 컨덕턴스                    | W/K      |
| $D_P$         | CV $P$의 대각 계수: 저장 및 유출 컨덕턴스의 합 | W/K      |
| $b_P$         | 이전 온도, 외기, 열원을 모은 알려진 우변       | W        |
| $Q_P$         | HVAC 또는 diffuser가 공급하는 열량             | W        |
| $c_i$         | 복사 self-term의 대각 계수                     | W/(m² K) |
| $g_i$         | 다른 표면 온도가 만드는 복사 구동 열유속       | W/m²     |

## 2. CV 형상과 경계 CV 크기

전체 격자 간격을 $\delta_x$라고 하면 내부 CV는 다음 크기를 갖는다.

$$
u_P=v_P=\delta_x.
$$

외기에 열린 면이 있는 boundary CV는 열린 방향의 길이를 절반으로 줄인다.

- edge CV는 한 축만 $\delta_x/2$이므로 체적이 내부 CV의 $1/2$이다.
- corner CV는 두 축이 모두 $\delta_x/2$이므로 체적이 내부 CV의 $1/4$이다.
- production floor plan은 직사각형 외곽을 사용한다.

production의 직사각형 외곽에서는 같은 면을 공유하는 두 CV의 접선 방향 길이가 항상 같다. 따라서 왼쪽 또는 오른쪽 면의 공유 면적은

$$
A_f=v_Pz=v_{N_f}z
$$

이고 위쪽 또는 아래쪽 면의 공유 면적은

$$
A_f=u_Pz=u_{N_f}z
$$

이다. 이 조건에서는

$$
\min(v_P,v_{N_f})=v_P=v_{N_f}
$$

이고 위아래 면도 동일하므로 `min`은 계산 결과에 아무 영향이 없다.

다만 현재 repository에는 오목한 모서리와 계단형 경계를 갖는 기존 weird floor-plan 테스트가 남아 있다. 여기서
`legacy irregular test`는 정식 테스트 이름 하나를 뜻하는 것이 아니라 다음 두 종류를 묶어 부르는 표현이다.

- `simulator_flexible_floor_plan_test.py`의
    `_create_dummy_floor_plan_weird_shape()`와 관련 회귀 테스트
- `tf_simulator_heat_balance_test.py`의
    `test_shared_face_area_is_the_narrower_of_the_two_cvs`

weird floor plan은 `0=실내 공기`, `1=벽`, `2=외기`인 다음과 같은 오목한 형상이다.

```text
2 2 2 2 2 2 2 2 2
2 1 1 1 2 1 1 1 2
2 1 0 1 1 1 0 1 2
2 1 0 0 1 0 0 1 2
2 1 0 0 1 0 0 1 2
2 1 1 1 1 1 1 1 2
          ^ 오목한 모서리와 계단형 경계가 존재
```

이런 경계에서는 한쪽 CV의 접선 길이가 $\delta_x$, 다른 쪽이 $\delta_x/2$일 수 있다. 예를 들어 수평으로 맞닿은 두
CV가 다음과 같다고 하자.

```text
접선 방향
   delta_x
P +--------+
  | shared |   N의 높이는 delta_x/2
  +----+---+--------+
       | N |
       +---+

실제 공유 길이 = delta_x/2
```

왼쪽 또는 오른쪽 면의 물리적 접촉 면적은 큰 CV의 전체 면적이 아니라 겹치는 부분만 사용해야 한다. 따라서 generic simulator는

$$
A_f=\min(v_P,v_{N_f})z
$$

또는

$$
A_f=\min(u_P,u_{N_f})z
$$

를 사용한다. 위 예에서 `min`을 사용하면 양쪽 모두 $A_f=(\delta_x/2)z$를 사용한다. `min`이 없으면 큰 CV는
$\delta_xz$, 작은 CV는 $(\delta_x/2)z$를 사용해 같은 공유 면의 열률을 서로 2배 다르게 계산한다.

따라서 차이는 다음처럼 정리된다.

- 직사각형 production 형상: 맞닿은 접선 길이가 같아 차이 0
- full-width와 half-width가 만나는 irregular face: 그 면의 owner-side 열률이 최대 2배
- 전체 건물 온도 영향: 해당 면의 개수, 재질, 온도차에 따라 달라지지만 국소 에너지 보존은 즉시 깨짐

production 입력이 항상 직사각형이라는 계약을 building validation에서 강제하고 기존 irregular API를 제거한다면
`min` 처리도 제거할 수 있다. 현재는 기존 형상 지원과 회귀 테스트를 유지하기 위해 필요하다.

## 3. 내부 공유 면의 열전도

두 CV 사이의 열은 owner CV 절반과 neighbor CV 절반을 차례로 통과한다. 면 $f$에 수직인 각 CV의 길이를
$\ell_{P,f}$와 $\ell_{N_f,f}$라고 하면 단위 면적당 열저항은 다음과 같다.

$$
R_{P,f}=\frac{\ell_{P,f}}{2k_P},
\qquad
R_{N_f,f}=\frac{\ell_{N_f,f}}{2k_{N_f}}.
$$

두 저항은 직렬이므로 면 컨덕턴스는 다음과 같다.

$$
\Gamma_f
=
\left(
\frac{\ell_{P,f}}{2k_P}
+\frac{\ell_{N_f,f}}{2k_{N_f}}
\right)^{-1}.
$$

면을 통해 owner CV로 들어오는 열률은

$$
Q_{P\leftarrow N_f}
=A_f\Gamma_f(T_{N_f}-T_P)
$$

이다. 같은 면을 이웃 CV 기준으로 계산하면

$$
Q_{P\leftarrow N_f}=-Q_{N_f\leftarrow P}
$$

가 성립한다.

기존 owner-cell 근사

$$
A_f\Gamma_f\approx\frac{k_PA_f}{\ell_{P,f}}
$$

은 균일한 물질에서는 같은 결과를 내지만, $k_P\ne k_{N_f}$인 물질 경계에서는 두 CV가 서로 다른 열유량을 계산한다. 특히 공기와
단열 외벽이 만나는 면에서 오차가 크게 발생한다.

## 4. 외기 경계면의 대류와 전도

외기에 열린 면에서는 외기 대류막과 boundary CV의 바깥쪽 half-cell을 모두 통과해야 한다.

$$
R_{film}=\frac{1}{h},
\qquad
R_{half}=\frac{\ell_{P,f}}{2k_P}.
$$

따라서 유효 외기 컨덕턴스는

$$
\Gamma_{\infty,f}
=
\left(
\frac{1}{h}+\frac{\ell_{P,f}}{2k_P}
\right)^{-1}
$$

이고 CV로 들어오는 열률은

$$
Q_{amb,f}
=A_f\Gamma_{\infty,f}(T_{amb}-T_P)
$$

이다.

이 식은 다음 조건을 만족한다.

$$
0<\Gamma_{\infty,f}<h.
$$

경계 물질의 열전도율이 매우 커지면 고체 열저항이 사라지므로

$$
\lim_{k_P\rightarrow\infty}\Gamma_{\infty,f}=h
$$

가 된다. 반대로 단열재에서는 half-cell 전도 저항이 지배적이므로 실제 외기 열교환은 $h$만 사용한 값보다 훨씬 작다.

## 5. 시간 저장항

implicit Euler 시간 이산화에서 CV의 에너지 저장항은 다음과 같다.

$$
Q_{storage}=C_P(T_P-T_P^{(-)}),
$$

여기서

$$
C_P=\frac{\rho_Pc_Pu_Pv_Pz}{\Delta t}.
$$

corner와 edge CV의 $1/4$, $1/2$ 체적은 이미 $u_Pv_P$에 반영된다. 따라서 별도의 경계 체적 보정 계수를 다시 곱하지
않는다.

## 6. 기본 공기 CV 열평형식

$I_P$를 내부 이웃과 연결된 면 집합, $B_P$를 외기에 열린 면 집합이라고 하자. 내부 질량과 복사를 제외한 보존형 열평형식은 다음과
같다.

$$
\sum_{f\in I_P}A_f\Gamma_f(T_{N_f}-T_P)
+\sum_{f\in B_P}A_f\Gamma_{\infty,f}(T_{amb}-T_P)
+Q_P
=C_P(T_P-T_P^{(-)}).
$$

$T_P$에 대해 정리하면 point relaxation에서 사용하는 업데이트 식을 얻는다.

$$
T_P
=
\frac{
\sum_{f\in I_P}A_f\Gamma_fT_{N_f}
+\sum_{f\in B_P}A_f\Gamma_{\infty,f}T_{amb}
+Q_P
+C_PT_P^{(-)}
}{
C_P
+\sum_{f\in I_P}A_f\Gamma_f
+\sum_{f\in B_P}A_f\Gamma_{\infty,f}
}.
$$

모든 항을 W 또는 W/K 단위로 유지하므로 owner CV의 열전도율로 식 전체를 나누지 않는다. 이 형태는 서로 다른 물질이 만나는 면에서도
그대로 사용할 수 있다.

두 구현은 같은 식을 서로 다른 반복 순서로 푼다.

- `simulator.py`: sweep 중 이미 갱신된 이웃 온도를 사용하는 Gauss-Seidel 방식
- `tf_simulator.py`: 이전 field 전체를 tensor shift하여 사용하는 Jacobi 방식

### 6.1 열평형식을 $\mathbf A\mathbf T=\mathbf b$로 변환하는 방법

각 CV의 열평형식에서 **현재 time step의 미지 온도**는 왼쪽으로 옮기고, 이전 time step 온도와 외기 온도, 입력 열원처럼
이미 알려진 값은 오른쪽에 둔다. 기본 공기 CV 식에서 면 전체 컨덕턴스를

$$
G_{P,N_f}=A_f\Gamma_f,
\qquad
G_{P,\infty,f}=A_f\Gamma_{\infty,f}
$$

로 짧게 쓰면

$$
C_P\left(T_P-T_P^{(-)}\right)
=
\sum_{f\in I_P}G_{P,N_f}\left(T_{N_f}-T_P\right)
+\sum_{f\in B_P}G_{P,\infty,f}\left(T_{amb}-T_P\right)
+Q_P
$$

이다. 우변의 괄호를 풀면

$$
\begin{aligned}
C_PT_P-C_PT_P^{(-)}
={}&
\sum_{f\in I_P}G_{P,N_f}T_{N_f}
-\left(\sum_{f\in I_P}G_{P,N_f}\right)T_P \\
&+\sum_{f\in B_P}G_{P,\infty,f}T_{amb}
-\left(\sum_{f\in B_P}G_{P,\infty,f}\right)T_P
+Q_P.
\end{aligned}
$$

현재 time-step의 미지 온도 $T_P$, $T_{N_f}$를 왼쪽으로 옮기면

$$
\left(
C_P+\sum_{f\in I_P}G_{P,N_f}
+\sum_{f\in B_P}G_{P,\infty,f}
\right)T_P
-\sum_{f\in I_P}G_{P,N_f}T_{N_f}
=
C_PT_P^{(-)}+Q_P
+\sum_{f\in B_P}G_{P,\infty,f}T_{amb}.
$$

이 식 하나가 $\mathbf A$의 한 행과 $\mathbf b$의 한 원소가 된다. $n_r\times n_c$ grid에서 2차원 CV
$(i,j)$를 row-major index

$$
p(i,j)=i n_c+j
$$

로 펼치면, CV $P$의 식은 matrix row $p=p(P)$에 들어간다. 구체적으로

$$
A_{p,p}
=C_P+\sum_{f\in I_P}G_{P,N_f}
+\sum_{f\in B_P}G_{P,\infty,f},
$$

$$
A_{p,p(N_f)}=-G_{P,N_f},
$$

$$
b_p=C_PT_P^{(-)}+Q_P
+\sum_{f\in B_P}G_{P,\infty,f}T_{amb}.
$$

따라서 물리 항과 matrix 위치는 다음처럼 대응한다.

| 열수지 항                                 | matrix 위치    |              coefficient |
| ----------------------------------------- | -------------- | -----------------------: |
| transient storage                         | $A_{p,p}$      |                   $+C_P$ |
| interior face에서 $T_P$에 곱해지는 유출항 | $A_{p,p}$      |             $+G_{P,N_f}$ |
| interior neighbor 온도 $T_{N_f}$          | $A_{p,p(N_f)}$ |             $-G_{P,N_f}$ |
| ambient face에서 $T_P$에 곱해지는 유출항  | $A_{p,p}$      |        $+G_{P,\infty,f}$ |
| 이전 step 온도                            | $b_p$          |          $+C_PT_P^{(-)}$ |
| HVAC, diffuser, solar 등의 net source     | $b_p$          |                   $+Q_P$ |
| 알려진 외기 온도                          | $b_p$          | $+G_{P,\infty,f}T_{amb}$ |

존재하지 않는 이웃에는 matrix column을 만들지 않는다. 외기 온도는 미지수가 아니므로 ambient face는 off-diagonal
column을 만들지 않고, 대각항과 우변만 증가시킨다. 모든 CV 행을 같은 방식으로 쌓으면

$$
\mathbf A\mathbf T=\mathbf b
$$

가 된다.

#### $2\times2$ CV 예제

네 활성 CV를 row-major 순서로 다음처럼 번호 붙이자.

```text
[1] -- [2]
 |      |
[3] -- [4]
```

내부 면 컨덕턴스를 $G_{12}$, $G_{13}$, $G_{24}$, $G_{34}$라고 하면

$$
\begin{bmatrix}
D_1 & -G_{12} & -G_{13} & 0 \\
-G_{12} & D_2 & 0 & -G_{24} \\
-G_{13} & 0 & D_3 & -G_{34} \\
0 & -G_{24} & -G_{34} & D_4
\end{bmatrix}
\begin{bmatrix}
T_1\\T_2\\T_3\\T_4
\end{bmatrix}
=
\begin{bmatrix}
b_1\\b_2\\b_3\\b_4
\end{bmatrix}.
$$

예를 들어 CV 1이 위쪽과 왼쪽 외기에 노출되어 있으면

$$
D_1
=C_1+G_{12}+G_{13}+G_{1,\infty,U}+G_{1,\infty,L},
$$

$$
b_1
=C_1T_1^{(-)}+Q_1
+\left(G_{1,\infty,U}+G_{1,\infty,L}\right)T_{amb}.
$$

CV 1과 면을 공유하지 않는 CV 4에 대해서는 $A_{1,4}=0$이다. 각 CV가 자기 자신과 최대 네 이웃에만 연결되므로
$\mathbf A$는 dense matrix가 아니라 sparse matrix다. 직사각형 grid를 grid-row 단위로 묶으면 개념적으로

$$
\mathbf A
=
\begin{bmatrix}
\mathbf B_0 & \mathbf C_0 & 0 & \cdots & 0\\
\mathbf C_0^T & \mathbf B_1 & \mathbf C_1 & \ddots & \vdots\\
0 & \mathbf C_1^T & \mathbf B_2 & \ddots & 0\\
\vdots & \ddots & \ddots & \ddots & \mathbf C_{n_r-2}\\
0 & \cdots & 0 & \mathbf C_{n_r-2}^T & \mathbf B_{n_r-1}
\end{bmatrix}
$$

와 같은 block-tridiagonal 형태다. $\mathbf B_i$는 같은 grid row의 left/right coupling이고,
$\mathbf C_i$는 인접 grid row 사이의 up/down coupling이다. 비직사각형 geometry에서는 일부
coefficient가 빠지지만 조립 원리는 같다.

고정 외기 CV를 전체 grid 미지수 벡터에 포함하는 현재 구현에서는 해당 exterior row를

$$
A_{p,p}=1,\qquad b_p=T_{amb}
$$

로 두어 Dirichlet 조건을 강제한다.

행렬을 $\mathbf D=\operatorname{diag}(A_{11},\ldots,A_{NN})$와
$\mathbf R=\mathbf A-\mathbf D$로 나누면 Jacobi update는

$$
\mathbf T^{(k+1)}
=
\mathbf D^{-1}\left(\mathbf b-\mathbf R\mathbf T^{(k)}\right)
$$

이다. Gauss-Seidel은 같은 row 식을 사용하되 현재 sweep에서 이미 갱신된 이웃 온도를 즉시 사용한다. Sparse direct
solve는 모든 row를 동시에 factorization하여 풀고, matrix-free PCG는 $\mathbf A$를 dense
array로 만들지 않고 $\mathbf A\mathbf T$ 연산으로 같은 시스템을 반복해 푼다.

따라서 point iteration, sparse direct solve와 PCG는 서로 다른 물리식을 푸는 것이 아니라 같은
$\mathbf A\mathbf T=\mathbf b$를 서로 다른 알고리즘으로 푼다. 계산된 온도장이 선형 시스템을 만족하는 정도는

$$
\mathbf r=\mathbf b-\mathbf A\mathbf T
$$

인 residual로 측정할 수 있다. Direct solve와 PCG는 residual을 직접 작게 만들지만, 현재
point-iteration timestep 종료 기준은 연속 sweep 사이의 최대 온도 변화
$\max_P|T_P^{(k+1)}-T_P^{(k)}|$다.

내부 질량과 복사는 outer Picard 반복 $k$에서 계수를 고정한 뒤 같은 방식으로 넣는다.

- 내부 질량 온도 $T_m^{(k)}$를 고정하면 $A_{p,p}$에 $G_m$을 더하고 $b_p$에 $G_mT_m^{(k)}$를 더한다.
- 복사 항을 $q_i=g_i^{(k)}-c_i^{(k)}T_i$로 분리하면 $A_{p,p}$에 $A_{rad}c_i^{(k)}$를 더하고
    $b_p$에 $A_{rad}g_i^{(k)}$를 더한다.

따라서 각 Picard 반복에서는 선형 시스템을 풀고, 새 온도로 질량 및 복사 계수를 갱신한 뒤 수렴할 때까지 반복한다.

## 7. 내부 질량 결합

공기 CV와 내부 질량 사이의 컨덕턴스는

$$
G_m=\frac{k_muv}{z}
$$

이고 내부 질량의 저장 컨덕턴스는

$$
C_m=\frac{\rho_mc_muvz}{\Delta t}
$$

이다. 내부 질량의 열평형식은

$$
G_m(T_P-T_m)=C_m(T_m-T_m^{(-)})
$$

이며, 새 내부 질량 온도는 다음과 같다.

$$
T_m
=
\frac{G_mT_P+C_mT_m^{(-)}}{G_m+C_m}.
$$

공기 CV에는 크기가 같고 부호가 반대인 항이 들어간다.

$$
Q_{mass}=G_m(T_m-T_P).
$$

공기 온도와 내부 질량 온도가 결합되어 있으므로 내부 질량을 켠 경우에는 외부 fixed-point 반복이 필요하다. 각 반복에서 내부 질량은
항상 이전 시간 단계의 $T_m^{(-)}$에서 다시 계산한다. 따라서 한 time step 안에서 반복 횟수만큼 질량 온도가 잘못 전진하는
문제를 방지한다.

## 8. 실내 장파복사 결합

복사 노드 $i$가 얻는 장파복사 열유속은 다음과 같다.

$$
q_i
=
\sum_{j\ne i}M_{ij}\sigma(T_j^4-T_i^4).
$$

다음 항등식을 사용한다.

$$
T_j^4-T_i^4
=
(T_i^2+T_j^2)(T_i+T_j)(T_j-T_i).
$$

표면 쌍의 정확한 복사 컨덕턴스를

$$
h_{ij}=\sigma(T_i^2+T_j^2)(T_i+T_j)
$$

로 정의하면 복사 열유속을 다음과 같이 분리할 수 있다.

$$
q_i
=
\sum_{j\ne i}M_{ij}h_{ij}(T_j-T_i)
=g_i-c_iT_i,
$$

여기서

$$
c_i=\sum_{j\ne i}M_{ij}h_{ij},
\qquad
g_i=\sum_{j\ne i}M_{ij}h_{ij}T_j.
$$

내벽 CV에서는 복사 면적을 $A_{rad}$라고 할 때 $A_{rad}c_i$를 대각항에, $A_{rad}g_i$를 우변에 더한다.

$$
(D_i+A_{rad}c_i)T_i=N_i+A_{rad}g_i.
$$

이 변환은 $T^4$를 근사한 것이 아니라 정확한 대수적 인수분해다. 다만 $c_i$와 $g_i$를 이전 Picard 반복 온도에서 계산해 다음
반복 동안 고정하므로 솔버 전체는 반암시적 반복이 된다.

내부 질량 표면은 질량 온도로 복사하지만 열유속은 같은 위치의 공기 CV로 전달된다. 공기 CV가 내부 질량 온도를 직접 미지수로 풀지 않으므로
내부 질량 복사 self-term은 공기 CV의 대각항으로 이동하지 않고 명시적 열원으로 유지한다.

서로 연결되지 않은 복사 enclosure는 별도 행렬 block으로 계산한다. TensorFlow 구현은 모든 표면 쌍의 온도 행렬을 만들지
않고 enclosure별로

$$
[1,T,T^2,T^3,T^4]
$$

의 weighted power sum을 행렬곱하여 $c_i$와 $g_i$를 계산한다.

## 9. NumPy/SciPy 희소 정확해 솔버와 TensorFlow PCG

내부 질량과 복사를 끄면 implicit Euler 열평형식은 선형 연립방정식이다.

$$
\mathbf A\mathbf T=\mathbf b.
$$

### 9.1 정확해가 푸는 대상

여기서 미지수 벡터 $\mathbf T$는 2차원 building grid를 row-major 순서로 펼친 벡터다. grid shape이
$n_r\times n_c$이면 CV $(i,j)$의 1차원 index는

$$
p(i,j)=i n_c+j
$$

이고 전체 미지수 개수는

$$
N=n_rn_c
$$

이다. 따라서

$$
\mathbf T
=
[T_{0,0},T_{0,1},\ldots,T_{n_r-1,n_c-1}]^T
$$

로 쓸 수 있다.

`NumpyScipySimulator`의 정확해 솔버는 연속 열방정식을 직접 푸는 것이 아니라, 현재 time step의 implicit
Euler 유한체적식을 NumPy로 조립해 얻은 $N\times N$ 희소 선형 시스템을 SciPy로 직접 푼다.

기본 열평형식을 다음처럼 쓴다.

$$
D_PT_P-
\sum_{f\in I_P}A_f\Gamma_fT_{N_f}
=b_P,
$$

여기서 복사와 내부 질량을 제외한 기본 대각항은

$$
D_P
=
C_P
+\sum_{f\in I_P}A_f\Gamma_f
+\sum_{f\in B_P}A_f\Gamma_{\infty,f}
$$

이다. $D_P$의 단위는 W/K이고 $b_P$의 단위는 W다. 따라서

$$
T_P=\frac{b_P+\sum_fA_f\Gamma_fT_{N_f}}{D_P}
$$

에서 온도 단위 K가 얻어진다.

### 9.2 행렬의 각 성분

활성 CV $P$의 대각 성분은

$$
A_{PP}
=
C_P
+\sum_{f\in I_P}A_f\Gamma_f
+\sum_{f\in B_P}A_f\Gamma_{\infty,f}
$$

이고 내부 이웃에 대한 비대각 성분은

$$
A_{P,N_f}=-A_f\Gamma_f
$$

이다. 우변은

$$
b_P
=
C_PT_P^{(-)}
+Q_P
+\sum_{f\in B_P}A_f\Gamma_{\infty,f}T_{amb}
$$

이다.

예를 들어 CV $P$에 왼쪽, 오른쪽, 위쪽, 아래쪽 이웃이 모두 있으면 한 행은

$$
\begin{aligned}
&\left(
C_P+A_L\Gamma_L+A_R\Gamma_R+A_U\Gamma_U+A_D\Gamma_D
\right)T_P \\
&\quad
-A_L\Gamma_LT_L-A_R\Gamma_RT_R
-A_U\Gamma_UT_U-A_D\Gamma_DT_D
=b_P
\end{aligned}
$$

가 된다. 따라서 한 행에는 일반적으로 다음 다섯 개의 nonzero가 있다.

```text
column P      :  D_P
column left   : -A_L Gamma_L
column right  : -A_R Gamma_R
column up     : -A_U Gamma_U
column down   : -A_D Gamma_D
```

edge와 corner CV는 존재하지 않는 interior neighbor column을 만들지 않고, 외기 경계 열교환은
$A_f\Gamma_{\infty,f}T_{amb}$ 형태로 우변에 들어간다.

#### 9.2.1 세 개 CV 예제로 $\mathbf A\mathbf T=\mathbf b$ 만들기

`finite difference method`와 $\mathbf A\mathbf T=\mathbf b$ 방법은 서로 다른 열모델이 아니다. 이
구현은 엄밀히 말하면 보존형 **유한체적 이산화**를 사용하며, 그 결과로 생긴 동일한 대수 방정식을

- Jacobi/Gauss-Seidel 반복으로 풀거나
- sparse direct solver로 풀 수 있다.

가장 간단한 1차원 세 CV를 생각하자.

```text
ambient -- A∞L Γ∞L -- [1] -- A12 Γ12 -- [2] -- A23 Γ23 -- [3] -- A∞R Γ∞R -- ambient
```

CV 1의 이산 열평형은

$$
A_{12}\Gamma_{12}(T_2-T_1)
+A_{\infty L}\Gamma_{\infty,L}(T_{amb}-T_1)
+Q_1
=C_1(T_1-T_1^{(-)})
$$

이다. 미지수인 새 온도를 왼쪽에, 이미 알려진 항을 오른쪽에 모으면

$$
\underbrace{(C_1+A_{12}\Gamma_{12}
+A_{\infty L}\Gamma_{\infty,L})}_{D_1}T_1
-A_{12}\Gamma_{12}T_2
=
\underbrace{C_1T_1^{(-)}+Q_1
+A_{\infty L}\Gamma_{\infty,L}T_{amb}}_{b_1}.
$$

같은 방식으로 CV 2와 CV 3을 정리하면

$$
-A_{12}\Gamma_{12}T_1
+(C_2+A_{12}\Gamma_{12}+A_{23}\Gamma_{23})T_2
-A_{23}\Gamma_{23}T_3
=C_2T_2^{(-)}+Q_2,
$$

$$
-A_{23}\Gamma_{23}T_2
+\underbrace{(C_3+A_{23}\Gamma_{23}
+A_{\infty R}\Gamma_{\infty,R})}_{D_3}T_3
=
\underbrace{C_3T_3^{(-)}+Q_3
+A_{\infty R}\Gamma_{\infty,R}T_{amb}}_{b_3}.
$$

세 식의 계수를 행별로 쌓으면 바로 다음 행렬이 된다.

$$
\underbrace{
\begin{bmatrix}
D_1 & -A_{12}\Gamma_{12} & 0 \\
-A_{12}\Gamma_{12} & D_2 & -A_{23}\Gamma_{23} \\
0 & -A_{23}\Gamma_{23} & D_3
\end{bmatrix}
}_{\mathbf A}
\underbrace{
\begin{bmatrix}
T_1\\T_2\\T_3
\end{bmatrix}
}_{\mathbf T}
=
\underbrace{
\begin{bmatrix}
b_1\\b_2\\b_3
\end{bmatrix}
}_{\mathbf b},
$$

여기서

$$
D_2=C_2+A_{12}\Gamma_{12}+A_{23}\Gamma_{23},
\qquad
b_2=C_2T_2^{(-)}+Q_2.
$$

즉 $A_f\Gamma_f$는 **두 노드를 연결하는 면 컨덕턴스**, $D$는 **한 행의 대각에 들어가는 저장 및 유출 컨덕턴스의
총합**이다. 반복식은 첫 행을 $T_1$에 대해 다시 나눈

$$
T_1=\frac{b_1+A_{12}\Gamma_{12}T_2}{D_1}
$$

을 여러 번 적용하는 것이고, direct solve는 세 식을 동시에 푸는 것이다.

### 9.3 외기 CV 고정 조건

외기 CV는 미지수로 풀지 않고 다음 조건으로 고정한다.

$$
A_{PP}=1,
\qquad
b_P=T_{amb}.
$$

즉 외기 CV의 원래 열평형 행을 제거하고 identity row로 교체한다. 외기 CV가 다른 CV와 coupling되어 있었다면 해당
coupling은 boundary convection 항으로 이미 처리되므로 외기 CV 온도는 항상 정확히 $T_{amb}$로 유지된다.

### 9.4 희소 행렬 조립과 직접 풀이

각 CV는 최대 네 개의 이웃과 연결되므로 $\mathbf A$는 희소 행렬이다. `NumpyScipySimulator`는 SciPy CSR
행렬을 만들고 다음 함수로 선형 시스템을 푼다.

```text
scipy.sparse.linalg.spsolve
```

조립 절차는 다음과 같다.

1. 모든 CV의 대각항 $D_P$를 sparse triplet의 `(row, column, value)`에 추가한다.
2. 실제 interior neighbor마다 $-A_f\Gamma_f$를 off-diagonal entry로 추가한다.
3. grid 바깥쪽으로 향하는 항은 unknown column을 만들지 않고 $A_f\Gamma_{\infty,f}T_{amb}$로 우변에
    더한다.
4. exterior CV의 행을 identity row로 교체하고 우변을 $T_{amb}$로 설정한다.
5. triplet을 CSR matrix로 변환한다.
6. `spsolve(A, b)`로 $\mathbf T$를 계산한다.

수식으로는

$$
\mathbf A=\mathbf L\mathbf U
$$

를 계산한 뒤

$$
\mathbf L\mathbf y=\mathbf b,
\qquad
\mathbf U\mathbf T=\mathbf y
$$

를 순서대로 푸는 것이다. 코드에서는 이 과정이 `spsolve(A, b)` 내부에서 수행되며 호출마다 필요한 factorization을 다시
계산한다. 명시적인 역행렬

$$
\mathbf T=\mathbf A^{-1}\mathbf b
$$

를 만들지 않는다. 역행렬을 직접 만드는 것보다 LU solve가 더 빠르고 수치적으로 안정적이며 sparse 구조도 유지할 수 있다.

### 9.5 이번 변경에서 LU cache를 사용하지 않는 이유

어떤 입력이 $\mathbf A$를 바꾸고 어떤 입력이 $\mathbf b$만 바꾸는지 구분해야 한다.

| 변경되는 값                   | $\mathbf A$ | $\mathbf b$ | 이유                                                    |
| ----------------------------- | ----------: | ----------: | ------------------------------------------------------- |
| 이전 time-step 온도           |          -- |        변경 | $C_PT_P^{(-)}$                                          |
| HVAC heating/cooling          |          -- |        변경 | $Q_P$                                                   |
| diffuser 또는 solar heat gain |          -- |        변경 | net source $Q_P$                                        |
| 외기 온도 $T_{amb}$           |          -- |        변경 | $A_f\Gamma_{\infty,f}T_{amb}$                           |
| 외기 대류계수 $h$             |        변경 |        변경 | $A_f\Gamma_{\infty,f}(h)$가 대각과 외기항에 모두 들어감 |
| 재질, 격자, $\Delta t$        |        변경 |   변경 가능 | $A_f\Gamma_f$, $C_P$ 변경                               |
| 복사 계수 $c_i,g_i$           |        변경 |        변경 | Picard 반복마다 현재 온도에서 재계산                    |
| 내부 질량 추정 온도           |          -- |        변경 | 현재 구현에서는 질량 온도가 공기식의 source로 들어감    |

따라서 heating/cooling, solar gain, 외기 온도 자체는 우변만 바꾼다. 반면 weather 입력 중 대류계수까지 바뀌면
행렬도 바뀐다. 복사를 켜면 거의 모든 Picard 반복에서 $c_i$가 달라져 대각행렬도 달라진다.

고정된 geometry, material, $\Delta t$, $h$로 기본 선형 문제를 여러 step 푸는 경우에는 LU 재사용이 이론적으로
가능하다. 하지만 현재 변경에서는 다음 이유로 cache를 넣지 않는다.

1. 복사 경로에서는 cache hit 가능성이 낮다.
2. weather가 $h$를 갱신하면 step 사이에도 행렬이 달라진다.
3. 행렬 동일성 비교와 cache invalidation 로직도 비용과 유지보수 부담이 있다.
4. 현재 격자 크기에서는 factorization 자체보다 TensorFlow 계수 조립과 반복 횟수가 더 큰 비용일 수 있으므로 실제
    end-to-end benchmark 없이 cache를 넣지 않는다.

`NumpyScipySimulator`는 매 direct solve마다 `spsolve`를 호출한다. 이후 고정 $h$, 복사 없음처럼
cache hit가 확실한 production workload를 별도로 측정해 유의미한 이득이 확인될 때만 `splu(A).solve(b)`
재사용을 고려하는 편이 안전하다.

`TFSimulator`는 SciPy를 import하지 않고 Tensor를 NumPy로 변환해 선형계를 풀지도 않는다.
`solver_method="fdm"`은 기존 Jacobi relaxation을 사용하고, `solver_method="cg"`는
TensorFlow tensor만 사용하는 matrix-free Jacobi- preconditioned conjugate
gradient(PCG)를 사용한다.

기본 선형 조건에서는 직접해 한 번으로 time step이 완료된다. `iteration_limit=1`이어도 수렴 성공으로 처리한다.

내부 질량을 켜면 직접해는 현재 질량 온도에 대한 공기 연립방정식을 정확히 풀고, 공기와 질량의 결합 fixed point를 위해 외부 반복을
수행한다. 복사를 켜면 현재 lagged 복사 계수에 대한 선형 시스템을 정확히 풀지만, 비선형 복사 fixed point를 위해 외부 반복이
필요하다.

간단한 end-to-end 비교는 `reference/heat_balance_benchmark.py`로 재현한다. 이 benchmark는 외기
온도, 대류계수, 양·음의 열원을 step마다 바꾸므로 정적인 행렬만 반복해서 푸는 특수 상황을 측정하지 않는다.

### 9.6 복사가 있을 때의 outer fixed-point solve

복사가 있으면 전체 시스템은 다음 비선형 형태가 된다.

$$
\mathbf A(\mathbf T)\mathbf T=\mathbf b(\mathbf T).
$$

현재 온도 추정값을 $\mathbf T^{(k)}$라고 하자. 먼저 이 온도에서 복사 계수를 계산한다.

$$
c_i^{(k)}=c_i(\mathbf T^{(k)}),
\qquad
g_i^{(k)}=g_i(\mathbf T^{(k)}).
$$

한 반복 동안 이 계수를 고정하면 다음 선형 시스템을 얻는다.

$$
\left(D_i+A_{rad}c_i^{(k)}\right)T_i^{(k+1)}
-\sum_jG_{ij}T_j^{(k+1)}
=b_i+A_{rad}g_i^{(k)}.
$$

행렬 형태는 다음과 같다.

$$
\mathbf A^{(k)}\mathbf T^{(k+1)}=\mathbf b^{(k)}.
$$

#### 9.6.1 enclosure별 복사 계수 계산

복사 교환 행렬의 off-diagonal 부분을 $M$이라고 하면

$$
q_i^{rad}
=
\sum_{j\ne i}M_{ij}\sigma(T_j^4-T_i^4)
$$

이다. 여기서 `exact secant`는 $\sigma T^4$ 곡선 위의 두 점을 잇는 **할선의 기울기**다.
$f(T)=\sigma T^4$라고 두면

$$
h_{ij}
=
\frac{f(T_j)-f(T_i)}{T_j-T_i}
=
\frac{\sigma(T_j^4-T_i^4)}{T_j-T_i}
$$

이고, 네제곱 차의 인수분해로 다음 식을 얻는다.

$$
h_{ij}
=
\sigma(T_i^2+T_j^2)(T_i+T_j).
$$

따라서

$$
\sigma(T_j^4-T_i^4)=h_{ij}(T_j-T_i)
$$

는 근사가 아니라 정확한 항등식이다. $T_i=T_j=T$이면 분모가 0인 할선식 대신 극한값인 접선 기울기

$$
\lim_{T_j\to T_i}h_{ij}=4\sigma T^3
$$

를 생각할 수 있다. 문헌에서 자주 쓰는 $4\sigma\bar T^3$는 두 온도가 가까울 때의 대표온도 근사이고, 여기의 $h_{ij}$는
온도차가 커도 네제곱 차를 정확히 재현한다.

$h_{ij}$를 전개하면

$$
h_{ij}
=
\sigma
\left(T_i^3+T_i^2T_j+T_iT_j^2+T_j^3\right)
$$

이다. 코드의 `moment_0`부터 `moment_4`는 시간 적분 모멘트나 통계적 모멘트가 아니다. 각 표면 $i$가 다른 표면들의 온도
거듭제곱을 복사 교환 가중치 $M_{ij}$로 합한 **weighted power sum**이다. 성분별 정의는

$$
m_{p,i}=\sum_{j\ne i}M_{ij}T_j^p,
\qquad p=0,1,2,3,4
$$

이고 벡터로 쓰면

$$
\mathbf m_p=M\mathbf T^p,
\qquad p=0,1,2,3,4
$$

로 정의한다. 여기서 $\mathbf T^0=\mathbf 1$이다. 그러면 복사 대각 계수는

$$
\mathbf c
=
\sigma
\left(
\mathbf T^3\odot\mathbf m_0
+\mathbf T^2\odot\mathbf m_1
+\mathbf T\odot\mathbf m_2
+\mathbf m_3
\right)
$$

이고 복사 구동항은

$$
\mathbf g
=
\sigma
\left(
\mathbf T^3\odot\mathbf m_1
+\mathbf T^2\odot\mathbf m_2
+\mathbf T\odot\mathbf m_3
+\mathbf m_4
\right)
$$

이다. 예를 들어 $c_i=\sum_jM_{ij}h_{ij}$에 전개된 $h_{ij}$를 대입하면

$$
\begin{aligned}
c_i
&=\sigma\sum_jM_{ij}
\left(T_i^3+T_i^2T_j+T_iT_j^2+T_j^3\right)\\
&=\sigma\left(
T_i^3m_{0,i}+T_i^2m_{1,i}+T_im_{2,i}+m_{3,i}
\right),
\end{aligned}
$$

가 되어 위 벡터식의 각 성분과 정확히 일치한다. $g_i=\sum_jM_{ij}h_{ij}T_j$도 같은 방식으로 $m_1$부터 $m_4$가
나온다.

이 방법은 각 $(i,j)$ 표면 쌍에 대한 $n\times n$ 온도 배열을 별도로 만들지 않고, enclosure마다

$$
M[\mathbf 1,\mathbf T,\mathbf T^2,\mathbf T^3,\mathbf T^4]
$$

한 번으로 다섯 weighted power sum을 동시에 계산한다. `moment`라는 변수명은 이 다섯 열을 짧게 부르기 위한 구현 이름일
뿐이다.

#### 9.6.2 복사 항을 행렬과 우변에 넣는 방법

복사가 없는 기본 행렬을 $\mathbf A_0$, 기본 우변을 $\mathbf b_0$라고 하자. 내벽 복사 면적을 대각행렬

$$
\mathbf S_{rad}
=
\operatorname{diag}(A_{rad,1},\ldots,A_{rad,N})
$$

로 쓰면 Picard 반복 $k$의 시스템은

$$
\mathbf A^{(k)}
=
\mathbf A_0
+\mathbf S_{rad}\operatorname{diag}(\mathbf c^{(k)})
$$

와

$$
\mathbf b^{(k)}
=
\mathbf b_0
+\mathbf S_{rad}\mathbf g^{(k)}
$$

로 조립된다.

즉 복사 self-term은 행렬 대각항을 증가시키고, 다른 표면 온도가 만드는 복사 유입은 우변 열원으로 들어간다. $c_i\ge0$이므로
복사 대각항은 음의 damping이 아니라 양의 안정화 항이다.

복사 enclosure의 off-diagonal coupling을 $\mathbf A$에 직접 넣지 않는 이유는 전도 행렬의 최대 5-point
sparse stencil을 유지하기 위해서다. 모든 복사 표면을 implicit coupling으로 넣으면 enclosure 크기의 dense
block이 생긴다.

#### 9.6.3 실제 Picard 반복 순서

각 Picard 반복의 절차는 다음과 같다.

1. 이전 time step의 온도 또는 직전 반복 온도를 초기값 $\mathbf T^{(0)}$로 둔다.
2. 현재 온도 $\mathbf T^{(k)}$에서 enclosure surface 온도를 gather한다.
3. enclosure별 $M[1,T,T^2,T^3,T^4]$를 계산한다.
4. moment로부터 $\mathbf c^{(k)}$, $\mathbf g^{(k)}$를 계산한다.
5. 복사 면적을 곱해 grid 위치의 W/K 대각항과 W 구동항으로 scatter한다.
6. $\mathbf A^{(k)}$, $\mathbf b^{(k)}$를 조립한다.
7. NumPy/SciPy backend는 `spsolve`, TensorFlow backend는 PCG로 선형 시스템을 풀어
    $\mathbf T^{(k+1)}$를 얻는다.
8. interior mass가 있으면 새 공기 온도로 질량 온도를 다시 계산한다.
9. 온도 변화량을 계산하고 수렴하지 않았으면 $k\leftarrow k+1$로 반복한다.

의사 코드는 다음과 같다.

```python
temperature = previous_temperature.copy()

for iteration in range(iteration_limit):
  coefficient, driving = radiation_terms(temperature)
  matrix = assemble_matrix(coefficient)
  rhs = assemble_rhs(driving, previous_temperature, input_q)
  new_temperature = sparse_direct_solve(matrix, rhs)

  if include_interior_mass:
    new_mass_temperature = update_mass_from_previous_step(
        new_temperature
    )

  max_delta = max(abs(new_temperature - temperature))
  if max_delta <= convergence_threshold:
    break

  temperature = new_temperature
```

#### 9.6.4 복사와 내부 질량을 함께 켠 경우

복사와 내부 질량을 함께 켜면 한 Picard 반복에서 사용하는 surface temperature vector는 내벽 CV의 공기 온도와 내부
질량 온도를 함께 포함한다.

$$
\boldsymbol\theta^{(k)}
=
\begin{bmatrix}
\mathbf T_{wall}^{(k)}\\
\mathbf T_m^{(k)}
\end{bmatrix}.
$$

먼저 $\boldsymbol\theta^{(k)}$로 모든 enclosure의 $\mathbf c^{(k)}$와
$\mathbf g^{(k)}$를 계산한다. 내벽 node에서는 그 표면 온도가 air-system 미지수 $T_i^{(k+1)}$ 자체이므로

$$
q_{rad,i}^{(k+1)}
=g_i^{(k)}-c_i^{(k)}T_i^{(k+1)}
$$

로 두어 $A_{rad,i}c_i^{(k)}$를 행렬 대각항에 넣는다. 반면 내부 질량 node의 복사 표면 온도는 air-system 미지수가
아니라 현재 질량 추정값 $T_{m,i}^{(k)}$이므로

$$
q_{rad,m,i}^{(k)}
=g_{m,i}^{(k)}-c_{m,i}^{(k)}T_{m,i}^{(k)}
$$

전체를 air equation의 명시적 우변 열원으로 넣는다. 즉 mass node의 복사 self-term을 air CV의 대각항에 잘못
더하지 않는다.

내부 질량이 없는 기본 전도·저장 행렬과 우변을 $\mathbf A_0$, $\mathbf b_0$라고 하고, air-mass 컨덕턴스를
grid 대각행렬 $\mathbf G_m$으로 쓰자. 내벽과 mass 복사면적 대각행렬을 각각 $\mathbf S_w$,
$\mathbf S_m$으로 쓰면 현재 구현의 air direct solve는 다음 형태다.

$$
\begin{aligned}
\Bigl[
\mathbf A_0
+\mathbf G_m
+\mathbf S_w\operatorname{diag}(\mathbf c_w^{(k)})
\Bigr]\mathbf T^{(k+1)}
=\;&\mathbf b_0
+\mathbf G_m\mathbf T_m^{(k)}\\
&+\mathbf S_w\mathbf g_w^{(k)}
+\mathbf S_m
\left(
\mathbf g_m^{(k)}
-\mathbf c_m^{(k)}\odot\mathbf T_m^{(k)}
\right).
\end{aligned}
$$

여기서 $\mathbf G_m$은 interior-mass mask 밖에서는 0이다. $\mathbf S_w$와 $\mathbf S_m$도 해당
복사 node가 scatter되는 air CV 위치 밖에서는 0이다. 따라서 행렬의 전도 off-diagonal은 기존 5-point
stencil을 그대로 유지한다.

이 선형 air system을 NumPy/SciPy direct solve 또는 TensorFlow PCG로 푼 다음, **이전 time
step** 질량 온도 $\mathbf T_m^{(-)}$에서 새 질량 추정값을 다시 계산한다.

$$
\mathbf T_m^{(k+1)}
=
(\mathbf G_m+\mathbf C_m)^{-1}
\left(
\mathbf G_m\mathbf T^{(k+1)}
+\mathbf C_m\mathbf T_m^{(-)}
\right).
$$

각 CV에서는 앞의 식이

$$
T_{m,i}^{(k+1)}
=
\frac{G_{m,i}T_i^{(k+1)}+C_{m,i}T_{m,i}^{(-)}}
{G_{m,i}+C_{m,i}}
$$

가 된다. 반복 $k$의 질량 추정값을 다음 반복의 air-mass source와 복사 surface temperature로 사용한다.

$$
(\mathbf T^{(k)},\mathbf T_m^{(k)})
\longrightarrow
(\mathbf c^{(k)},\mathbf g^{(k)})
\longrightarrow
\mathbf A^{(k)}\mathbf T^{(k+1)}=\mathbf b^{(k)}
\longrightarrow
\mathbf T_m^{(k+1)}.
$$

따라서 정확성의 범위를 구분해야 한다.

- NumPy/SciPy backend는 각 outer iteration의 air 선형계를 sparse direct solve한다.
- TensorFlow backend는 지정한 residual tolerance까지 PCG로 근사해를 구한다.
- 내부 질량은 주어진 새 air temperature에 대해 닫힌 형태 식으로 정확히 갱신한다.
- 그러나 복사 계수와 air-mass source는 직전 추정값에 의존하므로 전체 coupled timestep은 한 번의 선형 solve가
    아니라 fixed-point 수렴이 필요하다.
- 수렴점에서는 air balance, mass storage balance와 원래 $T^4$ 복사 balance를 동시에 만족한다.

현재 production 구현의 종료 조건은 기본적으로

$$
\Delta_T^{(k)}
=
\max_i
\left|T_i^{(k+1)}-T_i^{(k)}\right|
$$

가 `convergence_threshold` 이하인지 확인하는 것이다. 또한 설정된 경우 연속 반복의 `max_delta` 변화가 충분히 작은
상태가 일정 횟수 유지되는지도 확인한다.

현재 runtime 종료 조건은 원래 $T^4$ 식의 residual을 직접 계산하지 않는다. 대신 테스트에서 다음 비선형 residual을
별도로 검증한다.

$$
R_i(\mathbf T)
=
D_iT_i
-\sum_jG_{ij}T_j
-b_i
-A_{rad,i}q_i^{rad}(\mathbf T).
$$

수치적 정확도를 더 엄밀하게 관리하려면 runtime에서도

$$
\epsilon_R
=
\max_i\frac{|R_i|}{D_i}
$$

와 같은 등가 온도 residual을 온도 변화량과 함께 사용하는 것이 좋다.

#### 9.6.5 수렴점이 원래 $T^4$ 식을 만족하는 이유

수렴점에서는

$$
\mathbf T^{(k+1)}=\mathbf T^{(k)}=\mathbf T^*
$$

이므로

$$
g_i(\mathbf T^*)-c_i(\mathbf T^*)T_i^*
=
\sum_{j\ne i}M_{ij}\sigma
\left((T_j^*)^4-(T_i^*)^4\right).
$$

따라서 각 반복에서 푸는 문제는 lagged coefficient를 가진 선형 문제지만, 수렴한 온도는 원래 비선형 $T^4$ 이산 열평형식을
만족한다.

정확한 표현은 다음과 같다.

- 전체 비선형 복사 문제의 닫힌 형태 해석해는 아니다.
- 각 Picard 반복의 선형 시스템은 희소 LU로 정확하게 푼다.
- 반복이 수렴하면 원래 비선형 이산식의 수치적 정확해를 얻는다.

### 9.7 Newton 방법과의 차이

원래 비선형 residual을 직접 Newton 방법으로 풀 수도 있다.

$$
\mathbf J(\mathbf T^{(k)})\Delta\mathbf T
=-\mathbf R(\mathbf T^{(k)}),
\qquad
\mathbf T^{(k+1)}=\mathbf T^{(k)}+\lambda\Delta\mathbf T.
$$

복사 열유속의 Jacobian 성분은

$$
\frac{\partial q_i}{\partial T_i}
=-4\sigma T_i^3\sum_{j\ne i}M_{ij},
$$

$$
\frac{\partial q_i}{\partial T_j}
=4\sigma M_{ij}T_j^3
\qquad(i\ne j)
$$

이다. Newton 방법은 수렴점 근처에서 빠르지만 enclosure 안의 모든 복사 표면을 서로 연결하는 dense Jacobian
block을 만든다. 현재 Picard 직접해 방식은 복사 off-diagonal 항을 우변에 두어 전도 시스템의 sparse 구조를 유지한다.

## 10. 유한체적 반복 풀이와 직접 $\mathbf A\mathbf T=\mathbf b$ 풀이 비교

기본 TensorFlow 솔버는 Jacobi relaxation으로 같은 연립방정식의 해에 반복적으로 접근한다. 직접해 솔버는 SciPy
sparse solve로 같은 이산식을 한 번에 푼다. 따라서 비교 대상은 `finite difference physics`와
`matrix physics`가 아니라 **같은 유한체적 방정식에 대한 iterative linear solver와 direct linear
solver**다.

검증에는 다음 조건이 포함된다.

- 서로 다른 열전도율을 갖는 공기와 벽
- edge와 corner를 포함하는 직사각형 외곽
- legacy stepped floor plan에서 양쪽 CV의 공유 열률 보존
- 0이 아닌 내부 열원
- 초기 온도보다 높은 외기 온도
- 비균일한 내부 질량 초기 온도
- 복사만 켠 경우
- 내부 질량만 켠 경우
- 복사와 내부 질량을 함께 켠 경우

주요 허용 기준은 다음과 같다.

| 검증 항목                                  | 허용 기준                                      |
| ------------------------------------------ | ---------------------------------------------- |
| 공유 면 대칭성                             | 양쪽 CV의 $A_f\Gamma_f$가 일치                 |
| 물질 경계                                  | 단열성이 큰 쪽의 저항이 전체 컨덕턴스를 지배   |
| 직사각형 공유 면                           | `min` 적용 전후의 $A_f$가 동일                 |
| legacy stepped 공유 면                     | 작은 접선 길이를 사용해 양쪽 열률이 일치       |
| 외기 컨덕턴스                              | $0<\Gamma_{\infty,f}<h$                        |
| 직접해 에너지 잔차                         | 등가 온도 오차 $5\times10^{-5}$ K 미만         |
| 직접해의 relaxation fixed point            | 한 sweep 이동량 $5\times10^{-5}$ K 미만        |
| 기본 direct와 iterative 수렴해             | 최대 차이 $5\times10^{-4}$ K 미만              |
| 내부 질량 direct와 iterative 수렴해        | 공기 온도 최대 차이 $10^{-3}$ K 미만           |
| 복사 direct와 iterative 수렴해             | 최대 차이 $5\times10^{-4}$ K 미만              |
| 복사와 내부 질량 direct와 iterative 수렴해 | 공기 온도 최대 차이 $10^{-3}$ K 미만           |
| 내부 질량 온도                             | backend 사이 최대 차이 $5\times10^{-5}$ K 미만 |

직접해와 반복해가 bit 단위로 같지는 않다. TensorFlow relaxation은 `float32`로 계수를 조립하고 반복하므로 가장
가까운 표현 가능 온도에서 더 이상 움직이지 않을 수 있다. SciPy 직접해는 희소 시스템을 `float64`로 풀기 때문에 두 결과 사이에
수 mK 수준의 차이가 남는다.

### 10.1 간단한 시간 benchmark

다음 스크립트는 네 물리 조합 각각에 대해 Jacobi 반복 풀이와 sparse direct 풀이를 비교한다.

1. 복사 없음, 내부 질량 없음
2. 복사 없음, 내부 질량 있음
3. 복사 있음, 내부 질량 없음
4. 복사 있음, 내부 질량 있음

각 step에서 외기 온도, 외기 대류계수, 양의 heating source와 음의 cooling source를 바꾼다. 따라서 고정 행렬을
반복해 푸는 경우만 유리하게 측정하지 않는다. 실행 시간을 assert하지 않으므로 이 스크립트는 unit test가 아니라 수동
benchmark다.

```bash
poetry run python reference/heat_balance_benchmark.py \
  --steps 3 --repeats 1
```

스크립트 기본값은 전체 네 조합을 빠르게 확인하기 위한 1 step, 1 repeat다. 위 명령의 3 step은 50 W heating,
-30 W cooling, 15 W heat gain과 함께 외기 온도 및 대류계수를 순서대로 바꾼다. 더 안정적인 중앙값이 필요하면
`--repeats 3`처럼 늘릴 수 있지만 반복 풀이가 느린 조건에서는 실행 시간도 비례해 증가한다.

출력의 `speedup`은 `iterative_ms / direct_ms`다. 1보다 크면 direct solve가 빠르다. benchmark의
iterative path는 production의 `max_delta` 종료 조건을 사용하지 않는다. 각 sweep 후 direct 결과와 직접
비교해 다음 기준을 만족할 때까지 계속 계산한다.

- 내부 질량 없음: `air_diff_K <= 5e-4 K`
- 내부 질량 있음: `air_diff_K <= 1e-3 K`
- 내부 질량 온도: `mass_diff_K <= 5e-5 K`

이 값들은 production timestep의 `convergence_threshold`가 아니라, float32 Jacobi 결과를
float64 direct 참조해와 비교할 때 사용하는 별도 acceptance tolerance다. 모든 reference benchmark의
실제 timestep 수렴 기본값은 `1e-4 K`다.

`sweeps`는 이 기준에 도달하는 데 필요했던 평균 Jacobi sweep 수다. 표의 오차는 세 step에서 관측한 최대값이다. 절대
시간은 CPU, BLAS, TensorFlow와 SciPy build에 따라 달라진다.

2026년 9월 17일 로컬 CPU 실행 결과는 다음과 같다.

- CPU: AMD Ryzen 7 5800X
- TensorFlow 2.15.1, SciPy 1.15.3
- solver 비교용 균일 열전도율 $k=0.5\ \mathrm{W/(m\,K)}$
- 30초 time step, timestep \`convergence_threshold=10^{-4}$ K
- `--steps 3 --repeats 1`
- LU cache 없음: direct path는 매번 `spsolve` 호출

| interior mass | radiation | iterative ms/step | direct ms/step | speedup | sweeps |          air diff K | mass diff K |
| ------------: | --------: | ----------------: | -------------: | ------: | -----: | ------------------: | ----------: |
|           off |       off |             853.7 |           4.62 |   184.9 |  113.7 | $4.88\times10^{-4}$ |          -- |
|            on |       off |             828.6 |           8.33 |    99.5 |  104.3 | $9.77\times10^{-4}$ |           0 |
|           off |        on |            1068.2 |          18.24 |    58.6 |  113.7 | $4.88\times10^{-4}$ |          -- |
|            on |        on |            1071.3 |          20.32 |    52.7 |  108.7 | $9.77\times10^{-4}$ |           0 |

따라서 네 조합 모두 지정한 온도 오차 이내로 실제 수렴했으며, 같은 정확도 기준에서도 direct solve가 약 53--185배 빨랐다.
iterative 시간에는 매 sweep마다 direct 참조해와 오차를 비교하는 benchmark용 검사 비용도 포함되므로 production
Jacobi sweep만의 순수 시간보다는 조금 큰 값이다.

균일 열전도율을 사용한 이유는 solver 자체의 수렴과 물질 경계 오차를 분리하기 위해서다. 공기와 단열벽의 급격한 열전도율 차이,
half-cell 저항, 공유 면 보존은 이 파일의 별도 conductance 및 energy-balance 테스트에서 검증한다.

### 10.2 Direct solve의 radiation on/off 다시간 비교

Direct solver만 사용해 radiation을 켰을 때와 껐을 때의 실행 시간과 12시간 온도 trajectory를 비교하는
benchmark는 별도 파일로 분리했다.

```text
reference/direct_solver_radiation_multihour_benchmark.py
reference/direct_solver_radiation_multihour_benchmark.md
```

300초 timestep으로 12시간을 계산한 로컬 결과에서 radiation은 timestep 비용을 내부질량 없이 3.23배, 내부질량과
함께 2.51배 증가시켰다. Radiation on/off의 최대 국소 공기 온도 차이는 각각 0.66 K와 2.14 K였고, grid 평균
공기 온도의 최대 차이는 두 경우 모두 약 0.20 K였다. 상세 forcing, 지표 정의와 결과 해석은 별도 benchmark 문서를
참조한다.

### 10.3 Exact, Python iterative와 TF FDM의 다시간 비교

Direct $\mathbf A\mathbf T=\mathbf b$ 풀이, Python control-volume iterative sweep과
TensorFlow Jacobi FDM을 여러 geometry에서 독립적으로 실행하고, timestep timing boxplot과 온도장 및
exact 대비 차이 snapshot을 만드는 benchmark는 다음 파일로 분리했다.

```text
reference/exact_vs_fdm_multistep_visual_benchmark.py
reference/exact_vs_fdm_multistep_visual_benchmark.md
```

내부 질량과 복사의 on/off 네 조합 및 rectangular, irregular, fenestrated geometry를 포함한다. 각
geometry/case에서 Python iterative와 TF FDM을 exact discrete solution과 독립적으로 비교한다.
상세 실행 조건과 생성되는 CSV 및 PNG 목록은 별도 benchmark 문서를 참조한다.

### 10.4 현재 세 solver backend 비교와 sensitivity analysis

현재 architecture를 직접 비교하는 benchmark와 결과 문서는 다음과 같다.

```text
reference/three_solver_sensitivity_benchmark.py
reference/three_solver_sensitivity_benchmark.md
```

이 benchmark는 NumPy/SciPy sparse direct, TensorFlow PCG와 TensorFlow Jacobi FDM을
내부 질량 및 복사 on/off 네 조합에서 한 시간 동안 비교한다. 또한 active CV 수 100--4,356의 grid-size
sensitivity와 CG residual tolerance sensitivity를 포함한다.

2026년 9월 17일 CPU 실행에서 TF-PCG는 TF-FDM보다 7.2--14.5배 빨랐지만 현재 fixture 크기에서는
NumPy/SciPy direct보다 8.7--10.0배 느렸다. Base-physics size sweep에서는 약 1,156--2,500
active CV 사이에서 TF-PCG가 현재 Python/NumPy sparse assembly를 포함한 direct timestep보다
빨라졌다. 자세한 오차, CG iteration 수, trace-time 처리와 heatmap은 별도 결과 문서를 참조한다.

### 10.5 Device-resident FP32 PCG 비교

기존 backend를 변경하지 않고 전체 Picard timestep을 하나의 TensorFlow graph에서 실행하는
`tf_cg_resident_fp32` backend를 별도로 추가했다. 이 경로는 공기 및 내부 질량 온도를 `tf.Variable`로
device에 유지하고, PCG를 FP32와 기본 tolerance $10^{-8}$로 실행한다. CPU/GPU grid benchmark는
다음 명령으로 실행한다.

```bash
bash reference/run_application_grid_cpu_gpu.sh
```

결과 CSV는 완료된 `(device, case, grid, solver, repeat, configuration)` sample을 재사용하므로
중단 후 같은 명령을 실행하면 누락된 sample만 계산한다. 결합 결과에는 기존 FP64 PCG와 device-resident FP32
PCG가 별도 series로 저장된다.

기본 grid 목록은 interior size 8에서 2,048까지 증가한다. Runner는 각
`(device, solver, application)` 조합을 별도 Python process로 실행하므로 큰 grid에서 OOM 또는
solver failure가 발생해도 그 전에 성공한 sample은 CSV에 남고 다른 조합은 계속 실행된다. 실패 조합과 exit code는
`benchmark_failures.csv`에 기록한다.

면적 표시는 한 CV를 $10\text{ cm}\times10\text{ cm}$로 해석한 reference 값이다. CSV에는 다음 두 면적을
m²와 ft²로 모두 저장한다.

- `active_floor_area_*`: wall/boundary를 포함한 active CV 전체 footprint
- `interior_floor_area_*`: `interior_size^2`로 계산한 내부 floor footprint

Line plot의 위쪽 보조축과 boxplot의 grid tick에는 active footprint를 표시한다. 이 10 cm 값은 grid
규모를 실제 공간 크기로 읽기 위한 annotation이며 benchmark building material 및 기존 결과의 수치 이산화 간격을
자동으로 변경하지 않는다.

## 11. 테스트 실행 방법

열평형과 정확해 비교 테스트만 실행하려면 다음 명령을 사용한다.

```bash
poetry run pytest --disable-pytest-warnings -q \
  smart_control/simulator/tf_simulator_heat_balance_test.py
```

전체 회귀 테스트는 다음과 같다.

```bash
poetry run pytest --disable-pytest-warnings -q
```

현재 검증 결과는 다음과 같다.

```text
focused heat-balance tests: 27 passed
TF simulator + heat-balance tests: 76 passed
full test suite: 필요 시 별도 실행
```

## 12. 구현 위치

### `smart_control/simulator/simulator.py`

- CV별 실제 $u$, $v$ 계산
- 두 half-cell 저항과 실제 공유 면적을 사용한 컨덕턴스
- 외기 대류막과 boundary half-cell의 직렬 저항
- Gauss-Seidel CV 열평형 업데이트
- 내부 질량 결합 반복
- whole-field backend의 선형-only one-pass 종료 hook

### `smart_control/simulator/tf_simulator.py`

- CV 크기 및 방향별 컨덕턴스 tensor
- Jacobi finite-difference relaxation
- enclosure별 복사 대각 계수와 구동 열원 계산
- TensorFlow Jacobi FDM backend
- SciPy나 NumPy 변환이 없는 matrix-free Jacobi-PCG backend

### `smart_control/simulator/numpy_scipy_simulator.py`

- NumPy로 sparse heat-balance coefficient와 우변 조립
- SciPy `spsolve`를 이용한 reference direct solution
- 복사와 내부 질량 outer fixed-point 반복 지원

### `smart_control/simulator/tf_device_resident_simulator.py`

- 기존 `TFSimulator`와 독립적인 device-resident FP32 PCG backend
- 전체 Picard 반복과 PCG inner loop를 단일 `tf.function`에서 실행
- 공기 및 내부 질량 온도를 `tf.Variable`로 유지
- timestep 종료 시 선택적으로만 NumPy building state와 동기화

### `smart_control/simulator/tf_simulator_heat_balance_test.py`

- half-cell 및 공유 면 컨덕턴스 검증
- 외기 경계 컨덕턴스의 물리적 극한 검증
- 정상상태 에너지 보존 검증
- 정확해 잔차 및 fixed-point 검증
- 기본, 내부 질량, 복사, 내부 질량과 복사 조합 비교

### `reference/heat_balance_benchmark.py`

- 반복 풀이와 직접 풀이의 time-step당 실행 시간 비교
- 복사 on/off와 내부 질량 on/off의 네 조합 비교
- step마다 weather 계수와 heating/cooling source 변경

### `reference/direct_solver_radiation_multihour_benchmark.py`

- direct solver의 radiation on/off runtime 비교
- 12시간 air 및 interior-mass temperature trajectory 비교
- 동일 weather, HVAC와 boundary surface heat gain 적용

### `reference/exact_vs_fdm_multistep_visual_benchmark.py`

- exact discrete, Python iterative와 TF FDM의 독립적인 trajectory 비교
- rectangular, irregular, fenestrated floor plan 선택 지원
- 세 solver의 반복 timing boxplot 생성
- 첫, 중간, 마지막 timestep의 3-way temperature/difference heatmap 생성
- timestep별 exact 대비 air 및 interior-mass error CSV 저장

### `reference/three_solver_sensitivity_benchmark.py`

- NumPy/SciPy direct, TF-PCG와 TF-FDM의 네 물리 조합 multistep 비교
- 반복 timing boxplot과 direct-reference heatmap 생성
- active-CV grid-size 및 CG tolerance sensitivity 분석
- CG warm-up 시간과 warm timestep 시간 분리

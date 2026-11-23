# EnergyPlus와 SBSim 공통 VAV 모델링 계획

## 목표

하나의 평면 명세로 같은 존의 위치, 실내 면적·부피, 인접 관계를 가진 EnergyPlus 및 SBSim VAV 모델을 생성한다. 이번 단계의
완료 기준은 두 엔진에서 모델을 불러와 5분 스텝으로 실행할 수 있는지 확인하는 것이다. RL 학습 속도 측정은 모델 검토 뒤 진행한다.

## 공통 평면

- 입력은 존 수, 열 수, 실내 폭·깊이, 층고, 벽 두께, SBSim 셀 크기다. 북쪽 행부터 왼쪽에서 오른쪽으로 채우며, 마지막 행의 빈
    타일은 외기로 둔다. 평면은 단층 직사각형 존의 연결로 구성한다.
- 셀 경계에 실내 치수와 벽 두께를 정확히 맞춘다. 존 수로 건물 규모를 바꾸고, 셀 크기로 동일한 물리 평면의 CV 개수를 바꾼다. 기본
    검증 규모는 1, 5, 16존이다. 5존은 부분 행과 오목한 외벽도 확인한다.
- EnergyPlus의 구역 표면은 공통 평면의 실내 경계를 사용한다. 내부 벽은 양쪽 방의 표면을 `Surface`로 서로 참조하며 벽
    두께만큼 떨어져 있다. 이는 EnergyPlus가 지원하는 표현이다.
    [EnergyPlus 열구역 형상 문서](https://energyplus.readthedocs.io/en/v26.2.0/guides/input-output-reference/1.10-group-thermal-zone-descriptiongeometry.html)
- 선택적으로 노출된 각 벽에 창을 한 개 배치한다. 공통 평면에서 창의 폭과 가로 위치를 정하고, EnergyPlus에는 높이까지 있는 표면을
    생성한다. SBSim은 외벽의 창 CV와 기존 일사 경로를 사용한다.

## 모델 생성 순서

1. `floorplan.py`에서 평면 명세와 존·창 위치를 생성한다. 존 순서와 면적, 부피, 인접 관계, CV 개수를
    `floorplan.json`에 남긴다.
2. `energyplus_model.py`에서 공통 좌표로 `BuildingSurface:Detailed`와 창을 만든다. 중앙 공기 루프
    1개에 가변풍량 팬, DX 냉각 코일, 존별 단일 덕트 VAV 전기 재열을 연결한다. 고정 설계값을 실내 면적으로 스케일링한다.
    epJSON을 직접 생성하면 설치된 EnergyPlus가 해석하는 필드로 구성할 수 있으며, 필요하면 공식 CLI로 IDF를 변환할 수
    있다. Eppy는 기존 IDF 편집에, OpenStudio는 HVAC 및 형상 모델링 SDK가 필요한 작업에 적합하다.
    [EnergyPlus epJSON과 변환 문서](https://energyplus.readthedocs.io/en/v26.2.0/essentials/essentials.html)
3. `sbsim_model.py`에서 공통 배열을 `FloorPlanBasedBuilding`, 중앙 `AirHandler`와
    `Boiler`, 존별 `Vav`에 연결한다. `JaxDeviceResidentSimulator`를 정상 스텝 진입점에서 실행한다.
    생성된 방의 좌표와 공통 명세의 대응을 확인한다.
4. 생성 CLI와 실행 검증 CLI를 추가한다. 공통 geometry 검증은 존별 면적·부피, 내부 벽 쌍, 창 방향·폭, 존별 VAV 수를
    검사한다. EnergyPlus는 EPW를 사용해 하루를 실행하고 오류 건수와 스텝 및 VAV 출력을 검사한다. SBSim은 같은 날짜의
    짧은 실행으로 존 온도, VAV 풍량, 솔버 수렴을 확인한다. 창을 켠 경우도 확인한다.
5. `README.ko.md`에 생성·실행 명령, 사용한 버전, 검증 결과와 엔진별 물리 모델 차이를 기록한다. 이후 속도 검증은 동일한 5분
    간격, 존 순서, 에피소드 길이 및 스텝 수를 적용한 뒤, 초기화·JAX 컴파일·시뮬레이터 스텝·학습 업데이트 시간을 각각 측정한다.

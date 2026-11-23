# 공통 평면 VAV 모델 입력

하나의 평면 명세에서 SBSim의 격자와 EnergyPlus의 epJSON을 생성한다. 방 이름, 위치, 면적, 창과 인접 관계를 공유한다. 새
7일 물리·SAC 실험은 [SBSim과 EnergyPlus 벤치마크](../../grouped_jax_study/README_ko.md)에
있다.

- [SBSim 100존 사무실 평면](models/zone100_office/floorplan.json)
- [상세 EnergyPlus epJSON](models/zone100_detailed_office/model.epJSON)과
    [평면 명세](models/zone100_detailed_office/floorplan.json)
- [공통 평면 생성 코드](generate.py)와 [상세 EnergyPlus HVAC 생성 코드](detailed_vav_hvac.py)

저장소 루트에서 100존 상세 입력을 재생성할 수 있다.

```bash
.venv/bin/python -m benchmarks.eplus_rl_speed.matched_vav.generate \
  --zones 100 --window-width-m 2 --detailed-office \
  --out /tmp/zone100-detailed-office
```

# 측면 상반신 자세 테스트 (명세서 v9 기준)

팀원의 `posture_test.py`(정면 카메라·목 길이 비율·어깨/고개 기울기)를 **명세서 v9**(측면, 머리~명치, 귀·어깨, H 하나)에 맞춰 다시 만든 테스트 코드입니다.
0단계 실험(카메라 배치, `L_ref` 후보 비교, S 검증, 임계값 탐색)에 쓰도록 만들었습니다.

## 가상환경
python -m venv .venv
source .venv/Scripts/activate

python -m pip install --upgrade pip
python -m pip install mediapipe opencv-python numpy scikit-learn joblib
python -m pip install scikit-learn==1.6.1

## 파일

| 파일 | 역할 |
| --- | --- |
| `config.yaml` | 명세서 C5 설정 키. **전부 임시 placeholder(TODO(실험))**. 값은 여기서만 바꿉니다 |
| `posture_core.py` | 핵심 로직(MediaPipe 비의존): 캘리브레이션, H/S, 측정 불가, 상태 머신, 이벤트 |
| `mp_io.py` | MediaPipe Tasks 입출력 (팀원 코드의 모델 로딩 방식 그대로) |
| `posture_spec_test.py` | 실시간/영상 테스트 실행기 (로그 CSV, 요약) |
| `analyze_log.py` | 로그 분석: 감지율·측정 불가 비율, `L_ref` 후보 비교, S 검증, `T_forward` 참고 분포 |
| `multiposture_probe.py` | MultiPosture 모델(joblib)을 우리 환경에 대 보는 호환성 점검 |
| `tests/test_core.py` | 합성 데이터 단위 테스트 13개 (카메라 불필요) |

## 설치·실행

```bash
pip install -r requirements.txt
python -m unittest discover -s tests -v          # 먼저: 로직 테스트 (카메라 불필요)
python posture_spec_test.py --log                # 웹캠 + 로그 저장
python posture_spec_test.py --video 녹화.mp4 --auto-calib --no-gui --log   # 녹화 영상으로 재현
python analyze_log.py data/posture/log_*.csv
```

`pose_landmarker_lite.task`가 같은 폴더에 있으면 그대로 쓰고, 없으면 자동 다운로드합니다.

키: `c` 캘리브레이션 시작 → 요약 확인 후 `y` 저장 / `n` 재측정 · `1~6` 조건 라벨 · `d` 디버그(S 표시) · `p` 일시정지 · `q` 종료

## 명세서 대비 팀원 코드에서 바뀐 점

| 항목 | 팀원 코드 | 이 코드 (명세 v9) |
| --- | --- | --- |
| 카메라 | 정면 | **측면, 머리~명치**. 골반·하체는 읽지도 않음 |
| 사용 관절 | 양쪽 귀·어깨·눈 | **캘리브레이션에서 고정한 한쪽 귀·어깨**(반대쪽 대체 금지) |
| 지표 | 목 길이 비율, 머리 크기, 어깨선·눈선 기울기 | **H = d·(x_ear − x_shoulder)/L_ref_fixed** 하나. S는 로그 전용 |
| 판정 | 절댓값 비교 포함 | **부호 있는 비교** dH > T_forward (후방 이동은 주의 아님) |
| 기준 길이 | 매 프레임 계산 | **캘리브레이션에서 1회 측정해 고정** (후보 A/B/C 동시 기록) |
| 캘리브레이션 | 3초 평균, 자동 저장 | 중앙값, 변동 폭 로그, **사용자 확인(y) 후 저장**, side·d 고정 |
| 측정 불가 | 사람 미검출 안내뿐 | 가림·저신뢰·화면 이탈·비정상 배치·미검출 → 판별 중단, 주의 타이머 일시정지/초기화, K프레임 연속 후 재개 |
| 경고 문구 | 거북목/어깨/고개 기울임 등 | 명세 17.2 문구(H 기준). 진단 표현 없음 |
| 화면 | 좌우 반전 | **반전 없음**(d 부호가 헷갈리지 않게) |

## 0단계 실험 절차 (권장)

1. **카메라 배치**: 측면에서 머리~명치가 보이게 고정. 앞으로 숙이거나 머리를 내밀 때 귀·어깨가 회색 마진 박스 밖으로 나가지 않는지 확인(나가면 `out_of_frame`).
2. **캘리브레이션**(`c`): 편안한 자세 7초 → 요약(side, d, L_ref, H 변동) 확인 → `y`.
3. **조건별 기록**(같은 캘리브레이션 유지, 조건 키로 라벨 지정, 각 20~30초 이상, `--log`):
   - 고정 조건: `1 neutral` 기준 자세 / `2 lean_fwd` 상체를 숙임 / `3 head_fwd` 머리만 내밀기 / `4 recline` 뒤로 기댐
   - 변경 조건: `5 chair_moved` 기준 자세 유지한 채 의자만 앞뒤로 이동 / `6 camera_moved` 카메라만 살짝 이동 (**이때 재캘리브레이션 금지**)
4. `python analyze_log.py data/posture/log_*.csv` 로 확인:
   - 3번 항목: `L_ref` 후보 A/B/C 중 신호/잡음이 큰 쪽
   - 4번 항목: S가 자세 동작과 위치 변경을 구분하는지 (비율이 1에 가까우면 승격 불가)
   - 5번 항목: `T_forward` 후보 구간
5. 값이 정해지면 **`config.yaml` 한 곳**에만 반영하고, 실험이 끝나면 로그 CSV는 **삭제**합니다(프레임별 좌표 원본 저장 금지 원칙).

## MultiPosture 모델(`upperbody_posture_model.joblib`) 점검 결과

모델 파일을 열어 확인한 내용입니다(`data.csv` 자체와 학습 정확도 기록은 받지 못해 평가는 못 했습니다).

- RandomForest(300 트리), 입력 **51 특징 = 상체 17개 랜드마크 × (x, y, z)**, 출력 5클래스: `TUP`(상체 세움) / `TLF`(앞으로) / `TLB`(뒤로) / `TLR`(오른쪽) / `TLL`(왼쪽).
- 트리 임계값 범위로 보면 좌표는 **미터 단위, 힙 중심이 원점**인 형태입니다(예: 코 y ≈ −0.5, 어깨 y ≈ −0.4). 즉 MediaPipe **월드 랜드마크**와 같은 좌표계로 추정됩니다.
- 우리 환경과 맞지 않을 가능성이 큰 이유:
  1. **힙이 화면에 없습니다.** 월드 좌표의 원점이 힙 중심이라, 힙이 안 보이면 원점이 추정값입니다(명세: 화면 밖 관절 좌표는 신뢰 불가).
  2. 클래스에 **좌/우 기울임**이 있지만 측면 카메라로는 보이지 않습니다(명세 제외 항목).
  3. 특징 중 `x`·`z` 중요도가 높은데 학습 데이터의 카메라 방향과 우리의 측면 배치가 다를 수 있습니다.
  4. 이 모델은 "자세 유형 분류"일 뿐 H 판정을 대체하지 않습니다(명세 8.2).
- 점검 방법: `python multiposture_probe.py --model-file upperbody_posture_model.joblib`
  → 모델 라벨, 학습 분포 범위 밖 특징 수, 힙 visibility, **규칙 기반 상태 × 모델 라벨 교차표**를 출력합니다. 정확도 평가가 아니라 호환성 점검입니다.
- 모델은 scikit-learn **1.6.1**로 저장돼 있어 `requirements.txt`에 같은 버전을 적었습니다(다르면 경고가 뜹니다).

## 이 코드에서 정한 가정 (확인 필요)

- `d`: 코가 귀보다 화면 오른쪽이면 +1 (0단계에서 양방향 검증 필요).
- 사람이 검출되지 않으면 사유를 `ear_not_detected`로 기록.
- 가림(`occluded`)은 임시 휴리스틱(짧은 시간에 visibility 급락·복귀 반복). 판별 방식은 TBD.
- `unavailable_prolonged`는 측정 불가가 `unavailable_prolonged_sec` 이상 지속될 때, `recalibration_suggested`는 그 횟수가 반복될 때(안내만, 자동 갱신 없음).
- 캘리브레이션 "불안정" 기준(`calib_max_h_std`)은 임시값이며 우선 로그로 분포를 봅니다.

## 검증 상태

- 실행해서 확인함: `posture_core.py` 단위 테스트 13개 통과, 실행기·분석기는 가짜 MediaPipe와 합성 로그로 흐름 확인, MultiPosture 특징 생성(51개)과 예측 호출 확인.
- **아직 못 해본 것**: 실제 웹캠·MediaPipe로 실행(이 환경에는 카메라·mediapipe가 없음), 월드 랜드마크 좌표계가 학습 데이터와 같은지(점검 스크립트로 확인 필요), Orange Pi에서의 FPS.
  첫 실행에서 오류가 나면 에러 메시지를 보내 주세요.

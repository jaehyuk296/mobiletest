"""
webcam_test_v2.py — 팀원 webcam_test.py(MLP 39특징 분류기)를 '정확도를 직접 재 볼 수 있게' 고친 버전

원본과 달라진 점
  1) MediaPipe에는 '좌우 반전하지 않은' 원본 프레임을 넣고, 화면 표시용 복사본만 반전한다
     (원본은 반전한 프레임을 모델에 넣어 TLL/TLR 좌우와 x 좌표가 학습 때와 뒤집힐 수 있음)
  2) 신뢰도(visibility) 게이트: 코·양쪽 귀·양쪽 어깨가 기준 미만이면 예측하지 않고 UNAVAILABLE (명세: 불확실하면 측정 불가)
  3) 확률 이동평균(EMA)으로 프레임 깜빡임 완화
  4) 정답 라벨 키 입력 → 종료 시 혼동행렬·클래스별 정밀도/재현율 출력 (웹캠에서의 실제 정확도 측정)
  5) (선택, 미검증) --yaw-normalize: 어깨선이 화면 x축과 나란하도록 수직축 회전 → 카메라 각도(정면/측면) 차이를 줄이는 실험
특징 추출(extract_features)은 원본과 동일하다(어깨 중심 이동 + 어깨너비 정규화).

키: u=TUP  f=TLF  b=TLB  l=TLL  r=TLR  (지금 취하고 있는 자세를 정답으로 지정) | n=정답 없음 | ESC 종료
※ 라벨을 누른 구간만 정확도에 반영된다. 자세 전환 직후 1~2초는 라벨을 바꾸지 말고 유지한 뒤 누르면 정확하다.
"""
import argparse, collections, csv, os, time
import numpy as np

LANDMARK_IDS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]     # 학습 때 쓴 13개 (nose, eyes, ears, mouth, shoulders)
GATE_IDS = [0, 7, 8, 11, 12]                                    # 신뢰도 게이트에 쓰는 랜드마크: 코, 양쪽 귀, 양쪽 어깨
KEY2LABEL = {ord("u"): "TUP", ord("f"): "TLF", ord("b"): "TLB", ord("l"): "TLL", ord("r"): "TLR"}


def extract_features(landmarks, yaw_normalize=False):
    """원본 webcam_test.py 와 동일한 특징 (1, 39). yaw_normalize=True 면 정규화 전에 수직(y)축 회전을 추가."""
    coords = np.array([[landmarks[i].x, landmarks[i].y, landmarks[i].z] for i in LANDMARK_IDS], dtype=np.float64)
    left_shoulder, right_shoulder = coords[11], coords[12]
    center = (left_shoulder + right_shoulder) / 2
    width = np.linalg.norm(left_shoulder - right_shoulder)
    if not np.isfinite(coords).all() or width < 1e-6:
        return None
    coords = coords - center
    if yaw_normalize:                                           # 실험: 어깨 벡터가 x축과 나란해지도록 y축 회전 (미검증)
        v = right_shoulder - left_shoulder
        ang = np.arctan2(v[2], v[0])
        c, s = np.cos(ang), np.sin(ang)
        rot = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
        coords = coords @ rot.T
    coords = coords / width
    return coords.reshape(1, -1)


def visibility_ok(pose_landmarks, thr):
    return all(getattr(pose_landmarks[i], "visibility", 0.0) >= thr for i in GATE_IDS)


class ConfusionTracker:
    def __init__(self, classes):
        self.classes = list(classes)
        self.m = collections.Counter()
        self.n_unavail = 0
        self.n_labeled = 0

    def add(self, truth, pred):
        if truth is None:
            return
        self.n_labeled += 1
        if pred is None:
            self.n_unavail += 1
        else:
            self.m[(truth, pred)] += 1

    def report(self):
        cl = self.classes
        print("\n================ 웹캠 정확도 (라벨을 지정한 구간만) ================")
        print(f"라벨 지정 프레임 {self.n_labeled}, 그중 측정 불가 {self.n_unavail}")
        print("혼동행렬 (행=정답, 열=예측)")
        print("%-6s" % "" + "".join("%7s" % c for c in cl) + "  | 재현율")
        total_ok = total = 0
        for t in cl:
            row = [self.m[(t, p)] for p in cl]; n = sum(row)
            if n == 0 and not any(self.m[(t, p)] for p in cl):
                continue
            total += n; total_ok += self.m[(t, t)]
            print("%-6s" % t + "".join("%7d" % v for v in row) + "  | %5.1f%%" % (100 * self.m[(t, t)] / max(n, 1)))
        print("정밀도: " + ", ".join(f"{p} {100 * self.m[(p, p)] / max(sum(self.m[(t, p)] for t in cl), 1):.1f}%" for p in cl))
        if total:
            print(f"전체 정확도(측정 가능 프레임 기준) {100 * total_ok / total:.1f}%")
        print("※ 한 사람·한 자리에서 잰 값입니다. 사람/카메라 위치/조명을 바꿔 반복해야 일반화 여부를 알 수 있습니다.")


def main():
    import cv2
    import joblib
    import mediapipe as mp

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="upperbody_posture_model_final.joblib")
    ap.add_argument("--camera", type=int, default=1)
    ap.add_argument("--image-coords", action="store_true", help="월드 랜드마크 대신 이미지 정규화 좌표 사용 (학습 CSV 기준과 같을 때만!)")
    ap.add_argument("--vis-thr", type=float, default=0.5, help="신뢰도 게이트 (TODO(실험))")
    ap.add_argument("--ema", type=float, default=0.3, help="확률 이동평균 계수(0이면 사용 안 함)")
    ap.add_argument("--complexity", type=int, default=1, help="0 lite / 1 full / 2 heavy. 학습 데이터가 Heavy 기반이면 2로도 비교")
    ap.add_argument("--yaw-normalize", action="store_true", help="실험(미검증): 어깨선 기준 수직축 회전")
    ap.add_argument("--csv", help="(개발용) 프레임별 확률·정답 로그 저장 경로")
    a = ap.parse_args()

    model = joblib.load(a.model)
    classes = list(model.classes_)
    mp_pose, mp_draw = mp.solutions.pose, mp.solutions.drawing_utils
    cap = cv2.VideoCapture(a.camera, cv2.CAP_DSHOW) if os.name == "nt" else cv2.VideoCapture(a.camera)
    if not cap.isOpened():
        raise RuntimeError("웹캠을 열 수 없습니다. --camera 를 0, 1, 2로 바꿔 보세요.")

    tracker = ConfusionTracker(classes)
    truth, ema_p = None, None
    fh = writer = None
    if a.csv:
        fh = open(a.csv, "w", newline="", encoding="utf-8"); writer = csv.writer(fh)
        writer.writerow(["ts", "truth", "pred", "conf"] + [f"p_{c}" for c in classes])
    t0 = time.time()

    with mp_pose.Pose(static_image_mode=False, model_complexity=a.complexity, smooth_landmarks=True,
                      min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("웹캠 프레임을 읽지 못했습니다."); break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)          # ★ 반전하지 않은 프레임을 모델에 입력
            results = pose.process(rgb)

            pred, conf, text = None, 0.0, "UNAVAILABLE (no person)"
            if results.pose_landmarks:
                src = results.pose_landmarks if a.image_coords else results.pose_world_landmarks
                if not visibility_ok(results.pose_landmarks.landmark, a.vis_thr):
                    text, ema_p = "UNAVAILABLE (low visibility)", None
                elif src:
                    X = extract_features(src.landmark, a.yaw_normalize)
                    if X is not None:
                        p = model.predict_proba(X)[0]
                        ema_p = p if (ema_p is None or a.ema <= 0) else a.ema * p + (1 - a.ema) * ema_p
                        k = int(np.argmax(ema_p)); pred, conf = classes[k], float(ema_p[k])
                        top = np.argsort(ema_p)[::-1][:3]
                        text = f"{pred}: {conf * 100:.0f}%   (" + ", ".join(f"{classes[i]} {ema_p[i] * 100:.0f}" for i in top[1:]) + ")"
            else:
                ema_p = None

            tracker.add(truth, pred)
            if writer:
                writer.writerow([f"{time.time() - t0:.3f}", truth or "", pred or "", f"{conf:.3f}"] +
                                ([f"{x:.3f}" for x in ema_p] if (ema_p is not None and pred) else [""] * len(classes)))

            view = cv2.flip(frame, 1)                              # 표시용만 거울처럼 반전 (랜드마크 그림은 생략: 좌우가 어긋나므로)
            cv2.rectangle(view, (10, 10), (620, 95), (0, 0, 0), -1)
            color = (0, 255, 0) if pred else (0, 165, 255)
            cv2.putText(view, text, (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            cv2.putText(view, f"truth: {truth or '-'}   [u f b l r n]  ESC=quit", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.imshow("Posture classification (v2)", view)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break
            if key in KEY2LABEL:
                truth = KEY2LABEL[key]; print("정답 라벨 →", truth)
            elif key == ord("n"):
                truth = None; print("정답 라벨 해제")

    cap.release(); cv2.destroyAllWindows()
    if fh: fh.close()
    tracker.report()


if __name__ == "__main__":
    main()
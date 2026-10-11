"""
바자회 자세 경고 테스트 (MediaPipe Pose + 웹캠, 정면 카메라 전제)
설치:  pip install mediapipe opencv-python
실행:  python posture_test.py
키:    c = 기준 자세 다시 잡기,  q = 종료
"""
import math, os, time, urllib.request
import cv2
import mediapipe as mp
import joblib
import numpy as np
import pandas as pd
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision

# ---------------- 임의로 정한 기준값 (직접 해보면서 조정) ----------------
CALIB_SEC      = 3.0    # 처음 기준 자세 잡는 시간
HOLD_SEC       = 3.0    # 이 시간 이상 나쁜 자세가 유지되면 경고
COOLDOWN_SEC   = 10.0   # 같은 경고를 다시 하기까지 쉬는 시간
TURTLE_RATIO   = 0.80   # 목 길이가 기준의 80% 아래 → 거북목/숙임
CLOSE_RATIO    = 1.20   # 머리 크기가 기준의 120% 초과 → 너무 가까움
SHOULDER_DEG   = 6.0    # 어깨선이 기준보다 6도 이상 기울면 → 어깨 비대칭
HEAD_TILT_DEG  = 10.0   # 눈 선이 기준보다 10도 이상 기울면 → 고개 기울임
SMOOTH         = 0.3    # 값 흔들림 줄이기 (작을수록 부드러움)
# -------------------------------------------------------------------------

MODEL_URL  = ("https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
              "pose_landmarker_lite/float16/1/pose_landmarker_lite.task")
MODEL_PATH = "pose_landmarker_lite.task"

CLASSIFIER_PATH = "upperbody_posture_model.joblib"

artifact = joblib.load(CLASSIFIER_PATH)
classifier = artifact["model"]
feature_cols = artifact["feature_cols"]
upper_landmarks = artifact["upper_landmarks"]

# MediaPipe Pose 랜드마크 인덱스
LANDMARK_INDEX = {
    "nose": 0,
    "left_eye_inner": 1,
    "left_eye": 2,
    "left_eye_outer": 3,
    "right_eye_inner": 4,
    "right_eye": 5,
    "right_eye_outer": 6,
    "left_ear": 7,
    "right_ear": 8,
    "mouth_left": 9,
    "mouth_right": 10,
    "left_shoulder": 11,
    "right_shoulder": 12,
    "left_elbow": 13,
    "right_elbow": 14,
    "left_wrist": 15,
    "right_wrist": 16,
}

LABEL_TEXT = {
    "TUP": "바른 상체 자세",
    "TLF": "상체 앞으로 기울임",
    "TLB": "상체 뒤로 기울임",
    "TLR": "상체 오른쪽 기울임",
    "TLL": "상체 왼쪽 기울임",
}

NOSE, L_EYE, R_EYE, L_EAR, R_EAR, L_SH, R_SH = 0, 2, 5, 7, 8, 11, 12

WARN_TEXT = {
    "turtle":   "고개가 앞으로 나오거나 숙여졌어요. 턱을 당기고 등을 펴 주세요.",
    "close":    "화면에 너무 가까워요. 몸을 조금 뒤로 빼 주세요.",
    "shoulder": "어깨가 한쪽으로 기울었어요. 양쪽 어깨 높이를 맞춰 주세요.",
    "tilt":     "고개가 한쪽으로 기울었어요. 정면을 바라봐 주세요.",
}


def ensure_model():
    if not os.path.exists(MODEL_PATH):
        print("자세 인식 모델 다운로드 중...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

def predict_upperbody(lm):
    values = {}

    for name in upper_landmarks:
        idx = LANDMARK_INDEX[name]
        values[f"{name}_x"] = lm[idx].x
        values[f"{name}_y"] = lm[idx].y
        values[f"{name}_z"] = lm[idx].z

    # 학습할 때 사용한 특징 순서를 그대로 유지
    row = pd.DataFrame(
        [[values[col] for col in feature_cols]],
        columns=feature_cols
    )

    prediction = classifier.predict(row)[0]
    probability = classifier.predict_proba(row)[0].max()

    return prediction, probability

def measure(lm, w, h):
    p = lambda i: (lm[i].x * w, lm[i].y * h)
    dist = lambda a, b: math.hypot(a[0] - b[0], a[1] - b[1])
    angle = lambda a, b: math.degrees(math.atan2(b[1] - a[1], b[0] - a[0]))

    lsh, rsh, lear, rear = p(L_SH), p(R_SH), p(L_EAR), p(R_EAR)
    sh_w = dist(lsh, rsh) or 1.0
    return {
        "neck":     ((lsh[1] + rsh[1]) / 2 - (lear[1] + rear[1]) / 2) / sh_w,
        "head":     dist(lear, rear),
        "shoulder": angle(rsh, lsh),
        "tilt":     angle(p(R_EYE), p(L_EYE)),
    }


def judge(cur, base):
    return {
        "turtle":   cur["neck"] < base["neck"] * TURTLE_RATIO,
        "close":    cur["head"] > base["head"] * CLOSE_RATIO,
        "shoulder": abs(cur["shoulder"] - base["shoulder"]) > SHOULDER_DEG,
        "tilt":     abs(cur["tilt"] - base["tilt"]) > HEAD_TILT_DEG,
    }


def main():
    ensure_model()
    landmarker = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
        base_options=mp_tasks.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=vision.RunningMode.VIDEO, num_poses=1))

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("웹캠을 열 수 없어요. 카메라를 쓰는 다른 프로그램을 꺼 주세요.")
        return

    smooth, base, calib = None, None, []
    calib_start = t0 = time.time()
    bad_since, last_warn, last_missing = {}, {}, 0.0
    print(f"바른 자세로 {CALIB_SEC:.0f}초 동안 앉아 있어 주세요...")

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = landmarker.detect_for_video(
            mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), int((time.time() - t0) * 1000))
        now = time.time()

        if not res.pose_landmarks:
            if now - last_missing > 3:
                print("[!] 사람이 감지되지 않아요. 카메라 정면에 앉아 주세요.")
                last_missing = now
            cv2.imshow("posture test", cv2.flip(frame, 1))
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            continue

        lm = res.pose_landmarks[0]
        cur = measure(lm, w, h)
        posture_label, confidence = predict_upperbody(lm)
        smooth = cur if smooth is None else {k: smooth[k] + SMOOTH * (cur[k] - smooth[k]) for k in cur}

        for i in (NOSE, L_EYE, R_EYE, L_EAR, R_EAR, L_SH, R_SH):
            cv2.circle(frame, (int(lm[i].x * w), int(lm[i].y * h)), 5, (80, 200, 120), -1)
        view = cv2.flip(frame, 1)
        posture_text = LABEL_TEXT.get(posture_label, posture_label)
        cv2.putText(
            view,
            f"AI: {posture_text} ({confidence:.0%})",
            (20, 180),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 200, 0),
            2
        )

        if base is None:
            calib.append(cur)
            left = CALIB_SEC - (now - calib_start)
            cv2.putText(view, f"CALIBRATING... {max(left, 0):.1f}s", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 200, 255), 2)
            if left <= 0:
                base = {k: sum(c[k] for c in calib) / len(calib) for k in calib[0]}
                smooth = dict(base)
                print("기준 자세 저장 완료:", ", ".join(f"{k}={v:.2f}" for k, v in base.items()))
                print("이제 자세가 틀어지면 여기에 경고가 표시됩니다. (c: 재설정, q: 종료)\n")
        else:
            bad = judge(smooth, base)
            for k, is_bad in bad.items():
                if not is_bad:
                    bad_since.pop(k, None)
                    continue
                bad_since.setdefault(k, now)
                if now - bad_since[k] >= HOLD_SEC and now - last_warn.get(k, 0) >= COOLDOWN_SEC:
                    print(f"[{time.strftime('%H:%M:%S')}] ⚠ {WARN_TEXT[k]}")
                    last_warn[k] = now

            lines = [f"neck {smooth['neck']:.2f} / base {base['neck']:.2f}",
                     f"head {smooth['head']:.0f} / base {base['head']:.0f}",
                     f"shoulder {smooth['shoulder'] - base['shoulder']:+.1f} deg",
                     f"tilt {smooth['tilt'] - base['tilt']:+.1f} deg"]
            for j, (txt, key) in enumerate(zip(lines, ["turtle", "close", "shoulder", "tilt"])):
                color = (0, 0, 255) if bad[key] else (80, 200, 120)
                cv2.putText(view, txt, (20, 40 + j * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        cv2.imshow("posture test", view)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("c"):
            base, calib, calib_start = None, [], time.time()
            bad_since.clear(); last_warn.clear()
            print(f"\n기준 자세를 다시 잡습니다. 바른 자세로 {CALIB_SEC:.0f}초 앉아 있어 주세요...")

    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()


if __name__ == "__main__":
    main()
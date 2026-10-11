
import cv2
import mediapipe as mp
import numpy as np
import joblib

# =========================
# 설정
# =========================
MODEL_PATH = "upperbody_posture_model_final.joblib"
CAMERA_INDEX = 1

# 학습 CSV를 만들 때 사용한 좌표 기준과 동일하게 설정!
USE_WORLD_LANDMARKS = True

# 모델 불러오기
model = joblib.load(MODEL_PATH)

# 학습할 때 사용한 랜드마크 순서
LANDMARKS = [
    "nose",
    "left_eye_inner", "left_eye", "left_eye_outer",
    "right_eye_inner", "right_eye", "right_eye_outer",
    "left_ear", "right_ear",
    "mouth_left", "mouth_right",
    "left_shoulder", "right_shoulder"
]

# MediaPipe Pose 랜드마크 인덱스
LANDMARK_IDS = [
    0, 1, 2, 3, 4, 5, 6,
    7, 8, 9, 10, 11, 12
]

mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)

if not cap.isOpened():
    raise RuntimeError(
        "웹캠을 열 수 없습니다. CAMERA_INDEX를 0, 1, 2로 바꿔 보세요."
    )

def extract_features(landmarks):
    # 학습 때 사용한 13개 랜드마크의 x, y, z
    coords = np.array([
        [landmarks[i].x, landmarks[i].y, landmarks[i].z]
        for i in LANDMARK_IDS
    ], dtype=np.float64)

    # 어깨 중심점 계산
    left_shoulder = coords[11]
    right_shoulder = coords[12]
    center = (left_shoulder + right_shoulder) / 2

    # 어깨너비 계산
    width = np.linalg.norm(left_shoulder - right_shoulder)

    if not np.isfinite(coords).all() or width < 1e-6:
        return None

    # 학습 때와 동일한 정규화
    coords = (coords - center) / width

    # 모델 입력 형태: (1, 39)
    return coords.reshape(1, -1)

with mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    smooth_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as pose:

    while True:
        ret, frame = cap.read()

        if not ret:
            print("웹캠 프레임을 읽지 못했습니다.")
            break

        # 좌우 반전: 거울처럼 보이도록 표시
        frame = cv2.flip(frame, 1)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)

        text = "Posture: Detecting..."

        if results.pose_landmarks:
            # 관절 연결선 표시
            mp_draw.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS
            )

            if USE_WORLD_LANDMARKS:
                source = results.pose_world_landmarks
            else:
                source = results.pose_landmarks

            if source:
                X = extract_features(source.landmark)

                if X is not None:
                    probabilities = model.predict_proba(X)[0]
                    classes = model.classes_

                    best_idx = np.argmax(probabilities)
                    prediction = classes[best_idx]
                    confidence = probabilities[best_idx]

                    text = f"{prediction}: {confidence * 100:.1f}%"

                    # 상위 3개 예측 확률을 콘솔에 표시
                    top_indices = np.argsort(probabilities)[::-1][:3]
                    print(
                        " | ".join(
                            f"{classes[i]}: {probabilities[i] * 100:.1f}%"
                            for i in top_indices
                        ),
                        end="\r"
                    )


        cv2.rectangle(frame, (10, 10), (460, 65), (0, 0, 0), -1)
        cv2.putText(
            frame, text, (20, 47),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9, (0, 255, 0), 2
        )

        cv2.imshow("Real-time Posture Classification", frame)

        # ESC 키로 종료
        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()

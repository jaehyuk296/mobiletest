"""mp_io.py — MediaPipe Tasks(PoseLandmarker) 입출력 래퍼. 팀원 posture_test.py의 모델 로딩 방식을 그대로 사용."""
import os, time, urllib.request
import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision
from posture_core import Frame

MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
             "pose_landmarker_lite/float16/1/pose_landmarker_lite.task")
# MediaPipe Pose 표준 인덱스. 골반(23·24) 이하는 읽지도 않는다.
IDX = {"nose": 0, "eye_l": 2, "eye_r": 5, "ear_l": 7, "ear_r": 8, "sh_l": 11, "sh_r": 12}


def ensure_model(path):
    if not os.path.exists(path):
        print("자세 인식 모델 다운로드 중...", path)
        urllib.request.urlretrieve(MODEL_URL, path)


class Source:
    """웹캠 또는 영상 파일. read() -> (ok, bgr, ts_sec)"""
    def __init__(self, camera=0, video=None, width=None, height=None):
        self.is_file = video is not None
        self.cap = cv2.VideoCapture(video if self.is_file else camera)
        if not self.cap.isOpened():
            raise RuntimeError("카메라/영상을 열 수 없습니다. 다른 프로그램이 카메라를 쓰고 있지 않은지 확인하세요.")
        if width and not self.is_file:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        if height and not self.is_file:
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.t0 = time.time()

    def read(self):
        ok, frame = self.cap.read()
        if not ok:
            return False, None, 0.0
        ts = self.cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0 if self.is_file else time.time() - self.t0
        return True, frame, ts

    def release(self):
        self.cap.release()


class Detector:
    def __init__(self, model_path):
        ensure_model(model_path)
        self.lm = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
            base_options=mp_tasks.BaseOptions(model_asset_path=model_path),
            running_mode=vision.RunningMode.VIDEO, num_poses=1))
        self._last_ms = -1

    def detect(self, bgr, ts_sec):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        ms = max(int(ts_sec * 1000), self._last_ms + 1)      # 타임스탬프는 단조 증가해야 함
        self._last_ms = ms
        return self.lm.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), ms)

    def close(self):
        self.lm.close()


def to_frame(res, w, h, ts):
    """MediaPipe 결과 → posture_core.Frame (필요한 7개 랜드마크만 픽셀 좌표로 변환)."""
    if not res.pose_landmarks:
        return Frame(ts, w, h, None)
    lm = res.pose_landmarks[0]
    pts = {k: (lm[i].x * w, lm[i].y * h, float(getattr(lm[i], "visibility", 0.0) or 0.0)) for k, i in IDX.items()}
    return Frame(ts, w, h, pts)

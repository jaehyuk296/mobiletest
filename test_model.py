import joblib
import numpy as np

MODEL_PATH = "upperbody_posture_model.joblib"

artifact = joblib.load(MODEL_PATH)

model = artifact["model"]
feature_cols = artifact["feature_cols"]
upper_landmarks = artifact["upper_landmarks"]
labels = artifact["labels"]

print("모델 파일 불러오기 성공!")
print("사용 관절:", len(upper_landmarks))
print("입력 특징 수:", len(feature_cols))
print("분류 라벨:", list(model.classes_))

# 학습 모델에 임시 숫자 입력을 넣어 예측 기능 확인
sample = np.zeros((1, len(feature_cols)), dtype=float)

prediction = model.predict(sample)[0]
probabilities = model.predict_proba(sample)[0]

print("\n임시 입력 예측 결과:", prediction)
print("라벨별 예측 확률:")

for label, probability in zip(model.classes_, probabilities):
    print(f"{label}: {probability:.2%}")
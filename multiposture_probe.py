"""
multiposture_probe.py — MultiPosture(data.csv)로 학습한 RandomForest 모델을 우리 촬영 환경(측면, 머리~명치)에 대 보는 호환성 점검 도구
(명세서 v9 8.2: MultiPosture 모델은 규칙 기반을 대체하지 않는 '자세 유형 분류 실험'이며, 호환성 검증 전에는 확정 자원이 아님)

하는 일
  1) 웹캠/영상에서 MediaPipe '월드 랜드마크'(pose_world_landmarks, 힙 중심 원점·미터 단위)를 뽑아 모델 입력(51 특징)을 만든다.
  2) 모델 예측(TUP/TLF/TLB/TLR/TLL)을 화면에 표시한다.
  3) 입력 특징이 학습 데이터 분포 안에 있는지 점검한다 (모델 트리 임계값의 최소~최대 범위를 기준으로 한 대략적 검사).
  4) 처음 N초를 규칙 기반(H) 캘리브레이션으로 쓰고, 이후 '규칙 기반 상태(정상/주의/측정불가) × 모델 라벨' 교차표를 출력한다.
  5) 힙(23·24)의 visibility 를 같이 기록해 '골반이 안 보이는데 힙 기준 좌표를 쓰는' 문제의 크기를 본다.

실행:  python multiposture_probe.py --model-file ../upperbody_posture_model.joblib
"""
import argparse, collections, os, time
import numpy as np
import joblib
import cv2
from posture_core import Calibrator, PostureAnalyzer, load_config
from mp_io import Source, Detector, to_frame

HERE = os.path.dirname(os.path.abspath(__file__))


def build_feature_fn(bundle):
    """feature_cols(예: 'left_ear_x') → (world landmark 인덱스, 축). upper_landmarks 순서 = MediaPipe 인덱스 0~16."""
    names = bundle["upper_landmarks"]
    spec = []
    for col in bundle["feature_cols"]:
        name, axis = col.rsplit("_", 1)
        spec.append((names.index(name), axis))

    def feats(world):
        return np.array([[getattr(world[i], ax) for i, ax in spec]], dtype=float)
    return feats


def threshold_ranges(rf, cols):
    lo = np.full(len(cols), np.inf); hi = np.full(len(cols), -np.inf)
    for est in rf.estimators_:
        t = est.tree_
        for f, thr in zip(t.feature, t.threshold):
            if f >= 0:
                lo[f] = min(lo[f], thr); hi[f] = max(hi[f], thr)
    return lo, hi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-file", required=True)
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--video")
    ap.add_argument("--config", default=os.path.join(HERE, "config.yaml"))
    ap.add_argument("--mp-model", default=os.path.join(HERE, "pose_landmarker_lite.task"))
    ap.add_argument("--calib-sec", type=float, default=7.0, help="시작 후 이 시간 동안 기준 자세(규칙 기반 캘리브레이션)")
    ap.add_argument("--no-gui", action="store_true")
    a = ap.parse_args()

    bundle = joblib.load(a.model_file)
    rf, cols, labels = bundle["model"], bundle["feature_cols"], bundle["labels"]
    feats = build_feature_fn(bundle)
    lo, hi = threshold_ranges(rf, cols)
    cfg = load_config(a.config)
    src, det = Source(a.camera, a.video), Detector(a.mp_model)
    print("모델 클래스:", list(rf.classes_), "|", {k: v for k, v in labels.items()})
    print("※ 이 모델은 TLL/TLR(좌우 기울임)도 예측합니다. 측면 카메라로는 좌우 기울임을 볼 수 없습니다(명세서 제외 항목).")

    calib = Calibrator(cfg); analyzer = None; t0 = None
    cross = collections.Counter(); pred_cnt = collections.Counter()
    n = n_world = n_shift = 0; hip_vis = []; out_counts = []

    while True:
        ok, bgr, ts = src.read()
        if not ok: break
        h, w = bgr.shape[:2]
        res = det.detect(bgr, ts); frame = to_frame(res, w, h, ts); n += 1
        t0 = ts if t0 is None else t0

        # 규칙 기반 상태 (캘리브레이션 후)
        state = "calibrating"
        if analyzer is None:
            calib.add(frame)
            if ts - t0 >= a.calib_sec:
                cal, info = calib.finish()
                if cal is None:
                    print("캘리브레이션 실패:", info.get("error")); break
                analyzer = PostureAnalyzer(cfg, cal); print("규칙 기반 캘리브레이션 완료:", cal.side, cal.d)
        else:
            state = analyzer.update(frame).state

        label, prob, nout = "-", 0.0, 0
        if res.pose_world_landmarks and res.pose_landmarks:
            n_world += 1
            x = feats(res.pose_world_landmarks[0])
            nout = int(((x[0] < lo - 1e-9) | (x[0] > hi + 1e-9)).sum())          # 학습 분포 범위 밖 특징 수(대략)
            out_counts.append(nout); n_shift += nout > len(cols) * 0.3
            pr = rf.predict_proba(x)[0]; label = rf.classes_[int(np.argmax(pr))]; prob = float(pr.max())
            lm = res.pose_landmarks[0]; hip_vis.append((lm[23].visibility + lm[24].visibility) / 2)
            if analyzer is not None:
                cross[(state, label)] += 1
            pred_cnt[label] += 1

        if not a.no_gui:
            v = bgr.copy()
            cv2.putText(v, f"model: {label} ({prob:.2f})  out-of-range feats {nout}/{len(cols)}", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
            cv2.putText(v, f"rule state: {state}", (12, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (80, 200, 120), 2)
            if hip_vis: cv2.putText(v, f"hip visibility {hip_vis[-1]:.2f}", (12, 84), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 2)
            cv2.imshow("multiposture probe", v)
            if cv2.waitKey(1) & 0xFF == ord("q"): break

    print("\n================ 점검 결과 ================")
    print(f"프레임 {n}, 월드 랜드마크 있음 {n_world}")
    if n_world:
        print(f"모델 예측 분포: {dict(pred_cnt)}")
        print(f"학습 분포 범위 밖 특징 수 평균 {np.mean(out_counts):.1f}/{len(cols)}  | 30% 이상 벗어난 프레임 {n_shift / n_world:.1%}")
        print(f"힙(23·24) 평균 visibility {np.mean(hip_vis):.2f}  (낮으면: 힙 중심 원점 좌표가 추정값 → 학습 데이터와 분포가 다를 가능성)")
        states = sorted({s for s, _ in cross}); labs = sorted({l for _, l in cross})
        if states:
            print("\n교차표 (행: 규칙 기반 상태, 열: 모델 라벨)")
            print("%-12s" % "" + "".join("%8s" % l for l in labs))
            for s in states:
                print("%-12s" % s + "".join("%8d" % cross[(s, l)] for l in labs))
    print("\n판단 가이드: 범위 밖 특징이 많거나 힙 visibility 가 낮으면 이 모델을 우리 환경에 그대로 쓰기 어렵습니다(명세서 8.2).")
    print("             이 결과는 호환성 점검용이며, 정확도 평가가 아닙니다. 정확도는 라벨이 있는 우리 데이터로 별도 평가해야 합니다.")
    src.release(); det.close(); cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

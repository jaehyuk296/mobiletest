"""
h_monitor.py — 캘리브레이션 + 기준 대비 ΔH 실시간 확인 (명세 P1-3의 4~6단계 프로토타입)

- 기준 자세는 영상이 아니라 숫자(H_base, x_shoulder_base, L_ref_fixed, side, d)로만 저장한다.
- 모니터링 중 side/d/L_ref_fixed/H_base는 바뀌지 않는다. 자동 갱신 없음(다시 하려면 c 키).
- 고정된 side 의 귀·어깨만 쓴다. 반대쪽으로 대체하지 않는다.
- 판정은 부호 있는 비교(ΔH > T_forward). 절댓값 비교 금지. S는 이 파일에서 계산하지 않는다.
- 임계값은 전부 임시 placeholder(TODO(실험)). 확정값이 아니다.
- 화면 글자는 OpenCV가 한글을 못 그려서 영어로 표시한다.
- ※ 구조를 검토해 작성한 코드이며, 실제 카메라·MediaPipe 버전에서 실행 확인이 필요하다.

키: c=기준 자세 수집 시작 | y=미리보기 확인 후 저장 | r=다시 수집 | ESC=종료
"""
import argparse
import collections
import csv
import json
import math
import os
import statistics
import time
from datetime import datetime

import cv2
import mediapipe as mp

NOSE = 0
SIDES = {
    "left":  {"ear": 7, "shoulder": 11, "eye": 2},
    "right": {"ear": 8, "shoulder": 12, "eye": 5},
}
CALIB_FILE = "calibration.json"


def open_cam(idx):
    return cv2.VideoCapture(idx, cv2.CAP_DSHOW) if os.name == "nt" else cv2.VideoCapture(idx)


def read_side(lm, ids, w, h):
    ear, sh, eye = lm[ids["ear"]], lm[ids["shoulder"]], lm[ids["eye"]]
    return {"xe": ear.x * w, "ye": ear.y * h, "xs": sh.x * w, "ys": sh.y * h,
            "ex": eye.x * w, "ey": eye.y * h, "xn": lm[NOSE].x * w,
            "ve": ear.visibility, "vs": sh.visibility, "vey": eye.visibility}


def ref_len(mode, p, h):
    if mode == "C":
        return abs(p["ys"] - p["ye"])
    if mode == "A":
        return math.hypot(p["xe"] - p["ex"], p["ye"] - p["ey"])
    return float(h)  # B: 이미지 높이


def inside(p, w, h, mx, my):
    return all(mx <= x <= w - mx and my <= y <= h - my for x, y in ((p["xe"], p["ye"]), (p["xs"], p["ys"])))


def finish_calibration(samples, a, h):
    """수집한 샘플로 기준값 계산. 실패하면 (None, 사유)."""
    mean_vis = {s: (statistics.mean((p["ve"] + p["vs"]) / 2 for p in lst) if lst else 0.0)
                for s, lst in samples.items()}
    side = max(mean_vis, key=mean_vis.get)  # 양쪽 visibility 평균이 높은 쪽
    n_all = len(samples[side])
    good = [p for p in samples[side] if p["ok"]]
    if n_all < 10 or len(good) < 0.5 * n_all:   # 0.5 는 임시 기준 (TODO(실험))
        return None, f"stable frames too few ({len(good)}/{n_all}). retry"
    L = statistics.median(ref_len(a.l_ref_mode, p, h) for p in good)
    if L < 1e-6:
        return None, "L_ref too small. retry"
    # d: 코가 귀보다 오른쪽이면 오른쪽을 향함(+1). 판별법은 0단계에서 검증 필요(가정)
    d = 1 if statistics.median(p["xn"] - p["xe"] for p in good) > 0 else -1
    H = [d * (p["xe"] - p["xs"]) / L for p in good]
    q = statistics.quantiles(H, n=4) if len(H) >= 2 else [0, 0, 0]
    return {
        "H_base": statistics.median(H),
        "x_shoulder_base": statistics.median(p["xs"] for p in good),
        "L_ref_fixed": L, "side": side, "d": d,
        "l_ref_mode": a.l_ref_mode,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "_spread_iqr": q[2] - q[0], "_n_good": len(good), "_n_all": n_all,
        "_mean_vis": {k: round(v, 3) for k, v in mean_vis.items()},
    }, ""


def check_reason(lm, cal, w, h, mx, my, thr):
    """측정 불가 사유. 없으면 (None, 좌표)."""
    if lm is None:
        return "ear_not_detected", None   # 사람/랜드마크가 검출되지 않음
    p = read_side(lm, SIDES[cal["side"]], w, h)  # 고정된 side 만 사용
    if p["ve"] < thr or p["vs"] < thr:
        return "low_confidence", p
    if not inside(p, w, h, mx, my):
        return "out_of_frame", p
    if p["ye"] > p["ys"]:                         # 귀가 어깨보다 아래: 기하학적으로 비정상
        return "invalid_geometry", p
    return None, p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, default=1)
    ap.add_argument("--load", action="store_true", help=f"{CALIB_FILE} 이 있으면 불러와서 바로 모니터링")
    ap.add_argument("--subject", default="")
    ap.add_argument("--note", default="")
    ap.add_argument("--csv", help="(개발용) 프레임별 H·ΔH·상태 로그 경로. 좌표는 저장하지 않음")
    ap.add_argument("--l-ref-mode", default="C", choices=["A", "B", "C"])
    # ↓ 전부 임시 placeholder — 실험으로 확정 전까지 확정값으로 쓰지 말 것
    ap.add_argument("--vis-thr", type=float, default=0.5)         # T_vis        TODO(실험)
    ap.add_argument("--margin", type=float, default=0.05)         # frame_margin TODO(실험)
    ap.add_argument("--calib-sec", type=float, default=7.0)       # calib_duration_sec (제안 5~10초)
    ap.add_argument("--t-forward", type=float, default=0.10)      # T_forward    TODO(실험)
    ap.add_argument("--t-hold", type=float, default=5.0)          # T_hold_sec   TODO(실험)
    ap.add_argument("--t-reset", type=float, default=3.0)         # T_reset_sec  TODO(실험)
    ap.add_argument("--k-recover", type=int, default=5)           # K_recover    TODO(실험)
    ap.add_argument("--n-low-vis", type=int, default=5)           # N_low_vis    TODO(실험)
    ap.add_argument("--smooth", type=int, default=5)              # 이동 중앙값 창(가정)
    a = ap.parse_args()

    cap = open_cam(a.camera)
    if not cap.isOpened():
        raise RuntimeError(f"카메라 {a.camera}번을 열 수 없습니다.")

    cal, mode = None, "idle"   # idle / collecting / preview / monitor
    if a.load and os.path.exists(CALIB_FILE):
        with open(CALIB_FILE, encoding="utf-8") as f:
            cal = json.load(f)
        mode = "monitor"
        print("기준값 불러옴:", cal)

    fh = writer = None
    if a.csv:
        fh = open(a.csv, "w", newline="", encoding="utf-8")
        writer = csv.writer(fh)
        writer.writerow(["ts", "subject", "note", "side", "d", "H", "dH", "state", "reason", "fwd_sec"])

    samples, t_collect, pending = None, 0.0, None
    hist = collections.deque(maxlen=max(a.smooth, 1))
    state, reason, fwd_t, unavail_t, stable, low_run = "normal", None, 0.0, 0.0, 0, 0
    dH = H = 0.0
    msg, last_t, t0 = "press c to capture baseline posture", time.time(), time.time()

    try:
        with mp.solutions.pose.Pose(static_image_mode=False, model_complexity=1, smooth_landmarks=True,
                                    min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            while True:
                ok, frame = cap.read()
                if not ok:
                    print("프레임을 읽지 못했습니다."); break
                h, w = frame.shape[:2]
                now = time.time(); dt = now - last_t; last_t = now
                res = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))   # 반전하지 않은 프레임
                lm = res.pose_landmarks.landmark if res.pose_landmarks else None
                mx, my = a.margin * w, a.margin * h
                lines = []

                if mode == "collecting":
                    if lm is not None:
                        for s, ids in SIDES.items():
                            p = read_side(lm, ids, w, h)
                            p["ok"] = (p["ve"] >= a.vis_thr and p["vs"] >= a.vis_thr and inside(p, w, h, mx, my)
                                       and (a.l_ref_mode != "A" or p["vey"] >= a.vis_thr))
                            samples[s].append(p)
                    left = a.calib_sec - (now - t_collect)
                    lines = [f"CAPTURING baseline... hold your comfortable posture ({max(left, 0):.1f}s)"]
                    if left <= 0:
                        pending, why = finish_calibration(samples, a, h)
                        if pending is None:
                            mode, msg = "idle", "FAILED: " + why + "  (press c)"
                        else:
                            mode = "preview"
                            print("미리보기:", pending)
                elif mode == "preview":
                    lines = [f"PREVIEW side={pending['side']} d={pending['d']:+d}",
                             f"H_base={pending['H_base']:.3f} L_ref={pending['L_ref_fixed']:.1f}px "
                             f"spread(IQR)={pending['_spread_iqr']:.3f} frames={pending['_n_good']}/{pending['_n_all']}",
                             "y=save as baseline   r=retry   (this is a personal baseline, not a 'correct' posture)"]
                elif mode == "idle":
                    lines = [msg, "state: UNAVAILABLE (not_calibrated)"]
                else:  # monitor
                    raw, p = check_reason(lm, cal, w, h, mx, my, a.vis_thr)
                    if raw == "low_confidence":          # N_low_vis 프레임 연속일 때만 측정 불가
                        low_run += 1
                        raw = raw if low_run >= a.n_low_vis else "pending"
                    else:
                        low_run = 0
                    if raw == "pending":
                        pass                              # 아직 판단 보류: 이전 상태 유지
                    elif raw:                             # 측정 불가: 정상/주의 판단 안 함
                        state, reason = "unavailable", raw
                        unavail_t += dt; stable = 0
                        if unavail_t > a.t_reset:
                            fwd_t = 0.0                   # T_reset 초과: 주의 타이머 초기화 (그 전엔 일시정지)
                    else:
                        stable += 1
                        if state == "unavailable" and stable < a.k_recover:
                            pass                          # K_recover 연속 안정 전에는 판정 재개 안 함
                        else:
                            unavail_t, reason = 0.0, None
                            H = cal["d"] * (p["xe"] - p["xs"]) / cal["L_ref_fixed"]   # L_ref_fixed 재계산 없음
                            hist.append(H - cal["H_base"])
                            dH = statistics.median(hist)
                            if dH > a.t_forward:          # 부호 있는 비교 (절댓값 금지)
                                fwd_t += dt
                            else:
                                fwd_t = 0.0
                            state = "attention" if fwd_t >= a.t_hold else "normal"
                    col = {"normal": (0, 255, 0), "attention": (0, 0, 255), "unavailable": (0, 165, 255)}[state]
                    lines = [f"STATE: {state.upper()}" + (f" ({reason})" if reason else ""),
                             f"dH={dH:+.3f} (T_forward={a.t_forward})  forward_for={fwd_t:.1f}s / {a.t_hold}s",
                             f"side={cal['side']} d={cal['d']:+d} H_base={cal['H_base']:.3f}"]
                    if p:
                        c2 = col
                        cv2.circle(frame, (int(p["xe"]), int(p["ye"])), 6, c2, -1)
                        cv2.circle(frame, (int(p["xs"]), int(p["ys"])), 6, c2, -1)
                        cv2.line(frame, (int(p["xe"]), int(p["ye"])), (int(p["xs"]), int(p["ys"])), c2, 1)
                    if writer:
                        writer.writerow([f"{now - t0:.3f}", a.subject, a.note, cal["side"], cal["d"],
                                         f"{H:.4f}", f"{dH:.4f}", state, reason or "", f"{fwd_t:.2f}"])

                for i, t in enumerate(lines):
                    cv2.putText(frame, t, (10, 28 + 26 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                (255, 255, 255) if mode != "monitor" else col, 2)
                cv2.putText(frame, "c:capture  y:save  r:retry  ESC:quit", (10, h - 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                cv2.imshow("h_monitor", frame)

                key = cv2.waitKey(1) & 0xFF
                if key == 27:
                    break
                if key in (ord("c"), ord("r")) and mode in ("idle", "monitor", "preview"):
                    samples = {"left": [], "right": []}
                    t_collect, mode = time.time(), "collecting"
                elif key == ord("y") and mode == "preview":
                    cal = {k: v for k, v in pending.items() if not k.startswith("_")}
                    with open(CALIB_FILE, "w", encoding="utf-8") as f:
                        json.dump(cal, f, ensure_ascii=False, indent=2)
                    hist.clear()
                    state, reason, fwd_t, unavail_t, stable, low_run = "normal", None, 0.0, 0.0, 0, 0
                    mode = "monitor"
                    print("저장:", CALIB_FILE, cal)
    finally:
        cap.release(); cv2.destroyAllWindows()
        if fh:
            fh.close()


if __name__ == "__main__":
    main()
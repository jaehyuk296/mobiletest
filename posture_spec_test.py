"""
posture_spec_test.py — 명세서 v9 기준 측면 상반신 자세 테스트 (0단계 실험 겸용)

팀원 posture_test.py(정면 카메라 전제)를 아래처럼 바꿨습니다.
  - 측면 상반신(머리~명치): 귀·어깨만 사용, 고정된 쪽(side) 랜드마크만 사용
  - 판정 지표 H 하나 (S는 로그 전용), 부호 있는 비교(dH > T_forward)
  - 캘리브레이션: 중앙값, side/d/L_ref 고정, 사용자 확인(y) 후 저장
  - 측정 불가 처리 (가림·저신뢰·화면 이탈·비정상 배치) + 타이머 일시정지/초기화/K프레임 복귀
  - 로그 CSV(--log) 와 종료 시 요약(FPS·감지율·측정 불가 비율)

실행 예:
  python posture_spec_test.py --log                  # 웹캠, 로그 저장
  python posture_spec_test.py --video test.mp4 --auto-calib --no-gui --log
키:  c 캘리브레이션 시작 | y 저장 / n 다시 | 1~6 조건 라벨 | d 디버그(S 표시) | p 일시정지 | q 종료
"""
import argparse, csv, json, os, sys, time, collections, urllib.request
import cv2
import numpy as np
from posture_core import Calibrator, PostureAnalyzer, Calibration, load_config
from mp_io import Source, Detector, to_frame

HERE = os.path.dirname(os.path.abspath(__file__))

# 안내 문구 (명세서 v9 17.2: 진단·치료 표현 없음, H 기준)
TEXT = {
    "attention_started": "머리가 기준보다 앞쪽에 있는 상태가 이어지고 있어요. 턱을 살짝 당겨 볼까요?",
    "recovered": "좋아요. 기준 자세에 가까워졌어요.",
    "unavailable_prolonged": "지금은 자세를 확인하기 어려워요. 카메라에 머리와 어깨가 보이는지 확인해 주세요.",
    "recalibration_suggested": "기준 자세를 다시 설정해 볼까요? (자동으로 바뀌지 않아요. 'c' 키로 직접 시작하세요)",
}
REASON_EN = {  # OpenCV 기본 폰트는 한글을 못 그리므로 화면에는 영문 사용
    "low_confidence": "LOW CONFIDENCE", "occluded": "OCCLUDED", "out_of_frame": "OUT OF FRAME",
    "ear_not_detected": "EAR NOT DETECTED", "shoulder_not_detected": "SHOULDER NOT DETECTED",
    "invalid_geometry": "INVALID GEOMETRY", "not_calibrated": "NOT CALIBRATED"}
CSV_COLS = ["ts", "frame", "side", "d", "x_ear", "y_ear", "x_shoulder", "y_shoulder", "vis_ear", "vis_shoulder",
            "L_ref_A", "L_ref_B", "L_ref_C", "H", "dH", "H_A", "H_B", "H_C", "S", "state", "reason", "condition", "proc_ms"]


def speak(url, text, kind="alert"):
    """(선택) P3 /speak 연동 테스트: POST {text, kind}"""
    try:
        req = urllib.request.Request(url, data=json.dumps({"text": text, "kind": kind}).encode("utf-8"),
                                     headers={"Content-Type": "application/json"})
        urllib.request.urlopen(req, timeout=2).read()
    except Exception as e:
        print("  (speak 실패:", type(e).__name__, ")")


def put(img, text, y, color=(80, 200, 120), scale=0.6):
    cv2.putText(img, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--video", help="영상 파일(녹화 영상으로 재현 테스트)")
    ap.add_argument("--config", default=os.path.join(HERE, "config.yaml"))
    ap.add_argument("--model", default=os.path.join(HERE, "pose_landmarker_lite.task"))
    ap.add_argument("--log", action="store_true", help="프레임별 H/S 로그 CSV 저장(0단계 실험용, 실험 후 삭제!)")
    ap.add_argument("--log-dir", default=os.path.join(HERE, "data", "posture"))
    ap.add_argument("--calib-file", default=os.path.join(HERE, "calibration.json"))
    ap.add_argument("--load-calib", action="store_true", help="저장된 캘리브레이션 재사용")
    ap.add_argument("--auto-calib", action="store_true", help="시작 직후 자동 캘리브레이션+자동 확인(영상 파일/헤드리스용)")
    ap.add_argument("--no-gui", action="store_true")
    ap.add_argument("--condition", default="neutral", help="초기 조건 라벨")
    ap.add_argument("--tag", default="", help="로그 파일명에 붙일 꼬리표(예: 모델·카메라 배치 비교용 lite / full)")
    ap.add_argument("--speak-url", help="예: http://127.0.0.1:8081/speak (P3 TTS 연동 시험)")
    ap.add_argument("--width", type=int); ap.add_argument("--height", type=int)
    a = ap.parse_args()

    cfg = load_config(a.config)
    src, det = Source(a.camera, a.video, a.width, a.height), Detector(a.model)
    conds = cfg["conditions"]; cond = a.condition

    writer = fh = log_path = None
    if a.log:
        os.makedirs(a.log_dir, exist_ok=True)
        path = os.path.join(a.log_dir, time.strftime("log_%Y%m%d_%H%M%S") + (f"_{a.tag}" if a.tag else "") + ".csv")
        log_path = path
        fh = open(path, "w", newline="", encoding="utf-8"); writer = csv.writer(fh); writer.writerow(CSV_COLS)
        print("로그 저장:", path, "  ※ 개발용 실험 로그입니다. 실험이 끝나면 삭제하세요(프레임별 좌표 원본 저장 금지 원칙).")

    mode = "IDLE"                      # IDLE → CALIBRATING → CONFIRM → MONITORING
    calibrator, pending, analyzer, cal = None, None, None, None
    calib_start = 0.0
    if a.load_calib and os.path.exists(a.calib_file):
        d = json.load(open(a.calib_file, encoding="utf-8"))
        cal = Calibration(**d); analyzer = PostureAnalyzer(cfg, cal); mode = "MONITORING"
        print("저장된 캘리브레이션 로드:", cal.side, cal.d, cal.l_ref_mode)
        if log_path: json.dump(d, open(log_path + ".calib.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=str)
    if a.auto_calib and mode == "IDLE":
        mode, calibrator, calib_start = "CALIBRATING", Calibrator(cfg), None

    stats = collections.Counter(); state_time = collections.Counter(); reason_cnt = collections.Counter()
    n_frames = n_pose = 0; proc = []; last_ts = None; t_run0 = time.time(); paused = False; debug = False
    print("키: c=캘리브레이션 y/n=저장·재측정 1~6=조건 d=디버그 p=일시정지 q=종료" if not a.no_gui else "(no-gui)")
    if mode == "IDLE":
        print("'c' 키를 누르고 편안한 기준 자세로 앉아 있으세요.")

    while True:
        ok, bgr, ts = src.read()
        if not ok:
            break
        h, w = bgr.shape[:2]
        t1 = time.time()
        res = det.detect(bgr, ts)
        frame = to_frame(res, w, h, ts)
        pm = (time.time() - t1) * 1000; proc.append(pm)
        n_frames += 1; n_pose += 1 if frame.pts else 0
        r = None

        if mode == "CALIBRATING":
            if calib_start is None: calib_start = ts
            calibrator.add(frame)
            left = cfg["calib_duration_sec"] - (ts - calib_start)
            if left <= 0:
                pending, info = calibrator.finish()
                if pending is None:
                    print("캘리브레이션 실패:", info.get("error"), "→ 'c'로 다시 시도"); mode = "IDLE"
                else:
                    print("\n[기준값 요약] side=%s d=%+d  L_ref(A/B/C)=%s\n  H_base=%s  H변동(std)=%.3f%s" % (
                        pending.side, pending.d, {k: round(v, 1) for k, v in pending.L_ref.items()},
                        {k: round(v, 3) for k, v in pending.H_base.items()}, info["H_std"],
                        "  ⚠ 불안정(재측정 권장)" if info.get("unstable") else ""))
                    print("  d 판별 일치율=%.2f, 양쪽 visibility=%s" % (info["d_vote_ratio"], {k: round(v, 2) for k, v in info["mean_vis"].items()}))
                    mode = "CONFIRM"
                    if not a.auto_calib:
                        print("이 자세를 기준으로 저장할까요? (y/n)")
        elif mode == "MONITORING" and not paused:
            r = analyzer.update(frame)
            for ev in r.events:
                msg = TEXT.get(ev["type"], ev["type"])
                print(f"[{time.strftime('%H:%M:%S')}] {ev['type']}: {msg}")
                stats[ev["type"]] += 1
                if a.speak_url: speak(a.speak_url, msg)
            if last_ts is not None: state_time[r.state] += max(0, ts - last_ts)
            if r.reason: reason_cnt[r.reason] += 1
            stats["frames_" + r.state] += 1
        last_ts = ts

        if writer and r is not None:
            Lr = cal.L_ref if cal else {}
            writer.writerow([f"{ts:.3f}", n_frames, r.side, r.d,
                             *( [f"{r.ear[0]:.1f}", f"{r.ear[1]:.1f}", f"{r.shoulder[0]:.1f}", f"{r.shoulder[1]:.1f}",
                                 f"{r.ear[2]:.2f}", f"{r.shoulder[2]:.2f}"] if r.ear and r.shoulder else [""] * 6),
                             *(f"{Lr.get(k, ''):.2f}" if k in Lr else "" for k in "ABC"),
                             "" if r.H is None else f"{r.H:.4f}", "" if r.dH is None else f"{r.dH:.4f}",
                             *(f"{r.H_all[k]:.4f}" if k in r.H_all else "" for k in "ABC"),
                             "" if r.S is None else f"{r.S:.4f}", r.state, r.reason or "", cond, f"{pm:.1f}"])

        # ---------------- 화면 ----------------
        key = 255
        if not a.no_gui:
            view = bgr.copy()
            mx, my = int(w * cfg["frame_margin"]), int(h * cfg["frame_margin"])
            cv2.rectangle(view, (mx, my), (w - mx, h - my), (90, 90, 90), 1)           # 프레임 마진
            side = (cal or pending).side if (cal or pending) else None
            if frame.pts:
                for k, p in frame.pts.items():
                    if side and not (k == "nose" or k.endswith("_l" if side == "left" else "_r")):
                        continue                                                        # 고정된 쪽만 표시
                    cv2.circle(view, (int(p[0]), int(p[1])), 5, (80, 200, 120) if p[2] >= cfg["T_vis"] else (0, 0, 255), -1)
            put(view, f"mode {mode}  cond [{cond}]  proc {pm:.0f}ms", 24, (0, 200, 255))
            if mode == "CALIBRATING":
                put(view, f"CALIBRATING {max(0, left):.1f}s  (hold comfortable posture)", 52, (0, 200, 255))
            elif mode == "CONFIRM":
                put(view, "Save this baseline?  y = save   n = retry", 52, (0, 200, 255))
            elif mode == "MONITORING" and r is not None:
                col = {"normal": (80, 200, 120), "attention": (0, 0, 255), "unavailable": (0, 165, 255)}[r.state]
                put(view, f"STATE {r.state.upper()}" + (f" ({REASON_EN.get(r.reason, r.reason)})" if r.reason else ""), 52, col, 0.8)
                if r.H is not None:
                    put(view, f"H {r.H:+.3f}  dH {r.dH:+.3f}  (T_forward {cfg['T_forward']})  hold {r.duration_sec:.1f}/{cfg['T_hold_sec']}s", 80, col)
                if debug and r.S is not None:
                    put(view, f"[DEBUG] S {r.S:+.3f}  (validation only, NOT used)", 108, (255, 200, 0))
            cv2.imshow("posture spec test (side view)", view)
            key = cv2.waitKey(1) & 0xFF
        if mode == "CONFIRM" and a.auto_calib:       # 자동 모드: 요약 출력 후 바로 저장 확인
            key = ord("y")

        if key == ord("q"): break
        elif key == ord("c"):
            mode, calibrator, calib_start, analyzer, cal = "CALIBRATING", Calibrator(cfg), None, None, None
            print("\n기준 자세 측정 시작 — 편안한 자세로 %.0f초 앉아 있으세요." % cfg["calib_duration_sec"])
        elif key == ord("y") and mode == "CONFIRM":
            cal = pending; analyzer = PostureAnalyzer(cfg, cal); mode = "MONITORING"
            cal_dict = {k: getattr(cal, k) for k in ("side", "d", "l_ref_mode", "L_ref", "H_base", "x_shoulder_base", "info")}
            json.dump(cal_dict, open(a.calib_file, "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=str)
            if log_path:    # 이 로그를 만든 캘리브레이션을 로그 옆에 함께 보관(evaluate_sweep.py 가 사용)
                json.dump(cal_dict, open(log_path + ".calib.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=str)
            print("기준값 저장 완료(side·d·L_ref 는 재캘리브레이션 전까지 고정). 모니터링 시작.\n")
        elif key == ord("n") and mode == "CONFIRM":
            mode = "IDLE"; print("다시 측정하려면 'c'")
        elif key == ord("d"): debug = not debug
        elif key == ord("p"): paused = not paused; print("일시정지" if paused else "재개")
        elif ord("1") <= key <= ord("9") and key - ord("1") < len(conds):
            cond = conds[key - ord("1")]; print("조건 라벨 →", cond)

    # ---------------- 요약 ----------------
    dur = time.time() - t_run0
    print("\n================ 요약 ================")
    print(f"프레임 {n_frames}  | 평균 FPS {n_frames / max(dur, 1e-6):.1f}  | 처리 지연 평균 {np.mean(proc):.1f}ms (p95 {np.percentile(proc, 95):.1f}ms)")
    print(f"랜드마크 감지율 {n_pose / max(n_frames, 1):.1%}")
    tot = sum(state_time.values())
    if tot > 0:
        print("상태 시간(s): " + ", ".join(f"{k} {v:.1f}" for k, v in state_time.items()) +
              f"  | 측정 불가 비율 {state_time['unavailable'] / tot:.1%}")
        print("측정 불가 사유 프레임 수:", dict(reason_cnt))
    print("이벤트:", {k: v for k, v in stats.items() if not k.startswith("frames_")})
    src.release(); det.close(); cv2.destroyAllWindows()
    if fh: fh.close()


if __name__ == "__main__":
    main()

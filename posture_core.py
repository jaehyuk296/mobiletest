"""
posture_core.py — 명세서 v9 기준 자세 분석 핵심 로직 (MediaPipe에 의존하지 않음)

규칙 (명세서 v9 / 공통 계약 C2):
  - 판정 지표는 H 하나. S(어깨 화면 이동량)는 로그 전용 (판정·경고에 쓰지 않음).
  - H 판정은 부호 있는 비교: dH > T_forward 가 지속될 때만 주의 (절댓값 비교 금지).
  - side / d / L_ref 는 캘리브레이션 때만 정하고 모니터링 중 바꾸지 않는다.
  - 고정된 쪽 랜드마크가 나빠져도 반대쪽으로 대체하지 않고 '측정 불가'.
  - 골반·하체 좌표는 사용하지 않는다 (입력에 아예 받지 않음).
  - 모든 임계값은 config.yaml 에서 읽는다 (코드에 숫자를 박지 않음).
"""
from __future__ import annotations

import collections
import math
import statistics
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

Pt = Tuple[float, float, float]  # (x_px, y_px, visibility)

# 사용하는 랜드마크 키 (MediaPipe: nose=0, eye_l=2, eye_r=5, ear_l=7, ear_r=8, sh_l=11, sh_r=12)
SIDE_KEYS = {
    "left": {"ear": "ear_l", "sh": "sh_l", "eye": "eye_l"},
    "right": {"ear": "ear_r", "sh": "sh_r", "eye": "eye_r"},
}
LREF_MODES = ("A", "B", "C")


# ------------------------------------------------------------------ 설정
DEFAULTS = {
    "T_vis": 0.5, "N_low_vis": 5, "frame_margin": 0.03,
    "occlusion_flip_count": 4, "occlusion_window_sec": 3.0,
    "T_forward": 0.10, "T_hold_sec": 5.0, "T_reset_sec": 10.0, "K_recover": 5,
    "hysteresis_ratio": 0.8, "alert_interval_sec": 30.0, "smooth_window": 5,
    "unavailable_prolonged_sec": 10.0, "recalib_suggest_count": 3, "recalib_window_sec": 300.0,
    "calib_duration_sec": 7.0, "calib_min_frames": 30, "calib_max_h_std": 0.05,
    "l_ref_mode": "C", "l_ref_B_frac": 0.25,
    "conditions": ["neutral", "lean_fwd", "head_fwd", "recline", "chair_moved", "camera_moved"],
}


def load_config(path: Optional[str] = None) -> dict:
    cfg = dict(DEFAULTS)
    if path:
        import yaml  # pip install pyyaml
        with open(path, "r", encoding="utf-8") as f:
            cfg.update(yaml.safe_load(f) or {})
    return cfg


# ------------------------------------------------------------------ 데이터
@dataclass
class Frame:
    ts: float                         # 초
    width: int
    height: int
    pts: Optional[Dict[str, Pt]]      # 사람이 검출되지 않으면 None


@dataclass
class Calibration:
    side: str                         # "left" | "right"  (고정)
    d: int                            # +1: 사용자가 화면 오른쪽을 향함, -1: 왼쪽  (고정)
    l_ref_mode: str                   # 판정에 쓰는 L_ref 후보
    L_ref: Dict[str, float]           # 후보별 고정 기준 길이 (A/B/C)  (고정)
    H_base: Dict[str, float]          # 후보별 기준 H
    x_shoulder_base: float            # S 로그용 (판정에 사용 안 함)
    info: dict = field(default_factory=dict)


@dataclass
class Result:
    ts: float
    state: str                        # normal | attention | unavailable   (API 노출용)
    internal_state: str               # observing 포함
    reason: Optional[str]
    side: Optional[str]
    d: Optional[int]
    ear: Optional[Pt] = None
    shoulder: Optional[Pt] = None
    H: Optional[float] = None         # 판정에 쓰는 H (이동 중앙값 적용)
    dH: Optional[float] = None
    S: Optional[float] = None         # 검증 지표 (로그 전용)
    H_all: Dict[str, float] = field(default_factory=dict)    # 후보별 원시 H (로그)
    duration_sec: float = 0.0         # 주의 누적 시간
    events: List[dict] = field(default_factory=list)


# ------------------------------------------------------------------ 유틸
def _dist(a: Pt, b: Pt) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def l_ref_candidates(ear: Pt, sh: Pt, eye: Pt, height: int, b_frac: float) -> Dict[str, float]:
    """기준 길이 후보 (캘리브레이션에서 1회 측정해 고정하는 용도)."""
    return {
        "A": _dist(ear, eye),             # 귀-눈 거리 (머리 크기)
        "B": height * b_frac,             # 이미지 높이 기반 상수
        "C": abs(sh[1] - ear[1]),         # 귀-어깨 수직 거리 (기본안)
    }


def in_frame(p: Pt, w: int, h: int, margin: float) -> bool:
    mx, my = w * margin, h * margin
    return mx <= p[0] <= w - mx and my <= p[1] <= h - my


def _get_side(pts: Dict[str, Pt], side: str):
    k = SIDE_KEYS[side]
    return pts.get(k["ear"]), pts.get(k["sh"]), pts.get(k["eye"])


# ------------------------------------------------------------------ 캘리브레이션
class Calibrator:
    """기준 자세 프레임을 모아 side / d / L_ref(고정) / H_base 를 계산한다."""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.frames: List[Frame] = []

    def add(self, frame: Frame) -> None:
        self.frames.append(frame)

    def finish(self) -> Tuple[Optional[Calibration], dict]:
        cfg = self.cfg
        info: dict = {"n_frames": len(self.frames)}
        detected = [f for f in self.frames if f.pts]
        info["n_detected"] = len(detected)
        if not detected:
            info["error"] = "사람이 검출된 프레임이 없습니다."
            return None, info

        # 1) side 선택: 귀+어깨 visibility 평균이 높은 쪽 (캘리브레이션에서만 결정)
        mean_vis = {}
        for side in SIDE_KEYS:
            vals = []
            for f in detected:
                ear, sh, _ = _get_side(f.pts, side)
                if ear and sh:
                    vals.append((ear[2] + sh[2]) / 2)
            mean_vis[side] = statistics.fmean(vals) if vals else 0.0
        side = max(mean_vis, key=mean_vis.get)
        info["mean_vis"] = mean_vis
        info["side"] = side

        # 2) 안정 프레임: 신뢰도·화면 내·기하 조건 통과
        good = []
        for f in detected:
            ear, sh, eye = _get_side(f.pts, side)
            nose = f.pts.get("nose")
            if not (ear and sh and eye and nose):
                continue
            if min(ear[2], sh[2]) < cfg["T_vis"]:
                continue
            if not (in_frame(ear, f.width, f.height, cfg["frame_margin"]) and
                    in_frame(sh, f.width, f.height, cfg["frame_margin"])):
                continue
            if ear[1] >= sh[1]:
                continue
            good.append((f, ear, sh, eye, nose))
        info["n_good"] = len(good)
        if len(good) < cfg["calib_min_frames"]:
            info["error"] = f"안정 프레임이 부족합니다({len(good)}/{cfg['calib_min_frames']})."
            return None, info

        # 3) d 판별: 코가 귀보다 화면 오른쪽이면 +1 (가정 — 0단계에서 양방향 검증 필요)
        signs = [1 if g[4][0] > g[1][0] else -1 for g in good]
        d = 1 if sum(signs) > 0 else -1
        info["d"] = d
        info["d_vote_ratio"] = abs(sum(signs)) / len(signs)

        # 4) 후보별 L_ref 를 중앙값으로 1회 측정 → 고정
        per = [l_ref_candidates(g[1], g[2], g[3], g[0].height, cfg["l_ref_B_frac"]) for g in good]
        L_ref = {m: statistics.median(p[m] for p in per) for m in LREF_MODES}
        if min(L_ref.values()) <= 1e-6:
            info["error"] = "기준 길이가 0에 가깝습니다."
            return None, info

        dx = [d * (g[1][0] - g[2][0]) for g in good]            # d*(x_ear - x_shoulder)
        dx_med = statistics.median(dx)
        H_base = {m: dx_med / L_ref[m] for m in LREF_MODES}
        x_sh_base = statistics.median(g[2][0] for g in good)

        # 5) 변동 폭 로그 (재측정 기준은 TBD → 우선 값을 기록)
        h_series = [v / L_ref[cfg["l_ref_mode"]] for v in dx]
        info["H_std"] = statistics.pstdev(h_series)
        info["x_ear_range_px"] = (min(g[1][0] for g in good), max(g[1][0] for g in good))
        info["L_ref"] = L_ref
        info["H_base"] = H_base
        info["unstable"] = info["H_std"] > cfg["calib_max_h_std"]

        cal = Calibration(side=side, d=d, l_ref_mode=cfg["l_ref_mode"], L_ref=L_ref,
                          H_base=H_base, x_shoulder_base=x_sh_base, info=info)
        return cal, info



# ------------------------------------------------------------------ 주의 판정 타이머 (분석기·로그 재생 공용)
class AttentionTracker:
    """dH(부호 있음)가 T_forward 를 넘는 시간을 누적해 주의 여부를 판정한다.
    PostureAnalyzer 와 evaluate_sweep.py(로그 재생)가 같은 코드를 쓰도록 분리했다."""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.acc = 0.0
        self.active = False
        self.last_alert_ts: Optional[float] = None
        self.prev_good_ts: Optional[float] = None

    def pause(self, unavail_dur: float) -> None:
        """측정 불가 프레임: 누적 일시정지, 오래 지속되면 초기화."""
        self.prev_good_ts = None
        if unavail_dur >= self.cfg["T_reset_sec"]:
            self.acc, self.active = 0.0, False

    def step(self, ts: float, dH: float):
        """측정 가능 프레임 하나를 처리. (internal_state, events) 반환."""
        cfg = self.cfg
        events: List[dict] = []
        dt = 0.0 if self.prev_good_ts is None else min(ts - self.prev_good_ts, 1.0)
        self.prev_good_ts = ts
        T, hyst = cfg["T_forward"], cfg["hysteresis_ratio"]
        internal = "normal"
        if self.active:                                   # 주의 중: 히스테리시스 아래로 내려오면 해제
            if dH <= T * hyst:                            # 부호 있는 비교 (후방 이동은 주의가 아님)
                self.active, self.acc = False, 0.0
                events.append({"type": "recovered", "ts": ts, "delta": dH})
            else:
                self.acc += dt
                internal = "attention"
                if self.last_alert_ts is None or ts - self.last_alert_ts >= cfg["alert_interval_sec"]:
                    self.last_alert_ts = ts
                    events.append({"type": "attention_started", "ts": ts, "delta": dH, "repeat": True})
        else:
            if dH > T:                                    # 부호 있는 비교
                self.acc += dt
                internal = "observing"
                if self.acc >= cfg["T_hold_sec"]:
                    self.active = True
                    internal = "attention"
                    self.last_alert_ts = ts
                    events.append({"type": "attention_started", "ts": ts, "delta": dH})
            else:
                self.acc = 0.0
        return internal, events

# ------------------------------------------------------------------ 분석기
class PostureAnalyzer:
    """모니터링 중 프레임을 받아 상태를 판정한다. 기준값(side/d/L_ref)은 바꾸지 않는다."""

    def __init__(self, cfg: dict, cal: Optional[Calibration]):
        self.cfg = cfg
        self.cal = cal
        self._low_cnt = 0
        self._low_flips: collections.deque = collections.deque()
        self._prev_low = False
        self._in_unavail = False
        self._unavail_start: Optional[float] = None
        self._unavail_reason: Optional[str] = None
        self._prolonged_sent = False
        self._good_cnt = 0
        self._tracker = AttentionTracker(cfg)
        self._h_win: collections.deque = collections.deque(maxlen=max(1, int(cfg["smooth_window"])))
        self._prolonged_times: collections.deque = collections.deque()
        self._last: Optional[Result] = None

    @property
    def _attn_acc(self) -> float:          # 테스트·표시용 (타이머 값)
        return self._tracker.acc

    # -- 측정 가능 여부 -------------------------------------------------
    def _check(self, f: Frame) -> Optional[str]:
        """None=측정 가능, 'skip'=일시적 저신뢰(판정 보류), 그 외=측정 불가 사유."""
        cfg, cal = self.cfg, self.cal
        if not f.pts:
            return "ear_not_detected"
        ear, sh, _ = _get_side(f.pts, cal.side)        # 고정된 쪽만 사용 (반대쪽 대체 금지)
        if ear is None:
            return "ear_not_detected"
        if sh is None:
            return "shoulder_not_detected"
        m = cfg["frame_margin"]
        if not in_frame(ear, f.width, f.height, m) or not in_frame(sh, f.width, f.height, m):
            return "out_of_frame"

        low = min(ear[2], sh[2]) < cfg["T_vis"]
        if low and not self._prev_low:                  # 정상→저신뢰 전환 기록 (가림 의심 판단용)
            self._low_flips.append(f.ts)
        self._prev_low = low
        while self._low_flips and f.ts - self._low_flips[0] > cfg["occlusion_window_sec"]:
            self._low_flips.popleft()
        self._low_cnt = self._low_cnt + 1 if low else 0
        if len(self._low_flips) >= cfg["occlusion_flip_count"]:
            return "occluded"                           # TODO(실험) 가림 판별 방식 미확정(임시 휴리스틱)
        if self._low_cnt >= cfg["N_low_vis"]:
            return "low_confidence"
        if low:
            return "skip"
        if ear[1] >= sh[1]:
            return "invalid_geometry"
        return None

    # -- 메인 -----------------------------------------------------------
    def update(self, f: Frame) -> Result:
        cfg, cal = self.cfg, self.cal
        events: List[dict] = []

        if cal is None:
            r = Result(f.ts, "unavailable", "unavailable", "not_calibrated", None, None)
            self._last = r
            return r

        reason = self._check(f)
        ear, sh, _ = _get_side(f.pts, cal.side) if f.pts else (None, None, None)

        if reason == "skip" and self._last is not None:
            r = Result(**{**self._last.__dict__, "ts": f.ts, "events": []})     # 판정 보류(직전 상태 유지)
            return r

        if reason is not None:                           # ---- 측정 불가 ----
            if not self._in_unavail:
                self._in_unavail, self._unavail_start, self._prolonged_sent = True, f.ts, False
            self._unavail_reason = reason
            self._good_cnt = 0
            dur = f.ts - self._unavail_start
            self._tracker.pause(dur)                     # 주의 타이머 일시정지, 오래 지속되면 초기화
            if dur >= cfg["unavailable_prolonged_sec"] and not self._prolonged_sent:
                self._prolonged_sent = True
                events.append({"type": "unavailable_prolonged", "ts": f.ts})
                self._prolonged_times.append(f.ts)
                while self._prolonged_times and f.ts - self._prolonged_times[0] > cfg["recalib_window_sec"]:
                    self._prolonged_times.popleft()
                if len(self._prolonged_times) >= cfg["recalib_suggest_count"]:
                    events.append({"type": "recalibration_suggested", "ts": f.ts})   # 안내만, 자동 갱신 금지
            r = Result(f.ts, "unavailable", "unavailable", reason, cal.side, cal.d, ear, sh,
                       duration_sec=self._tracker.acc, events=events)
            self._last = r
            return r

        # ---- 측정 가능한 프레임 ----
        if self._in_unavail:                             # 판정 재개: 안정 프레임 K개 연속 필요
            self._good_cnt += 1
            if self._good_cnt < cfg["K_recover"]:
                r = Result(f.ts, "unavailable", "unavailable", self._unavail_reason, cal.side, cal.d,
                           ear, sh, duration_sec=self._tracker.acc)
                self._last = r
                return r
            self._in_unavail = False

        d, mode = cal.d, cal.l_ref_mode
        diff = d * (ear[0] - sh[0])
        H_all = {m: diff / cal.L_ref[m] for m in LREF_MODES}      # L_ref 는 고정값 사용 (재계산 금지)
        self._h_win.append(H_all[mode])
        H = statistics.median(self._h_win)
        dH = H - cal.H_base[mode]
        S = d * (sh[0] - cal.x_shoulder_base) / cal.L_ref[mode]   # 검증 지표: 로그 전용, 판정에 사용 안 함

        internal, events = self._tracker.step(f.ts, dH)

        state = "attention" if internal == "attention" else "normal"    # observing 은 API에 노출하지 않음
        r = Result(f.ts, state, internal, None, cal.side, cal.d, ear, sh, H, dH, S, H_all,
                   self._tracker.acc, events)
        self._last = r
        return r

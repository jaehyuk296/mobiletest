"""
analyze_log.py — 0단계 실험 로그(CSV) 요약 (명세서 v9 21장 / S_VALIDATION 용)

  python analyze_log.py data/posture/log_*.csv
  python analyze_log.py data/posture/log_*.csv --baseline neutral

출력:
  1) 성능·검출: FPS, 처리 지연, 랜드마크 감지율, 측정 불가 비율·사유 분포
  2) 조건별 H / dH / S 통계
  3) L_ref 후보(A/B/C) 비교: 잡음 대비 신호 (head_fwd 가 neutral 과 얼마나 구분되는가)
  4) S 검증: 자세 동작(lean/head/recline)에서의 변화 vs 위치 변경(chair/camera moved)에서의 변화
  5) T_forward 탐색 참고용 분포
※ 승격·임계값을 자동 결정하지 않습니다. 기준 수치는 TBD(실험 후 팀이 확정).
"""
import argparse, glob, sys
import numpy as np
import pandas as pd

POSTURE_CONDS = ["lean_fwd", "head_fwd", "recline"]
MOVE_CONDS = ["chair_moved", "camera_moved"]


def load(paths):
    files = []
    for p in paths:
        files += glob.glob(p)
    if not files:
        sys.exit("로그 파일이 없습니다.")
    df = pd.concat([pd.read_csv(f).assign(file=f) for f in files], ignore_index=True)
    for c in ["ts", "x_ear", "vis_ear", "H", "dH", "H_A", "H_B", "H_C", "S", "proc_ms"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["reason"] = df["reason"].fillna("")
    return df


def med(s):
    return float(np.nanmedian(s)) if len(s) else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("logs", nargs="+")
    ap.add_argument("--baseline", default="neutral")
    a = ap.parse_args()
    df = load(a.logs)
    ok = df[df["state"] != "unavailable"]

    print("=" * 60, "\n1) 성능·검출")
    dur = sum(g["ts"].max() - g["ts"].min() for _, g in df.groupby("file"))
    print(f"행 {len(df)}  | 기록 시간 {dur:.1f}s  | FPS(행/시간) {len(df) / max(dur, 1e-6):.1f}")
    print(f"처리 지연 평균 {df['proc_ms'].mean():.1f}ms  p95 {df['proc_ms'].quantile(.95):.1f}ms")
    print(f"귀 랜드마크 감지율 {df['x_ear'].notna().mean():.1%}")
    print(f"측정 불가 비율 {(df['state'] == 'unavailable').mean():.1%}")
    rc = df[df["reason"] != ""]["reason"].value_counts()
    if len(rc):
        print("측정 불가 사유:", {k: f"{v / len(df):.1%}" for k, v in rc.items()})

    print("=" * 60, "\n2) 조건별 통계 (측정 가능 프레임만)")
    rows = []
    for c, g in ok.groupby("condition"):
        rows.append([c, len(g), med(g["H"]), g["H"].std(), med(g["dH"]), g["dH"].quantile(.05), g["dH"].quantile(.95), med(g["S"]), g["S"].std()])
    t = pd.DataFrame(rows, columns=["condition", "n", "H_med", "H_std", "dH_med", "dH_p5", "dH_p95", "S_med", "S_std"])
    print(t.round(3).to_string(index=False) if len(t) else "(데이터 없음)")

    base = ok[ok["condition"] == a.baseline]
    print("=" * 60, "\n3) L_ref 후보 비교 — 신호(조건 간 H 차이) / 잡음(기준 조건 H 표준편차). 클수록 좋음")
    if len(base) < 5:
        print(f"'{a.baseline}' 조건 데이터가 부족합니다.")
    else:
        for tgt in ["head_fwd", "lean_fwd"]:
            g = ok[ok["condition"] == tgt]
            if len(g) < 5:
                print(f"- {tgt}: 데이터 없음"); continue
            out = {}
            for X in "ABC":
                sig = med(g[f"H_{X}"]) - med(base[f"H_{X}"]); noise = base[f"H_{X}"].std()
                out[X] = sig / noise if noise > 0 else float("nan")
            print(f"- {tgt} vs {a.baseline}:", {k: round(float(v), 2) for k, v in out.items()})
        print("(후보는 캘리브레이션 때 고정한 값으로 계산됨. 선택은 팀이 확정)")

    print("=" * 60, "\n4) S 검증 — 자세 동작 vs 위치 변경 (기준 조건 대비 중앙값 변화)")
    if len(base) < 5:
        print("기준 조건 데이터 부족")
    else:
        s0, h0 = med(base["S"]), med(base["dH"])
        eff = {}
        for c in POSTURE_CONDS + MOVE_CONDS:
            g = ok[ok["condition"] == c]
            if len(g) >= 5:
                eff[c] = (abs(med(g["S"]) - s0), abs(med(g["dH"]) - h0))
        if not eff:
            print("비교할 조건 데이터가 없습니다 (키 2~6으로 조건 라벨을 바꿔가며 기록).")
        else:
            print(f"{'condition':14s}{'|ΔS|':>8s}{'|ΔdH|':>8s}")
            for c, (ds, dh) in eff.items():
                print(f"{c:14s}{ds:8.3f}{dh:8.3f}   {'(자세 동작)' if c in POSTURE_CONDS else '(위치 변경)'}")
            ps = [eff[c][0] for c in POSTURE_CONDS if c in eff]; ms = [eff[c][0] for c in MOVE_CONDS if c in eff]
            if ps and ms:
                print(f"\nS: 자세 동작 최대 {max(ps):.3f} / 위치 변경 최대 {max(ms):.3f} → 비율 {max(ps) / max(max(ms), 1e-9):.2f}")
                ph = [eff[c][1] for c in POSTURE_CONDS if c in eff]; mh = [eff[c][1] for c in MOVE_CONDS if c in eff]
                print(f"H: 자세 동작 최대 {max(ph):.3f} / 위치 변경 최대 {max(mh):.3f} → 비율 {max(ph) / max(max(mh), 1e-9):.2f}")
                print("해석: 비율이 1에 가까우면 S는 자세 변화와 의자·카메라 이동을 구분하지 못함(승격 불가).")
                print("      승격 기준 수치는 TBD — 이 스크립트는 판정하지 않습니다.")
            else:
                print("\n위치 변경(chair_moved/camera_moved) 또는 자세 동작 데이터가 없어 비율을 계산할 수 없습니다.")

    print("=" * 60, "\n5) T_forward 탐색 참고 (dH 분포)")
    g_head = ok[ok["condition"] == "head_fwd"]
    if len(base) >= 5:
        print(f"{a.baseline}: p95 {base['dH'].quantile(.95):.3f}, p99 {base['dH'].quantile(.99):.3f}, max {base['dH'].max():.3f}")
    if len(g_head) >= 5:
        print(f"head_fwd: p5 {g_head['dH'].quantile(.05):.3f}, p50 {g_head['dH'].quantile(.5):.3f}")
        if len(base) >= 5:
            lo, hi = base["dH"].quantile(.99), g_head["dH"].quantile(.05)
            print(("→ 두 분포가 분리됨. 후보 구간(참고): %.3f ~ %.3f" % (lo, hi)) if lo < hi else
                  "→ 두 분포가 겹침. 오탐/미탐 trade-off 를 팀이 정해야 함")


if __name__ == "__main__":
    main()

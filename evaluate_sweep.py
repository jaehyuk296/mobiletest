"""
evaluate_sweep.py — 라벨이 붙은 0단계 로그를 재생해 판정 파라미터를 데이터로 비교한다 (자세 판정 정확도 개선용)

왜 필요한가
  임계값(T_forward, T_hold_sec 등)을 감으로 정하면 오탐/미탐을 알 수 없다. 로그의 조건 라벨(키 1~6)을 정답으로 보고
  파라미터 조합마다 "정답 자세에서 얼마나 잡고, 아닌 자세에서 얼마나 잘못 울리는지"를 계산한다.

입력
  python posture_spec_test.py --log 로 만든 CSV (조건 라벨이 기록돼 있어야 함).
  각 로그 옆의 <로그>.calib.json(같은 캘리브레이션의 H_base)을 사용한다. 없으면 --calib 로 지정하거나,
  없는 경우 baseline 조건(neutral)의 H 중앙값으로 대체한다(경고 표시).

정답 라벨 (기본값, 팀이 바꿔야 할 수 있음 — TODO)
  --pos    head_fwd                                  : 주의가 울려야 하는 조건
  --neg    neutral recline chair_moved camera_moved   : 울리면 안 되는 조건 (H는 후방·위치 변경에 반응하면 안 됨)
  그 외(예: lean_fwd)는 H만으로 정답이 불분명하므로 평가에서 제외한다.

실행
  python evaluate_sweep.py data/posture/log_*.csv
  python evaluate_sweep.py data/posture/log_*.csv --max-fa-per-hour 1 --loo --out sweep.csv

주의
  - 이 스크립트는 config.yaml 을 바꾸지 않는다. 결과를 보고 팀이 확정해 config.yaml 한 곳에만 반영한다.
  - 사용자·세션이 적으면 과적합된다. --loo(로그 파일 단위 교차 검증)로 일반화를 확인한다.
  - '측정 불가' 판정(T_vis 등)은 로그에 기록된 값을 그대로 쓴다. 이 값들을 바꾸려면 실시간 재실행이 필요하다.
  - 판정은 H 하나의 부호 있는 비교만 사용한다(S는 쓰지 않는다).
"""
import argparse, glob, itertools, json, os, statistics, sys
from collections import deque, defaultdict
import numpy as np
import pandas as pd
from posture_core import AttentionTracker, load_config

HERE = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------- 로드
def load_logs(paths, calib_override, baseline_cond):
    files = []
    for p in paths:
        files += sorted(glob.glob(p))
    files = [f for f in files if f.endswith(".csv")]
    if not files:
        sys.exit("로그 파일이 없습니다.")
    sessions = []
    for f in files:
        df = pd.read_csv(f)
        for c in ["ts", "H_A", "H_B", "H_C"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.sort_values("ts").reset_index(drop=True)
        calp = f + ".calib.json"
        base = None
        if calib_override and os.path.exists(calib_override):
            base = json.load(open(calib_override, encoding="utf-8"))["H_base"]
        elif os.path.exists(calp):
            base = json.load(open(calp, encoding="utf-8"))["H_base"]
        if base is None:
            ok = df[(df["state"] != "unavailable") & (df["condition"] == baseline_cond)]
            if len(ok) < 5:
                print(f"[건너뜀] {os.path.basename(f)}: 캘리브레이션 파일도 baseline 조건 데이터도 없음"); continue
            base = {m: float(np.nanmedian(ok[f"H_{m}"])) for m in "ABC"}
            print(f"[경고] {os.path.basename(f)}: .calib.json 없음 → '{baseline_cond}' 조건 H 중앙값을 기준으로 대체(실제 기준과 다를 수 있음)")
        sessions.append({"file": f, "df": df, "H_base": {k: float(v) for k, v in base.items()}})
    if not sessions:
        sys.exit("평가할 세션이 없습니다.")
    return sessions


def segments(df):
    """연속된 같은 condition 구간 [(cond, start_idx, end_idx_exclusive)]"""
    out, start = [], 0
    conds = df["condition"].tolist()
    for i in range(1, len(conds) + 1):
        if i == len(conds) or conds[i] != conds[start]:
            out.append((conds[start], start, i)); start = i
    return out


# ---------------------------------------------------------------- 재생
def smoothed_dH(sess, lref, smooth):
    """측정 가능 프레임에서만 이동 중앙값을 갱신(PostureAnalyzer 와 동일). 측정 불가 프레임은 None."""
    df = sess["df"]; hb = sess["H_base"][lref]
    col = df[f"H_{lref}"].tolist(); st = df["state"].tolist()
    win = deque(maxlen=max(1, int(smooth))); out = []
    for h, s in zip(col, st):
        if s == "unavailable" or h != h:          # h != h : NaN
            out.append(None); continue
        win.append(h)
        out.append(statistics.median(win) - hb)
    return out


def replay(sess, dH_list, cfg):
    """AttentionTracker 로 재생. 행마다 'attention' 여부와 새 경고 이벤트를 반환."""
    tr = AttentionTracker(cfg)
    ts = sess["df"]["ts"].tolist()
    attn, new_ev = [], []
    un_start = None
    for t, dh in zip(ts, dH_list):
        if dh is None:
            if un_start is None: un_start = t
            tr.pause(t - un_start)
            attn.append(False); new_ev.append(False); continue
        un_start = None
        internal, events = tr.step(t, dh)
        attn.append(internal == "attention")
        new_ev.append(any(e["type"] == "attention_started" and not e.get("repeat") for e in events))
    return attn, new_ev


def score_session(sess, attn, new_ev, pos, neg):
    df = sess["df"]; ts = df["ts"].tolist(); st = df["state"].tolist(); cond = df["condition"].tolist()
    r = dict(pos_segs=0, pos_det=0, lat=[], neg_ev=0, neg_dur=0.0, tp=0, fp=0, fn=0, tn=0, n_unavail=0, n_eval=0)
    for c, a, b in segments(df):
        if c in pos:
            r["pos_segs"] += 1
            idx = [i for i in range(a, b) if attn[i]]
            if idx:
                r["pos_det"] += 1; r["lat"].append(ts[idx[0]] - ts[a])
        elif c in neg:
            r["neg_ev"] += sum(new_ev[a:b])
            r["neg_dur"] += (ts[b - 1] - ts[a]) if b - a > 1 else 0.0
        else:
            continue
        for i in range(a, b):
            r["n_eval"] += 1
            if st[i] == "unavailable":
                r["n_unavail"] += 1; continue
            if c in pos:
                r["tp" if attn[i] else "fn"] += 1
            else:
                r["fp" if attn[i] else "tn"] += 1
    return r


def merge(rs):
    m = dict(pos_segs=0, pos_det=0, lat=[], neg_ev=0, neg_dur=0.0, tp=0, fp=0, fn=0, tn=0, n_unavail=0, n_eval=0)
    for r in rs:
        for k in m:
            m[k] = m[k] + r[k] if k != "lat" else m[k] + r[k]
    return m


def summarize(m):
    prec = m["tp"] / (m["tp"] + m["fp"]) if (m["tp"] + m["fp"]) else float("nan")
    rec = m["tp"] / (m["tp"] + m["fn"]) if (m["tp"] + m["fn"]) else float("nan")
    f1 = 2 * prec * rec / (prec + rec) if prec == prec and rec == rec and (prec + rec) > 0 else float("nan")
    return dict(
        seg_recall=m["pos_det"] / m["pos_segs"] if m["pos_segs"] else float("nan"),
        lat_med=float(np.median(m["lat"])) if m["lat"] else float("nan"),
        fa_per_hour=m["neg_ev"] / (m["neg_dur"] / 3600) if m["neg_dur"] > 0 else float("nan"),
        precision=prec, recall=rec, f1=f1,
        unavail=m["n_unavail"] / m["n_eval"] if m["n_eval"] else float("nan"))


def rank_key(s, budget):
    fa = s["fa_per_hour"]
    within = (fa == fa) and fa <= budget
    rec = s["seg_recall"] if s["seg_recall"] == s["seg_recall"] else -1
    lat = s["lat_med"] if s["lat_med"] == s["lat_med"] else 1e9
    f1 = s["f1"] if s["f1"] == s["f1"] else -1
    return (0 if within else 1, fa if not within else 0, -rec, lat, -f1)


# ---------------------------------------------------------------- 메인
def main():
    ap = argparse.ArgumentParser(description="라벨 로그 재생 기반 파라미터 비교 (자세 판정 정확도 개선)")
    ap.add_argument("logs", nargs="+")
    ap.add_argument("--config", default=os.path.join(HERE, "config.yaml"))
    ap.add_argument("--calib", help="모든 로그에 쓸 calibration.json (없으면 로그 옆 .calib.json)")
    ap.add_argument("--pos", nargs="+", default=["head_fwd"])
    ap.add_argument("--neg", nargs="+", default=["neutral", "recline", "chair_moved", "camera_moved"])
    ap.add_argument("--baseline", default="neutral")
    # 아래 격자는 '탐색 후보'일 뿐 확정값이 아니다(TODO(실험)). 데이터 범위에 맞게 바꿔 쓴다.
    ap.add_argument("--t-forward", nargs="+", type=float, default=None,
                    help="T_forward 절대값 후보(지정하면 모든 L_ref 에 동일 적용). 기본은 --tf-frac 사용")
    ap.add_argument("--tf-frac", nargs="+", type=float, default=[0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
                    help="T_forward 후보 = 비율 × (pos 조건의 dH 중앙값). L_ref 후보마다 H 크기가 달라 절대값 격자는 불공평하다")
    ap.add_argument("--t-hold", nargs="+", type=float, default=[2.0, 3.0, 5.0, 8.0])
    ap.add_argument("--smooth", nargs="+", type=int, default=[1, 3, 5, 9])
    ap.add_argument("--hyst", nargs="+", type=float, default=[0.6, 0.8, 0.9])
    ap.add_argument("--lref", nargs="+", default=["C", "A", "B"])
    ap.add_argument("--max-fa-per-hour", type=float, default=2.0, help="탐색 시 허용할 시간당 오경고 수(탐색용 가정, 확정값 아님)")
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--loo", action="store_true", help="로그 파일 단위 교차 검증(세션 2개 이상 필요)")
    ap.add_argument("--out", help="전체 결과 CSV 저장 경로")
    a = ap.parse_args()

    base_cfg = load_config(a.config)
    pos, neg = set(a.pos), set(a.neg)
    sessions = load_logs(a.logs, a.calib, a.baseline)

    # 데이터 요약 / 경고
    print("=" * 70, "\n데이터 요약")
    npos = nneg = 0
    for s in sessions:
        segs = segments(s["df"]); p = [x for x in segs if x[0] in pos]; n = [x for x in segs if x[0] in neg]
        npos += len(p); nneg += len(n)
        print(f"- {os.path.basename(s['file'])}: 정답 구간 {len(p)}개(pos) / {len(n)}개(neg), 제외 구간 {len(segs) - len(p) - len(n)}개")
    if npos < 5:
        print(f"⚠ pos 구간이 {npos}개뿐입니다. 결과가 우연일 수 있으니 같은 동작을 여러 번(여러 세션·여러 사람) 기록하세요.")
    if len(sessions) < 3:
        print(f"⚠ 세션이 {len(sessions)}개입니다. 과적합 위험이 큽니다(사람·날짜·카메라 배치를 바꿔 더 모으세요).")

    # T_forward 후보: L_ref 후보마다 H 의 크기(스케일)가 다르므로 기본은 'pos 조건 dH 중앙값 × 비율'로 만든다
    def tf_candidates(lr):
        if a.t_forward:
            return list(a.t_forward)
        vals = []
        for i, s_ in enumerate(sessions):
            dh = smoothed_dH(s_, lr, 1); cond = s_["df"]["condition"].tolist()
            vals += [d for d, c in zip(dh, cond) if d is not None and c in pos]
        sig = float(np.median(vals)) if vals else float("nan")
        if not (sig == sig) or sig <= 0:
            print(f"[경고] L_ref {lr}: pos 조건 신호를 계산할 수 없어 기본 절대값 격자를 씁니다."); return [0.05, 0.08, 0.10, 0.12, 0.15, 0.20]
        print(f"L_ref {lr}: pos 조건 dH 중앙값 {sig:.3f} → T_forward 후보 {[round(f * sig, 3) for f in a.tf_frac]}")
        return [round(f * sig, 4) for f in a.tf_frac]

    # 격자 평가 (평활화·L_ref 별로 dH 를 한 번만 계산)
    results = {}                                            # config -> [session별 raw]
    dh_cache = {(i, lr, sm): smoothed_dH(s, lr, sm) for i, s in enumerate(sessions) for lr in a.lref for sm in a.smooth}
    for lr in a.lref:
        for sm, tf, th, hy in itertools.product(a.smooth, tf_candidates(lr), a.t_hold, a.hyst):
            cfg = dict(base_cfg); cfg.update(T_forward=tf, T_hold_sec=th, hysteresis_ratio=hy, smooth_window=sm)
            per = []
            for i, s in enumerate(sessions):
                attn, nev = replay(s, dh_cache[(i, lr, sm)], cfg)
                per.append(score_session(s, attn, nev, pos, neg))
            results[(lr, sm, tf, th, hy)] = per

    rows = []
    for k, per in results.items():
        s = summarize(merge(per)); rows.append(dict(lref=k[0], smooth=k[1], T_forward=k[2], T_hold_sec=k[3], hysteresis=k[4], **s))
    res = pd.DataFrame(rows)
    res["_k"] = [rank_key(r, a.max_fa_per_hour) for r in rows]
    res = res.sort_values("_k").drop(columns="_k").reset_index(drop=True)
    if a.out:
        res.to_csv(a.out, index=False); print("전체 결과 저장:", a.out)

    pd.set_option("display.width", 200)
    fmt = res.head(a.top).copy()
    for c in ["seg_recall", "precision", "recall", "f1", "unavail"]:
        fmt[c] = (fmt[c] * 100).round(1)
    fmt["lat_med"] = fmt["lat_med"].round(1); fmt["fa_per_hour"] = fmt["fa_per_hour"].round(2)
    within = res[res["fa_per_hour"] <= a.max_fa_per_hour]
    print("=" * 70, f"\n상위 {a.top}개 (시간당 오경고 ≤ {a.max_fa_per_hour} 안에서 구간 재현율↑, 지연↓ 순; 범위 안에 없으면 오경고 적은 순)")
    print(f"범위 안 조합 {len(within)} / 전체 {len(res)}")
    print(fmt.to_string(index=False))
    print("\n열 설명: seg_recall=정답 구간 중 한 번이라도 주의가 울린 비율(%), lat_med=구간 시작→첫 주의까지 중앙값(초),")
    print("         fa_per_hour=울리면 안 되는 구간에서 새 경고가 난 횟수/시간, precision·recall·f1=프레임 단위(%, 지속시간 지연 때문에 recall 은 낮게 나옴),")
    print("         unavail=평가 구간 중 측정 불가 비율(%) — 파라미터와 무관, 카메라 배치·T_vis 영향")

    # 민감도: 한 파라미터씩 바꿨을 때 (최적 조합 기준)
    if len(res):
        best = res.iloc[0]
        print("=" * 70, "\n최적 조합 주변 민감도 (다른 값은 최적 조합에 고정)")
        for name in ["T_forward", "T_hold_sec", "smooth", "hysteresis"]:
            sub = res[res["lref"] == best["lref"]]
            for other in ["smooth", "T_forward", "T_hold_sec", "hysteresis"]:
                if other != name:
                    sub = sub[sub[other] == best[other]]
            sub = sub.sort_values(name)
            line = "  ".join(f"{r[name]}→재현율 {r['seg_recall']*100:.0f}%/오경고 {r['fa_per_hour']:.1f}" for _, r in sub.iterrows())
            print(f"- {name}: {line}")
        sub = res
        for other in ["smooth", "T_hold_sec", "hysteresis"]:
            sub = sub[sub[other] == best[other]]
        line = "  ".join(f"{lr}→(최선) 재현율 {g.iloc[0]['seg_recall']*100:.0f}%/오경고 {g.iloc[0]['fa_per_hour']:.1f}/T_forward {g.iloc[0]['T_forward']}"
                         for lr, g in ((lr, sub[sub['lref'] == lr]) for lr in a.lref) if len(g))
        print(f"- lref(각 후보의 T_forward 중 최선): {line}")

    # 교차 검증
    if a.loo:
        if len(sessions) < 2:
            print("\n--loo 는 세션이 2개 이상 필요합니다.")
        else:
            print("=" * 70, "\n로그 파일 단위 교차 검증 (나머지로 고른 조합을 빠진 로그에 적용)")
            agg = []
            for h in range(len(sessions)):
                tr_rows = []
                for k, per in results.items():
                    s = summarize(merge([p for i, p in enumerate(per) if i != h])); tr_rows.append((rank_key(s, a.max_fa_per_hour), k))
                bk = sorted(tr_rows)[0][1]
                s = summarize(merge([results[bk][h]])); agg.append(s)
                print(f"- 빠진 로그 {os.path.basename(sessions[h]['file'])}: 선택 {bk} → 재현율 {s['seg_recall']*100:.0f}%, 오경고 {s['fa_per_hour']:.1f}/h, 지연 {s['lat_med']:.1f}s")
            ok = [x for x in agg if x["seg_recall"] == x["seg_recall"]]
            if ok:
                print(f"평균: 재현율 {np.mean([x['seg_recall'] for x in ok])*100:.0f}%, 오경고 {np.nanmean([x['fa_per_hour'] for x in ok]):.1f}/h")

    print("\n※ 이 결과는 파라미터 후보를 좁히는 참고 자료입니다. 값 확정과 config.yaml 반영은 팀이 하고(P7 값 확정 기록표),")
    print("  확정 전에는 같은 방식으로 다른 사람·다른 날 데이터에서 다시 검증하세요.")


if __name__ == "__main__":
    main()

"""단위 테스트 (카메라·MediaPipe 없이 합성 랜드마크로 실행): python -m unittest discover -s tests -v"""
import os, sys, unittest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from posture_core import Calibrator, PostureAnalyzer, Frame, load_config

W, H_IMG = 640, 480


def make_pts(d=1, off=20.0, side="left", vis_main=0.9, vis_other=0.2, ear_y=240.0, sh_y=300.0, sh_x=320.0, shift=0.0):
    """측면 합성 랜드마크. off = d 방향으로 귀가 어깨보다 앞선 px."""
    pts = {}
    for s, suf in (("left", "_l"), ("right", "_r")):
        v = vis_main if s == side else vis_other
        dx = 0 if s == side else 6          # 반대쪽은 약간 다른 위치
        pts["ear" + suf] = (sh_x + shift + d * off + dx, ear_y, v)
        pts["sh" + suf] = (sh_x + shift + dx, sh_y, v)
        pts["eye" + suf] = (sh_x + shift + d * (off + 12) + dx, ear_y - 8, v)
    pts["nose"] = (sh_x + shift + d * (off + 30), ear_y + 10, vis_main)
    return pts


def fr(ts, **kw):
    return Frame(ts, W, H_IMG, make_pts(**kw))


def calibrate(cfg, d=1, side="left", n=40):
    c = Calibrator(cfg)
    for i in range(n):
        c.add(fr(i / 10, d=d, side=side))
    cal, info = c.finish()
    assert cal is not None, info
    return cal


class T(unittest.TestCase):
    def setUp(self):
        self.cfg = load_config()
        self.cfg.update(T_hold_sec=2.0, T_reset_sec=5.0, K_recover=3, N_low_vis=3, smooth_window=1,
                        unavailable_prolonged_sec=3.0, alert_interval_sec=100.0)

    def run_seq(self, an, frames):
        return [an.update(f) for f in frames]

    # 캘리브레이션
    def test_calib_side_and_d(self):
        for d in (1, -1):
            for side in ("left", "right"):
                cal = calibrate(self.cfg, d=d, side=side)
                self.assertEqual(cal.side, side)
                self.assertEqual(cal.d, d)

    # 앞으로 나오면 d와 무관하게 dH 증가
    def test_forward_increases_dH(self):
        for d in (1, -1):
            cal = calibrate(self.cfg, d=d)
            an = PostureAnalyzer(self.cfg, cal)
            r0 = an.update(fr(10, d=d, off=20))
            r1 = an.update(fr(10.1, d=d, off=45))
            self.assertGreater(r1.dH, r0.dH)

    # 후방 이동(dH 음수)은 주의가 아님 (절댓값 비교 금지)
    def test_backward_not_attention(self):
        cal = calibrate(self.cfg)
        an = PostureAnalyzer(self.cfg, cal)
        rs = self.run_seq(an, [fr(10 + i * 0.1, off=-10) for i in range(80)])
        self.assertTrue(all(r.state == "normal" for r in rs))
        self.assertLess(rs[-1].dH, 0)

    # 지속 시간 미달 → 주의 아님 / 초과 → 주의
    def test_hold_time(self):
        cal = calibrate(self.cfg)
        an = PostureAnalyzer(self.cfg, cal)
        rs = self.run_seq(an, [fr(10 + i * 0.1, off=60) for i in range(15)])        # 1.4초
        self.assertTrue(all(r.state == "normal" for r in rs))
        an = PostureAnalyzer(self.cfg, cal)
        rs = self.run_seq(an, [fr(10 + i * 0.1, off=60) for i in range(40)])        # 3.9초
        self.assertEqual(rs[-1].state, "attention")
        self.assertTrue(any(e["type"] == "attention_started" for r in rs for e in r.events))

    # 복귀 시 recovered 1회
    def test_recovered(self):
        cal = calibrate(self.cfg)
        an = PostureAnalyzer(self.cfg, cal)
        seq = [fr(10 + i * 0.1, off=60) for i in range(40)] + [fr(14 + i * 0.1, off=20) for i in range(10)]
        rs = self.run_seq(an, seq)
        ev = [e["type"] for r in rs for e in r.events]
        self.assertEqual(ev.count("recovered"), 1)
        self.assertEqual(rs[-1].state, "normal")

    # 측정 불가: 타이머 일시정지, T_reset 초과 시 초기화, K_recover 연속 후 재개
    def test_unavailable_pause_and_reset(self):
        cal = calibrate(self.cfg)
        an = PostureAnalyzer(self.cfg, cal)
        t = 10.0
        for i in range(10):                              # 1초 누적
            an.update(fr(t, off=60)); t += 0.1
        acc_before = an._attn_acc
        for i in range(20):                              # 2초 화면 이탈(T_reset=5 미만)
            r = an.update(fr(t, off=60, sh_x=5)); t += 0.1
            self.assertEqual(r.state, "unavailable")
        self.assertAlmostEqual(an._attn_acc, acc_before, places=6)       # 일시정지
        for i in range(70):                              # 7초 이탈 → 초기화
            an.update(fr(t, off=60, sh_x=5)); t += 0.1
        self.assertEqual(an._attn_acc, 0.0)

    def test_recover_needs_k_frames(self):
        cal = calibrate(self.cfg)
        an = PostureAnalyzer(self.cfg, cal)
        t = 10.0
        an.update(fr(t, off=20)); t += 0.1
        for _ in range(5):
            an.update(fr(t, off=20, sh_x=5)); t += 0.1
        states = []
        for _ in range(5):
            states.append(an.update(fr(t, off=20)).state); t += 0.1
        self.assertEqual(states[:2], ["unavailable", "unavailable"])    # K_recover=3 → 2프레임은 보류
        self.assertEqual(states[-1], "normal")

    # 고정된 쪽이 나빠져도 반대쪽으로 대체하지 않음
    def test_no_side_switch(self):
        cal = calibrate(self.cfg, side="left")
        an = PostureAnalyzer(self.cfg, cal)
        t, last = 10.0, None
        for _ in range(10):
            last = an.update(Frame(t, W, H_IMG, make_pts(side="left", vis_main=0.1, vis_other=0.95))); t += 0.1
        self.assertEqual(last.state, "unavailable")
        self.assertEqual(last.reason, "low_confidence")
        self.assertEqual(last.side, "left")

    # L_ref 고정: 모니터링 중 변하지 않음
    def test_lref_fixed(self):
        cal = calibrate(self.cfg)
        before = dict(cal.L_ref)
        an = PostureAnalyzer(self.cfg, cal)
        for i in range(30):
            an.update(fr(10 + i * 0.1, off=50, ear_y=200 + i, sh_y=300 - i))   # 머리/어깨 움직임
        self.assertEqual(cal.L_ref, before)

    def test_invalid_geometry(self):
        cal = calibrate(self.cfg)
        an = PostureAnalyzer(self.cfg, cal)
        r = an.update(fr(10, ear_y=320.0, sh_y=300.0))
        self.assertEqual((r.state, r.reason), ("unavailable", "invalid_geometry"))

    def test_out_of_frame(self):
        cal = calibrate(self.cfg)
        an = PostureAnalyzer(self.cfg, cal)
        r = an.update(fr(10, sh_x=2))
        self.assertEqual(r.reason, "out_of_frame")

    def test_not_calibrated_and_no_pose(self):
        an = PostureAnalyzer(self.cfg, None)
        self.assertEqual(an.update(fr(1)).reason, "not_calibrated")
        an = PostureAnalyzer(self.cfg, calibrate(self.cfg))
        self.assertEqual(an.update(Frame(1, W, H_IMG, None)).reason, "ear_not_detected")

    # S는 판정에 쓰지 않음: 어깨만 크게 움직여도(귀-어깨 상대 위치 동일) 상태 불변
    def test_S_not_used_for_decision(self):
        cal = calibrate(self.cfg)
        an = PostureAnalyzer(self.cfg, cal)
        rs = self.run_seq(an, [fr(10 + i * 0.1, off=20, shift=60) for i in range(60)])
        self.assertTrue(all(r.state == "normal" for r in rs))
        self.assertGreater(abs(rs[-1].S), 0.3)           # S 값은 변하지만 판정에는 영향 없음


if __name__ == "__main__":
    unittest.main()

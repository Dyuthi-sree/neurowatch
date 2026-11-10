import numpy as np
import pandas as pd
import pytest

from neurowatch.evaluation import alarm_events, causal_smooth, evaluate, score_record


def test_causal_smooth_uses_only_past():
    p = np.array([0, 0, 1, 1, 0, 0], float)
    np.testing.assert_allclose(causal_smooth(p, 3), [0, 0, 1 / 3, 2 / 3, 2 / 3, 1 / 3])
    q = p.copy()
    q[4:] = 1
    np.testing.assert_array_equal(causal_smooth(p, 3)[:4], causal_smooth(q, 3)[:4])


def test_alarm_events_merge_gap():
    starts = np.arange(0, 200, 2.0)
    d = np.zeros(len(starts), bool)
    d[[10, 11, 12]] = True       # decisions at 24, 26, 28 s
    d[20] = True                 # 44 s: gap 16 s <= 30 -> same event
    d[60] = True                 # 124 s: new event
    ev = alarm_events(starts, d)
    assert ev == [(24.0, 20.0, 44.0), (124.0, 120.0, 124.0)]


def test_score_detection_delay_and_false_alarm():
    starts = np.arange(0, 3600 - 3, 2.0)
    d = np.zeros(len(starts), bool)
    d[(starts >= 1006) & (starts <= 1030)] = True   # alarm raised at 1010 s
    d[(starts >= 2000) & (starts <= 2004)] = True   # nothing there -> false alarm
    s = score_record(starts, d, [(1000.0, 1040.0), (3000.0, 3010.0)], 1.0)
    assert (s.n_seizures, s.n_detected, s.n_false_alarms) == (2, 1, 1)
    assert s.delays == [pytest.approx(10.0)]


def test_alarm_before_onset_counts_with_zero_delay():
    starts = np.arange(0, 100, 2.0)
    d = (starts >= 40) & (starts <= 60)
    s = score_record(starts, d, [(50.0, 70.0)], 1.0)
    assert s.n_detected == 1 and s.delays == [0.0] and s.n_false_alarms == 0


def test_evaluate_pooled():
    starts = np.arange(0, 3600 - 3, 2.0)
    win = pd.DataFrame({"record": ["a"] * len(starts) + ["b"] * len(starts),
                        "patient": ["p1"] * len(starts) + ["p2"] * len(starts),
                        "start_s": np.r_[starts, starts]})
    p = np.zeros(len(win))
    p[(win.record == "a") & (win.start_s.between(500, 520))] = 1
    sz = pd.DataFrame({"record": ["a"], "start_s": [500.0], "end_s": [530.0]})
    dur = pd.Series({"a": 3600.0, "b": 3600.0})
    m = evaluate(win, p, sz, dur, theta=0.5, k=1)
    assert m["sensitivity"] == 1 and m["n_false_alarms"] == 0 and m["hours"] == 2
    assert m["median_delay_s"] == 4.0  # first window [500,504] decides at 504

"""Event-level seizure detection scoring.

Everything is computed per recording (files are not concatenated: there are gaps
between CHB-MIT files and the filter restarts in each file).

Post-processing (causal, usable online):
  * p_smooth[i] = mean of the window probabilities i-k+1 .. i  (fewer at file start)
  * decision at time t_i = start_i + WIN_S (the moment window i is complete): p_smooth >= theta
  * an alarm EVENT is a chain of positive decisions where consecutive positives are at
    most MERGE_GAP_S apart; the alarm is RAISED at the first decision time of the chain.
    The event's data span is [start of its first window, end of its last window].

Scoring:
  * a seizure [onset, offset] is DETECTED if the data span of any event overlaps it;
    detection delay = max(0, raise time of the first such event - onset)
  * an event overlapping no annotated seizure is a FALSE ALARM
  * FA/h = false alarms / total recorded hours (all hours of the evaluated recordings)
  * sensitivity = detected seizures / annotated seizures
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import WIN_S

MERGE_GAP_S = 30.0


def causal_smooth(p: np.ndarray, k: int) -> np.ndarray:
    if k <= 1:
        return np.asarray(p, float)
    c = np.cumsum(np.insert(np.asarray(p, float), 0, 0.0))
    i = np.arange(1, len(p) + 1)
    lo = np.maximum(i - k, 0)
    return (c[i] - c[lo]) / (i - lo)


def alarm_events(starts: np.ndarray, decision: np.ndarray, win_s: float = WIN_S,
                 merge_gap_s: float = MERGE_GAP_S) -> list[tuple[float, float, float]]:
    """Return [(raise_time, data_start, data_end), ...] for one recording."""
    idx = np.flatnonzero(decision)
    if len(idx) == 0:
        return []
    t = starts[idx] + win_s
    breaks = np.flatnonzero(np.diff(t) > merge_gap_s)
    first = np.r_[0, breaks + 1]
    last = np.r_[breaks, len(idx) - 1]
    return [(float(t[a]), float(starts[idx[a]]), float(t[b])) for a, b in zip(first, last)]


@dataclass
class RecordScore:
    n_seizures: int
    n_detected: int
    n_false_alarms: int
    delays: list
    hours: float


def score_record(starts, decision, seizures, hours) -> RecordScore:
    ev = alarm_events(np.asarray(starts, float), np.asarray(decision, bool))
    used = np.zeros(len(ev), bool)
    det, delays = 0, []
    for on, off in seizures:
        hits = [i for i, (_, a, b) in enumerate(ev) if a < off and b > on]
        if hits:
            det += 1
            delays.append(max(0.0, ev[hits[0]][0] - on))
            used[hits] = True
    return RecordScore(len(seizures), det, int((~used).sum()), delays, hours)


def evaluate(win: pd.DataFrame, prob: np.ndarray, seizures: pd.DataFrame, durations: pd.Series,
             theta: float, k: int) -> dict:
    """win: windows (record, start_s) row-aligned with prob; seizures: record,start_s,end_s;
    durations: record -> seconds. Returns pooled event-level metrics."""
    szmap = {r: list(zip(g.start_s, g.end_s)) for r, g in seizures.groupby("record")}
    tot = RecordScore(0, 0, 0, [], 0.0)
    per_patient: dict = {}
    w = win.assign(_p=prob)
    for rec, g in w.groupby("record", observed=True, sort=False):
        ps = causal_smooth(g._p.to_numpy(), k)
        s = score_record(g.start_s.to_numpy(), ps >= theta, szmap.get(rec, []),
                         durations[rec] / 3600.0)
        pt = g.patient.iloc[0]
        for acc in (tot, per_patient.setdefault(pt, RecordScore(0, 0, 0, [], 0.0))):
            acc.n_seizures += s.n_seizures
            acc.n_detected += s.n_detected
            acc.n_false_alarms += s.n_false_alarms
            acc.delays += s.delays
            acc.hours += s.hours
    return {**summarize(tot), "per_patient": {p: summarize(v) for p, v in per_patient.items()}}


def summarize(s: RecordScore) -> dict:
    return {
        "n_seizures": s.n_seizures, "n_detected": s.n_detected,
        "sensitivity": s.n_detected / s.n_seizures if s.n_seizures else float("nan"),
        "n_false_alarms": s.n_false_alarms, "hours": s.hours,
        "fa_per_hour": s.n_false_alarms / s.hours if s.hours else float("nan"),
        "median_delay_s": float(np.median(s.delays)) if s.delays else float("nan"),
    }

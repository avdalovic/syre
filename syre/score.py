"""Count detections the Sherlock way. Attack labels are used here only."""
from __future__ import annotations

import numpy as np
import pandas as pd

CLASSES = ("Industroyer", "Drift Off", "Control & Freeze", "ARP DoS")


def alarm_starts(alarm: np.ndarray) -> list[int]:
    a = np.asarray(alarm).astype(np.int8)
    return np.flatnonzero(np.diff(np.r_[0, a]) == 1).tolist()


def count(name: str, alarm: np.ndarray, windows) -> dict:
    """An attack is detected if an alarm starts inside it. Starts in a recovery phase are ignored.
    Every other start is a false alarm. TTD = first start inside the attack minus attack start."""
    attacks = [w for w in windows if not w.benign]
    benign = [w for w in windows if w.benign]
    ttd, false = {}, []
    for s in alarm_starts(alarm):
        hit = next((w for w in attacks if w.start <= s <= w.end), None)
        if hit is not None:
            ttd.setdefault(hit.event_id, s - hit.start)
        elif not any(w.end < s <= w.recovery_end for w in attacks):
            false.append(s)
    row = {"detector": name, "detected": f"{len(ttd)}/{len(attacks)}"}
    for c in CLASSES:
        ws = [w for w in attacks if w.cls == c]
        row[c] = f"{sum(w.event_id in ttd for w in ws)}/{len(ws)}"
    row["false alarms"] = len(false)
    row["during a benign event"] = sum(any(w.start <= s <= w.end for w in benign) for s in false)
    row["mean TTD s"] = round(float(np.mean(list(ttd.values()))), 1) if ttd else np.nan
    return row


def table(alarms: dict[str, np.ndarray], windows) -> pd.DataFrame:
    return pd.DataFrame([count(k, v, windows) for k, v in alarms.items()])

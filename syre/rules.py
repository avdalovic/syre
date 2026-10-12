"""The two control-center rules."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd


def command_rule(changes: pd.DataFrame, commands: pd.DataFrame, grid: np.ndarray, window_s: float) -> np.ndarray:
    """Alarm when a switch or element reports a new state with no matching command in the previous window_s."""
    ch = changes[changes["name"].str.match(r"^(switch|load|sgen|storage)\.\d+:is_(closed|connected)$")]
    alarm = np.zeros(len(grid), np.int8)
    for r in ch.itertuples():
        element, attr = r.name.split(":")[0], "closed" if r.name.endswith("is_closed") else "connected"
        m = commands[(commands["name"] == f"{element}:{attr}@cmd") & (commands["t"] <= r.t + 0.5) & (commands["t"] >= r.t - window_s)]
        k = math.floor(r.t - grid[0])
        if not (pd.to_numeric(m["value"], errors="coerce") == r.new).any() and 0 <= k < len(grid):
            alarm[k] = 1
    return alarm


def update_rule(cc_train: dict, cc_test: dict, grid: np.ndarray, gap_s: float) -> np.ndarray:
    """Alarm when a measurement (one that is sent periodically in training) has no periodic update for gap_s."""
    sent = {n for j, n in enumerate(cc_train["names"]) if np.isfinite(cc_train["periodic"][:, j]).any()}
    idx = [j for j, n in enumerate(cc_test["names"]) if n in sent]
    with np.errstate(invalid="ignore"):
        return ((grid[:, None] - cc_test["periodic"][:, idx]) > gap_s).any(axis=1).astype(np.int8)

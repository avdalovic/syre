"""Learn the laws on training data: one PySR fit per law type on rows pooled over its instances,
then a gate, then the simplest law that passes. Needs `pip install pysr` (and Julia, installed by PySR)."""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from .grid import TYPES, columns
from .laws import predict, terms


def cusum_alarms(r_fit: np.ndarray, r_hold: np.ndarray, s: float, g: float) -> int:
    """Holdout alarms of a CUSUM fit on the fit residuals (delta = mean + std, threshold = s * max fit CUSUM)."""
    pos = lambda v: v if np.isfinite(v) and v > 0 else 1e-6  # noqa: E731
    delta = pos(float(np.mean(r_fit) + np.std(r_fit)))
    c = peak = 0.0
    for v in r_fit:
        c = max(0.0, c + v - delta)
        peak = max(peak, c)
    thr = pos(s * peak)
    cap, c, n = pos(thr + g * delta), 0.0, 0
    for v in r_hold:
        c = min(max(0.0, c + v - delta), cap)
        n += c > thr
    return n


def rows(graph, ttype: str, stride: int, holdout: float):
    """Every stride-th aligned training second of each instance; the last share of each instance is the holdout."""
    fit, hold = [], []
    for inst in graph.instances(ttype):
        m = np.flatnonzero(inst.ok & inst.on)[::stride]
        nf = int((1 - holdout) * len(m))
        fit.append((inst.name, inst.X[m[:nf]], inst.y[m[:nf]]))
        hold.append((inst.name, inst.X[m[nf:]], inst.y[m[nf:]]))
    X = np.vstack([p[1] for p in fit] + [p[1] for p in hold])
    y = np.concatenate([p[2] for p in fit] + [p[2] for p in hold])
    n_fit = sum(len(p[2]) for p in fit)
    slices, f0, h0 = [], 0, n_fit
    for (name, _, yf), (_, _, yh) in zip(fit, hold):
        slices.append((name, np.r_[np.arange(f0, f0 + len(yf)), np.arange(h0, h0 + len(yh))]))
        f0, h0 = f0 + len(yf), h0 + len(yh)
    return X, y, n_fit, [s for s in slices if len(s[1])]


def exact(expr: str, cols: list[str], X: np.ndarray, y: np.ndarray, slices, rel: float) -> int:
    """Number of instances where the median error exceeds rel x the largest median |term|."""
    vals = terms(expr, cols, X)
    err = np.abs(y - np.sum(vals, axis=0))
    bad = 0
    for _, idx in slices:
        largest = max([float(np.median(np.abs(y[idx])))] + [float(np.median(np.abs(v[idx]))) for v in vals])
        bad += float(np.median(err[idx])) > rel * largest
    return bad


def pysr_front(X: np.ndarray, y: np.ndarray, n_fit: int, cols: list[str], ttype: str, f: dict) -> pd.DataFrame:
    from pysr import PySRRegressor
    seed = f["seed"] + sum((i + 1) * ord(ch) for i, ch in enumerate(ttype)) % 100000
    model = PySRRegressor(niterations=f["niterations"], binary_operators=["+", "-", "*"], unary_operators=[], maxsize=f["max_complexity"],
                          populations=f["populations"], parsimony=f["parsimony"], procs=f["procs"], timeout_in_seconds=int(f["timeout_minutes"] * 60),
                          temp_equation_file=True, random_state=seed, model_selection="score", verbosity=0, progress=False)
    xf, xh = pd.DataFrame(X[:n_fit], columns=cols), pd.DataFrame(X[n_fit:], columns=cols)
    model.fit(xf, y[:n_fit])
    front = model.equations_.reset_index(drop=True).copy()
    mse = []
    for i in range(len(front)):
        e = np.asarray(model.predict(xh, index=i), dtype=float) - y[n_fit:]
        mse.append(float(np.mean(e[np.isfinite(e)] ** 2)) if np.isfinite(e).any() else np.inf)
    front["holdout_mse"] = mse
    return front


def gate(front: pd.DataFrame, ttype: str, X, y, n_fit: int, slices, f: dict) -> list[dict]:
    """Keep a candidate if it uses an input, has no holdout CUSUM alarm and is exact at every instance."""
    cols, trail = columns(ttype), []
    for _, c in front.iterrows():
        expr, cx = str(c["sympy_format"]), float(c["complexity"])
        rec = {"type": ttype, "law": expr, "complexity": cx, "holdout_mse": c["holdout_mse"]}
        r = np.abs(y - predict(expr, cols, X))
        if not set(re.findall(r"\b[A-Za-z_]\w*\b", expr)) & set(cols):
            rec["result"] = "no input"
        elif cx > f["max_complexity"] or not np.isfinite(r).all():
            rec["result"] = "complexity or nonfinite"
        elif cusum_alarms(r[:n_fit], r[n_fit:], f["gate_s"], f["gate_g"]):
            rec["result"] = "holdout CUSUM alarm"
        elif exact(expr, cols, X, y, slices, f["exact_rel"]):
            rec["result"] = "not exact at every instance"
        else:
            rec["result"] = "passes"
        trail.append(rec)
    return trail


def simplest(trail: list[dict]) -> str | None:
    ok = [t for t in trail if t["result"] == "passes"]
    return min(ok, key=lambda t: (t["complexity"], t["holdout_mse"]))["law"] if ok else None


def fit_laws(f: dict, graph, out) -> dict[str, str]:
    chosen, trail = {}, []
    for ttype in TYPES:
        X, y, n_fit, slices = rows(graph, ttype, f["stride"], f["holdout"])
        print(f"{ttype}: {n_fit} fit rows, {len(y) - n_fit} holdout rows; PySR ...", flush=True)
        front = pysr_front(X, y, n_fit, columns(ttype), ttype, f)
        front.to_csv(out / f"front_{ttype}.csv", index=False)
        t = gate(front, ttype, X, y, n_fit, slices, f)
        trail += t
        if law := simplest(t):
            chosen[ttype] = law
        print(f"  -> {law or 'no law passes'}", flush=True)
    pd.DataFrame(trail).to_csv(out / "gate.csv", index=False)
    return chosen

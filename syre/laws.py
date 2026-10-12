"""Score the laws: normalized residual per law instance, one conformal threshold on the maximum."""
from __future__ import annotations

import math

import numpy as np
import sympy as sp

from .grid import columns


def terms(expr: str, cols: list[str], X: np.ndarray) -> list[np.ndarray]:
    """Value of each additive term of the law, per second."""
    syms = [sp.Symbol(c) for c in cols]
    out = []
    for term in sp.Add.make_args(sp.expand(sp.sympify(expr, locals=dict(zip(cols, syms))))):
        with np.errstate(all="ignore"):
            v = np.asarray(sp.lambdify(syms, term, "numpy")(*X.T), dtype=float)
        out.append(np.broadcast_to(v, (X.shape[0],)) if v.ndim == 0 else v)
    return out


def predict(expr: str, cols: list[str], X: np.ndarray) -> np.ndarray:
    syms = sp.symbols(cols)
    with np.errstate(all="ignore"):
        out = np.asarray(sp.lambdify(syms, sp.sympify(expr, locals=dict(zip(cols, syms))), "numpy")(*X.T), dtype=float)
    return np.broadcast_to(out, (X.shape[0],)).copy() if out.ndim == 0 else out


def rel(r: np.ndarray, size: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(size > 0, r / np.where(size > 0, size, 1.0), np.where(r > 0, np.inf, 0.0))


def tau_conf(scores: np.ndarray, alpha: float) -> float:
    """The ceil((n+1)(1-alpha))-th smallest training score; +inf if n is too small."""
    s = np.sort(scores)
    k = math.ceil((len(s) + 1) * (1.0 - alpha) - 1e-9)
    return math.inf if k > len(s) else float(s[max(k, 1) - 1])


def residuals(expr: str, cols: list[str], inst) -> dict:
    X = np.nan_to_num(inst.X)
    r = np.abs(inst.y - predict(expr, cols, X))
    M = np.max(np.vstack([np.abs(np.nan_to_num(inst.y))] + [np.abs(v) for v in terms(expr, cols, X)]), axis=0)
    return {"r": r, "M": M, "ok": inst.ok & inst.on & np.isfinite(r)}


def score(laws: dict[str, str], g_train, g_test, alpha: float, floor_pct: float, rho_q: float) -> dict:
    """z = |y - f(X)| / max(M, floor) / rho; M = largest |term| now; floor and rho per law type, from training."""
    out, ztr = [], []
    for ttype, expr in laws.items():
        cols = columns(ttype)
        group = []
        for i, j in zip(g_train.instances(ttype), g_test.instances(ttype), strict=True):
            assert i.name == j.name, (i.name, j.name)
            group.append((i, residuals(expr, cols, i), residuals(expr, cols, j)))
        floor = float(np.percentile(np.concatenate([tr["M"][tr["ok"]] for _, tr, _ in group]), floor_pct))
        rho = float(np.quantile(np.concatenate([rel(tr["r"][tr["ok"]], np.maximum(tr["M"][tr["ok"]], floor)) for _, tr, _ in group]), rho_q))
        for inst, tr, te in group:
            ztr.append(np.where(tr["ok"], rel(tr["r"], np.maximum(tr["M"], floor)) / rho, 0.0))
            z = np.where(te["ok"], rel(te["r"], np.maximum(te["M"], floor)) / rho, 0.0)
            out.append({"type": ttype, "instance": inst.name, "expr": expr, "floor": floor, "rho": rho, "z": z})
    tau = tau_conf(np.vstack(ztr).max(axis=0), alpha)
    Z = np.vstack([o["z"] for o in out])
    return {"instances": out, "Z": Z, "tau": tau, "alarm": (Z.max(axis=0) > tau).astype(np.int8)}

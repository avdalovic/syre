"""Detection trace of one law instance around one attack (paper Fig. 5 style)."""
from __future__ import annotations

import re

import numpy as np
import sympy as sp

from . import laws
from .grid import columns

PINK, YELLOW, MEAS, BLUE, GREEN, RED, GREY = "#f8d0d6", "#f7ebcb", "#000000", "#1f5fbf", "#2ca02c", "#d62728", "#555555"
UNIT = {"P": (1e-3, "P [kW]"), "Q": (1e-3, "Q [kvar]"), "V": (1.0, "V [p.u.]"), "I": (1.0, "I [A]")}
BEFORE, AFTER, HOLD = 1800, 2400, 10


def latex(ttype: str, expr: str) -> str:
    def name(s: str) -> str:
        m = re.fullmatch(r"S_(\w+)_([PQIV])", s)
        if m:
            return rf"\sum {m.group(2)}_{{\mathrm{{{m.group(1)}}}}}"
        return rf"{s[-1]}_{{\mathrm{{bus}}}}"
    cols = columns(ttype)
    syms = {c: sp.Symbol(c) for c in cols}
    rhs = sp.latex(sp.sympify(expr, locals=syms), symbol_names={s: name(c) for c, s in syms.items()})
    lhs = rf"{ttype[-1]}_{{\mathrm{{{'bus' if ttype.startswith('bus') else 'end'}}}}}"
    return f"${lhs} = {rhs}$"


def held(z: np.ndarray, ok: np.ndarray, on: np.ndarray) -> np.ndarray:
    """Each checked value held until the next checked second (at most HOLD s); gaps stay empty."""
    v = np.where(ok, z, np.nan)
    idx = np.where(np.isfinite(v), np.arange(len(v)), 0)
    np.maximum.accumulate(idx, out=idx)
    out = v[idx]
    out[(np.arange(len(v)) - idx > HOLD) | ~on] = np.nan
    return out


def trace(cfg, scen, g_tr, g_te, ttype: str, name: str, attack: str, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    d = cfg["detector"]
    res = laws.score(cfg["laws"], g_tr, g_te, d["alpha"], d["floor_percentile"], d["rho_quantile"])
    meta = next(i for i in res["instances"] if i["type"] == ttype and i["instance"] == name)
    inst = next(i for i in g_te.instances(ttype) if i.name == name)
    cols, expr = columns(ttype), meta["expr"]
    rr = laws.residuals(expr, cols, inst)
    # display: the 2 s rule over the law's own inputs only (identical z wherever the detector checks the law)
    used = {str(s) for s in sp.sympify(expr).free_symbols}
    ok = g_te.aligned(inst.target, [m for m in inst.members if m[1] in used]) & inst.on & np.isfinite(rr["r"])
    z = held(np.where(ok, laws.rel(rr["r"], np.maximum(rr["M"], meta["floor"])) / meta["rho"], 0.0), ok, inst.on)

    t0 = float(g_te.a.grid[0])
    windows = [w for w in scen.windows(t0) if not w.benign]
    w = next(x for x in windows if x.event_id == attack)
    lo, hi = max(w.start - BEFORE, 0), min(w.end + AFTER, len(z) - 1)
    sec = np.arange(lo, hi + 1)
    t = (sec - lo) / 3600
    unit, ylabel = UNIT[ttype[-1]]
    yhat = laws.predict(expr, cols, np.nan_to_num(inst.X))

    plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "dejavuserif", "font.size": 10})
    fig, ax = plt.subplots(3, 1, figsize=(5.0, 4.9), sharex=True, gridspec_kw={"height_ratios": [1.15, 1, 1]})
    for a in ax:
        for x in windows:
            if x.end >= lo and x.start <= hi:
                a.axvspan((max(x.start, lo) - lo) / 3600, (min(x.end, hi) - lo) / 3600, color=PINK if x is w else YELLOW, lw=0, zorder=0)
        a.margins(x=0)
    ax[0].plot(t, inst.y[sec] * unit, color=MEAS, lw=1.4)
    ax[0].plot(t, yhat[sec] * unit, color=BLUE, lw=1.2, ls=(0, (3, 2)))
    ax[0].set_ylabel(ylabel)
    ax[1].plot(t, np.abs(inst.y[sec] - yhat[sec]) * unit, color=GREY, lw=0.8)
    ax[1].set_ylabel("|err|")
    ax[1].set_ylim(bottom=0)
    ax[2].plot(t, np.clip(z[sec], 3e-3, None), color=GREEN, lw=1.1, drawstyle="steps-post")
    ax[2].axhline(res["tau"], color=RED, lw=1.2, ls=(0, (4, 2)))
    ax[2].set_yscale("log")
    ax[2].set_ylim(3e-3, max(np.nanmax(z[sec]) * 4, res["tau"] * 20))
    ax[2].set_yticks([1e-2, 1e0, 1e2])
    ax[2].set_ylabel("$z$")
    ax[2].set_xlabel("time [h]")
    for a in ax:
        a.yaxis.set_label_coords(-0.12, 0.5)
    handles = [Patch(color=PINK, label="sensor attacked"), Patch(color=YELLOW, label="other attack"),
               Line2D([], [], color=MEAS, lw=1.4, label="measured"), Line2D([], [], color=BLUE, lw=1.2, ls=(0, (3, 2)), label="law prediction"),
               Line2D([], [], color=GREEN, lw=1.1, label="residual"), Line2D([], [], color=RED, lw=1.2, ls=(0, (4, 2)), label="alert threshold")]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False, fontsize=8.5, bbox_to_anchor=(0.54, 0.935), handlelength=1.6, columnspacing=1.0)
    fig.text(0.54, 0.985, latex(ttype, expr), ha="center", va="top", fontsize=11)
    fig.subplots_adjust(left=0.16, right=0.97, top=0.83, bottom=0.10, hspace=0.12)
    path = out / f"trace_attack{attack}_{ttype}_{re.sub(r'[^A-Za-z0-9]+', '_', name).strip('_')}.pdf"
    fig.savefig(path)
    fig.savefig(path.with_suffix(".png"), dpi=300)
    return path

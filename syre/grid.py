"""Grid map from the scenario drawing, and the per-bus neighbour sums that the laws read."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

ATTR = {"P": "active_power", "Q": "reactive_power", "I": "current", "V": "voltage"}
SUMS = [f"S_{g}_{q}" for q in ("P", "Q") for g in ("line", "trafo", "load", "gen", "storage")] + \
       [f"S_{g}_{q}" for q in ("I", "V") for g in ("line", "trafo")]
TYPES = ["bus_P", "bus_Q", "bus_V", "end_P", "end_Q", "end_I", "end_V"]
KIND_ORDER = {"load": 0, "sgen": 1, "storage": 2, "ext_grid": 3}
NUM = r"[-\d.e]+"


def columns(ttype: str) -> list[str]:
    """Inputs of a law type: every neighbour sum plus the bus's own P, Q, V (not the target itself)."""
    return SUMS + [f"bus_{q}" for q in ("P", "Q", "V") if f"bus_{q}" != ttype]


def _pts(d: str) -> np.ndarray:
    return np.array([tuple(map(float, p.split(","))) for p in re.findall(rf"{NUM},{NUM}", d)])


def extract_map(svg: Path) -> dict[str, pd.DataFrame]:
    """Lines (end buses, end switches), transformers (hv/lv bus) and elements (bus) from docs/power-grid.svg."""
    text = svg.read_text()
    buses: dict[int, list] = {}
    for m in re.finditer(rf'<rect x="({NUM})" y="({NUM})" width="({NUM})" height="({NUM})"[^>]*grid-element bus[^>]*data-element="bus\.(\d+)"', text):
        x, y, w, h = map(float, m.groups()[:4])
        buses.setdefault(int(m.group(5)), []).append((x, y, x + w, y + h))

    def bus_at(p, tol=1.0):
        hits = [b for b, rects in buses.items() for (x0, y0, x1, y1) in rects if x0 - tol <= p[0] <= x1 + tol and y0 - tol <= p[1] <= y1 + tol]
        return hits[0] if len(hits) == 1 else (None if not hits else -1)

    lines = {int(m.group(2)): _pts(m.group(1)) for m in re.finditer(r'<path d="([^"]+)"[^>]*grid-element line[^>]*data-element="line\.(\d+)"', text)}
    switches = {int(m.group(3)): np.array([float(m.group(1)), float(m.group(2))])
                for m in re.finditer(rf'<g transform="translate\(({NUM}), ({NUM})\)" class="vis-element grid-element switch" data-element="switch\.(\d+)"', text)}
    lt = pd.DataFrame([{"line": k, "from_bus": bus_at(p[0]), "to_bus": bus_at(p[-1])} for k, p in sorted(lines.items())]).set_index("line")

    # a switch belongs to the nearest line path; its end is "from" if it lies in the first half of the path
    on_end: dict[tuple[int, str], list[int]] = {}
    for s, p in sorted(switches.items()):
        best = (np.inf, None, None)
        for k, q in lines.items():
            lengths = np.r_[0, np.cumsum(np.linalg.norm(np.diff(q, axis=0), axis=1))]
            for i in range(len(q) - 1):
                ab, ap = q[i + 1] - q[i], p - q[i]
                t = float(np.clip(ap @ ab / max(ab @ ab, 1e-12), 0, 1))
                d = float(np.linalg.norm(ap - t * ab))
                if d < best[0]:
                    best = (d, k, (lengths[i] + t * (lengths[i + 1] - lengths[i])) / max(lengths[-1], 1e-9))
        d, k, pos = best
        if k is not None and d < 1.0:
            on_end.setdefault((k, "from" if pos < 0.5 else "to"), []).append(s)
    for end in ("from", "to"):
        lt[f"switch_{end}"] = [(on_end.get((k, end)) or [np.nan])[0] if len(on_end.get((k, end), [])) <= 1 else -1 for k in lt.index]

    trows = []
    for m in re.finditer(rf'<g transform="translate\(({NUM}), ({NUM})\)" class="vis-element grid-element transformer" data-element="trafo\.(\d+)"[^>]*>(.*?)</g>', text, flags=re.S):
        ends = [_pts(d)[0] + np.array([float(m.group(1)), float(m.group(2))]) for d in re.findall(r'<path d="([^"]+)"', m.group(4))]
        if len(ends) == 2:
            up, lo = sorted(ends, key=lambda e: e[1])  # hv = upper end
            trows.append({"trafo": int(m.group(3)), "hv_bus": bus_at(up), "lv_bus": bus_at(lo)})
    tt = pd.DataFrame(trows).sort_values("trafo").set_index("trafo")

    # an element sits at the bus bar at the far end of the connector that starts at its symbol
    elems = {}
    for m in re.finditer(r'<path d="([^"]+)"[^>]*grid-element (load|external_grid)[^>]*data-element="((?:load|external_grid)\.\d+)"', text):
        elems[m.group(3)] = _pts(m.group(1)).mean(axis=0)
    for m in re.finditer(rf'<g id="((?:sgen|storage)\.\d+)" transform="translate\(({NUM}), ({NUM})\)"', text):
        elems[m.group(1)] = np.array([float(m.group(2)), float(m.group(3))])
    connectors = [_pts(m.group(1)) for m in re.finditer(r'<path d="([^"]+)"[^>]*class="vis-element grid-connector', text)]
    erows = []
    for name, p in elems.items():
        best = (np.inf, None)
        for c in connectors:
            for a, b in ((c[0], c[-1]), (c[-1], c[0])):
                d = float(np.linalg.norm(a - p))
                if d < best[0]:
                    best = (d, b)
        kind, idx = name.split(".")
        kind = "ext_grid" if kind == "external_grid" else kind
        erows.append({"kind": kind, "index": int(idx), "bus": bus_at(best[1]) if best[1] is not None and best[0] < 15 else None})
    et = pd.DataFrame(erows)
    et = et.iloc[sorted(range(len(et)), key=lambda i: (KIND_ORDER[et.kind.iloc[i]], et["index"].iloc[i]))].reset_index(drop=True)
    return {"lines": lt, "trafos": tt, "elements": et}


@dataclass
class Instance:
    name: str  # e.g. "bus.14" or "line.7 to (bus 10)"
    target: str  # IPAL point the law predicts
    y: np.ndarray
    X: np.ndarray  # neighbour sums, columns = columns(type)
    ok: np.ndarray  # all inputs present and received within align_s
    on: np.ndarray  # line end connected (always true for a bus)
    members: list = field(default_factory=list)  # (point, column, state point or None)


class Graph:
    """Neighbour sums per bus and line end, with the members that are connected at each second."""

    def __init__(self, state, grid_map: dict[str, pd.DataFrame], align_s: float):
        self.a, self.align_s = state, align_s
        self.lines, self.trafos, self.elems = grid_map["lines"], grid_map["trafos"], grid_map["elements"]
        self.buses = sorted(int(n.split(".")[1].split(":")[0]) for n in state.col if re.fullmatch(r"bus\.\d+:active_power", n))
        self._state: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    def connected(self, point: str) -> tuple[np.ndarray, np.ndarray]:
        """(connected, known) for a switch is_closed or element is_connected point."""
        if point not in self._state:
            v = self.a.v(point)
            self._state[point] = (v >= 0.5, np.isfinite(v))
        return self._state[point]

    def aligned(self, target: str, members: list) -> np.ndarray:
        """The 2 s rule: target and every connected member present and received within align_s."""
        n = len(self.a.grid)
        rmin, rmax = self.a.r(target).copy(), self.a.r(target).copy()
        ok = np.isfinite(self.a.v(target)) & np.isfinite(rmin)
        for point, _, st in members:
            conn, known = (np.ones(n, bool), np.ones(n, bool)) if st is None else self.connected(st)
            ok &= known & (~conn | (np.isfinite(self.a.v(point)) & np.isfinite(self.a.r(point))))
            rmin = np.where(conn, np.fmin(rmin, self.a.r(point)), rmin)
            rmax = np.where(conn, np.fmax(rmax, self.a.r(point)), rmax)
        with np.errstate(invalid="ignore"):
            return ok & ((rmax - rmin) <= self.align_s)

    def sums(self, members: list, cols: list[str]) -> np.ndarray:
        n = len(self.a.grid)
        out = {c: np.zeros(n) for c in cols}
        for point, col, st in members:
            conn = np.ones(n, bool) if st is None else self.connected(st)[0]
            v = self.a.v(point)
            out[col] += np.where(conn & np.isfinite(v), v, 0.0)
        return np.column_stack([out[c] for c in cols])

    def instances(self, ttype: str):
        n = len(self.a.grid)
        q_t, cols = ttype[-1], columns(ttype)
        has = lambda p: p in self.a.col  # noqa: E731
        for b in self.buses:
            ends = [(int(k), end, None if pd.isna(ln[f"switch_{end}"]) else int(ln[f"switch_{end}"]))
                    for k, ln in self.lines.iterrows() for end in ("from", "to") if ln[f"{end}_bus"] == b]
            targets = [(f"bus.{b}", f"bus.{b}:{ATTR[q_t]}", None)] if ttype.startswith("bus_") else \
                      [(f"line.{k} {end} (bus {b})", f"line.{k}:{ATTR[q_t]}_{end}", (k, end, sw)) for k, end, sw in ends]
            for name, target, own in targets:
                if not has(target):
                    continue
                members = []
                for k, end, sw in ends:
                    if own is not None and (k, end) == own[:2]:
                        continue  # leave the target out
                    members += [(f"line.{k}:{ATTR[q]}_{end}", f"S_line_{q}", None if sw is None else f"switch.{sw}:is_closed") for q in "PQIV"]
                for t, tr in self.trafos.iterrows():
                    for side in ("hv", "lv"):
                        if tr[f"{side}_bus"] == b:
                            members += [(f"trafo.{t}:{ATTR[q]}_{side}", f"S_trafo_{q}", None) for q in "PQIV"]
                for _, e in self.elems[self.elems.bus == b].iterrows():
                    for q in "PQ":
                        if e.kind == "ext_grid":
                            members.append((f"external_grid.{int(e['index'])}:{ATTR[q]}", f"S_gen_{q}", None))
                        else:
                            el = f"{e.kind}.{int(e['index'])}"
                            group = {"load": "load", "sgen": "gen", "storage": "storage"}[e.kind]
                            members.append((f"{el}:{ATTR[q]}", f"S_{group}_{q}", f"{el}:is_connected"))
                members += [(f"bus.{b}:{ATTR[q]}", f"bus_{q}", None) for q in "PQV" if f"bus_{q}" in cols]
                members = [m for m in members if has(m[0]) and (m[2] is None or has(m[2]))]
                on = np.ones(n, bool) if (own is None or own[2] is None) else self.connected(f"switch.{own[2]}:is_closed")[0]
                yield Instance(name, target, self.a.v(target), self.sums(members, cols), self.aligned(target, members), on, members)

"""Load a Sherlock scenario: IPAL state, inferred receive times, attack windows, control-center log."""
from __future__ import annotations

import gzip
import json
import math
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ATTACK_CLASSES = {"industroyer": "Industroyer", "control-and-freeze": "Control & Freeze",
                  "drift-off": "Drift Off", "arp-spoof": "ARP DoS"}


@dataclass
class Window:
    event_id: str
    cls: str
    benign: bool
    start: int  # test second
    end: int
    recovery_end: int


class State:
    """IPAL state at 1 s. Receive time of a point = last second its RTU sent a changed measurement."""

    def __init__(self, names: list[str], stamps: np.ndarray, values: np.ndarray, point_map: dict):
        self.names, self.grid, self.values = names, stamps, values
        self.col = {n: i for i, n in enumerate(names)}
        rtu = {f"{v['element']}:{v['attribute']}": k.split(".")[0] for k, v in point_map.items()}
        meas = {f"{v['element']}:{v['attribute']}" for v in point_map.values() if v["context"] == "MEASUREMENT"}
        self.recv = np.full(values.shape, np.nan)
        for r in sorted({rtu[n] for n in names if n in rtu}):
            pts = [self.col[n] for n in names if rtu.get(n) == r]
            mpts = [self.col[n] for n in names if rtu.get(n) == r and n in meas]
            if not mpts:
                continue
            V = values[:, mpts]
            with np.errstate(invalid="ignore"):
                tick = np.r_[False, ((V[1:] != V[:-1]) & np.isfinite(V[1:])).any(axis=1)]
            last = pd.Series(np.where(tick, stamps, np.nan)).ffill().to_numpy()
            self.recv[:, pts] = last[:, None]

    def v(self, name: str) -> np.ndarray:
        return self.values[:, self.col[name]]

    def r(self, name: str) -> np.ndarray:
        return self.recv[:, self.col[name]]


class Scenario:
    def __init__(self, cfg: dict, data_root: Path):
        self.name = cfg["scenario"]
        self.dir = Path(data_root) / self.name
        self.cache = self.dir / "cache"
        self.state_file = cfg["state_file"]

    def point_map(self) -> dict:
        return json.loads((self.dir / "raw" / "test" / "data-point-map.json").read_text())

    def state(self, split: str) -> State:
        path = self.cache / f"state_{split}.npz"
        if not path.exists():
            cache_state(self.dir / self.state_file.format(split=split), path)
        d = np.load(path)
        return State(d["names"].tolist(), d["stamps"], d["values"], self.point_map())

    def windows(self, t0: float) -> list[Window]:
        out = []
        for e in json.loads((self.dir / "ipal" / "test" / "events.json").read_text()):
            benign = "benign" in str(e["id"])
            key = str(e["description"]).split(":", 1)[0]
            start, end = math.floor(e["start"] - t0), math.floor(e["end"] - t0)
            rec = math.floor(e["recovery"] - t0) if e.get("recovery") else end
            out.append(Window(str(e["id"]), "Benign event" if benign else ATTACK_CLASSES.get(key, key), benign, start, end, rec))
        return out

    def control_center(self, split: str, grid: np.ndarray) -> dict:
        """Periodic receive times of measurements, reported state changes and commands (for the two rules)."""
        base = self.cache / f"cc_{split}"
        if not (base.with_suffix(".npz")).exists():
            cache_control_center(self.dir / "raw" / split / "control-center.zip", self.point_map(), grid, base)
        d = np.load(base.with_suffix(".npz"))
        return {"names": d["names"].tolist(), "periodic": d["periodic"],
                "changes": pd.read_csv(f"{base}_changes.csv"), "commands": pd.read_csv(f"{base}_commands.csv")}


def cache_state(src: Path, dst: Path) -> None:
    """IPAL state file -> npz (union of keys in first-seen order; booleans -> 0/1; None -> NaN)."""
    names, index, stamps, rows = [], {}, [], []
    with gzip.open(src, "rt") as handle:
        for line in handle:
            rec = json.loads(line)
            for k in rec["state"]:
                if k not in index:
                    index[k] = len(names)
                    names.append(k)
            row = np.full(len(names), np.nan)
            for k, v in rec["state"].items():
                row[index[k]] = np.nan if v is None else float(v)
            stamps.append(rec["timestamp"])
            rows.append(row)
    values = np.full((len(rows), len(names)), np.nan)
    for i, r in enumerate(rows):
        values[i, :len(r)] = r
    dst.parent.mkdir(parents=True, exist_ok=True)
    np.savez(dst, names=np.array(names), stamps=np.asarray(stamps, dtype=np.float64), values=values)


def cache_control_center(src: Path, point_map: dict, grid: np.ndarray, base: Path) -> None:
    names = {k: f"{v['element']}:{v['attribute']}" + ("@cmd" if v["context"] == "CONFIGURATION" else "") for k, v in point_map.items()}
    rec, cmd = [], []
    with zipfile.ZipFile(src) as z, z.open("control-center/ccx-notifications.jsonl") as handle:
        for line in handle:
            if b"DATA_POINT_RECEIVED" in line:
                d = json.loads(line)
                p = d["data"]
                rec.append((p["data_point_identifier"], p["protocol_data"]["cause_of_transmission"], p["protocol_data"]["type_id"],
                            d["timestamp"], np.nan if p["value"] is None else float(p["value"])))
            elif b"DATA_POINT_COMMAND_REPLY" in line:
                d = json.loads(line)
                p = d["data"]
                v = p["value"]
                v = float(v) if isinstance(v, (bool, int, float)) else (v.get("__wattson.data__", json.dumps(v))[:60] if isinstance(v, dict) else v)
                cmd.append((p["data_point_identifier"], d["timestamp"], v))
    rec = pd.DataFrame(rec, columns=["ioa", "cot", "type_id", "t", "value"]).sort_values("t", kind="stable")
    # measurement points the update rule may use (a name used by several IOAs: the last IOA wins)
    meas = {f"{v['element']}:{v['attribute']}" for v in point_map.values() if v["context"] == "MEASUREMENT"}
    by_name = {names.get(i, i): i for i in sorted(rec["ioa"].unique())}
    keep = sorted(n for n in by_name if n in meas and ":is_" not in n and not n.endswith("tap_position"))
    col = {by_name[n]: j for j, n in enumerate(keep)}
    periodic = np.full((len(grid), len(keep)), np.nan)
    changes = []
    for ioa, g in rec.groupby("ioa", sort=False):
        t, v = g["t"].to_numpy(), g["value"].to_numpy()
        if ioa in col:
            tp = g.loc[g.cot == 1, "t"].to_numpy()
            if len(tp):
                pp = np.searchsorted(tp, grid, side="right") - 1
                periodic[pp >= 0, col[ioa]] = tp[pp[pp >= 0]]
        if (g["type_id"] != 13).all():  # single points and step positions: value changes
            for i in np.flatnonzero(np.diff(v) != 0) + 1:
                changes.append({"name": names.get(ioa, ioa), "t": t[i], "old": v[i - 1], "new": v[i]})
    base.parent.mkdir(parents=True, exist_ok=True)
    np.savez(base.with_suffix(".npz"), names=np.array(keep), periodic=periodic)
    pd.DataFrame(changes, columns=["name", "t", "old", "new"]).to_csv(f"{base}_changes.csv", index=False)
    pd.DataFrame([(names.get(i, i), t, v) for i, t, v in cmd], columns=["name", "t", "value"]).to_csv(f"{base}_commands.csv", index=False)

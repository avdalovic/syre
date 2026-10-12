"""syre run | fit | plot  CONFIG"""
from __future__ import annotations

import argparse
import hashlib
import json
import tomllib
from pathlib import Path

import numpy as np
import pandas as pd

from . import laws, rules, score
from .data import Scenario
from .grid import Graph, extract_map


def load(config: str, data: str) -> tuple[dict, Scenario]:
    cfg = tomllib.loads(Path(config).read_text())
    return cfg, Scenario(cfg, Path(data))


def sha(a: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(a, dtype=np.int8).tobytes()).hexdigest()


def graphs(cfg: dict, scen: Scenario):
    grid_map = extract_map(scen.dir / "raw" / "test" / "docs" / "power-grid.svg")
    tr, te = scen.state("train"), scen.state("test")
    align = cfg["detector"]["align_s"]
    return grid_map, Graph(tr, grid_map, align), Graph(te, grid_map, align)


def run(args) -> int:
    cfg, scen = load(args.config, args.data)
    out = Path(args.out) / scen.name
    out.mkdir(parents=True, exist_ok=True)
    d = cfg["detector"]
    print(f"[{scen.name}] loading state and building the grid graph")
    grid_map, g_tr, g_te = graphs(cfg, scen)
    for k, v in grid_map.items():
        v.to_csv(out / f"map_{k}.csv")
    print(f"[{scen.name}] scoring {len(cfg['laws'])} law(s)")
    res = laws.score(cfg["laws"], g_tr, g_te, d["alpha"], d["floor_percentile"], d["rho_quantile"])
    grid = g_te.a.grid
    print(f"[{scen.name}] control-center rules")
    cc_tr, cc_te = scen.control_center("train", g_tr.a.grid), scen.control_center("test", grid)
    cmd = rules.command_rule(cc_te["changes"], cc_te["commands"], grid, cfg["rules"]["command_window_s"])
    upd = rules.update_rule(cc_tr, cc_te, grid, cfg["rules"]["update_gap_s"])
    alarms = {"laws": res["alarm"], "rules": cmd | upd, "laws + rules": res["alarm"] | cmd | upd}
    np.savez_compressed(out / "alarms.npz", laws=res["alarm"], command=cmd, update=upd, tau=res["tau"])
    inst = pd.DataFrame([{k: v for k, v in i.items() if k != "z"} for i in res["instances"]])
    inst.to_csv(out / "instances.csv", index=False)
    tab = score.table(alarms, scen.windows(float(grid[0])))
    tab.to_csv(out / "detections.csv", index=False)
    print(f"\ntau = {res['tau']:.4f}, {len(inst)} law instances\n")
    print(tab.to_string(index=False))
    exp = cfg.get("expected", {})
    if exp:
        checks = {"laws alarm": sha(res["alarm"]) == exp["laws_sha256"], "command rule": sha(cmd) == exp["command_sha256"],
                  "update rule": sha(upd) == exp["update_sha256"]}
        print("\nsame as the paper, second by second:", ", ".join(f"{k} {'yes' if v else 'NO'}" for k, v in checks.items()))
        (out / "check.json").write_text(json.dumps({"tau": res["tau"], **{k: bool(v) for k, v in checks.items()}}, indent=1))
        return 0 if all(checks.values()) else 1
    return 0


def fit(args) -> int:
    from .fit import fit_laws
    cfg, scen = load(args.config, args.data)
    out = Path(args.out) / scen.name / "fit"
    out.mkdir(parents=True, exist_ok=True)
    _, g_tr, _ = graphs(cfg, scen)
    chosen = fit_laws(cfg["fit"], g_tr, out)
    block = "[laws]\n" + "".join(f'{t} = "{e}"\n' for t, e in chosen.items())
    (out / "laws.toml").write_text(block)
    print(block)
    return 0


def plot(args) -> int:
    from .plot import trace
    cfg, scen = load(args.config, args.data)
    out = Path(args.out) / scen.name
    out.mkdir(parents=True, exist_ok=True)
    _, g_tr, g_te = graphs(cfg, scen)
    ttype, name = args.law.split(" ", 1)
    path = trace(cfg, scen, g_tr, g_te, ttype, name, args.attack, out)
    print(path)
    return 0


def main() -> int:
    p = argparse.ArgumentParser(prog="syre", description="SyRe: symbolic and readable intrusion detection for power networks")
    p.add_argument("--data", default="data", help="folder with the unzipped Sherlock scenarios")
    p.add_argument("--out", default="out")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("run", help="detect and count on the test split").add_argument("config")
    sub.add_parser("fit", help="re-learn the laws with PySR (slow)").add_argument("config")
    pp = sub.add_parser("plot", help="detection trace of one law instance around one attack")
    pp.add_argument("config")
    pp.add_argument("--attack", required=True)
    pp.add_argument("--law", required=True, help='e.g. "bus_P bus.14" or "end_V line.7 to (bus 10)"')
    args = p.parse_args()
    return {"run": run, "fit": fit, "plot": plot}[args.cmd](args)

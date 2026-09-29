"""Sensitivity analysis + scenario grid for the Luzardo strikeout model.

Outputs model/output/sensitivity.json and model/output/grid.json, consumed by charts.py.
"""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import numpy as np

from sim import Simulator, load_cfg, summarize

HERE = Path(__file__).resolve().parent
OUT = HERE / "output"
OUT.mkdir(exist_ok=True)


def run(cfg, n, seed=7):
    return summarize(Simulator(cfg, seed=seed).run(n))


def scale_caps(cfg, mult):
    c = copy.deepcopy(cfg)
    for s in c["workload_scenarios"].values():
        s["pitch_cap_tri"] = [x * mult for x in s["pitch_cap_tri"]]
    return c


def shift_k(cfg, dk):
    c = copy.deepcopy(cfg)
    c["pitcher"]["k_vs_L"] += dk
    c["pitcher"]["k_vs_R"] += dk
    return c


def set_state_weights(cfg, weights):
    c = copy.deepcopy(cfg)
    for k, w in weights.items():
        c["arm_states"][k]["weight"] = w
    return c


def set_workload_weights(cfg, weights):
    c = copy.deepcopy(cfg)
    for k, w in weights.items():
        c["workload_scenarios"][k]["weight"] = w
    return c


def scale_lineup_k(cfg, mult):
    c = copy.deepcopy(cfg)
    for b in c["lineup"]:
        b["k"] = b["k"] * mult
    return c


def env_k(cfg, mult):
    c = copy.deepcopy(cfg)
    c["environment"]["k_mult_total"] *= mult
    return c


def hooks_mult(cfg, mult):
    c = copy.deepcopy(cfg)
    c["hooks"]["p_hook_2runs_traffic"] = min(1.0, c["hooks"]["p_hook_2runs_traffic"] * mult)
    c["hooks"]["p_hook_traffic_per_pa"] = min(1.0, c["hooks"]["p_hook_traffic_per_pa"] * mult)
    return c


def main(n=60_000):
    base_cfg = load_cfg()
    base = run(base_cfg, n)
    line = base_cfg.get("reference_line", 2.5)
    key = str(line)

    tests = {
        "Pitch cap -20% / +20%": (scale_caps(base_cfg, 0.8), scale_caps(base_cfg, 1.2)),
        "Luzardo true K% -3pp / +3pp": (shift_k(base_cfg, -0.03), shift_k(base_cfg, 0.03)),
        "Braves K% vs LHP x0.9 / x1.1": (scale_lineup_k(base_cfg, 0.9), scale_lineup_k(base_cfg, 1.1)),
        "Environment K odds x0.95 / x1.05": (env_k(base_cfg, 0.95), env_k(base_cfg, 1.05)),
        "Trouble-hook aggressiveness x1.5 / x0.5": (hooks_mult(base_cfg, 1.5), hooks_mult(base_cfg, 0.5)),
    }
    # arm-state tilt: pessimistic vs optimistic
    st = base_cfg["arm_states"]
    names = list(st.keys())
    pess = {n_: st[n_]["weight"] for n_ in names}
    opt = dict(pess)
    if "sharp_max_effort" in st and "rusty" in st:
        pess["sharp_max_effort"] = max(0.0, st["sharp_max_effort"]["weight"] - 0.15)
        pess["rusty"] = st["rusty"]["weight"] + 0.15
        opt["sharp_max_effort"] = st["sharp_max_effort"]["weight"] + 0.15
        opt["rusty"] = max(0.0, st["rusty"]["weight"] - 0.15)
        tests["Arm state: 15pts sharp->rusty / rusty->sharp"] = (
            set_state_weights(base_cfg, pess), set_state_weights(base_cfg, opt))

    sens = {"base": {"mean_k": base["mean_k"], "p_over": base["over"][key]["p_over"], "line": line}, "tests": {}}
    for label, (lo_cfg, hi_cfg) in tests.items():
        lo = run(lo_cfg, n)
        hi = run(hi_cfg, n)
        sens["tests"][label] = {
            "low": {"mean_k": lo["mean_k"], "p_over": lo["over"][key]["p_over"]},
            "high": {"mean_k": hi["mean_k"], "p_over": hi["over"][key]["p_over"]},
        }
        print(label, round(lo["mean_k"], 3), round(hi["mean_k"], 3),
              round(lo["over"][key]["p_over"], 3), round(hi["over"][key]["p_over"], 3), flush=True)

    with open(OUT / "sensitivity.json", "w") as f:
        json.dump(sens, f, indent=2)

    # 2-D grid: pitch-cap multiplier x K% shift -> P(over line), mean K
    cap_mults = [0.7, 0.85, 1.0, 1.15, 1.3]
    k_shifts = [-0.04, -0.02, 0.0, 0.02, 0.04]
    grid = {"cap_mults": cap_mults, "k_shifts": k_shifts, "p_over": [], "mean_k": [], "line": line}
    for cm in cap_mults:
        row_p, row_m = [], []
        for dk in k_shifts:
            s = run(shift_k(scale_caps(base_cfg, cm), dk), n // 2)
            row_p.append(s["over"][key]["p_over"])
            row_m.append(s["mean_k"])
        grid["p_over"].append(row_p)
        grid["mean_k"].append(row_m)
    with open(OUT / "grid.json", "w") as f:
        json.dump(grid, f, indent=2)
    print(json.dumps(grid, indent=1))


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 60_000)

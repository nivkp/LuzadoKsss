"""
Monte Carlo strikeout model for Jesús Luzardo, NL Wild Card Series Game 1,
PHI @ ATL, Truist Park, Tuesday 2026-09-29 (2:08 p.m. ET).

The model is a plate-appearance-level game simulator built around the one thing
that dominates an "opener" prop: how many batters he is allowed to face.

    K_total = sum over batters faced of Bernoulli(p_K[batter, state])

where the number of batters faced is produced by an explicit hook model
(pitch cap, inning cap, batters-faced cap, trouble hook, injury exit, weather),
and p_K comes from a generalized log5 (odds-ratio) matchup of Luzardo's
true-talent K% against each hitter's K% vs LHP, shifted by a latent
"arm-state" variable (post-IL rust vs max-effort short-stint adrenaline).

All inputs live in assumptions.json so every number in the report is traceable.
"""
from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------
def odds(p: float) -> float:
    return p / (1.0 - p)


def prob(o: float) -> float:
    return o / (1.0 + o)


def log5(p_pitcher: float, p_batter: float, p_league: float) -> float:
    """Generalized log5 / odds-ratio matchup rate."""
    return prob(odds(p_pitcher) * odds(p_batter) / odds(p_league))


def shift_odds(p: float, mult: float) -> float:
    return prob(odds(p) * mult)


@dataclass
class Batter:
    slot: int
    name: str
    hand: str  # effective hand vs LHP: 'L' or 'R' (switch hitters bat R vs LHP)
    k: float  # regressed K% vs LHP
    bb: float  # regressed BB% vs LHP
    hr: float  # HR per PA vs LHP (regressed)


# ----------------------------------------------------------------------------
# pitches-per-PA distributions, conditional on PA outcome
# (shapes follow public pitch-count-by-outcome distributions; tuned so a
#  league-average pitcher lands at ~3.9 P/PA and a 28% K pitcher ~4.0)
# ----------------------------------------------------------------------------
PITCHES = {
    # outcome: (values, weights)
    "K": ([3, 4, 5, 6, 7, 8, 9, 10], [0.20, 0.27, 0.25, 0.15, 0.08, 0.03, 0.015, 0.005]),
    "BB": ([4, 5, 6, 7, 8, 9, 10], [0.17, 0.29, 0.34, 0.12, 0.05, 0.02, 0.01]),
    "HBP": ([1, 2, 3, 4, 5, 6], [0.22, 0.20, 0.20, 0.18, 0.12, 0.08]),
    "BIP": ([1, 2, 3, 4, 5, 6, 7, 8, 9], [0.17, 0.18, 0.19, 0.18, 0.13, 0.08, 0.04, 0.015, 0.005]),
}


def _prep_pitch_tables(rng_scale: float = 1.0):
    tables = {}
    for k, (vals, w) in PITCHES.items():
        w = np.array(w, dtype=float)
        w = w / w.sum()
        tables[k] = (np.array(vals), np.cumsum(w))
    return tables


PITCH_TABLES = _prep_pitch_tables()


def mean_pitches(outcome: str) -> float:
    vals, w = PITCHES[outcome]
    w = np.array(w) / np.sum(w)
    return float(np.dot(vals, w))


# ----------------------------------------------------------------------------
# simulation core
# ----------------------------------------------------------------------------
class Simulator:
    def __init__(self, cfg: dict, seed: int = 20260929):
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)
        self.lineup = [Batter(**b) for b in cfg["lineup"]]
        self.lg = cfg["league"]
        self.p = cfg["pitcher"]
        self.states = cfg["arm_states"]
        self.workload = cfg["workload_scenarios"]
        self.hooks = cfg["hooks"]
        self.env = cfg["environment"]

    # -- per-PA outcome probabilities ------------------------------------
    def pa_probs(self, b: Batter, k_true_shift: float, st: dict, tto: int):
        p = self.p
        lg = self.lg
        # pitcher K%/BB% vs this batter's side
        pk_side = p["k_vs_L"] if b.hand == "L" else p["k_vs_R"]
        pbb_side = p["bb_vs_L"] if b.hand == "L" else p["bb_vs_R"]
        lgk = lg["k_LHP_vs_L"] if b.hand == "L" else lg["k_LHP_vs_R"]
        lgbb = lg["bb_LHP_vs_L"] if b.hand == "L" else lg["bb_LHP_vs_R"]

        # parameter uncertainty on true talent (same draw for the whole game)
        pk_side = prob(odds(pk_side) * math.exp(k_true_shift))

        pk = log5(pk_side, b.k, lgk)
        pbb = log5(pbb_side, b.bb, lgbb)

        # multiplicative (odds-space) context adjustments for strikeouts
        k_mult = st["k_mult"] * self.env["k_mult_total"]
        k_mult *= self.p["tto_k_mult"][min(tto, len(self.p["tto_k_mult"])) - 1]
        pk = shift_odds(pk, k_mult)
        pbb = shift_odds(pbb, st["bb_mult"] * self.env["bb_mult_total"])

        phbp = p["hbp"]
        phr = log5(p["hr"], b.hr, lg["hr"]) * st.get("hr_mult", 1.0)
        rest = 1.0 - pk - pbb - phbp - phr
        if rest < 0.2:
            raise ValueError("probabilities too large for " + b.name)
        return pk, pbb, phbp, phr, rest

    # -- workload draw --------------------------------------------------------
    def draw_workload(self):
        sc = self.workload
        names = list(sc.keys())
        w = np.array([sc[n]["weight"] for n in names], dtype=float)
        w /= w.sum()
        name = names[self.rng.choice(len(names), p=w)]
        s = sc[name]
        lo, mode, hi = s["pitch_cap_tri"]
        cap = self.rng.triangular(lo, mode, hi)
        inn_vals, inn_w = zip(*s["inning_cap"].items())
        inn_w = np.array(inn_w, dtype=float) / np.sum(inn_w)
        inning_cap = int(self.rng.choice([int(x) for x in inn_vals], p=inn_w))
        bf_cap = s.get("bf_cap")
        if bf_cap is not None:
            vals, ww = zip(*bf_cap.items())
            ww = np.array(ww, dtype=float) / np.sum(ww)
            bf_cap = int(self.rng.choice([int(x) for x in vals], p=ww))
        return name, cap, inning_cap, bf_cap

    def draw_state(self):
        names = list(self.states.keys())
        w = np.array([self.states[n]["weight"] for n in names], dtype=float)
        w /= w.sum()
        name = names[self.rng.choice(len(names), p=w)]
        return name, self.states[name]

    def n_pitches(self, outcome: str) -> int:
        vals, cdf = PITCH_TABLES[outcome]
        u = self.rng.random()
        return int(vals[np.searchsorted(cdf, u)])

    # -- one game ---------------------------------------------------------------
    def sim_game(self):
        rng = self.rng
        wl_name, cap, inning_cap, bf_cap = self.draw_workload()
        st_name, st = self.draw_state()
        k_true_shift = rng.normal(0.0, self.p["k_logit_sd"])
        # the arm state also moves the pitch cap (a laboring arm gets less rope)
        cap = cap * st.get("cap_mult", 1.0)
        inning_cap = max(1, inning_cap + st.get("inning_cap_add", 0))

        # pre-game / in-game exogenous exits
        weather_cut_pitch = None
        if rng.random() < self.env["p_weather_interrupt_during_outing"]:
            weather_cut_pitch = rng.uniform(5, 60)
        injury_hazard = self.hooks["injury_exit_per_pitch"] * st.get("injury_hazard_mult", 1.0)

        outs = 0
        total_outs = 0
        bases = [False, False, False]
        inning = 1
        runs = 0
        runs_inning = 0
        pc = 0
        bf = 0
        ks = 0
        bbs = 0
        hits = 0
        slot = 0
        exit_reason = "cap"
        n = len(self.lineup)

        while True:
            b = self.lineup[slot % n]
            tto = slot // n + 1
            pk, pbb, phbp, phr, rest = self.pa_probs(b, k_true_shift, st, tto)
            u = rng.random()
            if u < pk:
                oc = "K"
            elif u < pk + pbb:
                oc = "BB"
            elif u < pk + pbb + phbp:
                oc = "HBP"
            elif u < pk + pbb + phbp + phr:
                oc = "HR"
            else:
                oc = "BIP"
            np_ = self.n_pitches("K" if oc == "K" else "BB" if oc == "BB" else "HBP" if oc == "HBP" else "BIP")
            pc += np_
            bf += 1
            slot += 1

            # -- resolve PA ------------------------------------------------------
            if oc == "K":
                ks += 1
                outs += 1
            elif oc in ("BB", "HBP"):
                bbs += 1
                if bases[0]:
                    if bases[1]:
                        if bases[2]:
                            runs += 1
                            runs_inning += 1
                        bases[2] = True
                    bases[1] = True
                bases[0] = True
            elif oc == "HR":
                hits += 1
                r = 1 + sum(bases)
                runs += r
                runs_inning += r
                bases = [False, False, False]
            else:  # ball in play, not HR
                if rng.random() < self.p["babip"]:
                    hits += 1
                    v = rng.random()
                    if v < 0.70:  # single
                        r = 0
                        if bases[2]:
                            r += 1
                        new3 = False
                        if bases[1]:
                            if rng.random() < 0.60:
                                r += 1
                            else:
                                new3 = True
                        new2 = bases[0]
                        bases = [True, new2, new3]
                    elif v < 0.965:  # double
                        r = int(bases[2]) + int(bases[1])
                        new3 = False
                        if bases[0]:
                            if rng.random() < 0.40:
                                r += 1
                            else:
                                new3 = True
                        bases = [False, True, new3]
                    else:  # triple
                        r = sum(bases)
                        bases = [False, False, True]
                    runs += r
                    runs_inning += r
                else:
                    # out in play
                    if bases[0] and outs < 2 and rng.random() < self.p["dp_rate_given_chance"]:
                        outs += 2
                        bases[0] = False
                        if outs < 3 and bases[2] and rng.random() < 0.3:
                            runs += 1
                            runs_inning += 1
                            bases[2] = False
                    else:
                        outs += 1
                        if outs < 3 and bases[2] and rng.random() < 0.45:
                            runs += 1
                            runs_inning += 1
                            bases[2] = False
                        if outs < 3 and bases[1] and rng.random() < 0.25:
                            bases[2] = True
                            bases[1] = False

            # -- exogenous exits (evaluated at PA boundary) -------------------------
            if weather_cut_pitch is not None and pc >= weather_cut_pitch:
                exit_reason = "weather"
                break
            if rng.random() < 1.0 - (1.0 - injury_hazard) ** np_:
                exit_reason = "injury"
                break

            inning_over = outs >= 3
            if inning_over:
                total_outs += 3
                innings_done = inning
                outs = 0
                bases = [False, False, False]
                inning += 1
                runs_inning = 0
                if innings_done >= inning_cap:
                    exit_reason = "inning_cap"
                    break
                if pc >= cap - self.hooks["between_innings_buffer"]:
                    exit_reason = "cap_between_innings"
                    break
                if bf_cap is not None and bf >= bf_cap:
                    exit_reason = "bf_cap"
                    break
                continue

            # mid-inning hooks
            if pc >= cap + self.hooks["mid_inning_overrun"]:
                exit_reason = "cap_mid_inning"
                break
            if bf_cap is not None and bf >= bf_cap + self.hooks["bf_cap_mid_inning_slack"]:
                exit_reason = "bf_cap"
                break
            # trouble hook: short playoff leash on a post-IL arm
            if runs >= self.hooks["runs_hard_hook"]:
                exit_reason = "trouble"
                break
            traffic = sum(bases)
            if runs_inning >= 2 and traffic >= 1 and rng.random() < self.hooks["p_hook_2runs_traffic"]:
                exit_reason = "trouble"
                break
            if traffic >= 2 and rng.random() < self.hooks["p_hook_traffic_per_pa"] * st.get("hook_mult", 1.0):
                exit_reason = "trouble"
                break

        # outs in a partially completed inning (0 when he exits between innings)
        outs_recorded = total_outs + min(outs, 2)
        return dict(
            k=ks, bf=bf, pc=pc, outs=outs_recorded, runs=runs, bb=bbs, hits=hits,
            exit=exit_reason, workload=wl_name, state=st_name,
        )

    def run(self, n: int):
        out = [self.sim_game() for _ in range(n)]
        return out


# ----------------------------------------------------------------------------
# reporting helpers
# ----------------------------------------------------------------------------
def american(p: float) -> str:
    if p <= 0 or p >= 1:
        return "n/a"
    if p >= 0.5:
        return f"-{round(100 * p / (1 - p))}"
    return f"+{round(100 * (1 - p) / p)}"


def summarize(results, label=""):
    k = np.array([r["k"] for r in results])
    bf = np.array([r["bf"] for r in results])
    pc = np.array([r["pc"] for r in results])
    outs = np.array([r["outs"] for r in results])
    s = {
        "label": label,
        "n": int(len(k)),
        "mean_k": float(k.mean()),
        "median_k": float(np.median(k)),
        "sd_k": float(k.std()),
        "mean_bf": float(bf.mean()),
        "median_bf": float(np.median(bf)),
        "mean_pitches": float(pc.mean()),
        "mean_ip": float(outs.mean() / 3.0),
        "pmf": {int(i): float((k == i).mean()) for i in range(0, 11)},
        "bf_pmf": {int(i): float((bf == i).mean()) for i in range(1, 25)},
        "over": {},
    }
    for line in (0.5, 1.5, 2.5, 3.5, 4.5, 5.5):
        p_over = float((k > line).mean())
        s["over"][str(line)] = {"p_over": p_over, "p_under": 1 - p_over,
                                "fair_over": american(p_over), "fair_under": american(1 - p_over)}
    exits = {}
    for r in results:
        exits[r["exit"]] = exits.get(r["exit"], 0) + 1
    s["exit_mix"] = {k_: v / len(results) for k_, v in sorted(exits.items(), key=lambda x: -x[1])}
    wl = {}
    for r in results:
        d = wl.setdefault(r["workload"], [])
        d.append(r["k"])
    s["by_workload"] = {k_: {"share": len(v) / len(results), "mean_k": float(np.mean(v))} for k_, v in wl.items()}
    stt = {}
    for r in results:
        d = stt.setdefault(r["state"], [])
        d.append(r["k"])
    s["by_state"] = {k_: {"share": len(v) / len(results), "mean_k": float(np.mean(v))} for k_, v in stt.items()}
    return s


def load_cfg(path: Path | None = None) -> dict:
    path = path or HERE / "assumptions.json"
    with open(path) as f:
        return json.load(f)


def matchup_table(cfg: dict):
    """Per-batter K probability, first time through, 'normal' arm state, no talent shift."""
    sim = Simulator(cfg)
    normal = cfg["arm_states"].get("normal", {"k_mult": 1.0, "bb_mult": 1.0})
    rows = []
    for b in sim.lineup:
        pk, pbb, phbp, phr, rest = sim.pa_probs(b, 0.0, normal, 1)
        rows.append({"slot": b.slot, "name": b.name, "hand": b.hand, "batter_k_vs_lhp": b.k,
                     "p_k": pk, "p_bb": pbb, "p_hr": phr})
    return rows


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 200_000
    cfg = load_cfg()
    sim = Simulator(cfg)
    res = sim.run(n)
    s = summarize(res, "base")
    s["reference_line"] = cfg.get("reference_line", 2.5)
    out = HERE / "output"
    out.mkdir(exist_ok=True)
    with open(out / "summary.json", "w") as f:
        json.dump(s, f, indent=2)
    with open(out / "matchups.json", "w") as f:
        json.dump(matchup_table(cfg), f, indent=2)
    print(json.dumps(s, indent=2))

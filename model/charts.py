"""Static PNG charts for the report (light surface, reference palette)."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE / "output"
FIG = HERE.parent / "analysis" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASE = "#c3c2b7"
S1 = "#2a78d6"  # blue
S2 = "#eb6834"  # orange
SEQ = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5",
       "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.edgecolor": BASE,
    "axes.labelcolor": INK2,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "axes.titlecolor": INK,
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
})


def _style(ax):
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(BASE)
    ax.spines["bottom"].set_linewidth(1)
    ax.grid(axis="y", color=GRID, linewidth=1, linestyle="-")
    ax.set_axisbelow(True)
    ax.tick_params(length=0)


def _rounded_bar(ax, x, h, width, color):
    """Bar with a 4px-ish rounded data end, square at baseline."""
    if h <= 0:
        return
    r = min(0.06, h * 0.25)
    ax.add_patch(FancyBboxPatch((x - width / 2, 0), width, h,
                                boxstyle=f"round,pad=0,rounding_size={r}",
                                mutation_aspect=1 / 0.08 * 0.01 if False else 1,
                                linewidth=0, facecolor=color))
    # square off the baseline corners
    ax.add_patch(plt.Rectangle((x - width / 2, 0), width, min(h, r * 1.2), linewidth=0, facecolor=color))


def k_distribution(summary, line, path):
    pmf = summary["pmf"]
    ks = list(range(0, 9))
    vals = [pmf.get(str(k), pmf.get(k, 0.0)) for k in ks]
    tail = sum(v for k, v in pmf.items() if int(k) >= 9)
    vals[-1] += tail
    fig, ax = plt.subplots(figsize=(8.2, 4.6), dpi=160)
    under_c, over_c = S1, S2
    for k, v in zip(ks, vals):
        ax.bar(k, v * 100, width=0.55, color=over_c if k > line else under_c, linewidth=0)
    _style(ax)
    ax.set_xticks(ks)
    ax.set_xticklabels([str(k) if k < 8 else "8+" for k in ks])
    ax.set_xlabel("Luzardo strikeouts (simulated)")
    ax.set_ylabel("Probability (%)")
    p_under = sum(v for k, v in zip(ks, vals) if k < line)
    p_over = 1 - p_under
    ax.set_title(f"Strikeout distribution — {summary['n']:,} simulated outings", loc="left", fontsize=13, pad=26)
    ax.text(0, 1.035, f"Mean {summary['mean_k']:.2f}  ·  median {summary['median_k']:.0f}  ·  "
                      f"P(under {line}) {p_under*100:.1f}%  ·  P(over {line}) {p_over*100:.1f}%",
            transform=ax.transAxes, color=INK2, fontsize=10)
    ymax = max(vals) * 100
    ax.set_ylim(0, ymax * 1.18)
    mode_k = ks[vals.index(max(vals))]
    ax.text(mode_k, max(vals) * 100 + ymax * 0.03, f"{max(vals)*100:.1f}%", ha="center", color=INK, fontsize=10)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=under_c, label=f"Under {line}"), Patch(color=over_c, label=f"Over {line}")],
              frameon=False, loc="upper right", labelcolor=INK2)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def bf_distribution(summary, path):
    pmf = summary["bf_pmf"]
    xs = [int(k) for k in pmf.keys()]
    vals = [pmf[k] for k in pmf.keys()]
    keep = [(x, v) for x, v in zip(xs, vals) if x <= 18]
    fig, ax = plt.subplots(figsize=(8.2, 4.2), dpi=160)
    for x, v in keep:
        ax.bar(x, v * 100, width=0.55, color=S1, linewidth=0)
    _style(ax)
    ax.set_xticks([x for x, _ in keep])
    ax.set_xlabel("Batters faced")
    ax.set_ylabel("Probability (%)")
    ax.set_title("How long the leash is: batters faced", loc="left", fontsize=13, pad=26)
    ax.text(0, 1.035, f"Mean {summary['mean_bf']:.1f} BF  ·  median {summary['median_bf']:.0f}  ·  "
                      f"mean {summary['mean_pitches']:.0f} pitches  ·  mean {summary['mean_ip']:.2f} IP",
            transform=ax.transAxes, color=INK2, fontsize=10)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def tornado(sens, path):
    base = sens["base"]["p_over"] * 100
    line = sens["base"]["line"]
    items = []
    for label, d in sens["tests"].items():
        lo = d["low"]["p_over"] * 100
        hi = d["high"]["p_over"] * 100
        items.append((label, lo, hi, abs(hi - lo)))
    items.sort(key=lambda t: t[3])
    fig, ax = plt.subplots(figsize=(8.8, 0.62 * len(items) + 1.6), dpi=160)
    for i, (label, lo, hi, _) in enumerate(items):
        ax.barh(i, lo - base, left=base, height=0.5, color=S1, linewidth=0)
        ax.barh(i, hi - base, left=base, height=0.5, color=S2, linewidth=0)
        ax.text(min(lo, hi) - 0.6, i, f"{min(lo, hi):.0f}%", va="center", ha="right", color=INK2, fontsize=9)
        ax.text(max(lo, hi) + 0.6, i, f"{max(lo, hi):.0f}%", va="center", ha="left", color=INK2, fontsize=9)
    ax.axvline(base, color=INK2, linewidth=1)
    ax.set_yticks(range(len(items)))
    ax.set_yticklabels([t[0] for t in items], color=INK2)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="x", color=GRID, linewidth=1)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)
    ax.set_xlabel(f"P(over {line} strikeouts), %")
    ax.set_title(f"What moves the number (base {base:.0f}%)", loc="left", fontsize=13, pad=12)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=S1, label="First setting (low / pessimistic)"),
                       Patch(color=S2, label="Second setting (high / optimistic)")],
              frameon=False, loc="lower right", labelcolor=INK2, fontsize=9)
    lo_all = min(min(t[1], t[2]) for t in items)
    hi_all = max(max(t[1], t[2]) for t in items)
    ax.set_xlim(lo_all - 7, hi_all + 7)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def grid_heatmap(grid, path):
    import numpy as np
    from matplotlib.colors import LinearSegmentedColormap
    cm = LinearSegmentedColormap.from_list("seqblue", SEQ)
    z = np.array(grid["p_over"]) * 100
    fig, ax = plt.subplots(figsize=(7.4, 4.8), dpi=160)
    im = ax.imshow(z, cmap=cm, vmin=0, vmax=max(60, z.max()), aspect="auto", origin="lower")
    ax.set_xticks(range(len(grid["k_shifts"])))
    ax.set_xticklabels([f"{d*100:+.0f}pp" for d in grid["k_shifts"]])
    ax.set_yticks(range(len(grid["cap_mults"])))
    ax.set_yticklabels([f"x{m:.2f}" for m in grid["cap_mults"]])
    ax.set_xlabel("Shift to Luzardo's true-talent K%")
    ax.set_ylabel("Pitch-cap multiplier")
    for i in range(z.shape[0]):
        for j in range(z.shape[1]):
            v = z[i, j]
            ax.text(j, i, f"{v:.0f}%", ha="center", va="center", fontsize=10,
                    color="#ffffff" if v > 35 else INK)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    ax.set_title(f"P(over {grid['line']}) across leash length × stuff", loc="left", fontsize=13, pad=12)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.outline.set_visible(False)
    cb.ax.tick_params(length=0, colors=MUTED)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def matchup_bars(matchups, path):
    names = [m["name"] for m in matchups]
    vals = [m["p_k"] * 100 for m in matchups]
    fig, ax = plt.subplots(figsize=(8.2, 4.4), dpi=160)
    for i, v in enumerate(vals):
        ax.barh(i, v, height=0.5, color=S1, linewidth=0)
        ax.text(v + 0.5, i, f"{v:.0f}%", va="center", color=INK2, fontsize=9)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels([f"{m['slot']}. {m['name']} ({m['hand']})" for m in matchups], color=INK2)
    ax.invert_yaxis()
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="x", color=GRID, linewidth=1)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)
    ax.set_xlabel("Per-PA strikeout probability, first time through (normal arm state)")
    ax.set_title("Matchup-level K probability, projected Braves lineup", loc="left", fontsize=13, pad=12)
    ax.set_xlim(0, max(vals) * 1.2)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


if __name__ == "__main__":
    summary = json.load(open(OUT / "summary.json"))
    line = summary.get("reference_line", 2.5)
    k_distribution(summary, line, FIG / "k_distribution.png")
    bf_distribution(summary, FIG / "bf_distribution.png")
    if (OUT / "sensitivity.json").exists():
        tornado(json.load(open(OUT / "sensitivity.json")), FIG / "tornado.png")
    if (OUT / "grid.json").exists():
        grid_heatmap(json.load(open(OUT / "grid.json")), FIG / "grid.png")
    if (OUT / "matchups.json").exists():
        matchup_bars(json.load(open(OUT / "matchups.json")), FIG / "matchups.png")
    print("charts written to", FIG)

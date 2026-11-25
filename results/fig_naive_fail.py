"""Candidate Figure 3 (Section 4): how the standard two-rung procedure fails.

Fits a line in log10(N) to the two cheapest rungs (5.25M, 10M) -- exactly what
screening practice does -- and extends it. Shows the sign error on conv and the
magnitude blow-up on diff against the measured curves.

Also prints a consistency check on the Section 4 claim that diff's naive fit
implies a break-even near 9 billion parameters.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

N = np.array([5.25, 10.0, 15.6, 20.6, 29.4, 91.3, 162.9])
Y = {"conv": np.array([8.1, 5.1, -3.4, -9.8, -10.2, -9.9, -9.0]),
     "diff": np.array([15.7, 21.8, 12.3, 6.2, 4.0, 1.5, 0.3])}
COL = {"conv": "#0072B2", "diff": "#D55E00"}
INK, MUT = "#37352f", "#8a8578"

def two_rung_line(y):
    """Least squares line in log10 N through the two cheapest rungs."""
    x = np.log10(N[:2])
    slope = (y[1] - y[0]) / (x[1] - x[0])
    return slope, y[1] - slope * x[1]        # slope, intercept at log10 N = 0

print("Section 4 consistency check")
for k, y in Y.items():
    sl, ic = two_rung_line(y)
    pred = sl * np.log10(29.4) + ic
    root = -ic / sl                          # log10 N where the line hits zero
    print(f"  {k}: slope {sl:+.2f} pp/decade | pred@29.4M {pred:+.2f} "
          f"(actual {y[4]:+.1f}) | line hits zero at N = {10**root:,.0f}M "
          f"{'(BELOW the window - no break-even above it)' if sl > 0 else ''}")

fig, axes = plt.subplots(1, 2, figsize=(6.8, 3.1), dpi=220)
fig.patch.set_facecolor("white")
grid = np.geomspace(5.0, 175, 300)

for ax, k in zip(axes, ("conv", "diff")):
    y, c = Y[k], COL[k]
    sl, ic = two_rung_line(y)
    ax.set_facecolor("white")
    ax.axhline(0, color=MUT, lw=0.8, zorder=1)

    ax.plot(grid, sl * np.log10(grid) + ic, color=MUT, lw=1.5, ls=(0, (5, 3)),
            zorder=2, label="line fit to the 2 cheapest rungs")
    ax.plot(N, y, "-", color=c, lw=1.0, alpha=0.35, zorder=3)
    ax.plot(N, y, "o", color=c, ms=5.5, zorder=4, label="measured")
    ax.plot(N[:2], y[:2], "o", color=MUT, ms=5.5, zorder=5,
            label="rungs used by the fit")

    pred = sl * np.log10(29.4) + ic
    ax.annotate("", xy=(29.4, y[4]), xytext=(29.4, pred),
                arrowprops=dict(arrowstyle="<->", color=INK, lw=1.0, alpha=0.75))
    ax.plot([29.4], [pred], "x", color=INK, ms=7, mew=1.6, zorder=6)
    ax.text(33, (pred + y[4]) / 2, f"{abs(pred - y[4]):.0f} pp\noff",
            fontsize=8, color=INK, va="center", linespacing=1.4)

    ax.set_xscale("log")
    ax.set_xlim(4.8, 200)
    ax.set_xticks([5.25, 10, 29.4, 91.3, 162.9])
    ax.set_xticklabels(["5.25", "10", "29.4", "91", "163"], fontsize=8)
    ax.minorticks_off()
    ax.tick_params(axis="y", labelsize=8)
    ax.grid(axis="y", color="#000000", alpha=0.08, lw=0.6)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(MUT)
    ax.set_xlabel("parameters (millions, log scale)", fontsize=9, color=INK)

axes[0].set_ylabel("$\\Delta$ vs baseline (%)", fontsize=9.5, color=INK)
axes[0].set_ylim(-17.5, 14)
axes[1].set_ylim(-3, 40)
axes[0].set_title("conv: the sign is wrong", fontsize=9, color=INK, pad=5)
axes[1].set_title("diff: the magnitude is wrong", fontsize=9, color=INK, pad=5)
axes[0].text(5.05, -17.0,
             "predicts $+0.1$, truth $-10.2$: the best\nmechanism in the pool is discarded",
             fontsize=7.8, color=INK, linespacing=1.5, ha="left", va="bottom")
axes[1].legend(loc="upper left", fontsize=7.4, frameon=False, borderaxespad=0.2,
               labelspacing=0.3, handlelength=2.2)

fig.tight_layout()
out = os.path.splitext(os.path.abspath(__file__))[0]
fig.savefig(out + ".png", bbox_inches="tight")
fig.savefig(out + ".pdf", bbox_inches="tight")
print("saved", out + ".png / .pdf")

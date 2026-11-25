"""Candidate Figure 2 (Section 7): the disguising pair.

Two admissible curves with OPPOSITE asymptotes (+1 / -1 pp) that are
indistinguishable on the measured ladder. Reproduces mc_verify.py section C
exactly, then draws it. Prints the published numbers so the figure is
verifiably faithful (max rung gap 0.072pp, ~-22pp at 10M, optimal test errs 43%).

Design note: with alpha ~ 0.3 the asymptote is approached far too slowly to show
on any sane axis, so the figure does NOT pretend to plot convergence. Left panel
= what the experiment can see (one apparent curve). Right panel = the limit the
two curves actually have. That contrast is the theorem.
"""
import os
import sys
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

N = np.array([10.0, 15.6, 20.6, 29.4, 91.3])          # rungs, millions
m, s, sigma = len(N), 2, 0.5
Phi = lambda z: 0.5 * math.erfc(-z / math.sqrt(2))
u_of = lambda al: N ** (-al)

A0, A1 = 0.30, 0.34                                    # the two decay rates
t = 2.0 * sigma                                        # asymptotes +/- 1.0 pp

coef, *_ = np.linalg.lstsq(np.column_stack([u_of(A0), -u_of(A1)]),
                           -2 * t * np.ones(m), rcond=None)
b0, b1 = float(coef[0]), float(coef[1])
mu0 = t + b0 * u_of(A0)        # asymptote +1 pp  -> HURTS at scale
mu1 = -t + b1 * u_of(A1)       # asymptote -1 pp  -> HELPS at scale

d = np.sqrt(s) * np.linalg.norm(mu0 - mu1) / sigma
gap = np.abs(mu0 - mu1).max()
print(f"asymptotes {+t:+.2f} / {-t:+.2f} pp")
print(f"value at 10M rung:  {mu0[0]:+.2f} vs {mu1[0]:+.2f} pp")
print(f"max gap on rungs:   {gap:.4f} pp   (paper: 0.072)")
print(f"gap at a 163M rung: {abs((t + b0*162.9**-A0) - (-t + b1*162.9**-A1)):.4f} pp"
      f"   (paper: 0.19)")
print(f"optimal-test error: {Phi(-d/2):.4f}                (paper: 0.43)")

COL_HURT, COL_HELP = "#D55E00", "#0072B2"      # Okabe-Ito, same as Figure 1
INK, MUT = "#37352f", "#8a8578"

# Default: one-column layout (camera-ready, 25 Nov 2026), drawn at its printed size
# (ACL column = 3.03 in) so the text prints at the stated point sizes.
# "--wide" reproduces the earlier two-column (figure*) version.
WIDE = "--wide" in sys.argv
if WIDE:
    FIGSIZE = (6.8, 3.7)
    F = dict(leg=8, note=8.2, tick=8.5, lab=9.5, title=9, ans=8.4, inf=9)
    LW, MS, MSR = (2.6, 1.5), (6.5, 3.0), 7.5
    NOTE = ("Two different curves are drawn here.\n"
            f"They differ by at most {gap:.3f} pp on every rung\n"
            "— a seventh of the seed noise ($\\sigma = 0.5$ pp).")
    NOTE_Y, RATIOS, TICK_NUDGE = -6.6, [3.2, 1.35], {}
else:
    FIGSIZE = (3.15, 2.5)
    F = dict(leg=6.4, note=6.4, tick=6.4, lab=7.2, title=7.0, ans=6.6, inf=7.0)
    LW, MS, MSR = (2.0, 1.15), (4.6, 2.1), 5.4
    NOTE = ("Two different curves are drawn here.\n"
            f"They differ by at most {gap:.3f} pp\n"
            "on every rung, a seventh of the\n"
            "seed noise ($\\sigma = 0.5$ pp).")
    # the 15.6M and 20.6M rungs sit close on a log axis: nudge their labels apart (points)
    NOTE_Y, RATIOS, TICK_NUDGE = -11.0, [3.2, 1.1], {1: -3.0, 2: +2.5}

fig, (ax, axr) = plt.subplots(
    1, 2, figsize=FIGSIZE, dpi=220, sharey=True,
    gridspec_kw=dict(width_ratios=RATIOS, wspace=0.05))
fig.patch.set_facecolor("white")
for a in (ax, axr):
    a.set_facecolor("white")
    a.axhline(0, color=MUT, lw=0.8, zorder=1)
    a.grid(axis="y", color="#000000", alpha=0.08, lw=0.6)
    for sp in ("top", "right"):
        a.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        a.spines[sp].set_color(MUT)

# ---- left: what the ladder can actually see -------------------------------
zg = np.geomspace(N[0], N[-1], 300)
ax.plot(zg, t + b0 * zg ** (-A0), color=COL_HURT, lw=LW[0], zorder=3,
        label="a curve whose limit is $+1\\%$")
ax.plot(zg, -t + b1 * zg ** (-A1), color=COL_HELP, lw=LW[1], ls=(0, (5, 2.5)), zorder=4,
        label="a curve whose limit is $-1\\%$")
ax.plot(N, mu0, "o", color=COL_HURT, ms=MS[0], zorder=5)
ax.plot(N, mu1, "o", ms=MS[1], color="white", zorder=6)
ax.legend(loc="upper right", fontsize=F["leg"], frameon=False, borderaxespad=0.3,
          handlelength=2.6, labelspacing=0.35)

ax.annotate(NOTE,
            xy=(20.6, mu0[2]), xytext=(10.6, NOTE_Y), color=INK, fontsize=F["note"],
            linespacing=1.6, ha="left",
            arrowprops=dict(arrowstyle="-", color=MUT, lw=0.8,
                            connectionstyle="arc3,rad=0.2"))
ax.set_xscale("log")
ax.set_xlim(9.2, 100)
ax.set_xticks(N)
ax.set_xticklabels(["10", "15.6", "20.6", "29.4", "91.3"], fontsize=F["tick"])
for i, dx in TICK_NUDGE.items():
    lbl = ax.get_xticklabels()[i]
    lbl.set_transform(lbl.get_transform()
                      + matplotlib.transforms.ScaledTranslation(dx / 72, 0, fig.dpi_scale_trans))
ax.minorticks_off()
ax.tick_params(axis="y", labelsize=F["tick"])
ax.set_xlabel("parameters (millions, log scale)", fontsize=F["lab"], color=INK)
ax.set_ylabel("$\\Delta$ vs baseline (%)", fontsize=F["lab"], color=INK)
ax.set_title("what the ladder measures", fontsize=F["title"], color=INK, pad=6)

# ---- right: the limit they actually have ----------------------------------
axr.set_xlim(0, 1)
# faded connectors: this region is extrapolation, not data
for yv, c in ((+t, COL_HURT), (-t, COL_HELP)):
    axr.plot([0.0, 0.46], [mu0[-1] if c == COL_HURT else mu1[-1], yv],
             color=c, lw=1.0, ls=(0, (1.5, 2.5)), alpha=0.45, zorder=2)
axr.plot([0.46], [+t], "o", color=COL_HURT, ms=MSR, zorder=5)
axr.plot([0.46], [-t], "o", color=COL_HELP, ms=MSR, zorder=5)
axr.annotate("$a=+1$: hurts", xy=(0.50, +t), xytext=(0.60, +t + 3.6),
             color=COL_HURT, fontsize=F["ans"], ha="left", va="center",
             arrowprops=dict(arrowstyle="-", color=COL_HURT, lw=0.7, alpha=0.7))
axr.annotate("$a=-1$: helps", xy=(0.50, -t), xytext=(0.60, -t - 4.4),
             color=COL_HELP, fontsize=F["ans"], ha="left", va="center",
             arrowprops=dict(arrowstyle="-", color=COL_HELP, lw=0.7, alpha=0.7))
axr.set_xticks([0.46])
axr.set_xticklabels(["$N \\to \\infty$"], fontsize=F["inf"])
axr.set_title("the answer", fontsize=F["title"], color=INK, pad=6)
axr.spines["left"].set_visible(False)
axr.tick_params(axis="y", length=0)

ax.set_ylim(-24.5, 7.5)
axr.set_xlim(0, 1.55)
# NB: no summary line baked into the image -- the LaTeX caption carries it.

out = os.path.splitext(os.path.abspath(__file__))[0]
fig.savefig(out + ".png", bbox_inches="tight")
fig.savefig(out + ".pdf", bbox_inches="tight")
print("saved", out + ".png / .pdf")

"""Paper figure for Section 6: measured effect curves + derived-form fits
Delta(N) = a + b*N^(-alpha), with conv's crossover formula vs the observed flip."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

N = np.array([5.25, 10.0, 15.6, 20.6, 29.4, 91.3, 162.9])
curves = {
    "conv": np.array([8.1, 5.1, -3.4, -9.8, -10.2, -9.9, -9.0]),
    "diff": np.array([15.7, 21.8, 12.3, 6.2, 4.0, 1.5, 0.3]),
    "mtp":  np.array([7.1, 10.0, 6.3, 6.0, 5.7, 2.7, 2.2]),
}
COL = {"conv": "#0072B2", "diff": "#D55E00", "mtp": "#009E73"}  # Okabe-Ito, CVD-validated
INK, MUT = "#37352f", "#8a8578"

def fit_free(Ns, ys):
    best = None
    for alpha in np.arange(0.02, 3.5001, 0.001):
        u = Ns ** (-alpha)
        A = np.column_stack([np.ones_like(u), u])
        coef, *_ = np.linalg.lstsq(A, ys, rcond=None)
        sse = float(((A @ coef - ys) ** 2).sum())
        if best is None or sse < best[0]:
            best = (sse, float(alpha), float(coef[0]), float(coef[1]))
    return best

fig, ax = plt.subplots(figsize=(6.8, 4.2), dpi=220)
fig.patch.set_facecolor("white")
ax.set_facecolor("white")

# observed conv flip bracket + crossover formula marker
ax.axvspan(10.0, 15.6, color="#000000", alpha=0.06, zorder=0)
ax.axvline(11.7, color=COL["conv"], lw=1.1, ls=(0, (4, 3)), alpha=0.8, zorder=1)
ax.axhline(0, color=MUT, lw=0.8, zorder=1)

W = slice(1, 7)  # fit window 10M-163M (monotone suffix; 5.25M rung excluded)
grid = np.geomspace(10.0, 162.9, 300)
labels = {}
for name, y in curves.items():
    sse, al, a, b = fit_free(N[W], y[W])
    labels[name] = (al, a)
    c = COL[name]
    ax.plot(grid, a + b * grid ** (-al), color=c, lw=2.0, zorder=3)
    ax.plot(N[W], y[W], "o", color=c, ms=6, zorder=4,
            label=f"{name}  ($\\alpha$={al:.2f}, plateau {a:+.1f})")
    ax.plot(N[:1], y[:1], "o", ms=6, mfc="white", mec=c, mew=1.4, zorder=4)

# direct labels at the right ends
ends = {"conv": (-9.0, 0), "diff": (0.3, -1.3), "mtp": (2.2, 1.3)}
for name, (yv, dy) in ends.items():
    ax.annotate(name, xy=(162.9, yv), xytext=(178, yv + dy),
                color=INK, fontsize=9, va="center")

ax.text(4.55, 23.6, "shaded: conv flip, measured (10–15.6M)\n"
        "dashed: crossover formula $N^{*}$ = 11.7M",
        color=INK, fontsize=8, ha="left", va="top", linespacing=1.5)
ax.annotate("hollow = pre-turn rung,\nexcluded by the suffix rule",
            xy=(5.25, 8.1), xytext=(4.7, -5.5), color=MUT, fontsize=7.5,
            arrowprops=dict(arrowstyle="-", color=MUT, lw=0.8))

ax.set_xscale("log")
ax.set_xlim(4.3, 275)
ax.set_ylim(-13.5, 24)
ax.set_xticks(N)
ax.set_xticklabels(["5.25", "10", "15.6", "20.6", "29.4", "91", "163"], fontsize=8.5)
ax.tick_params(axis="y", labelsize=8.5)
ax.minorticks_off()
ax.set_xlabel("parameters (millions, log scale)", fontsize=9.5, color=INK)
ax.set_ylabel("$\\Delta$ vs baseline (%)", fontsize=9.5, color=INK)
ax.grid(axis="y", color="#000000", alpha=0.08, lw=0.6)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
for s in ("left", "bottom"):
    ax.spines[s].set_color(MUT)
ax.legend(loc="upper right", fontsize=8, frameon=False, borderaxespad=0.2)

fig.tight_layout()
out = os.path.splitext(os.path.abspath(__file__))[0]
fig.savefig(out + ".png", bbox_inches="tight")
fig.savefig(out + ".pdf", bbox_inches="tight")
print("saved", out + ".png / .pdf")
print({k: (round(v[0], 2), round(v[1], 1)) for k, v in labels.items()})

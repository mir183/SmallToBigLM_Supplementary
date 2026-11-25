"""Level 1 refit: fit the DERIVED form Delta(N) = a + b*N^(-alpha) to the measured
effect curves.  Produces the numbers for the paper's new theory section:
  1. per-mechanism free-alpha fits on the monotone window 10M-163M (descriptive refit)
  2. shared-alpha joint fit (the 'constants-not-exponents' picture) vs free alpha
  3. fits on <=91M -> forecast 163M (the predictive check, derived parameterization)
  4. 2x-budget floor shift: freeze alpha from 1x, refit (a,b) on the 2x rungs
"""
import numpy as np

N = np.array([5.25, 10.0, 15.6, 20.6, 29.4, 91.3, 162.9])   # params, millions
curves = {
    "conv": np.array([8.1, 5.1, -3.4, -9.8, -10.2, -9.9, -9.0]),
    "diff": np.array([15.7, 21.8, 12.3, 6.2, 4.0, 1.5, 0.3]),
    "mtp":  np.array([7.1, 10.0, 6.3, 6.0, 5.7, 2.7, 2.2]),
}
GRID = np.arange(0.02, 3.5001, 0.001)

def ab_for_alpha(Ns, ys, alpha):
    u = Ns ** (-alpha)
    A = np.column_stack([np.ones_like(u), u])
    coef, *_ = np.linalg.lstsq(A, ys, rcond=None)
    sse = float(((A @ coef - ys) ** 2).sum())
    return sse, float(coef[0]), float(coef[1])

def fit_free(Ns, ys):
    best = None
    for alpha in GRID:
        sse, a, b = ab_for_alpha(Ns, ys, alpha)
        if best is None or sse < best[0]:
            best = (sse, float(alpha), a, b)
    return best  # sse, alpha, a, b

def nstar(alpha, a, b):
    return (b / (-a)) ** (1 / alpha) if (a < 0 < b) or (b < 0 < a) else float("nan")

W = slice(1, 7)   # 10M..163M  (monotone window, drops the pre-turn 5.25M rung)
P = slice(1, 6)   # 10M..91M   (fit window for the 163M forecast check)

print("=" * 78)
print("1) FREE-ALPHA REFIT, derived form, window 10M-163M (6 rungs)")
print(f"{'mech':6s} {'alpha':>6s} {'a(floor)':>9s} {'b':>9s} {'rmse':>6s} {'N*':>7s}")
free = {}
for k, y in curves.items():
    sse, al, a, b = fit_free(N[W], y[W])
    rmse = (sse / len(y[W])) ** 0.5
    free[k] = (al, a, b, rmse)
    print(f"{k:6s} {al:6.3f} {a:+9.2f} {b:+9.1f} {rmse:6.2f} {nstar(al,a,b):7.1f}")

print()
print("=" * 78)
print("2) SHARED-ALPHA JOINT FIT (same window) -- the shared-exponent picture")
best = None
for alpha in GRID:
    tot, parts = 0.0, {}
    for k, y in curves.items():
        sse, a, b = ab_for_alpha(N[W], y[W], alpha)
        tot += sse
        parts[k] = (a, b, sse)
    if best is None or tot < best[0]:
        best = (tot, float(alpha), dict(parts))
tot, al_sh, parts = best
npts = sum(len(curves[k][W]) for k in curves)
print(f"shared alpha = {al_sh:.3f}   overall rmse = {(tot/npts)**0.5:.2f}")
print(f"{'mech':6s} {'a':>9s} {'b':>9s} {'rmse_shared':>11s} {'rmse_free':>10s}")
for k in curves:
    a, b, sse = parts[k]
    print(f"{k:6s} {a:+9.2f} {b:+9.1f} {(sse/len(curves[k][W]))**0.5:11.2f} "
          f"{free[k][3]:10.2f}")

print()
print("=" * 78)
print("3) PREDICTIVE CHECK: fit 10M-91M, forecast 163M (derived parameterization)")
print(f"{'mech':6s} {'alpha':>6s} {'a':>8s} {'pred163':>8s} {'actual':>7s} {'err_pp':>7s}")
for k, y in curves.items():
    sse, al, a, b = fit_free(N[P], y[P])
    pred = a + b * 162.9 ** (-al)
    print(f"{k:6s} {al:6.3f} {a:+8.2f} {pred:+8.2f} {y[6]:+7.1f} {abs(pred-y[6]):7.2f}")
    if k == "conv":
        print(f"       conv crossover from this fit: N* = {nstar(al,a,b):.1f}M "
              f"(observed flip between 10 and 15.6M)")

print()
print("=" * 78)
print("4) 2x-BUDGET FLOOR SHIFT: alpha frozen at the 1x value, refit (a,b) on 2x rungs")
N2 = np.array([10.0, 29.4, 91.3])
two = {"conv": np.array([0.42, -2.93, -0.52]),   # 91M cell: 1 seed
       "diff": np.array([9.0, 2.0, 1.8])}
for k, y2 in two.items():
    al = free[k][0]
    sse, a2, b2 = ab_for_alpha(N2, y2, al)
    print(f"{k:6s} alpha={al:5.3f} frozen:  floor a: {free[k][1]:+6.2f} (1x) -> "
          f"{a2:+6.2f} (2x)   [b: {free[k][2]:+8.1f} -> {b2:+8.1f}]"
          + ("   N*_2x = %.1fM" % nstar(al, a2, b2) if not np.isnan(nstar(al, a2, b2)) else ""))
print("(conv 91M @2x is a single seed; treat the 2x conv floor as +/-1pp)")

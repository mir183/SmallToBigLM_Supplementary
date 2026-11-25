"""Level 2, unknown-alpha theorem: numbers for both halves.

(1) LOWER BOUND (Theorem 2). Adversary now picks two decay speeds. Separation of the
    two-point pair (+t, b0, alpha0) vs (-t, b1, alpha1), minimized over b0, b1, is the
    residual of projecting the constant onto span{u(alpha0), u(alpha1)}:
        error >= Phi(-t / SE2),  SE2 = sigma / sqrt(s*m*(1 - R2)),
        R2 = sup over pairs of |P_span 1|^2 / m   (two-regressor collinearity).
    Near-identical pairs give span{u, du/dalpha} (the local/Fisher version).

(2) UPPER BOUND (Theorem 3, the uniform certificate test). For every alpha in a grid,
    run OLS of the data on [1, u(alpha)] -> intercept ahat(alpha), its exact sd SE(alpha).
    Declare '+' iff ahat(alpha) > z*SE(alpha) for ALL alpha; '-' symmetrically; else ABSTAIN.
    Type-I is exact and one-line: the 'for all' includes the true alpha, where
    ahat ~ N(a, SE^2) exactly. Here: run it on our REAL mean curves (window 10-91M).
"""
import numpy as np

N = np.array([10.0, 15.6, 20.6, 29.4, 91.3])
m, s = len(N), 2
curves = {
    "conv": np.array([5.1, -3.4, -9.8, -10.2, -9.9]),
    "diff": np.array([21.8, 12.3, 6.2, 4.0, 1.5]),
    "mtp":  np.array([10.0, 6.3, 6.0, 5.7, 2.7]),
}
ALPHAS = np.arange(0.30, 3.5001, 0.005)

def u_of(alpha):
    return N ** (-alpha)

def r2_span(X):
    """squared norm fraction of the ones-vector captured by span(X columns)."""
    ones = np.ones(m)
    coef, *_ = np.linalg.lstsq(X, ones, rcond=None)
    return float((X @ coef) @ (X @ coef)) / m

# ---------- (1) lower bound geometry ----------
rho2_single = {al: r2_span(u_of(al).reshape(-1, 1)) for al in (0.80, 1.57, 2.65)}
# pair sup over a coarse grid (off-diagonal), plus the derivative-augmented local version
pair_grid = np.arange(0.30, 3.5001, 0.02)
best = (0.0, None)
for i, a0 in enumerate(pair_grid):
    u0 = u_of(a0)
    for a1 in pair_grid[i + 1:]:
        if a1 - a0 < 0.04:
            continue
        r2 = r2_span(np.column_stack([u0, u_of(a1)]))
        if r2 > best[0]:
            best = (r2, (a0, a1))
R2_pair, argpair = best
loc = {}
for al in (0.80, 1.57, 2.65):
    du = -np.log(N) * u_of(al)
    loc[al] = r2_span(np.column_stack([u_of(al), du]))
R2_sup = max(R2_pair, max(loc.values()))

print("(1) LOWER BOUND, window 10-91M, m=5, s=2  (needed |a|/sigma for 2-sigma calls)")
print(f"    pair sup R2 = {R2_pair:.4f} at alphas {argpair}")
for al in (0.80, 1.57, 2.65):
    r1 = rho2_single[al]
    need_known = 2 / np.sqrt(s * m * (1 - r1))
    need_local = 2 / np.sqrt(s * m * (1 - loc[al]))
    print(f"    alpha={al:4.2f}: known-alpha need {need_known:4.2f}*sigma | "
          f"local unknown-alpha (u,du) R2={loc[al]:.4f} -> need {need_local:5.2f}*sigma "
          f"(x{need_local/need_known:.1f})")
need_glob = 2 / np.sqrt(s * m * (1 - R2_sup))
print(f"    GLOBAL unknown-alpha (worst pair, R2={R2_sup:.4f}): need {need_glob:.2f}*sigma")
print("    (as alpha_lo -> 0 the sup -> 1: with no lower bound on decay speed,")
print("     sign is unidentifiable at ANY effect size)")

# ---------- (2) the uniform certificate test on the real curves ----------
print("\n(2) UNIFORM TEST on our real 10-91M mean curves, alpha in [0.3, 3.5], z=1.645")
z = 1.645
for sigma in (0.5, 1.0):
    print(f"    sigma = {sigma}pp:")
    for name, y in curves.items():
        lo_stat, hi_stat = np.inf, -np.inf
        a_min, a_max = np.inf, -np.inf
        for al in ALPHAS:
            X = np.column_stack([np.ones(m), u_of(al)])
            coef, *_ = np.linalg.lstsq(X, y, rcond=None)
            ahat = float(coef[0])
            r1 = r2_span(u_of(al).reshape(-1, 1))
            # exact OLS intercept sd for design [1,u], s seeds, noise sigma:
            u = u_of(al)
            se = sigma * np.sqrt((u @ u) / (s * (m * (u @ u) - u.sum() ** 2)))
            lo_stat = min(lo_stat, ahat / se)
            hi_stat = max(hi_stat, ahat / se)
            a_min, a_max = min(a_min, ahat), max(a_max, ahat)
        verdict = ("declare NEGATIVE (win certified)" if hi_stat < -z else
                   "declare POSITIVE (loss certified)" if lo_stat > z else "ABSTAIN")
        print(f"      {name:5s}: ahat(alpha) ranges [{a_min:+7.2f}, {a_max:+7.2f}], "
              f"t-stat range [{lo_stat:+7.2f}, {hi_stat:+7.2f}]  -> {verdict}")

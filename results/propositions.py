"""Level 2 continued: numbers for (a) Proposition 2's sign-failure horizon on the
registered diff linear fit, and (b) the window-width rule (practical inverse).

Geometry fact used in (b): with m log-spaced rungs across W decades and decay alpha,
u_i is a geometric sequence with ratio 10^(alpha*W/(m-1)), so rho^2 (and hence the
disguise factor) depends only on the product alpha*W and m -- 'decades of decay
visible in the window'.
"""
import numpy as np

# ---- (a) Prop 2: the registered linear fit for diff, fit window 10-91M ----
N = np.array([10.0, 15.6, 20.6, 29.4, 91.3])
diff = np.array([21.8, 12.3, 6.2, 4.0, 1.5])
x = np.log10(N)
A = np.column_stack([np.ones_like(x), x])
(c0, c1), *_ = np.linalg.lstsq(A, diff, rcond=None)
x_dag = -c0 / c1
print(f"(a) diff linear fit on 10-91M: intercept {c0:+.2f}, slope {c1:+.2f} per decade")
print(f"    sign-failure horizon x+ = {x_dag:.3f} -> N+ = {10**x_dag:.0f}M")
print(f"    line at 163M: {c0 + c1*np.log10(162.9):+.2f}%  (registered: -6.63%, truth +0.30%)")
print(f"    -> the pre-registered 163M call lay beyond the horizon; sign-wrong was forced.")

# ---- (b) window-width rule ----
def one_minus_rho2(alphaW, m):
    """1 - rho^2 for m log-spaced rungs with alpha*W decades of decay in-window."""
    i = np.arange(m)
    u = 10.0 ** (-alphaW * i / (m - 1))     # scale drops out of rho^2
    return 1 - u.sum() ** 2 / (m * (u ** 2).sum())

m, s = 5, 2
print(f"\n(b) window rule, m={m} rungs, s={s} seeds: sign callable (2-sigma) iff")
print("    |a|/sigma >= 2 / sqrt(s*m*(1-rho^2(alpha*W)))")
print(f"{'alpha*W':>8s} {'1-rho^2':>8s} {'needed |a|/sigma':>17s}")
for aW in (0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0):
    om = one_minus_rho2(aW, m)
    print(f"{aW:8.2f} {om:8.3f} {2/np.sqrt(s*m*om):17.2f}")

print("\nour three interventions on the 10->91M window (W = 0.96 decades):")
for name, al in (("conv", 2.65), ("diff", 1.57), ("mtp", 0.80)):
    aW = al * 0.96
    om = one_minus_rho2(aW, m)
    print(f"  {name}: alpha*W = {aW:4.2f} -> needed |a| >= {2/np.sqrt(s*m*om):.2f}*sigma")

print("\nwidening the window at fixed alpha (diff, alpha=1.57), m=5, s=2:")
for W in (0.96, 1.5, 2.0, 3.0):
    om = one_minus_rho2(1.57 * W, m)
    print(f"  W = {W:4.2f} decades -> callable when |a| >= {2/np.sqrt(s*m*om):.2f}*sigma")

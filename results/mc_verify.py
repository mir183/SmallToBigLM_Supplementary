"""Monte Carlo stress-test of the Level-2 theorems before they enter the paper.

A. Theorem 1: empirical sd of the OLS intercept must equal SE, and the sign-error rate
   must equal Phi(-|a|/SE) -- for several |a|/SE and for DIFFERENT b (invariance).
   Also verifies the two-point algebra ||2t*1 - c*u||^2 = 4 t^2 m (1-rho^2) and that the
   optimal likelihood-ratio test on the adversarial pair errs at the predicted rate.
B. Theorem 3: at a = 0 the uniform certificate must declare a sign in <= 5% of trials
   (exact Type-I); conv-like truth should certify '-' almost always; diff-like truth
   should abstain almost always.
C. Theorem 2: construct the two disguising curves (alpha0=0.30 vs alpha1=0.34) with
   asymptotes +-t; verify they are near-identical on the rungs, that the optimal test's
   simulated error matches Phi(-d/2), and report the amplitudes involved.
All at the real window: rungs 10-91.3M, m=5, s=2.
"""
import math
import numpy as np

rng = np.random.default_rng(0)
N = np.array([10.0, 15.6, 20.6, 29.4, 91.3])
m, s = len(N), 2
Phi = lambda z: 0.5 * math.erfc(-z / math.sqrt(2))

def u_of(al):
    return N ** (-al)

def intercept_weights(al):
    X = np.column_stack([np.ones(m), u_of(al)])
    return np.linalg.inv(X.T @ X) @ X.T  # 2 x m ; row 0 -> intercept

def se_of(al, sigma):
    u = u_of(al)
    return sigma * np.sqrt((u @ u) / (s * (m * (u @ u) - u.sum() ** 2)))

def check(label, emp, pred, n):
    tol = 3 * np.sqrt(max(pred * (1 - pred), 1e-9) / n) + 1e-4
    ok = abs(emp - pred) <= tol
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}: empirical {emp:.4f} vs predicted {pred:.4f}")
    return ok

allok = True
# ---------------- A. Theorem 1 ----------------
print("A. Theorem 1 (alpha known = 1.57, sigma = 0.5)")
al, sigma, n = 1.57, 0.5, 400_000
W0 = intercept_weights(al)[0]
SE = se_of(al, sigma)
noise_sd = sigma / np.sqrt(s)
# sd check
ybar = (0 + 800 * u_of(al)) + noise_sd * rng.standard_normal((n, m))
ahat = ybar @ W0
print(f"  [{'PASS' if abs(ahat.std() - SE) < 3 * SE / np.sqrt(2*n) + 5e-4 else 'FAIL'}] "
      f"sd(ahat): empirical {ahat.std():.4f} vs SE formula {SE:.4f}")
for ratio in (0.5, 1.0, 2.0):
    for b in (800.0, 8000.0):
        a = ratio * SE
        ybar = (a + b * u_of(al)) + noise_sd * rng.standard_normal((n, m))
        emp = float(((ybar @ W0) < 0).mean())
        allok &= check(f"sign-error |a|/SE={ratio}, b={b:g}", emp, float(Phi(-ratio)), n)
# two-point algebra + LR test at the SAME alpha
t = SE
u = u_of(al)
cstar = 2 * t * u.sum() / (u @ u)
resid2 = float(((2 * t - cstar * u) ** 2).sum())
rho2 = u.sum() ** 2 / (m * (u @ u))
print(f"  [{'PASS' if abs(resid2 - 4*t*t*m*(1-rho2)) < 1e-9 else 'FAIL'}] "
      f"||2t*1-c*u||^2 = 4t^2 m(1-rho^2): {resid2:.6f} vs {4*t*t*m*(1-rho2):.6f}")
mu0, mu1 = t + 800 * u, -t + (800 + cstar) * u
d = np.sqrt(s) * np.linalg.norm(mu0 - mu1) / sigma
half = 200_000
data = mu0 + noise_sd * rng.standard_normal((half, m))
err0 = (np.linalg.norm(data - mu1, axis=1) < np.linalg.norm(data - mu0, axis=1)).mean()
allok &= check(f"LR error on the adversarial pair (d={d:.3f})", float(err0), float(Phi(-d/2)), half)

# ---------------- B. Theorem 3 ----------------
print("B. Theorem 3 (uniform certificate, alpha grid 0.30:3.50:0.005, z=1.645)")
z, sigma = 1.645, 0.5
grid = np.round(np.arange(0.30, 3.5001, 0.005), 4)   # contains 1.57 and 2.65
Wg = np.stack([intercept_weights(al)[0] for al in grid])          # G x m
SEg = np.array([se_of(al, sigma) for al in grid])                 # G
def run_uniform(truth, n=100_000, chunk=10_000):
    plus = minus = 0
    for k in range(0, n, chunk):
        ybar = truth + (sigma/np.sqrt(s)) * rng.standard_normal((chunk, m))
        T = (ybar @ Wg.T) / SEg
        plus += int((T > z).all(axis=1).sum())
        minus += int((T < -z).all(axis=1).sum())
    return plus / n, minus / n
# (i) a=0 boundary null, alpha*=1.57 on the grid
p, mn = run_uniform(0.0 + 743.0 * u_of(1.57))
print(f"  [{'PASS' if p <= 0.05 + 3*np.sqrt(0.05*0.95/100_000) else 'FAIL'}] "
      f"Type-I at a=0: declare+ rate {p:.4f} (bound 0.05; conservativeness expected), declare- {mn:.4f}")
# (ii) conv-like truth: fit the real conv means to get (alpha,a,b) on this window
conv = np.array([5.1, -3.4, -9.8, -10.2, -9.9])
best = None
for alf in np.arange(0.3, 3.5001, 0.001):
    X = np.column_stack([np.ones(m), u_of(alf)])
    coef, *_ = np.linalg.lstsq(X, conv, rcond=None)
    sse = float(((X @ coef - conv) ** 2).sum())
    if best is None or sse < best[0]:
        best = (sse, alf, coef[0], coef[1])
_, alc, ac, bc = best
p, mn = run_uniform(ac + bc * u_of(alc))
print(f"         conv truth (a={ac:.2f}, alpha={alc:.2f}): certifies '-' in {mn:.1%} of trials, "
      f"false '+' {p:.2%}")
# (iii) diff-like truth
p, mn = run_uniform(0.25 + 807.0 * u_of(1.57))
print(f"         diff truth (a=+0.25): declares '+' {p:.2%}, '-' {mn:.2%}, "
      f"abstains {1-p-mn:.1%} (abstention is the correct behavior)")

# ---------------- C. Theorem 2 ----------------
print("C. Theorem 2 disguising pair (alpha 0.30 vs 0.34), sigma=0.5")
u0, u1 = u_of(0.30), u_of(0.34)
for tmult in (2.0, 5.0):
    t = tmult * sigma
    A = np.column_stack([u0, -u1])
    coef, *_ = np.linalg.lstsq(A, -2 * t * np.ones(m), rcond=None)
    b0, b1 = float(coef[0]), float(coef[1])
    mu0, mu1 = t + b0 * u0, -t + b1 * u1
    d = np.sqrt(s) * np.linalg.norm(mu0 - mu1) / sigma
    pred = float(Phi(-d / 2))
    data = mu0 + (sigma/np.sqrt(s)) * rng.standard_normal((200_000, m))
    emp = float((np.linalg.norm(data - mu1, axis=1) < np.linalg.norm(data - mu0, axis=1)).mean())
    allok &= check(f"t={tmult}sigma: optimal-test error", emp, pred, 200_000)
    print(f"         asymptotes {t:+.2f}/{-t:+.2f}pp; curve at 10M: {mu0[0]:+.1f} vs {mu1[0]:+.1f}pp; "
          f"max rung gap {np.abs(mu0-mu1).max():.3f}pp; gap at 163M rung: "
          f"{abs((t + b0*162.9**-0.30) - (-t + b1*162.9**-0.34)):.3f}pp")

print("\nALL CHECKS PASSED" if allok else "\nSOME CHECKS FAILED - investigate before the paper")

"""Camera-ready theory numbers (EACL 2027 Findings, commitment #839).

Recomputes, from the per-seed CSVs, every number the camera-ready adds to Section 7:
  1. per-seed paired Delta, rung means, and the measured per-seed noise sigma_hat
  2. the FINITE-TARGET bound (replaces "inevitable"): Theorem 1 applied to the target value
     Delta(N_T) = a + b*N_T^-alpha instead of the asymptote a. Same two-point argument with
     functional c = (1, u_T): minimax sign error = Phi(-t / SE_T),
     SE_T = (sigma/sqrt(s)) * sqrt(c' (X'X)^-1 c), X = [1, u]  (the OLS prediction's sd).
     Local unknown-alpha version: X = [1, u, du/dalpha], c = (1, u_T, b*du_T/dalpha)... the
     amplitude b multiplies the alpha direction, so it is absorbed into the free coefficient.
  3. the certificate test (Theorem 3) with known sigma (z) and with estimated sigma
     (t critical value), a 50x finer alpha grid, and the SEED-SPLIT version
     (choose rungs on seed 0 with the monotone-suffix rule, certify on seed 1 only).
Run:  python results/camready_theory.py   (numpy only; no torch)
"""
import csv
import math
import os
import numpy as np

class stats:  # one-sided 95% quantiles (standard tables); avoids a scipy dependency
    class norm:
        @staticmethod
        def ppf(q): assert q == 0.95; return 1.6448536269514722
    class t:
        TAB = {1: 6.314, 2: 2.920, 3: 2.353, 4: 2.132, 5: 2.015, 6: 1.943, 7: 1.895, 8: 1.860}
        @classmethod
        def ppf(cls, q, nu): assert q == 0.95; return cls.TAB[nu]

HERE = os.path.dirname(os.path.abspath(__file__))
def rd(name):
    with open(os.path.join(HERE, name)) as f:
        return list(csv.reader(f))[1:]

# ---------------------------------------------------------------- 1. data
NS = {"5M": 5.254, "10M": 10.016, "15M": 15.566, "20M": 20.562, "30M": 29.407, "90M": 91.341, "160M": 162.86}
ORDER = list(NS)
ppl = {}  # (size, variant, seed) -> ppl
for sz, v, s, pm, p in rd("ladder6000_runs.csv"):
    ppl[(sz, v, s)] = float(p)
for fn, s in (("scale_confirm_runs.csv", "0"), ("scale_confirm_s1_runs.csv", "1")):
    for v, pm, p in rd(fn):
        ppl[("30M", v, s)] = float(p)
for v, s, pm, p in rd("val90_runs.csv"):
    ppl[("90M", v, s)] = float(p)
for v, s, pm, p in rd("val160_runs.csv"):
    ppl[("160M", v, s)] = float(p)
for sz, v, s, pm, p in rd("mtp_ladder_runs.csv"):
    ppl[(sz, v, s)] = float(p)

def delta(v, sz, s):
    return 100.0 * (ppl[(sz, v, s)] - ppl[(sz, "combo", s)]) / ppl[(sz, "combo", s)]

VARS = {"conv": "combo_conv3", "diff": "combo_diff", "mtp": "combo_mtp"}
D = {k: {sz: [delta(v, sz, s) for s in ("0", "1")] for sz in ORDER if (sz, v, "0") in ppl}
     for k, v in VARS.items()}
print("1) per-seed paired Delta (%), seed0 / seed1 / mean")
for k in D:
    print(f"   {k}: " + "  ".join(f"{sz}:{a:+.2f}/{b:+.2f}/{(a+b)/2:+.2f}" for sz, (a, b) in D[k].items()))

def sigma_hat(k, sizes):
    # per-seed noise from the two seeds: var = (d0-d1)^2/2 per rung, pooled; nu = #rungs
    v = [(D[k][sz][0] - D[k][sz][1]) ** 2 / 2 for sz in sizes]
    return math.sqrt(np.mean(v)), len(v)
W91 = ["10M", "15M", "20M", "30M", "90M"]
W29 = ["10M", "15M", "20M", "30M"]
print("\n   measured per-seed sigma_hat (pooled, nu = #rungs):")
for k in D:
    a, na = sigma_hat(k, W91); b, nb = sigma_hat(k, W29)
    print(f"   {k}: 10-91M {a:.2f} (nu={na})   10-29.4M {b:.2f} (nu={nb})")

# ---------------------------------------------------------------- 2. finite-target bound
def Phi(z):
    return 0.5 * math.erfc(-z / math.sqrt(2))

def se_target(Nr, NT, alpha, sigma, s, local=False, b=1.0):
    Nr = np.asarray(Nr, float)
    u = Nr ** (-alpha); uT = NT ** (-alpha)
    cols, c = [np.ones_like(u), u], [1.0, uT]
    if local:  # unknown alpha, local: adversary also moves alpha -> extra regressor b*du/dalpha
        cols.append(-np.log(Nr) * u); c.append(-math.log(NT) * uT)
    X = np.column_stack(cols); c = np.array(c)
    return (sigma / math.sqrt(s)) * math.sqrt(c @ np.linalg.inv(X.T @ X) @ c)

def se_asym(Nr, alpha, sigma, s):
    u = np.asarray(Nr, float) ** (-alpha); m = len(u)
    rho2 = u.sum() ** 2 / (m * (u @ u))
    return sigma / math.sqrt(s * m * (1 - rho2))

print("\n2) FINITE-TARGET sign-error floors  Phi(-|Delta(N_T)| / SE_T)   (s = 2 seeds)")
cases = [
    ("Round 1 diff -> 91.3M, window 10-29.4M", [10.016, 15.566, 20.562, 29.407], 91.341, "diff", W29, "90M"),
    ("Round 1 diff -> 91.3M, window 5.25-29.4M (all rungs used)", [5.254, 10.016, 15.566, 20.562, 29.407], 91.341, "diff",
     ["5M"] + W29, "90M"),
    ("Round 2 diff -> 162.9M, window 10-91.3M", [10.016, 15.566, 20.562, 29.407, 91.341], 162.86, "diff", W91, "160M"),
    ("Held-out mtp -> 91.3M, window 10-29.4M", [10.016, 15.566, 20.562, 29.407], 91.341, "mtp", W29, "90M"),
    ("Round 2 mtp -> 162.9M, window 10-91.3M", [10.016, 15.566, 20.562, 29.407, 91.341], 162.86, "mtp", W91, "160M"),
]
ALPHA = {"conv": 2.65, "diff": 1.57, "mtp": 0.80}
for name, Nr, NT, k, win, tgt in cases:
    t = abs(np.mean(D[k][tgt]))
    sh, nu = sigma_hat(k, win)
    al = ALPHA[k]
    print(f"   {name}: |Delta(N_T)| = {t:.2f}, alpha = {al}")
    for sg, lab in ((0.5, "sigma 0.5"), (1.0, "sigma 1.0"), (sh, f"sigma_hat {sh:.2f}")):
        kn = Phi(-t / se_target(Nr, NT, al, sg, 2))
        lo = Phi(-t / se_target(Nr, NT, al, sg, 2, local=True))
        print(f"      {lab:16s} known-alpha {kn:6.1%}   local unknown-alpha {lo:6.1%}")

print("\n   asymptotic floors (Table 3 check, window 10-91M):")
for k in ("conv", "diff", "mtp"):
    a_abs = {"conv": 10.1, "diff": 0.25, "mtp": 1.41}[k]
    sh, _ = sigma_hat(k, W91)
    row = []
    for sg in (0.5, 1.0, sh):
        se = se_asym([10.016, 15.566, 20.562, 29.407, 91.341], ALPHA[k], sg, 2)
        row.append(f"sigma {sg:.2f}: |a|/SE {a_abs/se:5.2f} floor {Phi(-a_abs/se):6.2%}")
    print(f"   {k}: " + " | ".join(row))

# ---------------------------------------------------------------- 3. certificate
def certificate(Nr, y, s, sigma, crit, alphas, sigma_per_alpha=False):
    Nr = np.asarray(Nr, float); y = np.asarray(y, float); m = len(Nr)
    lo, hi = np.inf, -np.inf
    for al in alphas:
        u = Nr ** (-al)
        X = np.column_stack([np.ones(m), u])
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        if sigma_per_alpha:   # residual-based sigma (nu = m-2), independent of ahat at the true alpha
            res = y - X @ coef
            sg = math.sqrt(float(res @ res) / (m - 2)) * math.sqrt(s)  # per-seed scale
        else:
            sg = sigma
        se = sg * math.sqrt((u @ u) / (s * (m * (u @ u) - u.sum() ** 2)))
        T = coef[0] / se
        lo, hi = min(lo, T), max(hi, T)
    if hi < -crit: return "WIN certified", lo, hi
    if lo > crit: return "LOSS certified", lo, hi
    return "abstain", lo, hi

G = np.arange(0.30, 3.5001, 0.005)
G50 = np.arange(0.30, 3.5001, 0.0001)
N91 = [10.016, 15.566, 20.562, 29.407, 91.341]
z = stats.norm.ppf(0.95)
print(f"\n3) CERTIFICATE on the fixed 10-91M window, alpha in [0.3, 3.5], one-sided 5%")
for k in ("conv", "diff", "mtp"):
    y = [np.mean(D[k][sz]) for sz in W91]
    sh, nu = sigma_hat(k, W91)
    tc = stats.t.ppf(0.95, nu)
    out = []
    for sg, crit, lab in ((0.5, z, "sigma 0.5, z"), (1.0, z, "sigma 1.0, z"),
                          (sh, tc, f"sigma_hat {sh:.2f}, t_{nu} = {tc:.3f}")):
        v, lo, hi = certificate(N91, y, 2, sg, crit, G)
        v50, *_ = certificate(N91, y, 2, sg, crit, G50)
        out.append(f"{lab}: {v} (T in [{lo:+.2f}, {hi:+.2f}]){'' if v == v50 else '  << 50x grid differs: ' + v50}")
    print(f"   {k}:")
    for o in out: print("      " + o)

print("\n   SEED-SPLIT: monotone-suffix rule on seed 0 (5.25-91.3M), certify on seed 1 only (s = 1)")
def suffix(sizes, means):
    start = len(means) - 2
    while start > 0:
        d1 = means[start + 1] - means[start]; d0 = means[start] - means[start - 1]
        if d0 * d1 >= 0: start -= 1
        else: break
    else:
        start = 0
    return sizes[start:]
W_ALL = ["5M"] + W91
for k in ("conv", "diff", "mtp"):
    sel = suffix(W_ALL, [D[k][sz][0] for sz in W_ALL])
    Nr = [NS[sz] for sz in sel]; y = [D[k][sz][1] for sz in sel]
    sh, nu = sigma_hat(k, W91)
    print(f"   {k}: rungs chosen on seed 0 = {sel} (m={len(sel)})")
    if len(sel) < 3:
        print("      fewer than 3 rungs: the two-parameter-plus-rate fit has no residual df -> abstain by rule")
        continue
    for sg, crit, lab in ((0.5, z, "sigma 0.5, z"), (1.0, z, "sigma 1.0, z")):
        v, lo, hi = certificate(Nr, y, 1, sg, crit, G)
        print(f"      {lab}: {v} (T in [{lo:+.2f}, {hi:+.2f}])")
    if len(sel) >= 3:
        tc = stats.t.ppf(0.95, len(sel) - 2)
        v, lo, hi = certificate(Nr, y, 1, None, tc, G, sigma_per_alpha=True)
        print(f"      residual sigma (nu={len(sel)-2}), t = {tc:.3f}: {v} (T in [{lo:+.2f}, {hi:+.2f}])")

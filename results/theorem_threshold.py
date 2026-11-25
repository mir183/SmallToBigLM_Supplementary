"""Level 2, first numbers: the sign-identifiability threshold on OUR ladder.

Model: y_ij = a + b*N_i^(-alpha) + eps_ij, eps ~ N(0, sigma^2), i=1..m rungs, j=1..s seeds.
Question: when can ANY method determine sign(a)?

Both halves reduce to the same geometric quantity: with u_i = N_i^(-alpha) and
rho^2 = (sum u)^2 / (m * sum u^2)  (cos^2 of the angle between the constant vector and u),

  SE(a_hat) = sigma / sqrt(s * m * (1 - rho^2))        [GLS intercept standard error]

- Impossibility (Le Cam two-point + Pinsker): if |a| < SE/2, worst-case sign error >= 1/4
  for every test (adversary moves b to hide the asymptote's sign: the min separation of the
  two hypotheses is exactly the projection residual, 4a^2 * m * (1-rho^2)).
- Possibility: |a| >= 2*SE -> thresholding the LS estimate errs with prob <= ~2.3%.
So the threshold is SE(a_hat) up to a factor ~4, and 1/(1-rho^2) is the 'disguise factor':
slow decay (small alpha) makes N^(-alpha) mimic a constant inside the window and inflates it.
"""
import math
import numpy as np

rungs = np.array([10.0, 15.6, 20.6, 29.4, 91.3])   # the <=91M fit window (m=5)
s = 2                                               # seeds

def Phi(z):  # standard normal CDF
    return 0.5 * math.erfc(-z / math.sqrt(2))

def threshold(alpha, sigma):
    u = rungs ** (-alpha)
    m = len(u)
    rho2 = u.sum() ** 2 / (m * (u ** 2).sum())
    se = sigma / np.sqrt(s * m * (1 - rho2))
    return rho2, se

print(f"window {rungs.tolist()}M, s={s} seeds; thresholds in % (relative perplexity change)")
print("THEOREM (known alpha): minimax sign error = Phi(-|a| / SE(a_hat)), exactly.")
print(f"{'mech':5s} {'alpha':>6s} {'|a| fitted':>10s} {'sigma':>6s} {'rho^2':>6s} "
      f"{'SE(a)':>6s} {'|a|/SE':>7s} {'error floor':>11s}")
cases = [
    ("conv", 2.65, 10.1), ("diff", 1.57, 0.25), ("mtp", 0.80, 1.41),
]
for sigma in (0.5, 1.0):
    for name, alpha, a_abs in cases:
        rho2, se = threshold(alpha, sigma)
        r = a_abs / se
        floor = Phi(-r)
        print(f"{name:5s} {alpha:6.2f} {a_abs:10.2f} {sigma:6.1f} {rho2:6.3f} "
              f"{se:6.2f} {r:7.2f} {floor:10.1%}")
    print()

print("disguise factor 1/(1-rho^2) by alpha (same window):")
for alpha in (0.2, 0.5, 0.8, 1.57, 2.65):
    rho2, _ = threshold(alpha, 1.0)
    print(f"  alpha={alpha:4.2f}:  1/(1-rho^2) = {1/(1-rho2):6.1f}")

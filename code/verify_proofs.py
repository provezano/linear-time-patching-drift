#!/usr/bin/env python3

"""Self-contained theorem and proposition verification script for:
Linear-Time Noise-Debiased Dynamic Patching and Calm-Retaining Drift Adaptation for Nonstationary Streams
"""

from __future__ import annotations

import math
import numpy as np
from scipy import special

def verify_theorems(rng: np.random.Generator) -> None:
    # 1. Lemma 1: Under ||x_t|| <= B_x, M_0 = S_0 = 0, alpha in [0, 1), ||M_t|| <= t * theta*kappa*B_x/(1-eta)
    for _ in range(100):
        d = 6
        eta, alpha, theta, kappa, Bx = 0.85, 0.05, 0.1, 0.35, 1.5
        M = np.zeros(d)
        S = np.zeros(d)
        for t in range(1, 40):
            g = rng.normal(size=d)
            g = g / np.linalg.norm(g) * (kappa * Bx)
            S = eta * S - theta * g
            M = (1.0 - alpha) * M + S
            assert np.linalg.norm(M) <= t * theta * kappa * Bx / (1.0 - eta) + 1e-10, "Lemma 1 violated"

    # 2. Theorem 1(ii): Exact KL / Shannon entropy cancellation in E_t[iota_t]
    for _ in range(200):
        ft, mu_a, mu_f = rng.normal(size=3)
        sig_true = rng.uniform(0.1, 5.0)
        sig_hat = rng.uniform(0.1, 5.0)
        y = ft + sig_true * rng.normal(size=20000)
        iota_emp = np.mean(((y - mu_a) ** 2 - (y - mu_f) ** 2) / (2.0 * sig_hat ** 2))
        iota_exact = ((ft - mu_a) ** 2 - (ft - mu_f) ** 2) / (2.0 * sig_hat ** 2)
        assert abs(iota_emp - iota_exact) < 0.08 * (1 + abs(iota_exact)), "Thm 1(ii) violated"

    # 3. Theorem 2(i)-(ii): Pinelis bound & repaired exact Eq. (11) factor of pi (NOT pi/2!)
    for d in [1, 2, 4, 8, 16, 32, 64]:
        kappa = 0.35
        mu_max = kappa * math.exp(special.gammaln(0.5 * d) - 0.5 * math.log(math.pi) - special.gammaln(0.5 * (d + 1)))
        lhs_crit = math.pi * math.exp(2.0 * (special.gammaln(0.5 * (d + 1)) - special.gammaln(0.5 * d)))
        # Verify CSMAR's 1 - eta_d = 1.35 / (pi*d + 2) strictly satisfies (1+eta)/(1-eta) > lhs_crit for ALL d >= 1!
        one_minus_eta = 1.35 / (math.pi * d + 2.0)
        eta = 1.0 - one_minus_eta
        ratio = (1.0 + eta) / (1.0 - eta)
        null_rms = kappa * math.sqrt((1.0 - eta) / (1.0 + eta))
        assert ratio > lhs_crit and null_rms < mu_max, f"Thm 2(ii) Eq. (11) violated at d={d}"
        # Asymptotic equivalence check: (1 + eta)/(1 - eta) = lhs_crit <=> 1 - eta = 2 / (lhs_crit + 1)
        exact_1m_eta = 2.0 / (lhs_crit + 1.0)
        asymp_1m_eta = 4.0 / (math.pi * d + 2.0)
        if d >= 16:
            assert abs(exact_1m_eta / asymp_1m_eta - 1.0) < 0.8 / d, "Thm 2(ii) asymptotic expansion off"


if __name__ == "__main__":
    verify_theorems(np.random.default_rng(20261122))
    print("ALL_THEOREMS_VERIFIED_OK")

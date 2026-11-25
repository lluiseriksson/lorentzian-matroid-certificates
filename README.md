# Exact-Q Lorentzian Inertia Certificates, Coincident Down-Up Spectral-Gap Phase Transitions, and Machine-Checked Subgaussian Matrix Concentration for Matroids

**Author:** Lluis Eriksson (prepared with AI assistance — Google Antigravity)  
**License:** MIT (`LICENSE`)  
**Lean 4 Toolchain:** `leanprover/lean4:v4.35.0` (zero external dependencies, 18/18 theorems, 0 `sorry`, 0 `Classical.choice`, 0 warnings)  
**Certificate Schema:** `LORENTZ-MATROID-CERT-1.0` (pure Python stdlib `fractions.Fraction` only, zero floating-point trust)

## Overview

Advancing Eren Ercan (`arXiv:2611.00626`, Nov 2026), Brändén & Huh (*Ann. Math.* 2020), Anari–Liu–Oveis Gharan–Vinzant (STOC 2019), and Cryan–Guo–Mousa (*Ann. Probab.* 2021), this repository establishes four exact-$\mathbb{Q}$ and machine-checked results:

1. **Exact Coincident Lorentzian / Down-Up Spectral-Gap Phase Transition (Theorem 3.1):** On the two-cluster weighted graph family $G_{\varepsilon,m} = (K_m \sqcup K_m) + \varepsilon K_{m,m}$ ($m \ge 2$, $\varepsilon > 0$), the $2$-homogeneous generating polynomial $f_{\varepsilon,m}$ is Lorentzian ($\lambda_2(H_{\varepsilon,m}) \le 0$) **if and only if** the $2 \to 1 \to 2$ down-up spectral gap satisfies Ercan's optimal threshold $\gamma(\varepsilon, m) \ge 1/2$, with exact rational critical coupling $\varepsilon_c(m) = (m-1)/m$, exact piecewise spectral gap $\gamma(\varepsilon,m) = \frac{m\varepsilon}{(m-1)+m\varepsilon}$ ($\varepsilon \le 1$) and $\frac{1}{2} + \frac{1}{2((m-1)+m\varepsilon)}$ ($\varepsilon \ge 1$), and unique global maximum $\frac{m}{2m-1}$ at the unweighted complete graph $K_{2m}$ ($\varepsilon = 1$).
2. **Exact-$\mathbb{Q}$ Congruence Inertia & Down-Up Spectral-Gap Certificates Across Non-Representable Matroids (Theorem 3.2):** Via zero-float exact rational congruence $Q H Q^\top = \mathrm{diag}(D)$ ($Q Q^{-1} = I$), we certify complete $M$-convexity plus **all** $(r-2)$-link Lorentzian Hessians and strict excess down-up spectral gaps $\gamma(P_{r\to\ell\to r}) > 1 - \ell/r$ across uniform $U_{3,6}$ ($\gamma = 1/2$), graphic $M(K_4)$ ($\gamma = 3/4$ at $\ell=1$), Fano $F_7$, non-Fano $F_7^-$, and the **non-representable Vámos matroid $V_8$** (all 28 rank-2 links certified in 20.2 ms; exact gap $\gamma = 9/14 > 1/2$ at $\ell=2$; certified bracket $[46/125, 37/100] > 1/4$ at $\ell=3$).
3. **Zero-Float Exact-$\mathbb{Q}$ Matrix Subgaussian Tail Certificates (Theorem 3.3):** Combining Ercan's matrix entropy contraction with exact rational operator-norm PSD certificates $c^2 I_d - (A_e - A_f)^2 = Q^{-1} D (Q^{-1})^\top \succeq 0$ over all exchangeable pairs and exact rational Taylor-envelope denominators $T_K(x) = \sum_{k=0}^K x^k/k!$, we certify non-commuting Hermitian basis-sum tail bounds over $M(K_4)$ and non-representable $V_8$ purely over $\mathbb{Q}$, cross-checked against exact exhaustive enumeration over all bases.
4. **Pure-Stdlib Verifier `LORENTZ-MATROID-CERT-1.0` (Theorem 3.4):** Independently verifies all 16 exact-$\mathbb{Q}$ certificates with zero floats and rejects **48/48** live adversarial red-team attacks.

## Quick Verification

```bash
python scripts/verify_all.py
```

Re-run all experiments and regenerate figures:

```bash
python scripts/run_all_experiments.py
python scripts/make_figures.py
python scripts/verify_all.py
```

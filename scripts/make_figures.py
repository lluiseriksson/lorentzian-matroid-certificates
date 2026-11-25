"""Generate 3 publication-grade 300-DPI figures for Paper 11.

Author: Lluis Eriksson
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
FIGDIR = ROOT / "paper" / "figures"

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "legend.fontsize": 8.5,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "axes.grid": True,
    "grid.alpha": 0.28,
})


def make_fig1() -> None:
    d = json.loads((RESULTS / "exp2_coincident_phase_curve.json").read_text(encoding="utf-8"))
    curves = d["curves"]
    colors = {"2": "#1f77b4", "3": "#ff7f0e", "4": "#2ca02c", "6": "#d62728", "10": "#9467bd"}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.2), constrained_layout=True)

    for m_str in ["2", "3", "4", "6", "10"]:
        c = curves[m_str]
        eps = [p["eps"] for p in c["points"]]
        gap = [p["exact_gap"] for p in c["points"]]
        lam2 = [p["lambda_2"] for p in c["points"]]
        col = colors[m_str]
        lbl = rf"$m={m_str}$ ($\varepsilon_c={c['eps_c']}$)"
        ax1.plot(eps, gap, color=col, lw=2.0, label=lbl)
        ax1.scatter([c["eps_c_float"]], [0.5], color=col, s=36, zorder=5, edgecolor="black", lw=0.6)
        ax1.scatter([1.0], [c["peak_gap_float"]], color=col, marker="^", s=38, zorder=5, edgecolor="black", lw=0.6)
        ax2.plot(eps, lam2, color=col, lw=2.0, label=lbl)
        ax2.scatter([c["eps_c_float"]], [0.0], color=col, s=36, zorder=5, edgecolor="black", lw=0.6)

    ax1.axhline(0.5, color="black", ls="--", lw=1.2, label=r"Ercan bound $1-\ell/r = 1/2$")
    ax1.axvline(1.0, color="gray", ls=":", lw=1.1, label=r"Peak $\varepsilon=1$ ($K_{2m}$)")
    ax1.axhspan(0.0, 0.5, color="#ffdddd", alpha=0.25)
    ax1.axhspan(0.5, 0.72, color="#ddffdd", alpha=0.22)
    ax1.set_xlim(0.05, 2.5)
    ax1.set_ylim(0.0, 0.72)
    ax1.set_xlabel(r"Cross-cluster coupling $\varepsilon$")
    ax1.set_ylabel(r"Down-up spectral gap $\gamma(P_{2\to 1\to 2}; \varepsilon, m)$")
    ax1.set_title(r"(a) Exact Piecewise Spectral-Gap Phase Transition")
    ax1.legend(loc="lower right", frameon=True, framealpha=0.92)

    ax2.axhline(0.0, color="black", ls="--", lw=1.2, label=r"Lorentzian boundary $\lambda_2(H)=0$")
    ax2.axhspan(0.0, 8.5, color="#ffdddd", alpha=0.25)
    ax2.axhspan(-1.3, 0.0, color="#ddffdd", alpha=0.22)
    ax2.set_xlim(0.05, 2.5)
    ax2.set_ylim(-1.3, 8.5)
    ax2.set_xlabel(r"Cross-cluster coupling $\varepsilon$")
    ax2.set_ylabel(r"Second Hessian eigenvalue $\lambda_2(H_{\varepsilon,m})$")
    ax2.set_title(r"(b) Coincident Lorentzian Inertia Threshold $\varepsilon_c(m)=\frac{m-1}{m}$")
    ax2.text(1.38, 2.2, "Non-Lorentzian\n" + r"($\lambda_2 > 0, \gamma < 1/2$)", fontsize=8.5,
             bbox=dict(boxstyle="round,pad=0.3", fc="#fff0f0", ec="#cc5555", alpha=0.92))
    ax2.text(1.30, -0.72, r"Lorentzian ($\lambda_2 \leq 0, \gamma \geq 1/2$)", fontsize=8.5,
             bbox=dict(boxstyle="round,pad=0.3", fc="#f0fff0", ec="#2ca02c", alpha=0.92))
    ax2.legend(loc="upper right", frameon=True, framealpha=0.92)

    FIGDIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGDIR / "fig1_coincident_phase_transition.png", bbox_inches="tight")
    plt.close(fig)


def make_fig2() -> None:
    d3 = json.loads((RESULTS / "exp3_matroid_spectral_atlas.json").read_text(encoding="utf-8"))
    matroids = d3["matroids"]

    labels = []
    univ_vals = []
    true_vals = []
    excess_vals = []
    for m in matroids:
        for g in m["down_up_gaps"]:
            labels.append(f"${m['name']}$\n($\\ell={g['ell']}$)")
            univ_vals.append(g["universal_gap"])
            true_vals.append(g["true_gap"])
            excess_vals.append(g["excess_ratio"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.8, 4.3), constrained_layout=True)

    x = np.arange(len(labels))
    w = 0.36
    ax1.bar(x - w / 2, univ_vals, width=w, color="#8faadc", edgecolor="black", lw=0.7,
            label=r"Ercan universal bound $1 - \ell/r$")
    bars = ax1.bar(x + w / 2, true_vals, width=w, color="#2e7d32", edgecolor="black", lw=0.7,
                   label=r"Exact-$\mathbb{Q}$ certified gap $\gamma(P_{r\to\ell\to r})$")
    for idx, b in enumerate(bars):
        h = b.get_height()
        ax1.text(b.get_x() + b.get_width() / 2, h + 0.015, f"{h:.3f}", ha="center", va="bottom", fontsize=7.2, rotation=45)

    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=7.8)
    ax1.set_ylim(0, 1.08)
    ax1.set_ylabel("Spectral gap")
    ax1.set_title("(a) Certified Spectral Gaps vs. Ercan Universal Bound")
    ax1.legend(loc="upper right", frameon=True, framealpha=0.92)

    # Panel (b): Excess ratio gamma / (1 - ell/r) sorted
    colors_b = ["#c62828" if "V_8" in lb else "#1565c0" for lb in labels]
    ax2.barh(x, excess_vals, color=colors_b, edgecolor="black", lw=0.7, height=0.62)
    ax2.axvline(1.0, color="black", ls="--", lw=1.2, label=r"Universal baseline ($1.0\times$)")
    for idx, val in enumerate(excess_vals):
        ax2.text(val + 0.02, idx, f"{val:.3f}x", va="center", ha="left", fontsize=8.0)
    ax2.set_yticks(x)
    ax2.set_yticklabels([lb.replace("\n", " ") for lb in labels], fontsize=8.0)
    ax2.set_xlim(0.85, 1.72)
    ax2.set_xlabel(r"Certified excess factor $\gamma(P_{r\to\ell\to r}) / (1 - \ell/r)$")
    ax2.set_title(r"(b) Excess Factor Above $1-\ell/r$ (Red: Non-Representable $V_8$)")
    ax2.legend(loc="lower right", frameon=True, framealpha=0.92)

    fig.savefig(FIGDIR / "fig2_matroid_spectral_atlas.png", bbox_inches="tight")
    plt.close(fig)


def make_fig3() -> None:
    d4 = json.loads((RESULTS / "exp4_matrix_concentration_exact_enum.json").read_text(encoding="utf-8"))
    models = d4["models"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.2), constrained_layout=True)

    for ax, key, title in [
        (ax1, "M(K_4)", r"(a) Graphic Matroid $M(K_4)$ ($r=3, |\mathcal{B}|=16, d=2$)"),
        (ax2, "V_8", r"(b) Non-Representable Vámos $V_8$ ($r=4, |\mathcal{B}|=65, d=2$)"),
    ]:
        pts = models[key]["points"]
        t = [p["t"] for p in pts]
        emp = [max(p["empirical_tail"], 1e-6) for p in pts]
        lap = [p["exact_laplace_bound"] for p in pts]
        sg = [p["ercan_subgaussian_bound"] for p in pts]
        rat = [p["rational_taylor8_bound"] for p in pts]

        ax.semilogy(t, rat, "s-", color="#d62728", lw=1.8, ms=4.5,
                    label=r"Exact-$\mathbb{Q}$ Taylor-$8$ cert $d / T_8(t^2/(rc^2))$")
        ax.semilogy(t, sg, "--", color="#1f77b4", lw=2.0,
                    label=r"Ercan subgaussian $d \exp(-t^2/(rc^2))$")
        ax.semilogy(t, lap, "-.", color="#2ca02c", lw=1.8,
                    label=r"Exact matrix Laplace $\inf_\theta e^{-\theta t}\mathbb{E}\mathrm{Tr}\,e^{\theta \tilde F}$")
        # Plot empirical tail only where positive, with marker for zero transition
        t_pos = [p["t"] for p in pts if p["empirical_tail"] > 0]
        e_pos = [p["empirical_tail"] for p in pts if p["empirical_tail"] > 0]
        ax.semilogy(t_pos, e_pos, "ko-", lw=1.8, ms=5.0, label=r"Exact exhaustive empirical tail")
        ax.set_ylim(1e-4, 1.4)
        ax.set_xlim(0.2, 4.05)
        ax.set_xlabel(r"Deviation threshold $t$")
        ax.set_ylabel(r"Upper tail probability / bound")
        ax.set_title(title)
        ax.legend(loc="lower left", frameon=True, framealpha=0.92, fontsize=7.8)

    fig.savefig(FIGDIR / "fig3_matrix_concentration_tails.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    make_fig1()
    make_fig2()
    make_fig3()
    print("[FIGURES] Generated 3 publication-grade 300-DPI figures in paper/figures/.")


if __name__ == "__main__":
    main()

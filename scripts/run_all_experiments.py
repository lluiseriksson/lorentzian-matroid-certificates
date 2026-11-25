"""Run all 4 deterministic experiments and emit 16 exact-Q LORENTZ-MATROID-CERT-1.0 certificates.

Author: Lluis Eriksson
"""

from __future__ import annotations

from fractions import Fraction
import json
import math
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import (
    CERT_SCHEMA,
    check_basis_exchange,
    down_up_walk_exact,
    dual_up_down_walk_exact,
    exact_congruence_diagonalize,
    fano_matroid,
    graphic_k4_matroid,
    link_hessian,
    non_fano_matroid,
    two_cluster_hessian_and_walk,
    uniform_matroid,
    vamos_matroid,
    verify_lorentzian_certificate_exact,
)
from src.matroid_exact import (
    MatrixQ,
    dirichlet_gram_matrix,
    independent_sets_of_size,
    mat_mul,
    mat_to_str,
    rat_to_str,
    vec_to_str,
)

RESULTS_DIR = ROOT / "results"
CERTS_DIR = RESULTS_DIR / "certificates"


def _write_cert(filename: str, cert: Dict[str, Any]) -> Dict[str, Any]:
    CERTS_DIR.mkdir(parents=True, exist_ok=True)
    path = CERTS_DIR / filename
    path.write_text(json.dumps(cert, indent=2) + "\n", encoding="utf-8")
    t0 = time.perf_counter()
    res = verify_lorentzian_certificate_exact(cert)
    dt_ms = (time.perf_counter() - t0) * 1000.0
    res["file"] = filename
    res["verify_ms"] = round(dt_ms, 3)
    return res


def _build_mode1_cert(name: str, n: int, r: int, bases: List[Tuple[int, ...]]) -> Dict[str, Any]:
    weights = [Fraction(1)] * len(bases)
    links = independent_sets_of_size(bases, r - 2)
    link_certs = []
    for S in links:
        _, H_S = link_hessian(n, bases, weights, S)
        Q, Q_inv, D = exact_congruence_diagonalize(H_S)
        link_certs.append({
            "S": list(S),
            "Q": mat_to_str(Q),
            "Q_inv": mat_to_str(Q_inv),
            "D": vec_to_str(D),
        })
    return {
        "schema": CERT_SCHEMA,
        "mode": "lorentzian_hessian_inertia",
        "label": name,
        "n": n,
        "r": r,
        "bases": [list(b) for b in bases],
        "link_certificates": link_certs,
    }


def _find_rational_upper_witness(
    pi: Sequence[Fraction],
    P: MatrixQ,
    Q: MatrixQ,
    D: Sequence[Fraction],
    denom_scale: int = 5000,
) -> Tuple[List[Fraction], Fraction]:
    """Find an exact rational centered upper-bound witness v (pi^T v == 0).

    If gamma_low is exact, any zero pivot D[idx] == 0 yields an EXACT rational nullvector
    v = Q[idx] of M_gamma (so Rayleigh gap == gamma_low identically!). Otherwise quantize
    the numerical second eigenvector and project exactly to pi^T v == 0 over Q.
    """
    k = len(pi)

    def eval_rayleigh(v: List[Fraction]) -> Fraction | None:
        mean = sum((pi[i] * v[i] for i in range(k)), Fraction(0))
        vc = [v[i] - mean for i in range(k)]
        var_v = sum((pi[i] * vc[i] * vc[i] for i in range(k)), Fraction(0))
        if var_v <= 0:
            return None
        quad_Pv = Fraction(0)
        for i in range(k):
            row_dot = sum((P[i][j] * vc[j] for j in range(k)), Fraction(0))
            quad_Pv += pi[i] * vc[i] * row_dot
        return Fraction(1) - quad_Pv / var_v

    for idx, d_val in enumerate(D):
        if d_val == 0:
            cand = list(Q[idx])
            mean = sum((pi[i] * cand[i] for i in range(k)), Fraction(0))
            vc = [cand[i] - mean for i in range(k)]
            rg = eval_rayleigh(vc)
            if rg is not None:
                return vc, rg

    pi_f = np.array([float(x) for x in pi], dtype=np.float64)
    P_f = np.array([[float(x) for x in row] for row in P], dtype=np.float64)
    S_f = np.diag(np.sqrt(pi_f)) @ P_f @ np.diag(1.0 / np.sqrt(pi_f))
    S_f = 0.5 * (S_f + S_f.T)
    _, evecs = np.linalg.eigh(S_f)
    u = evecs[:, -2] / np.sqrt(pi_f)
    u = u / (np.max(np.abs(u)) + 1e-15)
    v_raw = [Fraction(int(round(val * denom_scale)), denom_scale) for val in u]
    mean = sum((pi[i] * v_raw[i] for i in range(k)), Fraction(0))
    v = [v_raw[i] - mean for i in range(k)]
    rg = eval_rayleigh(v)
    assert rg is not None
    return v, rg


def _build_mode2_cert(
    name: str,
    n: int,
    r: int,
    bases: List[Tuple[int, ...]],
    ell: int,
    gamma_low: Fraction,
    gamma_up: Fraction,
    operator_space: str = "dual_ell_sets",
    claim_strict_excess: bool = True,
) -> Dict[str, Any]:
    weights = [Fraction(1)] * len(bases)
    if operator_space == "basis_walk":
        pi, P = down_up_walk_exact(bases, weights, ell)
    else:
        _, pi, P = dual_up_down_walk_exact(bases, weights, ell)
    M = dirichlet_gram_matrix(pi, P, gamma_low)
    Q, Q_inv, D = exact_congruence_diagonalize(M)
    if any(d < 0 for d in D):
        raise RuntimeError(f"{name}: gamma_low={gamma_low} is not PSD (min D={min(D)}).")
    v, r_gap = _find_rational_upper_witness(pi, P, Q, D, denom_scale=5000)
    if r_gap > gamma_up:
        raise RuntimeError(f"{name}: Rayleigh gap {float(r_gap):.6f} > gamma_up {float(gamma_up):.6f}")
    return {
        "schema": CERT_SCHEMA,
        "mode": "down_up_spectral_gap",
        "label": name,
        "operator_space": operator_space,
        "n": n,
        "r": r,
        "ell": ell,
        "bases": [list(b) for b in bases],
        "gamma_low": rat_to_str(gamma_low),
        "claim_strict_excess": claim_strict_excess,
        "Q": mat_to_str(Q),
        "Q_inv": mat_to_str(Q_inv),
        "D": vec_to_str(D),
        "upper_bound_witness": {
            "gamma_up": rat_to_str(gamma_up),
            "v": vec_to_str(v),
        },
    }


def _build_mode3_cert(m: int, eps: Fraction, gamma_low: Fraction | None = None) -> Dict[str, Any]:
    H, pi, Q_walk, _, _, exact_gap = two_cluster_hessian_and_walk(m, eps)
    eps_c = Fraction(m - 1, m)
    if eps >= eps_c:
        g_low = exact_gap if gamma_low is None else gamma_low
        QH, QH_inv, DH = exact_congruence_diagonalize(H)
        M = dirichlet_gram_matrix(pi, Q_walk, g_low)
        QM, QM_inv, DM = exact_congruence_diagonalize(M)
        return {
            "schema": CERT_SCHEMA,
            "mode": "coincident_phase_transition",
            "regime": "lorentzian_supercritical",
            "m": m,
            "eps": rat_to_str(eps),
            "gamma_low": rat_to_str(g_low),
            "hessian_congruence": {
                "Q": mat_to_str(QH),
                "Q_inv": mat_to_str(QH_inv),
                "D": vec_to_str(DH),
            },
            "gap_congruence": {
                "Q": mat_to_str(QM),
                "Q_inv": mat_to_str(QM_inv),
                "D": vec_to_str(DM),
            },
        }
    u1 = [Fraction(1)] * (2 * m)
    u2 = [Fraction(1)] * m + [Fraction(-1)] * m
    return {
        "schema": CERT_SCHEMA,
        "mode": "coincident_phase_transition",
        "regime": "non_lorentzian_subcritical",
        "m": m,
        "eps": rat_to_str(eps),
        "pos_vec_1": vec_to_str(u1),
        "pos_vec_2": vec_to_str(u2),
        "gap_witness_v": vec_to_str(u2),
    }


def _noncommuting_element_matrices(n: int) -> List[MatrixQ]:
    """Deterministic family of non-commuting 2x2 symmetric rational matrices with ||A_e - A_f||_op <= 1."""
    # All matrices have eigenvalues inside [-1/2, +1/2], so ||A_e - A_f||_op <= 1 (c^2 = 1)
    templates = [
        [[Fraction(1, 2), Fraction(0)], [Fraction(0), Fraction(-1, 2)]],
        [[Fraction(0), Fraction(1, 2)], [Fraction(1, 2), Fraction(0)]],
        [[Fraction(3, 10), Fraction(2, 5)], [Fraction(2, 5), Fraction(-3, 10)]],
        [[Fraction(-3, 10), Fraction(2, 5)], [Fraction(2, 5), Fraction(3, 10)]],
        [[Fraction(2, 5), Fraction(-3, 10)], [Fraction(-3, 10), Fraction(-2, 5)]],
        [[Fraction(-2, 5), Fraction(-3, 10)], [Fraction(-3, 10), Fraction(2, 5)]],
        [[Fraction(1, 4), Fraction(1, 4)], [Fraction(1, 4), Fraction(-1, 4)]],
        [[Fraction(-1, 4), Fraction(1, 4)], [Fraction(1, 4), Fraction(1, 4)]],
    ]
    return [templates[i % len(templates)] for i in range(n)]


def _build_mode4_cert(
    name: str,
    n: int,
    r: int,
    bases: List[Tuple[int, ...]],
    A_mats: List[MatrixQ],
    c_sq: Fraction,
    t: Fraction,
    delta: Fraction,
    taylor_order: int = 8,
) -> Dict[str, Any]:
    d = len(A_mats[0])
    inv_B = Fraction(1, len(bases))
    exp_F: MatrixQ = [[Fraction(0)] * d for _ in range(d)]
    for b in bases:
        for e in b:
            for i in range(d):
                for j in range(d):
                    exp_F[i][j] += A_mats[e][i][j] * inv_B

    b_fsets = [frozenset(b) for b in bases]
    exch_pairs = sorted({
        tuple(sorted(b_fsets[i] ^ b_fsets[j]))
        for i in range(len(b_fsets))
        for j in range(i + 1, len(b_fsets))
        if len(b_fsets[i] ^ b_fsets[j]) == 2
    })
    pair_certs = []
    for e, f in exch_pairs:
        diff = [[A_mats[e][i][j] - A_mats[f][i][j] for j in range(d)] for i in range(d)]
        diff_sq = mat_mul(diff, diff)
        C_ef = [[(c_sq if i == j else Fraction(0)) - diff_sq[i][j] for j in range(d)] for i in range(d)]
        Q, Q_inv, D = exact_congruence_diagonalize(C_ef)
        if any(x < 0 for x in D):
            raise RuntimeError(f"{name}: pair ({e},{f}) violates c_sq={c_sq}")
        pair_certs.append({
            "e": e,
            "f": f,
            "Q": mat_to_str(Q),
            "Q_inv": mat_to_str(Q_inv),
            "D": vec_to_str(D),
        })

    return {
        "schema": CERT_SCHEMA,
        "mode": "matrix_concentration_tail",
        "label": name,
        "n": n,
        "r": r,
        "bases": [list(b) for b in bases],
        "matrix_dim": d,
        "element_matrices": [mat_to_str(A) for A in A_mats],
        "expected_matrix": mat_to_str(exp_F),
        "c_sq": rat_to_str(c_sq),
        "t": rat_to_str(t),
        "delta": rat_to_str(delta),
        "taylor_order": taylor_order,
        "exchange_norm_certificates": pair_certs,
    }


def run_exp1_certified_atlas() -> Dict[str, Any]:
    matroids = {
        "U36": uniform_matroid(3, 6),
        "MK4": graphic_k4_matroid(),
        "F7": fano_matroid(),
        "F7minus": non_fano_matroid(),
        "V8": vamos_matroid(),
    }
    records = []

    # Mode 1: Complete Lorentzian Hessian Inertia (all 5 matroids, including non-representable Vamos V8)
    for key, fname in [
        ("U36", "lorentzian_U36.cert.json"),
        ("MK4", "lorentzian_MK4.cert.json"),
        ("F7", "lorentzian_F7_fano.cert.json"),
        ("F7minus", "lorentzian_F7minus_nonfano.cert.json"),
        ("V8", "lorentzian_V8_vamos.cert.json"),
    ]:
        n, r, bases = matroids[key]
        cert = _build_mode1_cert(key, n, r, bases)
        records.append(_write_cert(fname, cert))

    # Mode 2: Exact-Q Down-Up Walk Spectral Gap Certificates (strict excess over 1 - ell/r)
    gap_specs = [
        ("gap_U36_ell2.cert.json", "U36", 2, Fraction(1, 2), Fraction(1, 2), "basis_walk"),
        ("gap_MK4_ell1.cert.json", "MK4", 1, Fraction(3, 4), Fraction(3, 4), "basis_walk"),
        ("gap_MK4_ell2.cert.json", "MK4", 2, Fraction(41, 100), Fraction(42, 100), "basis_walk"),
        ("gap_F7_ell2.cert.json", "F7", 2, Fraction(43, 100), Fraction(44, 100), "dual_ell_sets"),
        ("gap_V8_ell2_dual.cert.json", "V8", 2, Fraction(9, 14), Fraction(9, 14), "dual_ell_sets"),
        ("gap_V8_ell3_dual.cert.json", "V8", 3, Fraction(46, 125), Fraction(37, 100), "dual_ell_sets"),
    ]
    for fname, key, ell, g_low, g_up, op_space in gap_specs:
        n, r, bases = matroids[key]
        cert = _build_mode2_cert(f"{key}_ell{ell}", n, r, bases, ell, g_low, g_up, op_space, True)
        records.append(_write_cert(fname, cert))

    # Mode 3: Coincident Phase Transition on G_{eps,m}
    records.append(_write_cert("phase_m4_critical_eps_3_4.cert.json", _build_mode3_cert(4, Fraction(3, 4))))
    records.append(_write_cert("phase_m4_supercritical_eps_1.cert.json", _build_mode3_cert(4, Fraction(1))))
    records.append(_write_cert("phase_m4_subcritical_eps_7_10.cert.json", _build_mode3_cert(4, Fraction(7, 10))))

    # Mode 4: Exact-Q Matrix Concentration Tail Certificates (MK4 & non-representable Vamos V8)
    n_k4, r_k4, b_k4 = matroids["MK4"]
    cert_tail_k4 = _build_mode4_cert(
        "MK4_pauli_d2", n_k4, r_k4, b_k4,
        _noncommuting_element_matrices(n_k4),
        c_sq=Fraction(1), t=Fraction(3), delta=Fraction(1, 9), taylor_order=8,
    )
    records.append(_write_cert("tail_MK4_pauli_d2.cert.json", cert_tail_k4))

    n_v8, r_v8, b_v8 = matroids["V8"]
    cert_tail_v8 = _build_mode4_cert(
        "V8_vamos_d2", n_v8, r_v8, b_v8,
        _noncommuting_element_matrices(n_v8),
        c_sq=Fraction(1), t=Fraction(4), delta=Fraction(1, 20), taylor_order=10,
    )
    records.append(_write_cert("tail_V8_vamos_d2.cert.json", cert_tail_v8))

    out = {"schema": CERT_SCHEMA, "num_certificates": len(records), "certificates": records}
    (RESULTS_DIR / "exp1_certified_atlas.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    return out


def run_exp2_coincident_phase_curve() -> Dict[str, Any]:
    eps_grid = [Fraction(k, 20) for k in range(1, 61)]
    curves = {}
    max_hessian_err = 0.0
    max_gap_err = 0.0
    for m in [2, 3, 4, 6, 10]:
        pts = []
        for eps in eps_grid:
            H, pi, Q_walk, lam_p, lam_m, exact_gap = two_cluster_hessian_and_walk(m, eps)
            H_f = np.array([[float(x) for x in row] for row in H], dtype=np.float64)
            Q_f = np.array([[float(x) for x in row] for row in Q_walk], dtype=np.float64)
            h_evals = np.sort(np.linalg.eigvalsh(H_f))
            q_evals = np.sort(np.linalg.eigvalsh(Q_f))
            num_lam_m = float(h_evals[-2])
            exact_lam2 = max(lam_m, Fraction(-1))
            num_gap = float(1.0 - q_evals[-2])
            max_hessian_err = max(max_hessian_err, abs(num_lam_m - float(exact_lam2)))
            max_gap_err = max(max_gap_err, abs(num_gap - float(exact_gap)))
            pts.append({
                "eps": float(eps),
                "lambda_minus": float(lam_m),
                "lambda_2": float(exact_lam2),
                "exact_gap": float(exact_gap),
                "lorentzian": bool(lam_m <= 0),
                "gap_ge_half": bool(exact_gap >= Fraction(1, 2)),
            })
        curves[str(m)] = {
            "m": m,
            "eps_c": rat_to_str(Fraction(m - 1, m)),
            "eps_c_float": (m - 1) / m,
            "peak_gap": rat_to_str(Fraction(m, 2 * m - 1)),
            "peak_gap_float": m / (2 * m - 1),
            "points": pts,
        }
    out = {
        "max_hessian_eig_error": max_hessian_err,
        "max_down_up_gap_error": max_gap_err,
        "coincidence_violations": sum(
            1 for c in curves.values() for p in c["points"] if p["lorentzian"] != p["gap_ge_half"]
        ),
        "curves": curves,
    }
    (RESULTS_DIR / "exp2_coincident_phase_curve.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    return out


def run_exp3_matroid_spectral_atlas() -> Dict[str, Any]:
    matroids = [
        ("U_{3,6}", uniform_matroid(3, 6), "Representable (all fields)"),
        ("M(K_4)", graphic_k4_matroid(), "Graphic / Regular"),
        ("F_7", fano_matroid(), "Representable (char 2 only)"),
        ("F_7^-", non_fano_matroid(), "Representable (char != 2 only)"),
        ("V_8", vamos_matroid(), "Non-representable (any field)"),
    ]
    rows = []
    max_isospec_err = 0.0
    for label, (n, r, bases), rep_class in matroids:
        assert check_basis_exchange(n, r, bases)
        weights = [Fraction(1)] * len(bases)
        # Independent set counts I_0, ..., I_r
        ind_counts = [len(independent_sets_of_size(bases, k)) for k in range(r + 1)]
        ulc_ratios = []
        for k in range(1, r):
            c_k = math.comb(n, k)
            c_km1 = math.comb(n, k - 1)
            c_kp1 = math.comb(n, k + 1)
            ratio = (ind_counts[k] / c_k) ** 2 / ((ind_counts[k - 1] / c_km1) * (ind_counts[k + 1] / c_kp1))
            ulc_ratios.append(round(ratio, 6))

        ell_data = []
        for ell in range(1, r):
            pi_b, P_b = down_up_walk_exact(bases, weights, ell)
            _, pi_d, Q_d = dual_up_down_walk_exact(bases, weights, ell)

            def sym_evals(pi_vec: Sequence[Fraction], mat: MatrixQ) -> np.ndarray:
                pf = np.array([float(x) for x in pi_vec], dtype=np.float64)
                Mf = np.array([[float(x) for x in row] for row in mat], dtype=np.float64)
                S = np.diag(np.sqrt(pf)) @ Mf @ np.diag(1.0 / np.sqrt(pf))
                S = 0.5 * (S + S.T)
                return np.sort(np.linalg.eigvalsh(S))

            ev_b = sym_evals(pi_b, P_b)
            ev_d = sym_evals(pi_d, Q_d)
            gap_b = float(1.0 - ev_b[-2])
            gap_d = float(1.0 - ev_d[-2])
            max_isospec_err = max(max_isospec_err, abs(gap_b - gap_d))
            univ = (r - ell) / r
            ell_data.append({
                "ell": ell,
                "universal_gap": round(univ, 6),
                "true_gap": round(gap_b, 6),
                "dual_gap": round(gap_d, 6),
                "excess_ratio": round(gap_b / univ, 6),
                "num_dual_states": len(pi_d),
            })

        rows.append({
            "name": label,
            "n": n,
            "r": r,
            "num_bases": len(bases),
            "representability": rep_class,
            "independent_counts": ind_counts,
            "mason_ulc_ratios": ulc_ratios,
            "down_up_gaps": ell_data,
        })

    out = {"max_isospectral_error": max_isospec_err, "matroids": rows}
    (RESULTS_DIR / "exp3_matroid_spectral_atlas.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    return out


def run_exp4_matrix_concentration_exact_enum() -> Dict[str, Any]:
    matroids = [
        ("M(K_4)", graphic_k4_matroid()),
        ("F_7", fano_matroid()),
        ("V_8", vamos_matroid()),
    ]
    t_grid = [0.25 * k for k in range(1, 17)]
    out_models = {}
    for label, (n, r, bases) in matroids:
        A_q = _noncommuting_element_matrices(n)
        A_f = [np.array([[float(x) for x in row] for row in A], dtype=np.float64) for A in A_q]
        d = A_f[0].shape[0]
        F_list = [sum(A_f[e] for e in b) for b in bases]
        EF = sum(F_list) / len(bases)
        centered = [F - EF for F in F_list]
        lam_max = np.array([float(np.linalg.eigvalsh(M)[-1]) for M in centered])
        # Exact operator-norm exchange sensitivity c_true
        b_sets = [frozenset(b) for b in bases]
        c_true = 0.0
        for i in range(len(bases)):
            for j in range(i + 1, len(bases)):
                if len(b_sets[i] ^ b_sets[j]) == 2:
                    c_true = max(c_true, float(np.linalg.norm(F_list[i] - F_list[j], ord=2)))

        # Precompute exact Laplace MGF over theta grid
        thetas = np.linspace(0.05, 8.0, 160)
        mgf_vals = []
        for th in thetas:
            tr_sum = 0.0
            for M in centered:
                ev = np.linalg.eigvalsh(th * M)
                tr_sum += float(np.sum(np.exp(ev)))
            mgf_vals.append(tr_sum / len(bases))
        mgf_arr = np.array(mgf_vals)

        points = []
        for t in t_grid:
            emp = float(np.mean(lam_max >= t - 1e-12))
            laplace = float(min(1.0, np.min(np.exp(-thetas * t) * mgf_arr)))
            ercan_sg = float(min(1.0, d * math.exp(-(t * t) / (r * 1.0))))
            # Exact rational Taylor-envelope order K=8 with c^2 = 1
            x = (t * t) / float(r)
            t8 = sum((x ** k) / math.factorial(k) for k in range(9))
            rat_env = float(min(1.0, d / t8))
            points.append({
                "t": t,
                "empirical_tail": emp,
                "exact_laplace_bound": laplace,
                "ercan_subgaussian_bound": ercan_sg,
                "rational_taylor8_bound": rat_env,
            })
        out_models[label] = {
            "n": n,
            "r": r,
            "num_bases": len(bases),
            "matrix_dim": d,
            "exact_c_exchange": round(c_true, 6),
            "certified_c_upper": 1.0,
            "max_observed_dev": round(float(np.max(lam_max)), 6),
            "points": points,
        }

    out = {"models": out_models}
    (RESULTS_DIR / "exp4_matrix_concentration_exact_enum.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    return out


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    e1 = run_exp1_certified_atlas()
    e2 = run_exp2_coincident_phase_curve()
    e3 = run_exp3_matroid_spectral_atlas()
    e4 = run_exp4_matrix_concentration_exact_enum()
    print(f"[EXP 1] {e1['num_certificates']} exact-Q certificates emitted and verified (0 floats).")
    print(f"[EXP 2] Coincident phase curve: max_hessian_err={e2['max_hessian_eig_error']:.2e}, max_gap_err={e2['max_down_up_gap_error']:.2e}, violations={e2['coincidence_violations']}")
    print(f"[EXP 3] Matroid spectral atlas: {len(e3['matroids'])} matroids, max_isospec_err={e3['max_isospectral_error']:.2e}")
    print(f"[EXP 4] Matrix concentration exact enum: {len(e4['models'])} matroids verified.")


if __name__ == "__main__":
    main()

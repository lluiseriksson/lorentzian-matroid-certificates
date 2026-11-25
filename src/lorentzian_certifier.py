"""Pure-stdlib zero-floating-point-trust certificate verifier (LORENTZ-MATROID-CERT-1.0).

Supports 4 mutually exclusive verification modes over exact rationals Q (fractions.Fraction):
  1. lorentzian_hessian_inertia   : Complete M-convexity + all (r-2)-link Hessian inertia certificates.
  2. down_up_spectral_gap         : Exact reversible down-up walk Dirichlet-Gram PSD + Rayleigh bracket.
  3. coincident_phase_transition  : Sharp two-cluster G_{eps,m} Lorentzian / spectral-gap dichotomy.
  4. matrix_concentration_tail    : Exact exchange-norm PSD + Taylor-envelope subgaussian tail bound.

Author: Lluis Eriksson
"""

from __future__ import annotations

from fractions import Fraction
from itertools import combinations
import re
from typing import Any, Dict, List, Sequence, Tuple

from .matroid_exact import (
    MatrixQ,
    check_basis_exchange,
    dirichlet_gram_matrix,
    down_up_walk_exact,
    dual_up_down_walk_exact,
    independent_sets_of_size,
    link_hessian,
    mat_equal,
    mat_eye,
    mat_is_symmetric,
    mat_mul,
    mat_transpose,
    two_cluster_hessian_and_walk,
)

CERT_SCHEMA = "LORENTZ-MATROID-CERT-1.0"
VALID_MODES = {
    "lorentzian_hessian_inertia",
    "down_up_spectral_gap",
    "coincident_phase_transition",
    "matrix_concentration_tail",
}

_RAT_RE = re.compile(r"\A-?[0-9]+(?:/[1-9][0-9]*)?\Z")


class CertificateVerificationError(ValueError):
    """Raised whenever an exact-Q certificate fails any mathematical or schema check."""


def _assert_no_floats(obj: Any, path: str = "root") -> None:
    if isinstance(obj, float):
        raise CertificateVerificationError(f"Floating-point value forbidden at {path}: {obj!r}")
    if isinstance(obj, dict):
        for k, v in obj.items():
            _assert_no_floats(v, f"{path}.{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            _assert_no_floats(v, f"{path}[{i}]")


def _parse_int(val: Any, name: str) -> int:
    if isinstance(val, bool) or not isinstance(val, int):
        raise CertificateVerificationError(f"{name} must be an exact int, got {type(val).__name__}")
    return val


def _parse_rat(val: Any, name: str) -> Fraction:
    if isinstance(val, bool):
        raise CertificateVerificationError(f"{name} cannot be bool.")
    if isinstance(val, int):
        return Fraction(val)
    if isinstance(val, str):
        if not _RAT_RE.match(val):
            raise CertificateVerificationError(f"{name} is not a canonical rational string: {val!r}")
        if "/" in val:
            num_s, den_s = val.split("/")
            return Fraction(int(num_s), int(den_s))
        return Fraction(int(val))
    raise CertificateVerificationError(f"{name} must be exact int or rational string, got {type(val).__name__}")


def _parse_vec(val: Any, length: int, name: str) -> List[Fraction]:
    if not isinstance(val, list) or len(val) != length:
        raise CertificateVerificationError(f"{name} must be a list of length {length}.")
    return [_parse_rat(x, f"{name}[{i}]") for i, x in enumerate(val)]


def _parse_mat(val: Any, n: int, name: str) -> MatrixQ:
    if not isinstance(val, list) or len(val) != n:
        raise CertificateVerificationError(f"{name} must be an {n}x{n} matrix.")
    out: MatrixQ = []
    for i, row in enumerate(val):
        if not isinstance(row, list) or len(row) != n:
            raise CertificateVerificationError(f"{name}[{i}] must have length {n}.")
        out.append([_parse_rat(x, f"{name}[{i}][{j}]") for j, x in enumerate(row)])
    return out


def _verify_congruence(
    H: MatrixQ,
    Q_raw: Any,
    Q_inv_raw: Any,
    D_raw: Any,
    label: str,
) -> List[Fraction]:
    n = len(H)
    if not mat_is_symmetric(H):
        raise CertificateVerificationError(f"{label}: target matrix is not symmetric.")
    Q = _parse_mat(Q_raw, n, f"{label}.Q")
    Q_inv = _parse_mat(Q_inv_raw, n, f"{label}.Q_inv")
    D = _parse_vec(D_raw, n, f"{label}.D")
    # 1. Verify Q is nonsingular over Q via Q * Q_inv == I_n
    if not mat_equal(mat_mul(Q, Q_inv), mat_eye(n)):
        raise CertificateVerificationError(f"{label}: Q * Q_inv != I (congruence matrix not invertible).")
    # 2. Verify Q * H * Q^T == diag(D)
    QHQt = mat_mul(mat_mul(Q, H), mat_transpose(Q))
    diag_D = [[D[i] if i == j else Fraction(0) for j in range(n)] for i in range(n)]
    if not mat_equal(QHQt, diag_D):
        raise CertificateVerificationError(f"{label}: Q * H * Q^T != diag(D).")
    return D


def _parse_bases_and_weights(cert: Dict[str, Any]) -> Tuple[int, int, List[Tuple[int, ...]], List[Fraction]]:
    n = _parse_int(cert.get("n"), "n")
    r = _parse_int(cert.get("r"), "r")
    raw_bases = cert.get("bases")
    if not isinstance(raw_bases, list) or not raw_bases:
        raise CertificateVerificationError("bases must be a non-empty list of r-tuples.")
    bases: List[Tuple[int, ...]] = []
    for idx, b in enumerate(raw_bases):
        if not isinstance(b, list) or len(b) != r:
            raise CertificateVerificationError(f"bases[{idx}] must be a list of length {r}.")
        b_int = tuple(_parse_int(e, f"bases[{idx}]") for e in b)
        if tuple(sorted(b_int)) != b_int or len(set(b_int)) != r:
            raise CertificateVerificationError(f"bases[{idx}] must be strictly increasing distinct indices.")
        if any(e < 0 or e >= n for e in b_int):
            raise CertificateVerificationError(f"bases[{idx}] contains out-of-range element.")
        bases.append(b_int)
    if len(set(bases)) != len(bases):
        raise CertificateVerificationError("Duplicate bases are forbidden.")

    raw_weights = cert.get("weights")
    if raw_weights is None:
        weights = [Fraction(1)] * len(bases)
    else:
        weights = _parse_vec(raw_weights, len(bases), "weights")
        if any(w <= 0 for w in weights):
            raise CertificateVerificationError("All basis weights must be strictly positive.")
    return n, r, bases, weights


def _verify_mode1_lorentzian_hessian(cert: Dict[str, Any]) -> Dict[str, Any]:
    n, r, bases, weights = _parse_bases_and_weights(cert)
    if r < 2:
        raise CertificateVerificationError("Lorentzian Hessian certification requires rank r >= 2.")
    # 1. Independently verify M-convexity (matroid basis-exchange axiom)
    if not check_basis_exchange(n, r, bases):
        raise CertificateVerificationError("Basis family fails matroid basis-exchange axiom (not M-convex).")
    # 2. Independently enumerate ALL (r - 2)-element independent sets
    expected_links = independent_sets_of_size(bases, r - 2)
    raw_links = cert.get("link_certificates")
    if not isinstance(raw_links, list) or len(raw_links) != len(expected_links):
        raise CertificateVerificationError(
            f"link_certificates must cover all {len(expected_links)} independent sets of size r-2."
        )
    seen_links = set()
    total_links_verified = 0
    for idx, lc in enumerate(raw_links):
        if not isinstance(lc, dict):
            raise CertificateVerificationError(f"link_certificates[{idx}] must be a dict.")
        raw_S = lc.get("S")
        if not isinstance(raw_S, list) or len(raw_S) != r - 2:
            raise CertificateVerificationError(f"link_certificates[{idx}].S must have length {r - 2}.")
        S = tuple(_parse_int(e, f"link_certificates[{idx}].S") for e in raw_S)
        if S in seen_links:
            raise CertificateVerificationError(f"Duplicate link set S={S}.")
        seen_links.add(S)
        _, H_S = link_hessian(n, bases, weights, S)
        D = _verify_congruence(H_S, lc.get("Q"), lc.get("Q_inv"), lc.get("D"), f"link[{S}]")
        n_pos = sum(1 for d in D if d > 0)
        if n_pos > 1:
            raise CertificateVerificationError(
                f"link[{S}] fails Lorentzian inertia: found {n_pos} positive pivots (need <= 1)."
            )
        total_links_verified += 1
    if seen_links != set(expected_links):
        raise CertificateVerificationError("link_certificates set mismatch with true I_{r-2}.")
    return {
        "schema": CERT_SCHEMA,
        "mode": "lorentzian_hessian_inertia",
        "verdict": "LORENTZIAN_CERTIFIED",
        "n": n,
        "r": r,
        "num_bases": len(bases),
        "num_links_verified": total_links_verified,
    }


def _verify_mode2_down_up_gap(cert: Dict[str, Any]) -> Dict[str, Any]:
    n, r, bases, weights = _parse_bases_and_weights(cert)
    ell = _parse_int(cert.get("ell"), "ell")
    if ell < 0 or ell >= r:
        raise CertificateVerificationError("Require 0 <= ell < r.")
    if not check_basis_exchange(n, r, bases):
        raise CertificateVerificationError("Basis family fails matroid basis-exchange axiom.")

    gamma_low = _parse_rat(cert.get("gamma_low"), "gamma_low")
    if gamma_low <= 0 or gamma_low > 1:
        raise CertificateVerificationError("gamma_low must lie in (0, 1].")
    universal_gap = Fraction(r - ell, r)
    if gamma_low < universal_gap:
        raise CertificateVerificationError(
            f"gamma_low ({gamma_low}) is below Ercan's universal matroid bound 1 - ell/r = {universal_gap}."
        )
    claim_strict_excess = cert.get("claim_strict_excess", False)
    if not isinstance(claim_strict_excess, bool):
        raise CertificateVerificationError("claim_strict_excess must be bool.")
    if claim_strict_excess and not (gamma_low > universal_gap):
        raise CertificateVerificationError("claim_strict_excess=True requires gamma_low > 1 - ell/r.")

    op_space = cert.get("operator_space", "basis_walk")
    if op_space == "basis_walk":
        pi, P = down_up_walk_exact(bases, weights, ell)
    elif op_space == "dual_ell_sets":
        _, pi, P = dual_up_down_walk_exact(bases, weights, ell)
    else:
        raise CertificateVerificationError("operator_space must be 'basis_walk' or 'dual_ell_sets'.")

    M = dirichlet_gram_matrix(pi, P, gamma_low)
    D = _verify_congruence(M, cert.get("Q"), cert.get("Q_inv"), cert.get("D"), "M_gamma")
    if any(d < 0 for d in D):
        raise CertificateVerificationError("Dirichlet-Gram certificate failed: negative pivot in D.")

    gamma_up_val = None
    if "upper_bound_witness" in cert and cert["upper_bound_witness"] is not None:
        ub = cert["upper_bound_witness"]
        if not isinstance(ub, dict):
            raise CertificateVerificationError("upper_bound_witness must be a dict.")
        gamma_up = _parse_rat(ub.get("gamma_up"), "gamma_up")
        if gamma_up < gamma_low:
            raise CertificateVerificationError("gamma_up cannot be smaller than gamma_low.")
        v = _parse_vec(ub.get("v"), len(pi), "upper_bound_witness.v")
        # Require pi^T v == 0 and v^T Pi v > 0
        mean_v = sum((pi[i] * v[i] for i in range(len(pi))), Fraction(0))
        if mean_v != 0:
            raise CertificateVerificationError("Upper-bound witness v must satisfy pi^T v == 0.")
        var_v = sum((pi[i] * v[i] * v[i] for i in range(len(pi))), Fraction(0))
        if var_v <= 0:
            raise CertificateVerificationError("Upper-bound witness v must have strictly positive variance.")
        quad_Pv = Fraction(0)
        for i in range(len(pi)):
            row_dot = sum((P[i][j] * v[j] for j in range(len(pi))), Fraction(0))
            quad_Pv += pi[i] * v[i] * row_dot
        rayleigh_gap = Fraction(1) - quad_Pv / var_v
        if rayleigh_gap > gamma_up:
            raise CertificateVerificationError(
                f"Rayleigh gap {rayleigh_gap} exceeds claimed gamma_up {gamma_up}."
            )
        gamma_up_val = str(gamma_up)

    return {
        "schema": CERT_SCHEMA,
        "mode": "down_up_spectral_gap",
        "verdict": "SPECTRAL_GAP_CERTIFIED",
        "operator_space": op_space,
        "n": n,
        "r": r,
        "ell": ell,
        "universal_gap": str(universal_gap),
        "gamma_low": str(gamma_low),
        "gamma_up": gamma_up_val,
        "strict_excess": gamma_low > universal_gap,
    }


def _verify_mode3_coincident_phase_transition(cert: Dict[str, Any]) -> Dict[str, Any]:
    m = _parse_int(cert.get("m"), "m")
    if m < 2:
        raise CertificateVerificationError("Require cluster size m >= 2.")
    eps = _parse_rat(cert.get("eps"), "eps")
    if eps <= 0:
        raise CertificateVerificationError("Require coupling eps > 0.")
    regime = cert.get("regime")
    if regime not in {"lorentzian_supercritical", "non_lorentzian_subcritical"}:
        raise CertificateVerificationError(
            "regime must be 'lorentzian_supercritical' or 'non_lorentzian_subcritical'."
        )

    H, pi, Q_walk, _, _, exact_gap = two_cluster_hessian_and_walk(m, eps)
    eps_c = Fraction(m - 1, m)
    n = 2 * m

    if regime == "lorentzian_supercritical":
        if eps < eps_c:
            raise CertificateVerificationError(
                f"eps={eps} < eps_c={eps_c} cannot be certified as lorentzian_supercritical."
            )
        h_cert = cert.get("hessian_congruence")
        if not isinstance(h_cert, dict):
            raise CertificateVerificationError("Missing hessian_congruence.")
        D_H = _verify_congruence(H, h_cert.get("Q"), h_cert.get("Q_inv"), h_cert.get("D"), "H_eps_m")
        if sum(1 for d in D_H if d > 0) != 1:
            raise CertificateVerificationError("Lorentzian Hessian must have exactly 1 positive pivot.")

        gamma_low = _parse_rat(cert.get("gamma_low"), "gamma_low")
        if gamma_low < Fraction(1, 2) or gamma_low > 1:
            raise CertificateVerificationError("Supercritical regime requires gamma_low in [1/2, 1].")
        M = dirichlet_gram_matrix(pi, Q_walk, gamma_low)
        g_cert = cert.get("gap_congruence")
        if not isinstance(g_cert, dict):
            raise CertificateVerificationError("Missing gap_congruence.")
        D_M = _verify_congruence(M, g_cert.get("Q"), g_cert.get("Q_inv"), g_cert.get("D"), "M_gamma")
        if any(d < 0 for d in D_M):
            raise CertificateVerificationError("Supercritical Dirichlet-Gram certificate has negative pivot.")
        verdict = "LORENTZIAN_SUPERCRITICAL_CERTIFIED"
    else:
        if eps >= eps_c:
            raise CertificateVerificationError(
                f"eps={eps} >= eps_c={eps_c} cannot be certified as non_lorentzian_subcritical."
            )
        # Verify 2-dimensional strictly positive subspace of H (proving lambda_2(H) > 0)
        u1 = _parse_vec(cert.get("pos_vec_1"), n, "pos_vec_1")
        u2 = _parse_vec(cert.get("pos_vec_2"), n, "pos_vec_2")

        def quad(u: Sequence[Fraction], v: Sequence[Fraction], A: MatrixQ) -> Fraction:
            s = Fraction(0)
            for i in range(n):
                if u[i] == 0:
                    continue
                for j in range(n):
                    if v[j] != 0 and A[i][j] != 0:
                        s += u[i] * A[i][j] * v[j]
            return s

        # Check Gram matrix of (u1, u2) under H is strictly positive definite (Sylvester 2x2)
        g11 = quad(u1, u1, H)
        g22 = quad(u2, u2, H)
        g12 = quad(u1, u2, H)
        if not (g11 > 0 and (g11 * g22 - g12 * g12) > 0):
            raise CertificateVerificationError("pos_vec_1, pos_vec_2 do not span a 2D positive subspace of H.")

        # Check sub-critical gap witness v with pi^T v == 0 and Rayleigh gap < 1/2
        v = _parse_vec(cert.get("gap_witness_v"), n, "gap_witness_v")
        if sum((pi[i] * v[i] for i in range(n)), Fraction(0)) != 0:
            raise CertificateVerificationError("gap_witness_v must satisfy pi^T v == 0.")
        var_v = sum((pi[i] * v[i] * v[i] for i in range(n)), Fraction(0))
        if var_v <= 0:
            raise CertificateVerificationError("gap_witness_v must be nonzero.")
        quad_Qv = Fraction(0)
        for i in range(n):
            row_dot = sum((Q_walk[i][j] * v[j] for j in range(n)), Fraction(0))
            quad_Qv += pi[i] * v[i] * row_dot
        rayleigh_gap = Fraction(1) - quad_Qv / var_v
        if not (rayleigh_gap < Fraction(1, 2)):
            raise CertificateVerificationError("gap_witness_v fails to certify gap < 1/2.")
        gamma_low = rayleigh_gap
        verdict = "NON_LORENTZIAN_SUBCRITICAL_CERTIFIED"

    return {
        "schema": CERT_SCHEMA,
        "mode": "coincident_phase_transition",
        "verdict": verdict,
        "m": m,
        "eps": str(eps),
        "eps_c": str(eps_c),
        "exact_gap": str(exact_gap),
        "certified_gap": str(gamma_low),
    }


def _verify_mode4_matrix_concentration(cert: Dict[str, Any]) -> Dict[str, Any]:
    n, r, bases, weights = _parse_bases_and_weights(cert)
    if not check_basis_exchange(n, r, bases):
        raise CertificateVerificationError("Basis family fails matroid basis-exchange axiom.")
    if any(w != weights[0] for w in weights):
        raise CertificateVerificationError("Uniform basis weights required in matrix_concentration_tail mode.")

    d = _parse_int(cert.get("matrix_dim"), "matrix_dim")
    if d < 1:
        raise CertificateVerificationError("matrix_dim must be >= 1.")
    raw_mats = cert.get("element_matrices")
    if not isinstance(raw_mats, list) or len(raw_mats) != n:
        raise CertificateVerificationError(f"element_matrices must contain {n} symmetric {d}x{d} matrices.")
    A_mats: List[MatrixQ] = []
    for e, raw_A in enumerate(raw_mats):
        A = _parse_mat(raw_A, d, f"element_matrices[{e}]")
        if not mat_is_symmetric(A):
            raise CertificateVerificationError(f"element_matrices[{e}] is not symmetric.")
        A_mats.append(A)

    c_sq = _parse_rat(cert.get("c_sq"), "c_sq")
    t = _parse_rat(cert.get("t"), "t")
    delta = _parse_rat(cert.get("delta"), "delta")
    taylor_order = _parse_int(cert.get("taylor_order"), "taylor_order")
    if c_sq <= 0 or t <= 0 or not (0 < delta < 1) or taylor_order < 4:
        raise CertificateVerificationError("Require c_sq > 0, t > 0, 0 < delta < 1, taylor_order >= 4.")

    # 1. Independently compute exact expected basis sum E[F(B)]
    exp_F: MatrixQ = [[Fraction(0)] * d for _ in range(d)]
    inv_B = Fraction(1, len(bases))
    for b in bases:
        for e in b:
            Ae = A_mats[e]
            for i in range(d):
                for j in range(d):
                    exp_F[i][j] += Ae[i][j] * inv_B
    claimed_exp = _parse_mat(cert.get("expected_matrix"), d, "expected_matrix")
    if not mat_equal(exp_F, claimed_exp):
        raise CertificateVerificationError("expected_matrix does not match exact basis expectation E[F(B)].")

    # 2. Independently find ALL exchangeable element pairs (e < f)
    b_fsets = [frozenset(b) for b in bases]
    exch_pairs = set()
    for i in range(len(b_fsets)):
        bi = b_fsets[i]
        for j in range(i + 1, len(b_fsets)):
            sym = bi ^ b_fsets[j]
            if len(sym) == 2:
                e, f = sorted(sym)
                exch_pairs.add((e, f))

    raw_pair_certs = cert.get("exchange_norm_certificates")
    if not isinstance(raw_pair_certs, list) or len(raw_pair_certs) != len(exch_pairs):
        raise CertificateVerificationError(
            f"exchange_norm_certificates must cover all {len(exch_pairs)} exchangeable pairs."
        )
    seen_pairs = set()
    for idx, pc in enumerate(raw_pair_certs):
        if not isinstance(pc, dict):
            raise CertificateVerificationError(f"exchange_norm_certificates[{idx}] must be a dict.")
        e = _parse_int(pc.get("e"), f"pair[{idx}].e")
        f = _parse_int(pc.get("f"), f"pair[{idx}].f")
        if not (0 <= e < f < n):
            raise CertificateVerificationError(f"Invalid pair ({e}, {f}).")
        if (e, f) in seen_pairs:
            raise CertificateVerificationError(f"Duplicate pair ({e}, {f}).")
        seen_pairs.add((e, f))
        diff = [[A_mats[e][i][j] - A_mats[f][i][j] for j in range(d)] for i in range(d)]
        diff_sq = mat_mul(diff, diff)
        C_ef = [
            [(c_sq if i == j else Fraction(0)) - diff_sq[i][j] for j in range(d)]
            for i in range(d)
        ]
        D_ef = _verify_congruence(C_ef, pc.get("Q"), pc.get("Q_inv"), pc.get("D"), f"C_({e},{f})")
        if any(x < 0 for x in D_ef):
            raise CertificateVerificationError(f"Pair ({e},{f}) failed operator-norm PSD bound c^2 I - (A_e-A_f)^2 >= 0.")
    if seen_pairs != exch_pairs:
        raise CertificateVerificationError("exchange_norm_certificates pair set mismatch.")

    # 3. Exact rational Taylor-envelope subgaussian tail check:
    # From Ercan's ell/r matrix entropy contraction, v_sg = r * c^2 / 2 => x = t^2 / (2 * v_sg) = t^2 / (r * c^2)
    x = (t * t) / (Fraction(r) * c_sq)
    taylor_sum = Fraction(0)
    term = Fraction(1)
    for k in range(taylor_order + 1):
        if k > 0:
            term = term * x / Fraction(k)
        taylor_sum += term
    rational_tail_bound = Fraction(d) / taylor_sum
    if rational_tail_bound > delta:
        raise CertificateVerificationError(
            f"Certified rational tail bound {rational_tail_bound} exceeds target delta {delta}."
        )

    return {
        "schema": CERT_SCHEMA,
        "mode": "matrix_concentration_tail",
        "verdict": "MATRIX_TAIL_CERTIFIED",
        "n": n,
        "r": r,
        "matrix_dim": d,
        "num_exchangeable_pairs": len(exch_pairs),
        "exponent_x": str(x),
        "taylor_order": taylor_order,
        "rational_tail_bound": str(rational_tail_bound),
        "delta": str(delta),
    }


def verify_lorentzian_certificate_exact(cert: Dict[str, Any]) -> Dict[str, Any]:
    """Verify a LORENTZ-MATROID-CERT-1.0 JSON dictionary using exact rational arithmetic only."""
    if not isinstance(cert, dict):
        raise CertificateVerificationError("Certificate must be a JSON object.")
    _assert_no_floats(cert)
    if cert.get("schema") != CERT_SCHEMA:
        raise CertificateVerificationError(f"Unsupported schema: {cert.get('schema')!r}")
    mode = cert.get("mode")
    if mode not in VALID_MODES:
        raise CertificateVerificationError(f"Unknown mode: {mode!r}")

    if mode == "lorentzian_hessian_inertia":
        return _verify_mode1_lorentzian_hessian(cert)
    if mode == "down_up_spectral_gap":
        return _verify_mode2_down_up_gap(cert)
    if mode == "coincident_phase_transition":
        return _verify_mode3_coincident_phase_transition(cert)
    return _verify_mode4_matrix_concentration(cert)

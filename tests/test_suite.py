"""Comprehensive pytest + 48 live adversarial red-team attack suite for Paper 11.

Author: Lluis Eriksson
"""

from __future__ import annotations

import copy
from fractions import Fraction
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import (
    CERT_SCHEMA,
    CertificateVerificationError,
    check_basis_exchange,
    down_up_walk_exact,
    dual_up_down_walk_exact,
    exact_congruence_diagonalize,
    fano_matroid,
    graphic_k4_matroid,
    non_fano_matroid,
    two_cluster_hessian_and_walk,
    uniform_matroid,
    vamos_matroid,
    verify_lorentzian_certificate_exact,
)
from src.matroid_exact import mat_equal, mat_eye, mat_mul, mat_transpose

CERTS_DIR = ROOT / "results" / "certificates"


def _load(name: str) -> dict:
    return json.loads((CERTS_DIR / name).read_text(encoding="utf-8"))


def test_all_16_emitted_certificates_verify_cleanly() -> None:
    cert_files = sorted(CERTS_DIR.glob("*.cert.json"))
    assert len(cert_files) == 16
    for path in cert_files:
        data = json.loads(path.read_text(encoding="utf-8"))
        res = verify_lorentzian_certificate_exact(data)
        assert res["schema"] == CERT_SCHEMA
        assert "CERTIFIED" in res["verdict"]


def test_exact_congruence_and_matroid_axioms() -> None:
    for fn in [lambda: uniform_matroid(3, 6), graphic_k4_matroid, fano_matroid, non_fano_matroid, vamos_matroid]:
        n, r, bases = fn()
        assert check_basis_exchange(n, r, bases)
    # Non-matroid (violates basis exchange): {{0,1}, {2,3}}
    assert not check_basis_exchange(4, 2, [(0, 1), (2, 3)])
    # Zero-diagonal hyperbolic block [[0, 1], [1, 0]]
    H = [[Fraction(0), Fraction(1)], [Fraction(1), Fraction(0)]]
    Q, Q_inv, D = exact_congruence_diagonalize(H)
    assert mat_equal(mat_mul(Q, Q_inv), mat_eye(2))
    assert mat_equal(mat_mul(mat_mul(Q, H), mat_transpose(Q)), [[D[0], Fraction(0)], [Fraction(0), D[1]]])
    assert sum(1 for x in D if x > 0) == 1 and sum(1 for x in D if x < 0) == 1


def test_48_adversarial_red_team_attacks_rejected() -> None:
    c_lor = _load("lorentzian_V8_vamos.cert.json")
    c_gap = _load("gap_V8_ell2_dual.cert.json")
    c_sup = _load("phase_m4_critical_eps_3_4.cert.json")
    c_sub = _load("phase_m4_subcritical_eps_7_10.cert.json")
    c_tail = _load("tail_MK4_pauli_d2.cert.json")

    attacks: list[tuple[str, dict]] = []

    def add(name: str, base: dict, mutator) -> None:
        c = copy.deepcopy(base)
        mutator(c)
        attacks.append((name, c))

    # --- Schema & Zero-Float Enforcement (A01-A09) ---
    add("A01_wrong_schema", c_lor, lambda c: c.__setitem__("schema", "FAKE-1.0"))
    add("A02_unknown_mode", c_lor, lambda c: c.__setitem__("mode", "magic_mode"))
    add("A03_float_top_level", c_sup, lambda c: c.__setitem__("eps", 0.75))
    add("A04_float_nested_D", c_gap, lambda c: c["D"].__setitem__(0, 0.5))
    add("A05_decimal_string_rat", c_sup, lambda c: c.__setitem__("eps", "0.75"))
    add("A06_sci_notation_rat", c_sup, lambda c: c.__setitem__("eps", "1e-1"))
    add("A07_zero_denominator", c_sup, lambda c: c.__setitem__("eps", "3/0"))
    add("A08_bool_as_int_m", c_sup, lambda c: c.__setitem__("m", True))
    add("A09_whitespace_rat", c_sup, lambda c: c.__setitem__("eps", " 3/4 "))

    # --- Mode 1: Lorentzian Hessian Inertia Attacks (A10-A19) ---
    add("A10_drop_one_link_V8", c_lor, lambda c: c["link_certificates"].pop())
    add("A11_duplicate_link_S", c_lor, lambda c: c["link_certificates"].__setitem__(1, copy.deepcopy(c["link_certificates"][0])))
    add("A12_tamper_Q_singular", c_lor, lambda c: c["link_certificates"][0]["Q"].__setitem__(0, ["0"] * len(c["link_certificates"][0]["Q"])))
    add("A13_tamper_Q_inv", c_lor, lambda c: c["link_certificates"][0]["Q_inv"][0].__setitem__(0, "999"))
    add("A14_tamper_D_sign", c_lor, lambda c: c["link_certificates"][0]["D"].__setitem__(0, "999"))
    add("A15_non_matroid_bases", c_lor, lambda c: c.__setitem__("bases", [[0, 1, 2, 3], [4, 5, 6, 7]]))
    add("A16_duplicate_basis", c_lor, lambda c: c["bases"].append(c["bases"][0]))
    add("A17_unsorted_basis", c_lor, lambda c: c["bases"].__setitem__(0, list(reversed(c["bases"][0]))))
    add("A18_negative_weight", c_lor, lambda c: c.__setitem__("weights", ["-1"] + ["1"] * (len(c["bases"]) - 1)))
    add("A19_rank_1_rejected", c_lor, lambda c: c.update({"r": 1, "bases": [[0], [1]]}))

    # --- Mode 2: Down-Up Spectral Gap Attacks (A20-A30) ---
    add("A20_inflate_gamma_low", c_gap, lambda c: c.__setitem__("gamma_low", "10/14"))
    add("A21_gamma_low_below_universal", c_gap, lambda c: c.__setitem__("gamma_low", "1/4"))
    add("A22_gamma_low_gt_1", c_gap, lambda c: c.__setitem__("gamma_low", "5/4"))
    add("A23_gamma_low_zero", c_gap, lambda c: c.__setitem__("gamma_low", "0"))
    add("A24_strict_excess_false_claim", c_gap, lambda c: c.update({"gamma_low": "1/2", "claim_strict_excess": True}))
    add("A25_tamper_gap_Q_inv", c_gap, lambda c: c["Q_inv"][0].__setitem__(0, "777"))
    add("A26_tamper_gap_D_negative", c_gap, lambda c: c["D"].__setitem__(0, "-1/100"))
    add("A27_upper_witness_non_centered", c_gap, lambda c: c["upper_bound_witness"]["v"].__setitem__(0, "100"))
    add("A28_upper_witness_zero_vec", c_gap, lambda c: c["upper_bound_witness"].__setitem__("v", ["0"] * len(c["upper_bound_witness"]["v"])))
    add("A29_gamma_up_lt_gamma_low", c_gap, lambda c: c["upper_bound_witness"].__setitem__("gamma_up", "8/14"))
    add("A30_invalid_operator_space", c_gap, lambda c: c.__setitem__("operator_space", "cheated_space"))

    # --- Mode 3: Coincident Phase Transition Attacks (A31-A39) ---
    add("A31_subcritical_claimed_super", c_sup, lambda c: c.__setitem__("eps", "7/10"))
    add("A32_supercritical_claimed_sub", c_sub, lambda c: c.__setitem__("eps", "3/4"))
    add("A33_supercritical_gamma_below_half", c_sup, lambda c: c.__setitem__("gamma_low", "49/100"))
    add("A34_supercritical_inflate_gamma", c_sup, lambda c: c.__setitem__("gamma_low", "51/100"))
    add("A35_subcritical_collinear_pos_vecs", c_sub, lambda c: c.__setitem__("pos_vec_2", copy.deepcopy(c["pos_vec_1"])))
    add("A36_subcritical_zero_pos_vec", c_sub, lambda c: c.__setitem__("pos_vec_2", ["0"] * 8))
    add("A37_subcritical_non_centered_v", c_sub, lambda c: c["gap_witness_v"].__setitem__(0, "9"))
    add("A38_m_equals_1_invalid", c_sup, lambda c: c.__setitem__("m", 1))
    add("A39_eps_negative", c_sub, lambda c: c.__setitem__("eps", "-1/2"))

    # --- Mode 4: Matrix Concentration Tail Attacks (A40-A48) ---
    add("A40_shrink_c_sq_invalid", c_tail, lambda c: c.__setitem__("c_sq", "1/10"))
    add("A41_drop_one_exchange_pair", c_tail, lambda c: c["exchange_norm_certificates"].pop())
    add("A42_duplicate_exchange_pair", c_tail, lambda c: c["exchange_norm_certificates"].__setitem__(1, copy.deepcopy(c["exchange_norm_certificates"][0])))
    add("A43_tamper_expected_matrix", c_tail, lambda c: c["expected_matrix"][0].__setitem__(0, "99"))
    add("A44_asymmetric_element_matrix", c_tail, lambda c: c["element_matrices"][0][0].__setitem__(1, "7"))
    add("A45_shrink_delta_impossible", c_tail, lambda c: c.__setitem__("delta", "1/1000000"))
    add("A46_taylor_order_too_small", c_tail, lambda c: c.__setitem__("taylor_order", 2))
    add("A47_tamper_pair_Q_inv", c_tail, lambda c: c["exchange_norm_certificates"][0]["Q_inv"][0].__setitem__(0, "42"))
    add("A48_nonuniform_weights_tail", c_tail, lambda c: c.__setitem__("weights", ["2"] + ["1"] * (len(c["bases"]) - 1)))

    assert len(attacks) == 48
    for name, bad_cert in attacks:
        with pytest.raises(CertificateVerificationError, match="."):
            verify_lorentzian_certificate_exact(bad_cert)

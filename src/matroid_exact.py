"""Pure-stdlib exact-Q linear algebra, matroid constructors, and congruence diagonalization.

Zero floating-point operations anywhere in this module (fractions.Fraction only).
Author: Lluis Eriksson
"""

from __future__ import annotations

from fractions import Fraction
from itertools import combinations
from typing import Dict, List, Sequence, Tuple

MatrixQ = List[List[Fraction]]


def rat_to_str(x: Fraction) -> str:
    return str(x.numerator) if x.denominator == 1 else f"{x.numerator}/{x.denominator}"


def mat_to_str(M: MatrixQ) -> List[List[str]]:
    return [[rat_to_str(x) for x in row] for row in M]


def vec_to_str(v: Sequence[Fraction]) -> List[str]:
    return [rat_to_str(x) for x in v]


def mat_eye(n: int) -> MatrixQ:
    return [[Fraction(1) if i == j else Fraction(0) for j in range(n)] for i in range(n)]


def mat_transpose(A: MatrixQ) -> MatrixQ:
    if not A:
        return []
    m, n = len(A), len(A[0])
    return [[A[i][j] for i in range(m)] for j in range(n)]


def mat_mul(A: MatrixQ, B: MatrixQ) -> MatrixQ:
    m = len(A)
    k = len(B)
    n = len(B[0]) if k > 0 else 0
    out: MatrixQ = [[Fraction(0)] * n for _ in range(m)]
    for i in range(m):
        row_a = A[i]
        row_out = out[i]
        for t in range(k):
            a_it = row_a[t]
            if a_it == 0:
                continue
            row_b = B[t]
            for j in range(n):
                if row_b[j] != 0:
                    row_out[j] += a_it * row_b[j]
    return out


def mat_is_symmetric(A: MatrixQ) -> bool:
    n = len(A)
    for i in range(n):
        if len(A[i]) != n:
            return False
        for j in range(i + 1, n):
            if A[i][j] != A[j][i]:
                return False
    return True


def mat_equal(A: MatrixQ, B: MatrixQ) -> bool:
    if len(A) != len(B):
        return False
    for r1, r2 in zip(A, B):
        if len(r1) != len(r2):
            return False
        for a, b in zip(r1, r2):
            if a != b:
                return False
    return True


def exact_congruence_diagonalize(H: MatrixQ) -> Tuple[MatrixQ, MatrixQ, List[Fraction]]:
    """Compute exact rational invertible Q, Q_inv, and diagonal D such that
    Q * H * Q^T == diag(D) and Q * Q_inv == I_n.

    Works for ANY symmetric rational matrix H, including zero-diagonal
    multiaffine matroid Hessians, via elementary swaps, shears, and eliminations.
    """
    n = len(H)
    if not mat_is_symmetric(H):
        raise ValueError("Matrix must be square and symmetric over Q.")
    A = [[Fraction(x) for x in row] for row in H]
    Q = mat_eye(n)
    Q_inv = mat_eye(n)

    for k in range(n):
        if A[k][k] == 0:
            # 1. Look for a nonzero diagonal entry j > k
            piv_j = None
            for j in range(k + 1, n):
                if A[j][j] != 0:
                    piv_j = j
                    break
            if piv_j is not None:
                # Swap rows/cols k <-> piv_j in A; swap rows in Q; swap cols in Q_inv
                A[k], A[piv_j] = A[piv_j], A[k]
                for r in range(n):
                    A[r][k], A[r][piv_j] = A[r][piv_j], A[r][k]
                Q[k], Q[piv_j] = Q[piv_j], Q[k]
                for r in range(n):
                    Q_inv[r][k], Q_inv[r][piv_j] = Q_inv[r][piv_j], Q_inv[r][k]
            else:
                # 2. Look for a nonzero off-diagonal entry A[k][j] with j > k
                off_j = None
                for j in range(k + 1, n):
                    if A[k][j] != 0:
                        off_j = j
                        break
                if off_j is None:
                    # Entire active row/col k is zero
                    continue
                # Elementary shear E = I + E_{k, off_j}:
                # row k += row off_j, col k += col off_j in A -> new A[k][k] = 2 * A[k][off_j] != 0
                for c in range(n):
                    A[k][c] += A[off_j][c]
                for r in range(n):
                    A[r][k] += A[r][off_j]
                for c in range(n):
                    Q[k][c] += Q[off_j][c]
                # Q_inv <- Q_inv * (I - E_{k, off_j}): col off_j -= col k
                for r in range(n):
                    Q_inv[r][off_j] -= Q_inv[r][k]

        piv = A[k][k]
        if piv == 0:
            continue
        for i in range(k + 1, n):
            if A[i][k] == 0:
                continue
            alpha = A[i][k] / piv
            # E = I - alpha * E_{i, k}: row i -= alpha * row k, col i -= alpha * col k
            for c in range(k, n):
                if A[k][c] != 0:
                    A[i][c] -= alpha * A[k][c]
            for r in range(k, n):
                if A[r][k] != 0:
                    A[r][i] -= alpha * A[r][k]
            for c in range(n):
                if Q[k][c] != 0:
                    Q[i][c] -= alpha * Q[k][c]
            # Q_inv <- Q_inv * (I + alpha * E_{i, k}): col k += alpha * col i
            for r in range(n):
                if Q_inv[r][i] != 0:
                    Q_inv[r][k] += alpha * Q_inv[r][i]

    D = [A[i][i] for i in range(n)]
    return Q, Q_inv, D


def check_basis_exchange(n: int, r: int, bases: Sequence[Tuple[int, ...]]) -> bool:
    """Verify the exact matroid basis-exchange axiom (M-convexity on 0/1 cube)."""
    if n < 1 or r < 1 or r > n or not bases:
        return False
    bset = set()
    for b in bases:
        if len(b) != r:
            return False
        if tuple(sorted(b)) != tuple(b) or len(set(b)) != r:
            return False
        if any(e < 0 or e >= n for e in b):
            return False
        bset.add(tuple(b))
    if len(bset) != len(bases):
        return False
    b_fsets = [frozenset(b) for b in bases]
    b_lookup = set(b_fsets)
    for b1 in b_fsets:
        for b2 in b_fsets:
            if b1 == b2:
                continue
            diff12 = b1 - b2
            diff21 = b2 - b1
            for e in diff12:
                found = False
                for f in diff21:
                    if (b1 - {e}) | {f} in b_lookup:
                        found = True
                        break
                if not found:
                    return False
    return True


def uniform_matroid(r: int, n: int) -> Tuple[int, int, List[Tuple[int, ...]]]:
    return n, r, [tuple(c) for c in combinations(range(n), r)]


def graphic_k4_matroid() -> Tuple[int, int, List[Tuple[int, ...]]]:
    """Graphic matroid M(K_4): 6 edges, rank 3, 16 spanning trees."""
    edges = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    bases = []
    for comb in combinations(range(6), 3):
        # Check if 3 edges on 4 vertices form a spanning tree (connected)
        parent = list(range(4))

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        acyclic = True
        for e_idx in comb:
            u, v = edges[e_idx]
            ru, rv = find(u), find(v)
            if ru == rv:
                acyclic = False
                break
            parent[ru] = rv
        if acyclic:
            bases.append(comb)
    return 6, 3, bases


def fano_matroid() -> Tuple[int, int, List[Tuple[int, ...]]]:
    """Fano matroid F_7: 7 elements, rank 3, 28 bases (representable only over char 2)."""
    lines = {
        (0, 1, 2),
        (0, 3, 4),
        (0, 5, 6),
        (1, 3, 5),
        (1, 4, 6),
        (2, 3, 6),
        (2, 4, 5),
    }
    bases = [c for c in combinations(range(7), 3) if c not in lines]
    return 7, 3, bases


def non_fano_matroid() -> Tuple[int, int, List[Tuple[int, ...]]]:
    """Non-Fano matroid F_7^-: 7 elements, rank 3, 29 bases (F_7 with one line relaxed)."""
    lines = {
        (0, 1, 2),
        (0, 3, 4),
        (0, 5, 6),
        (1, 3, 5),
        (1, 4, 6),
        (2, 3, 6),
    }
    bases = [c for c in combinations(range(7), 3) if c not in lines]
    return 7, 3, bases


def vamos_matroid() -> Tuple[int, int, List[Tuple[int, ...]]]:
    """Vamos matroid V_8: 8 elements, rank 4, 65 bases (non-representable over every field)."""
    circuit_hyperplanes = {
        (0, 1, 2, 3),
        (0, 1, 4, 5),
        (2, 3, 4, 5),
        (0, 1, 6, 7),
        (2, 3, 6, 7),
    }
    bases = [c for c in combinations(range(8), 4) if c not in circuit_hyperplanes]
    return 8, 4, bases


def independent_sets_of_size(bases: Sequence[Tuple[int, ...]], k: int) -> List[Tuple[int, ...]]:
    """Return sorted list of all independent sets of size k."""
    s = set()
    for b in bases:
        for sub in combinations(b, k):
            s.add(sub)
    return sorted(s)


def link_hessian(
    n: int,
    bases: Sequence[Tuple[int, ...]],
    weights: Sequence[Fraction],
    S: Tuple[int, ...],
) -> Tuple[List[int], MatrixQ]:
    """Compute exact rational Hessian of partial_S f_M on active elements outside S.

    For S of size r - 2, partial_S f_M(x) = sum_{B superset S} w_B * x_i * x_j,
    so H_S[a, b] = w_{S U {i, j}} for i != j and 0 on the diagonal.
    """
    S_set = set(S)
    active = [e for e in range(n) if e not in S_set]
    idx = {e: k for k, e in enumerate(active)}
    m = len(active)
    H: MatrixQ = [[Fraction(0)] * m for _ in range(m)]
    for b, w in zip(bases, weights):
        b_set = set(b)
        if S_set.issubset(b_set):
            rem = sorted(b_set - S_set)
            if len(rem) != 2:
                raise ValueError("S must have size r - 2.")
            ia, ib = idx[rem[0]], idx[rem[1]]
            H[ia][ib] += w
            H[ib][ia] += w
    return active, H


def down_up_walk_exact(
    bases: Sequence[Tuple[int, ...]],
    weights: Sequence[Fraction],
    ell: int,
) -> Tuple[List[Fraction], MatrixQ]:
    """Build exact stationary distribution pi and reversible down-up walk P_{r -> ell -> r}."""
    num_b = len(bases)
    total_w = sum(weights, Fraction(0))
    if total_w <= 0:
        raise ValueError("Total weight must be positive.")
    pi = [w / total_w for w in weights]
    r = len(bases[0])
    if ell < 0 or ell >= r:
        raise ValueError("Need 0 <= ell < r.")

    # Map each ell-subset S to list of (basis_index, weight)
    fibers: Dict[Tuple[int, ...], List[int]] = {}
    for idx_b, b in enumerate(bases):
        for sub in combinations(b, ell):
            fibers.setdefault(sub, []).append(idx_b)

    # Number of ell-subsets per basis is C(r, ell)
    n_sub = len(list(combinations(range(r), ell)))
    inv_nsub = Fraction(1, n_sub)

    P: MatrixQ = [[Fraction(0)] * num_b for _ in range(num_b)]
    for sub, members in fibers.items():
        fiber_w = sum((weights[i] for i in members), Fraction(0))
        for i in members:
            for j in members:
                P[i][j] += inv_nsub * (weights[j] / fiber_w)
    return pi, P


def dual_up_down_walk_exact(
    bases: Sequence[Tuple[int, ...]],
    weights: Sequence[Fraction],
    ell: int,
) -> Tuple[List[Tuple[int, ...]], List[Fraction], MatrixQ]:
    """Build exact isospectral dual up-down walk Q_{ell -> r -> ell} on ell-independent sets."""
    r = len(bases[0])
    if ell < 0 or ell >= r:
        raise ValueError("Need 0 <= ell < r.")
    n_sub = len(list(combinations(range(r), ell)))
    inv_nsub = Fraction(1, n_sub)

    fibers: Dict[Tuple[int, ...], List[int]] = {}
    for idx_b, b in enumerate(bases):
        for sub in combinations(b, ell):
            fibers.setdefault(sub, []).append(idx_b)

    states = sorted(fibers.keys())
    s_idx = {s: i for i, s in enumerate(states)}
    k = len(states)
    fiber_w = [sum((weights[b] for b in fibers[s]), Fraction(0)) for s in states]
    total_fw = sum(fiber_w, Fraction(0))
    pi_dual = [w / total_fw for w in fiber_w]

    # Basis -> list of state indices
    b_to_states: List[List[int]] = [
        [s_idx[sub] for sub in combinations(b, ell)] for b in bases
    ]

    Q: MatrixQ = [[Fraction(0)] * k for _ in range(k)]
    for i, s in enumerate(states):
        fw = fiber_w[i]
        for b_id in fibers[s]:
            prob_up = weights[b_id] / fw
            for j in b_to_states[b_id]:
                Q[i][j] += prob_up * inv_nsub
    return states, pi_dual, Q


def dirichlet_gram_matrix(
    pi: Sequence[Fraction],
    P: MatrixQ,
    gamma: Fraction,
) -> MatrixQ:
    """Exact symmetric rational matrix M_gamma = (1 - gamma)*Pi - Pi*P + pi*pi^T."""
    n = len(pi)
    one_minus_g = Fraction(1) - gamma
    M: MatrixQ = [[Fraction(0)] * n for _ in range(n)]
    for i in range(n):
        pi_i = pi[i]
        for j in range(n):
            val = pi_i * pi[j] - pi_i * P[i][j]
            if i == j:
                val += one_minus_g * pi_i
            M[i][j] = val
    return M


def two_cluster_hessian_and_walk(
    m: int,
    eps: Fraction,
) -> Tuple[MatrixQ, List[Fraction], MatrixQ, Fraction, Fraction, Fraction]:
    """Exact Hessian H_{eps,m} and dual vertex walk Q_{1->2->1} for G_{eps,m} = (K_m + K_m) + eps K_{m,m}.

    Returns (H, pi, Q_walk, lambda_plus, lambda_minus, exact_gap).
    """
    if m < 2 or eps <= 0:
        raise ValueError("Require m >= 2 and eps > 0.")
    n = 2 * m
    H: MatrixQ = [[Fraction(0)] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i == j:
                H[i][j] = Fraction(0)
            elif (i < m and j < m) or (i >= m and j >= m):
                H[i][j] = Fraction(1)
            else:
                H[i][j] = eps
    deg = Fraction(m - 1) + Fraction(m) * eps
    pi = [Fraction(1, n)] * n
    # Q_{1->2->1} = (1/2) * (I + D^{-1} H)
    Q: MatrixQ = [[Fraction(0)] * n for _ in range(n)]
    half = Fraction(1, 2)
    for i in range(n):
        for j in range(n):
            val = half * (H[i][j] / deg)
            if i == j:
                val += half
            Q[i][j] = val
    lam_plus = deg
    lam_minus = Fraction(m - 1) - Fraction(m) * eps
    if eps <= 1:
        exact_gap = (Fraction(m) * eps) / deg
    else:
        exact_gap = half + Fraction(1, 2) / deg
    return H, pi, Q, lam_plus, lam_minus, exact_gap

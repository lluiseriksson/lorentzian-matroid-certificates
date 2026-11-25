"""End-to-end Gate-2 verification script for Paper 11.

Checks:
  1. Lean 4 (`lake build` + `AuditAxioms.lean`): 18/18 theorems, 0 warnings, 0 sorry, 0 Classical.choice.
  2. All 16 exact-Q LORENTZ-MATROID-CERT-1.0 certificates independently verified (0 floats).
  3. Full pytest suite (including 48/48 live adversarial red-team attacks).
  4. SHA256SUMS.json manifest integrity (or writes manifest with `--write-manifest`).

Author: Lluis Eriksson
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import verify_lorentzian_certificate_exact

EXPECTED_LEAN_THEOREMS = [
    "LorentzianMatroid.coincident_threshold_pos",
    "LorentzianMatroid.hessian_lambda_plus_pos",
    "LorentzianMatroid.hessian_lambda_minus_lorentzian_iff",
    "LorentzianMatroid.down_up_mu2_half_iff",
    "LorentzianMatroid.coincident_phase_transition_equiv",
    "LorentzianMatroid.coincident_exact_equality_at_critical",
    "LorentzianMatroid.piecewise_crossover_at_one",
    "LorentzianMatroid.complete_graph_gap_strictly_above_half",
    "LorentzianMatroid.ldl_psd_quadratic_form_nonneg",
    "LorentzianMatroid.ldl_lorentzian_hyperplane_nonpos",
    "LorentzianMatroid.reversible_gram_identity",
    "LorentzianMatroid.ercan_universal_gap_pos",
    "LorentzianMatroid.vamos_excess_gap_strict",
    "LorentzianMatroid.mason_ulc_elementary_symmetric",
    "LorentzianMatroid.exp_taylor_lower_bound_monotone",
    "LorentzianMatroid.subgaussian_taylor_tail_soundness",
    "LorentzianMatroid.exchange_operator_norm_sq_bound",
    "LorentzianMatroid.verdict_exclusivity",
]

MANIFEST_SKIP_DIRS = {".git", ".lake", "__pycache__", ".pytest_cache"}
MANIFEST_SKIP_EXTS = {".aux", ".bbl", ".blg", ".log", ".out", ".fls", ".fdb_latexmk", ".synctex.gz", ".pyc"}


def check_lean() -> None:
    lean_dir = ROOT / "lean"
    p1 = subprocess.run(["lake", "build"], cwd=lean_dir, capture_output=True, text=True)
    if p1.returncode != 0:
        raise RuntimeError(f"lake build failed:\n{p1.stdout}\n{p1.stderr}")
    if "warning:" in (p1.stdout + p1.stderr).lower():
        raise RuntimeError(f"Lean warnings detected:\n{p1.stdout}\n{p1.stderr}")
    p2 = subprocess.run(["lake", "env", "lean", "AuditAxioms.lean"], cwd=lean_dir, capture_output=True, text=True)
    if p2.returncode != 0:
        raise RuntimeError(f"AuditAxioms failed:\n{p2.stdout}\n{p2.stderr}")
    out = p2.stdout + p2.stderr
    for banned in ("sorryAx", "Classical.choice", "Lean.ofReduceBool", "Lean.trustCompiler"):
        if banned in out:
            raise RuntimeError(f"Forbidden Lean axiom detected: {banned}")
    for thm in EXPECTED_LEAN_THEOREMS:
        if thm not in out:
            raise RuntimeError(f"Missing Lean theorem in AuditAxioms output: {thm}")
    print(f"[PASS] Lean 4 kernel: {len(EXPECTED_LEAN_THEOREMS)}/18 theorems, 0 warnings, 0 sorry, 0 Classical.choice")


def check_certs() -> None:
    certs = sorted((ROOT / "results" / "certificates").glob("*.cert.json"))
    if len(certs) != 16:
        raise RuntimeError(f"Expected 16 certificates, found {len(certs)}")
    for path in certs:
        data = json.loads(path.read_text(encoding="utf-8"))
        verify_lorentzian_certificate_exact(data)
    print(f"[PASS] Exact-Q certificates: {len(certs)}/16 independently verified (0 floats)")


def check_pytest() -> None:
    p = subprocess.run([sys.executable, "-m", "pytest", "-q"], cwd=ROOT, capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"pytest failed:\n{p.stdout}\n{p.stderr}")
    summary = p.stdout.strip().splitlines()[-1]
    print(f"[PASS] pytest: {summary}")


def _collect_manifest_files() -> dict[str, str]:
    hashes: dict[str, str] = {}
    for path in sorted(ROOT.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(ROOT)
        if any(part in MANIFEST_SKIP_DIRS for part in rel.parts):
            continue
        if rel.name == "SHA256SUMS.json" or path.suffix in MANIFEST_SKIP_EXTS:
            continue
        if rel.parts[0] == "paper" and rel.name == "main.pdf":
            continue
        h = hashlib.sha256(path.read_bytes()).hexdigest()
        hashes[rel.as_posix()] = h
    return hashes


def check_or_write_manifest(write: bool) -> None:
    manifest_path = ROOT / "SHA256SUMS.json"
    current = _collect_manifest_files()
    if write:
        manifest_path.write_text(json.dumps(current, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"[PASS] Wrote SHA256SUMS.json ({len(current)} files)")
        return
    if not manifest_path.exists():
        raise RuntimeError("SHA256SUMS.json missing; run with --write-manifest first.")
    recorded = json.loads(manifest_path.read_text(encoding="utf-8"))
    if recorded != current:
        missing = sorted(set(recorded) - set(current))
        extra = sorted(set(current) - set(recorded))
        changed = sorted(k for k in set(recorded) & set(current) if recorded[k] != current[k])
        raise RuntimeError(f"SHA256SUMS mismatch! missing={missing}, extra={extra}, changed={changed}")
    print(f"[PASS] SHA256SUMS.json: {len(current)}/{len(current)} hashes match")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write-manifest", action="store_true")
    args = ap.parse_args()
    check_lean()
    check_certs()
    check_pytest()
    check_or_write_manifest(args.write_manifest)
    print("=== ALL PAPER 11 GATE-2 VERIFICATION CHECKS PASSED ===")


if __name__ == "__main__":
    main()

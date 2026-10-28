"""Environment-neutral verifier contracts for CCV.

The contract exposes only the public, task-independent comparison semantics.
It must never contain held-out examples, labels, predictions, or per-example
feedback.  Environment adapters may implement ``get_verifier_contract`` to
provide their own contract without coupling the regularization core to a benchmark.
"""
from __future__ import annotations

from copy import deepcopy


_FALLBACK_CONTRACT = {
    "id": "environment_verifier_unspecified",
    "public": True,
    "scope": "Only failures confirmed by the environment verifier can reject an update.",
    "normalization": [],
    "equivalences": [],
    "ignored": [],
    "attack_requirements": [
        "Predict an old-pass/candidate-fail outcome under the environment verifier.",
        "Do not rely on differences that the verifier treats as equivalent.",
    ],
}


def normalize_verifier_contract(value: dict | None) -> dict:
    """Return a compact JSON-safe verifier contract with stable keys."""
    if value is None:
        return deepcopy(_FALLBACK_CONTRACT)
    if not isinstance(value, dict):
        raise TypeError("verifier contract must be a dictionary")
    contract = {
        "id": str(value.get("id", "environment_verifier_unspecified") or
                  "environment_verifier_unspecified"),
        "public": bool(value.get("public", True)),
        "scope": str(value.get("scope", "") or ""),
        "normalization": [str(row) for row in value.get("normalization", []) or []],
        "equivalences": [str(row) for row in value.get("equivalences", []) or []],
        "type_rule": str(value.get("type_rule", "") or ""),
        "ignored": [str(row) for row in value.get("ignored", []) or []],
        "attack_requirements": [
            str(row) for row in value.get("attack_requirements", []) or []
        ],
    }
    if not contract["attack_requirements"]:
        contract["attack_requirements"] = list(
            _FALLBACK_CONTRACT["attack_requirements"]
        )
    return contract


def resolve_verifier_contract(adapter) -> dict:
    """Read a public verifier contract from an arbitrary environment adapter."""
    provider = getattr(adapter, "get_verifier_contract", None)
    if not callable(provider):
        return normalize_verifier_contract(None)
    return normalize_verifier_contract(provider())

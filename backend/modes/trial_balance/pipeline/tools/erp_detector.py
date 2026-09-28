"""Deterministic ERP identification (TB Multi-ERP Adaptability upgrade,
Phase 2 -- see modes/trial_balance/documentation/
TB_Multi_ERP_Adaptability_Implementation_Upgrade_Design.md, sections 11/22).

Purely additive: not wired into input_dispatch.py/classify_workbooks or any
existing parser yet -- wiring happens once ERP adapters exist (later
phases). Operates per-file on an already-loaded {sheet_name: grid} dict,
mirroring input_header_detect.pick_tb_sheet's own shape, so it composes
naturally with input_scenario_b._load_workbook_sheets without re-implementing
workbook loading.

Section 1.6.1 finding, which shapes the tier ordering below: every real
SAP/Oracle/Tally file received so far is already client-pre-standardized
before handoff -- none carry raw ERP-internal fingerprints (BUKRS/RLDNR/
HKONT for SAP, multi-segment account combinations for Oracle). Structural
fingerprinting is kept here as a fallback for a future batch of raw,
unconverted exports, but for this batch only the path/filename-convention
tier is expected to actually fire.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from modes.trial_balance.pipeline.tools.tb_erp_contracts import ERPType

_MAX_SCAN_ROWS = 15  # matches input_header_detect.locate_header's own scan depth


def _normalize_cell(value) -> str:
    if value is None:
        return ""
    return str(value).strip().lower()


def _all_normalized_tokens(sheets: Dict[str, list], max_rows: int = _MAX_SCAN_ROWS) -> set:
    tokens = set()
    for grid in sheets.values():
        for row in grid[:max_rows]:
            for cell in row:
                norm = _normalize_cell(cell)
                if norm:
                    tokens.add(norm)
    return tokens


def _path_tokens(path: Path) -> set:
    tokens = set()
    for part in path.parts:
        tokens.add(part.strip().lower())
        tokens.update(t for t in part.replace("_", " ").replace("-", " ").lower().split() if t)
    return tokens


# Tier 1: raw ERP-internal technical field names (section 11) -- only ever
# expected on an UNCONVERTED export; section 1.6.1 confirms none of the
# standardized real files carry these.
_SAP_STRUCTURAL_FINGERPRINTS = {
    "bukrs", "rldnr", "hkont", "racct", "gjahr", "monat", "poper", "waers", "prctr", "xbilk",
}
_ORACLE_STRUCTURAL_FINGERPRINTS = {"account combination", "natural account", "report definition"}
_TALLY_STRUCTURAL_FINGERPRINTS = {"primary group"}  # "ledger"/"group" alone are too generic to be a fingerprint

# Tier 2: human-readable column labels that still lean ERP-specific (weaker
# than tier 1, but stronger than a bare path guess).
_SAP_ALIAS_SIGNALS = {"company code", "profit centre", "profit center", "financial statement version"}
_ORACLE_ALIAS_SIGNALS = {"cost centre", "cost center", "balancing segment"}
_TALLY_ALIAS_SIGNALS = {"ledger name"}

# Tier 3: path/filename convention (section 1.6.1's actual recommendation
# for this batch) -- the ERP name appearing as a directory segment or
# filename token. Weakest signal: an artifact of how files were organized
# for handoff, not derived from file content.
_ERP_PATH_TOKENS = {
    ERPType.SAP: {"sap"},
    ERPType.ORACLE: {"oracle"},
    ERPType.TALLY: {"tally"},
    ERPType.GENERIC: {"in_house_or_other", "in house or other", "inhouse", "in-house"},
}

_FINGERPRINT_BY_ERP = {
    ERPType.SAP: _SAP_STRUCTURAL_FINGERPRINTS,
    ERPType.ORACLE: _ORACLE_STRUCTURAL_FINGERPRINTS,
    ERPType.TALLY: _TALLY_STRUCTURAL_FINGERPRINTS,
}
_ALIAS_BY_ERP = {
    ERPType.SAP: _SAP_ALIAS_SIGNALS,
    ERPType.ORACLE: _ORACLE_ALIAS_SIGNALS,
    ERPType.TALLY: _TALLY_ALIAS_SIGNALS,
}


@dataclass(frozen=True)
class ERPDetectionResult:
    """Deliverable shape per design doc section 22 Phase 2: {"erp_type",
    "confidence", "evidence"}. `evidence` is human-readable strings naming
    which signal(s) fired, for the same reason ClassificationEvidence and
    ERPSourceMetadata exist -- an auditor or developer must be able to see
    WHY a document was classified a given way, not just the answer."""

    erp_type: ERPType
    confidence: float
    evidence: List[str] = field(default_factory=list)


def detect_erp_type(
    path: Path, sheets: Dict[str, list], *, explicit_erp_type: Optional[ERPType] = None,
) -> ERPDetectionResult:
    """Deterministic ERP identification (section 11) -- no LLM (section 25).
    Tries, in order: an explicit caller-supplied selection, structural
    fingerprints, ERP-leaning column aliases, then path/filename convention.
    Returns ERPType.UNKNOWN with 0.0 confidence if nothing matches -- a
    real, expected outcome for a pre-standardized export with no other
    signal (section 1.6.1), not a bug to suppress."""
    if explicit_erp_type is not None:
        return ERPDetectionResult(explicit_erp_type, 1.0, [f"explicit_selection: {explicit_erp_type.value}"])

    tokens = _all_normalized_tokens(sheets)

    for erp_type, fingerprints in _FINGERPRINT_BY_ERP.items():
        matched = fingerprints & tokens
        if matched:
            return ERPDetectionResult(erp_type, 0.95, [f"structural_fingerprint: {', '.join(sorted(matched))}"])

    for erp_type, aliases in _ALIAS_BY_ERP.items():
        matched = aliases & tokens
        if matched:
            return ERPDetectionResult(erp_type, 0.6, [f"column_alias: {', '.join(sorted(matched))}"])

    path_tokens = _path_tokens(path)
    for erp_type, markers in _ERP_PATH_TOKENS.items():
        matched = markers & path_tokens
        if matched:
            evidence = f"path_token: {', '.join(sorted(matched))!r} in {path.name!r}'s path"
            return ERPDetectionResult(erp_type, 0.4, [evidence])

    return ERPDetectionResult(ERPType.UNKNOWN, 0.0, ["no_signal_matched"])

"""Scope-aware quality gate (TB Multi-ERP Adaptability upgrade, Phase 6 --
see modes/trial_balance/documentation/
TB_Multi_ERP_Adaptability_Implementation_Upgrade_Design.md, sections
7.4/16/17/22).

Purely additive: not wired into quality_gate.py, input_dispatch.py, or any
existing parser/adapter yet. `quality_gate.py` already does row-level
arithmetic, duplicate GL codes, grouping coverage, and the ingestion-error-
catalog checks (section 16, "Keep it") -- this module adds exactly what's
missing from section 16's list that this project can actually implement
with the data it has: document-level grand total (section 7.4), scope-aware
interpretation of it (section 16.2), opening-balance validation across
fiscal years (section 16.3), and the P&L-opening check (section 16.4).

Control-account checks (section 16.5) and true per-row currency-mixing
validation need data sources this project doesn't have (subsidiary-ledger
data, per-row currency columns beyond ScopeMetadata's single captured
value) -- not implemented here, not faked. Suspense/clearing checks
(section 16.6) are already covered per-ERP in tally_adapter.py's
_detect_suspense_accounts and sap_adapter.py's _detect_not_assigned_accounts
-- not duplicated here.

This is the first module to actually implement the section-17 Validation
Severity Model (PASS/WARNING/CONFIRMATION_REQUIRED/BLOCKING_ERROR/
AUDIT_RISK) -- both tally_adapter.py and sap_adapter.py's Finding types are
deliberately narrower, deferring the full model to here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional

from modes.trial_balance.pipeline.tools.tb_erp_contracts import ClassificationEvidence, ScopeMetadata, ScopeType
from modes.trial_balance.pipeline.tools.tb_models import TBRow

ValidationSeverity = Literal["PASS", "WARNING", "CONFIRMATION_REQUIRED", "BLOCKING_ERROR", "AUDIT_RISK"]

DEFAULT_TOLERANCE = 0.01
_PL_LABELS = {"pl", "p&l", "p & l", "p and l"}

# scope_types where a non-balancing grand total is the EXPECTED shape, not
# a finding -- a slice/subset of the ledger is not required to net to zero
# (section 16.2).
_SLICE_SCOPE_TYPES = {
    ScopeType.ENTITY_SLICE,
    ScopeType.ACCOUNT_SLICE,
    ScopeType.PROFIT_CENTER_SLICE,
    ScopeType.SEGMENT_SLICE,
    ScopeType.CONSOLIDATED,
    ScopeType.MULTI_ENTITY,
}


@dataclass(frozen=True)
class ValidationFinding:
    """The section-17 Validation Severity Model, actually implemented here
    (both tally_adapter.py and sap_adapter.py deliberately deferred this,
    using a narrower adapter-local vocabulary instead -- see their own
    docstrings)."""

    severity: ValidationSeverity
    code: str
    message: str


@dataclass(frozen=True)
class ScopeAwareQualityReport:
    total_debit: float
    total_credit: float
    grand_total_diff: float
    findings: List[ValidationFinding] = field(default_factory=list)


def check_grand_total(tb_rows: List[TBRow]) -> tuple:
    total_debit = sum(r.debit for r in tb_rows)
    total_credit = sum(r.credit for r in tb_rows)
    return total_debit, total_credit, abs(total_debit - total_credit)


def _evaluate_grand_total(
    total_debit: float, total_credit: float, diff: float, scope: ScopeMetadata, tolerance: float,
) -> Optional[ValidationFinding]:
    """Section 7.4 (grand-total check was explicitly absent before this
    phase) + section 16.2 (must be scope-aware, not blind). A slice scope
    is EXPECTED to have a non-zero total -- only FULL_LEDGER and UNKNOWN
    scope are evaluated here."""
    if scope.scope_type in _SLICE_SCOPE_TYPES:
        return None
    if diff <= tolerance:
        return None
    if scope.scope_type == ScopeType.UNKNOWN:
        return ValidationFinding(
            severity="CONFIRMATION_REQUIRED",
            code="grand_total_unknown_scope",
            message=(
                f"Total Debit ({total_debit:,.2f}) != Total Credit ({total_credit:,.2f}), diff={diff:,.2f}, "
                "and this document's scope could not be determined -- confirm whether this is a "
                "full ledger (where this would be a real imbalance) or an intentional slice."
            ),
        )
    # scope_type == FULL_LEDGER
    return ValidationFinding(
        severity="AUDIT_RISK",
        code="grand_total_imbalance",
        message=(
            f"Total Debit ({total_debit:,.2f}) != Total Credit ({total_credit:,.2f}), diff={diff:,.2f}, "
            "in a document scoped as a full ledger -- flag for audit review."
        ),
    )


def check_pl_opening(
    tb_rows: List[TBRow], classification_by_gl_code: Dict[str, ClassificationEvidence],
    tolerance: float = DEFAULT_TOLERANCE,
) -> List[ValidationFinding]:
    """Section 16.4: P&L accounts should normally open at zero (closed to
    equity at prior year-end). Only evaluates rows whose ClassificationEvidence
    actually carries a bs_pl value -- silent (no finding) when classification
    wasn't available at all, never a false positive from missing data."""
    findings = []
    for row in tb_rows:
        evidence = classification_by_gl_code.get(row.gl_code)
        bs_pl = evidence.account_classification.get("bs_pl") if evidence else None
        if bs_pl and bs_pl.strip().lower() in _PL_LABELS and abs(row.opening) > tolerance:
            findings.append(ValidationFinding(
                severity="WARNING",
                code="pl_account_nonzero_opening",
                message=(
                    f"'{row.gl_name}' (gl_code={row.gl_code}) is classified P&L but has a non-zero "
                    f"opening balance ({row.opening:,.2f}) -- P&L accounts should normally open at zero."
                ),
            ))
    return findings


def check_opening_against_prior_year(
    tb_rows: List[TBRow], prior_year_tb_rows: List[TBRow], tolerance: float = DEFAULT_TOLERANCE,
) -> List[ValidationFinding]:
    """Section 16.3: current Opening should equal prior-year Closing,
    account by account. Only compares gl_codes present in both years --
    a code appearing in only one year is a different, coverage-related
    question, not an opening-mismatch one."""
    prior_by_code = {r.gl_code: r for r in prior_year_tb_rows}
    findings = []
    for row in tb_rows:
        prior = prior_by_code.get(row.gl_code)
        if prior is None:
            continue
        diff = abs(prior.closing - row.opening)
        if diff > tolerance:
            findings.append(ValidationFinding(
                severity="AUDIT_RISK",
                code="opening_prior_year_closing_mismatch",
                message=(
                    f"'{row.gl_name}' (gl_code={row.gl_code}): prior-year closing ({prior.closing:,.2f}) "
                    f"!= current opening ({row.opening:,.2f}), diff={diff:,.2f}."
                ),
            ))
    return findings


def run_scope_aware_quality_gate(
    tb_rows: List[TBRow], scope: ScopeMetadata, classification_by_gl_code: Dict[str, ClassificationEvidence],
    *, prior_year_tb_rows: Optional[List[TBRow]] = None, tolerance: float = DEFAULT_TOLERANCE,
) -> ScopeAwareQualityReport:
    total_debit, total_credit, diff = check_grand_total(tb_rows)
    findings: List[ValidationFinding] = []

    grand_total_finding = _evaluate_grand_total(total_debit, total_credit, diff, scope, tolerance)
    if grand_total_finding:
        findings.append(grand_total_finding)

    findings.extend(check_pl_opening(tb_rows, classification_by_gl_code, tolerance))

    if prior_year_tb_rows is not None:
        findings.extend(check_opening_against_prior_year(tb_rows, prior_year_tb_rows, tolerance))

    return ScopeAwareQualityReport(
        total_debit=total_debit, total_credit=total_credit, grand_total_diff=diff, findings=findings,
    )

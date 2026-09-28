"""SAP ERP adapter (TB Multi-ERP Adaptability upgrade, Phase 5 -- see
modes/trial_balance/documentation/
TB_Multi_ERP_Adaptability_Implementation_Upgrade_Design.md, sections 5.3/22).

Purely additive: not wired into input_dispatch.py or any existing parser
yet. Operates on already-loaded (grid, header_row, cols, tb_rows) -- the
same shapes pick_tb_sheet/build_tb_rows_from_grid already produce -- and
composes ERP-neutral contracts from tb_erp_contracts.py (Phase 1) around
them, same entry-point shape as tally_adapter.py (Phase 3).

Real-fixture inspection (Terra_Clean, IOC_Global) found classification data
(Company Code/Business Area/BS-PL/Sub Head/Main Head/Account Type) commonly
embedded directly on the TB sheet's own row cells, side-by-side with the TB
columns -- the same pattern found for Tally in Phase 3. None of the raw SAP
technical fields (BUKRS/RLDNR/HKONT/MONAT/POPER) appear in any real file
(confirmed in Phase 2) -- this batch is client-flattened, per section 1.6.1.

SAP-specific fields (Company Code, Business Area, Ledger, Account Group,
Profit Centre, Segment, FS Version, Period, Currency) are scanned via a
self-contained local alias vocabulary, deliberately NOT added to
input_header_detect.py's shared FIELD_ALIASES -- Phase 3 already found that
a new generic-sounding alias there can have cross-ERP blast radius.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Literal, Optional

from modes.trial_balance.pipeline.tools.document_metadata import DocumentMetadata
from modes.trial_balance.pipeline.tools.tb_erp_contracts import (
    ClassificationEvidence,
    ERPSourceMetadata,
    ERPType,
    ScopeMetadata,
    ScopeType,
)
from modes.trial_balance.pipeline.tools.tb_models import GroupingHint, TBRow

FindingSeverity = Literal["info", "warning", "audit_risk"]

_SAP_FIELD_ALIASES = {
    "company_code": {"company code", "co. code", "co code"},
    "business_area": {"business area"},
    "ledger": {"ledger"},
    "account_group": {"account group"},
    "profit_center": {"profit centre", "profit center"},
    "segment": {"segment"},
    "financial_statement_version": {"financial statement version", "fs version"},
    "period": {"period", "posting period"},
    "currency": {"currency", "curr.", "curr"},
}

_NOT_ASSIGNED_KEYWORDS = ("not assigned", "unassigned", "not applicable")
_EXTENDED_FISCAL_PERIODS = {"13", "14", "15", "16"}


def _norm(value) -> str:
    return str(value).strip().lower() if value not in (None, "") else ""


def _scan_sap_field_columns(header_row_cells: list) -> Dict[str, int]:
    found = {}
    for idx, cell in enumerate(header_row_cells):
        norm = _norm(cell)
        if not norm:
            continue
        for field_name, aliases in _SAP_FIELD_ALIASES.items():
            if field_name not in found and norm in aliases:
                found[field_name] = idx
    return found


def _row_field(row: Optional[list], field_map: dict, field_name: str) -> Optional[str]:
    if row is None:
        return None
    idx = field_map.get(field_name)
    if idx is None or idx >= len(row) or row[idx] in (None, ""):
        return None
    return str(row[idx]).strip()


@dataclass(frozen=True)
class SAPAdapterFinding:
    """Same narrower, adapter-local vocabulary as TallyAdapterFinding
    (tally_adapter.py) -- not the full section-17 Validation Severity
    Model, which is Phase 6's own scope-aware quality gate."""

    severity: FindingSeverity
    code: str
    message: str


@dataclass(frozen=True)
class SAPAdapterResult:
    tb_rows: List[TBRow]
    classification_by_gl_code: Dict[str, ClassificationEvidence]
    scope: ScopeMetadata
    source_metadata: List[ERPSourceMetadata]
    findings: List[SAPAdapterFinding]


def _build_classification_evidence(
    row: Optional[list], cols: dict, sap_cols: dict, tb_row: TBRow, hint: Optional[GroupingHint],
) -> ClassificationEvidence:
    account_classification = {}
    for field_name in ("bs_pl", "account_type", "main_head", "sub_head_1", "sub_head_2"):
        value = _row_field(row, cols, field_name) or (hint.known_fields.get(field_name) if hint else None)
        if value:
            account_classification[field_name] = value
    account_group = _row_field(row, sap_cols, "account_group")
    if account_group:
        account_classification["account_group"] = account_group

    financial_statement = {}
    fs_version = _row_field(row, sap_cols, "financial_statement_version")
    if fs_version:
        financial_statement["financial_statement_version"] = fs_version

    business_dimensions = {}
    for field_name in ("business_area", "profit_center", "segment"):
        value = _row_field(row, sap_cols, field_name)
        if value:
            business_dimensions[field_name] = value

    return ClassificationEvidence(
        erp_type=ERPType.SAP,
        account_identity={"gl_code": tb_row.gl_code, "gl_name": tb_row.gl_name},
        account_classification=account_classification,
        financial_statement=financial_statement,
        business_dimensions=business_dimensions,
    )


def _detect_not_assigned_accounts(
    tb_rows: List[TBRow], classification_by_gl_code: Dict[str, ClassificationEvidence],
) -> List[SAPAdapterFinding]:
    findings = []
    for row in tb_rows:
        evidence = classification_by_gl_code.get(row.gl_code)
        values = " ".join(evidence.account_classification.values()) if evidence else ""
        haystack = f"{row.gl_name} {values}".lower()
        if any(keyword in haystack for keyword in _NOT_ASSIGNED_KEYWORDS):
            findings.append(SAPAdapterFinding(
                severity="audit_risk",
                code="sap_not_assigned_account",
                message=(
                    f"'{row.gl_name}' (gl_code={row.gl_code}) reads as a not-assigned/unassigned "
                    "account -- flag for audit review (suspense/clearing checks)."
                ),
            ))
    return findings


def _detect_extended_fiscal_periods(grid: list, header_row: int, sap_cols: dict) -> Optional[SAPAdapterFinding]:
    """Periods 13-16 (SAP's year-end adjustment periods) are real and
    common (section 5.3), but omitting or wrongly including them changes
    the TB total -- best-effort: only checks when a Period column is
    actually present (none of the real fixtures inspected carry one;
    section 1.6.1's flattening pattern applies here too)."""
    period_idx = sap_cols.get("period")
    if period_idx is None:
        return None
    found_periods = set()
    for row in grid[header_row + 1:]:
        if period_idx < len(row) and row[period_idx] not in (None, ""):
            found_periods.add(str(row[period_idx]).strip())
    extended = found_periods & _EXTENDED_FISCAL_PERIODS
    if not extended:
        return None
    return SAPAdapterFinding(
        severity="warning",
        code="sap_extended_fiscal_periods_present",
        message=(
            f"Periods {sorted(extended)} (13-16, year-end adjustment periods) are present -- "
            "confirm whether these should be included in this Trial Balance total."
        ),
    )


def _detect_multiple_company_codes(grid: list, header_row: int, sap_cols: dict) -> Optional[SAPAdapterFinding]:
    """A genuine, checkable multi-entity signal per section 18's scope
    model -- a document mixing multiple Company Codes is a MULTI_ENTITY
    scope, not a single full-ledger TB, and grand-total checks (Phase 6)
    must not treat it as one."""
    code_idx = sap_cols.get("company_code")
    if code_idx is None:
        return None
    codes = set()
    for row in grid[header_row + 1:]:
        if code_idx < len(row) and row[code_idx] not in (None, ""):
            codes.add(str(row[code_idx]).strip())
    if len(codes) <= 1:
        return None
    return SAPAdapterFinding(
        severity="warning",
        code="sap_multiple_company_codes",
        message=(
            f"{len(codes)} distinct Company Codes found in one document ({sorted(codes)}) -- "
            "this is a multi-entity scope, not a single full-ledger Trial Balance."
        ),
    )


def _build_scope_metadata(
    metadata: Optional[DocumentMetadata], grid: list, header_row: int, sap_cols: dict,
    multi_company_finding: Optional[SAPAdapterFinding],
) -> ScopeMetadata:
    company_code = ledger = currency = None
    for row in grid[header_row + 1:]:
        company_code = company_code or _row_field(row, sap_cols, "company_code")
        ledger = ledger or _row_field(row, sap_cols, "ledger")
        currency = currency or _row_field(row, sap_cols, "currency")
        if company_code and ledger and currency:
            break
    return ScopeMetadata(
        entity=metadata.entity_name if metadata else None,
        company_code=company_code,
        ledger=ledger,
        fiscal_year=metadata.financial_year if metadata else None,
        currency=currency,
        scope_type=ScopeType.MULTI_ENTITY if multi_company_finding else ScopeType.FULL_LEDGER,
    )


def _build_source_metadata(
    path: Path, sheet_name: str, header_row: int, tb_rows: List[TBRow],
    classification_by_gl_code: Dict[str, ClassificationEvidence],
) -> List[ERPSourceMetadata]:
    result = []
    for i, row in enumerate(tb_rows):
        evidence = classification_by_gl_code.get(row.gl_code)
        result.append(ERPSourceMetadata(
            source_file_id=path.name,
            source_sheet=sheet_name,
            source_row_number=header_row + 2 + i,
            source_account_identifier=row.gl_code,
            source_raw_account_name=row.gl_name,
            source_raw_group=evidence.account_classification.get("account_group") if evidence else None,
            erp_type=ERPType.SAP,
        ))
    return result


def run_sap_adapter(
    path: Path, grid: list, header_row: int, cols: dict, tb_rows: List[TBRow],
    *, grouping_hints: Optional[Dict[str, GroupingHint]] = None,
    metadata: Optional[DocumentMetadata] = None, sheet_name: str = "",
) -> SAPAdapterResult:
    grouping_hints = grouping_hints or {}
    sap_cols = _scan_sap_field_columns(grid[header_row]) if header_row < len(grid) else {}

    rows_by_gl_code = {}
    for row in grid[header_row + 1:]:
        gl_code_val = _row_field(row, cols, "gl_code")
        if gl_code_val:
            rows_by_gl_code[gl_code_val] = row

    classification_by_gl_code = {
        tb_row.gl_code: _build_classification_evidence(
            rows_by_gl_code.get(tb_row.gl_code), cols, sap_cols, tb_row, grouping_hints.get(tb_row.gl_code),
        )
        for tb_row in tb_rows
    }

    findings: List[SAPAdapterFinding] = []
    findings.extend(_detect_not_assigned_accounts(tb_rows, classification_by_gl_code))
    extended_periods_finding = _detect_extended_fiscal_periods(grid, header_row, sap_cols)
    if extended_periods_finding:
        findings.append(extended_periods_finding)
    multi_company_finding = _detect_multiple_company_codes(grid, header_row, sap_cols)
    if multi_company_finding:
        findings.append(multi_company_finding)

    return SAPAdapterResult(
        tb_rows=tb_rows,
        classification_by_gl_code=classification_by_gl_code,
        scope=_build_scope_metadata(metadata, grid, header_row, sap_cols, multi_company_finding),
        source_metadata=_build_source_metadata(path, sheet_name, header_row, tb_rows, classification_by_gl_code),
        findings=findings,
    )

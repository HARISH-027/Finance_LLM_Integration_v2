"""Tally ERP adapter (TB Multi-ERP Adaptability upgrade, Phase 3 -- see
modes/trial_balance/documentation/
TB_Multi_ERP_Adaptability_Implementation_Upgrade_Design.md, sections 5.1/22).

Purely additive: not wired into input_dispatch.py or any existing parser
yet. Operates on already-loaded (grid, header_row, cols, tb_rows) -- the
same shapes pick_tb_sheet/build_tb_rows_from_grid already produce -- and
composes ERP-neutral contracts from tb_erp_contracts.py (Phase 1) around
them. Key rule (section 22 Phase 3): never require gl_code != NULL for
Tally -- already honored by every function this module reuses (Phase 0's
classify_workbooks fix made gl_name a valid alternate row identity).

Section 5.1's "Ledger -> Group -> Parent Group -> Primary Group" hierarchy
is commonly carried directly on the TB sheet's own row cells for Tally
exports (confirmed against the real Gurugram fixture: Head/Sub-Head are
columns on the TB sheet itself, not a separate Grouping file) -- so
classification here reads row-embedded columns first, falling back to a
supplied GroupingHint only when the row itself doesn't carry a field.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Literal, Optional

from modes.trial_balance.pipeline.tools.document_metadata import DocumentMetadata
from modes.trial_balance.pipeline.tools.input_header_detect import _find_split_balance
from modes.trial_balance.pipeline.tools.tb_erp_contracts import (
    ClassificationEvidence,
    ERPSourceMetadata,
    ERPType,
    ScopeMetadata,
    ScopeType,
)
from modes.trial_balance.pipeline.tools.tb_models import GroupingHint, TBRow

FindingSeverity = Literal["info", "warning", "audit_risk"]

_GROUP_HIERARCHY_FIELDS = ("main_head", "sub_head_1", "sub_head_2")
_SUSPENSE_KEYWORDS = ("suspense", "misc", "miscellaneous", "unclassified", "unidentified", "clearing")


@dataclass(frozen=True)
class TallyAdapterFinding:
    """Deliberately NOT the full section-17 Validation Severity Model
    (PASS/WARNING/CONFIRMATION_REQUIRED/BLOCKING_ERROR/AUDIT_RISK) -- that's
    Phase 6's own scope-aware quality gate. This is a narrower, adapter-
    local vocabulary just for what this module itself can determine."""

    severity: FindingSeverity
    code: str
    message: str


@dataclass(frozen=True)
class TallyAdapterResult:
    tb_rows: List[TBRow]
    classification_by_gl_code: Dict[str, ClassificationEvidence]
    scope: ScopeMetadata
    source_metadata: List[ERPSourceMetadata]
    findings: List[TallyAdapterFinding]


PERIOD_LOCKING_NOTE = TallyAdapterFinding(
    severity="info",
    code="tally_no_period_locking",
    message=(
        "Tally has no period locking -- any past voucher can be silently altered after "
        "the fact. Treat this document's historical accuracy as a data-trust note, not a guarantee."
    ),
)


def _row_field(row: Optional[list], cols: dict, field_name: str) -> Optional[str]:
    if row is None:
        return None
    idx = cols.get(field_name)
    if idx is None or idx >= len(row) or row[idx] in (None, ""):
        return None
    return str(row[idx]).strip()


def _extract_group_fields(row: Optional[list], cols: dict, hint: Optional[GroupingHint]) -> Dict[str, str]:
    """Row-embedded columns are the primary source (Tally commonly carries
    Head/Sub-Head directly on the TB sheet itself); a companion GroupingHint
    is only consulted per-field when the row doesn't carry it. Field-keyed
    (not positional) so a missing middle field (e.g. no Sub Head 1 column,
    but Main Head and Sub Head 2 both present) can never be mislabeled."""
    fields = {}
    for field_name in _GROUP_HIERARCHY_FIELDS:
        value = _row_field(row, cols, field_name)
        if not value and hint:
            value = hint.known_fields.get(field_name)
        if value:
            fields[field_name] = value
    return fields


def _extract_group_hierarchy(row: Optional[list], cols: dict, hint: Optional[GroupingHint]) -> List[str]:
    return list(_extract_group_fields(row, cols, hint).values())


def _build_classification_evidence(
    gl_name: str, row: Optional[list], cols: dict, hint: Optional[GroupingHint],
) -> ClassificationEvidence:
    fields = _extract_group_fields(row, cols, hint)
    account_classification = dict(fields)
    path = list(fields.values())
    if path:
        account_classification["erp_group"] = path[-1]
        account_classification["erp_group_path"] = " -> ".join(path)
    return ClassificationEvidence(
        erp_type=ERPType.TALLY,
        account_identity={"gl_name": gl_name},
        account_classification=account_classification,
    )


def _detect_missing_opening_or_movement_columns(
    grid: list, header_row: int, cols: dict,
) -> Optional[TallyAdapterFinding]:
    """Section 1.5.3: opening-balance/movement columns can be switched OFF
    in the Tally export UI -- must be flagged, not silently treated as a
    genuine zero balance."""
    split = _find_split_balance(grid, header_row, cols)
    missing = []
    if "opening" not in cols and "opening" not in split:
        missing.append("opening balance")
    if "debit" not in cols and "credit" not in cols and "closing" not in split:
        # closing-via-split-balance already IS the year's movement in
        # signed form; only flag when there's no debit/credit AND no
        # split-balance fallback covering it either.
        missing.append("debit/credit movement")
    if not missing:
        return None
    return TallyAdapterFinding(
        severity="warning",
        code="tally_missing_opening_or_movement_columns",
        message=(
            f"{', '.join(missing)} column(s) not found in this export -- do not assume zero; "
            "confirm whether they were switched off in the source Tally UI."
        ),
    )


def _detect_grouped_view(
    tb_rows: List[TBRow], grid: list, header_row: int, cols: dict,
) -> Optional[TallyAdapterFinding]:
    """A 'grouped' Tally export (rows = Groups, not Ledgers) is not
    equivalent to a ledger-wise TB (section 5.1). Heuristic: collect every
    distinct Head/Sub-Head/Sub-Sub-Head label seen anywhere on the sheet,
    then check how many TB rows' own gl_name exactly match one of those
    group labels -- a real ledger-wise export should show ~zero overlap."""
    group_labels = set()
    for row in grid[header_row + 1:]:
        for value in _extract_group_hierarchy(row, cols, None):
            group_labels.add(value.strip().lower())
    if not group_labels or not tb_rows:
        return None
    overlapping = sum(1 for r in tb_rows if r.gl_name.strip().lower() in group_labels)
    if overlapping / len(tb_rows) > 0.1:
        return TallyAdapterFinding(
            severity="warning",
            code="tally_possible_grouped_view",
            message=(
                f"{overlapping}/{len(tb_rows)} row account names match known Head/Sub-Head labels -- "
                "this document may be a grouped/rolled-up view rather than a ledger-wise Trial Balance."
            ),
        )
    return None


def _detect_suspense_accounts(
    tb_rows: List[TBRow], classification_by_gl_code: Dict[str, ClassificationEvidence],
) -> List[TallyAdapterFinding]:
    findings = []
    for row in tb_rows:
        evidence = classification_by_gl_code.get(row.gl_code)
        group_text = evidence.account_classification.get("erp_group_path", "") if evidence else ""
        haystack = f"{row.gl_name} {group_text}".lower()
        if any(keyword in haystack for keyword in _SUSPENSE_KEYWORDS):
            findings.append(TallyAdapterFinding(
                severity="audit_risk",
                code="tally_suspense_or_misc_account",
                message=(
                    f"'{row.gl_name}' (gl_code={row.gl_code}) reads as a suspense/miscellaneous/"
                    "clearing account -- flag for audit review."
                ),
            ))
    return findings


def _build_scope_metadata(
    metadata: Optional[DocumentMetadata], grouped_view_finding: Optional[TallyAdapterFinding],
) -> ScopeMetadata:
    return ScopeMetadata(
        entity=metadata.entity_name if metadata else None,
        fiscal_year=metadata.financial_year if metadata else None,
        scope_type=ScopeType.UNKNOWN if grouped_view_finding else ScopeType.FULL_LEDGER,
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
            source_raw_group=evidence.account_classification.get("erp_group") if evidence else None,
            erp_type=ERPType.TALLY,
        ))
    return result


def run_tally_adapter(
    path: Path, grid: list, header_row: int, cols: dict, tb_rows: List[TBRow],
    *, grouping_hints: Optional[Dict[str, GroupingHint]] = None,
    metadata: Optional[DocumentMetadata] = None, sheet_name: str = "",
) -> TallyAdapterResult:
    grouping_hints = grouping_hints or {}
    rows_by_gl_code = {}
    for row in grid[header_row + 1:]:
        gl_code_val = _row_field(row, cols, "gl_code")
        gl_name_val = _row_field(row, cols, "gl_name")
        key = gl_code_val or gl_name_val
        if key:
            rows_by_gl_code[key] = row

    classification_by_gl_code = {
        tb_row.gl_code: _build_classification_evidence(
            tb_row.gl_name, rows_by_gl_code.get(tb_row.gl_code), cols, grouping_hints.get(tb_row.gl_code),
        )
        for tb_row in tb_rows
    }

    findings: List[TallyAdapterFinding] = []
    missing_cols_finding = _detect_missing_opening_or_movement_columns(grid, header_row, cols)
    if missing_cols_finding:
        findings.append(missing_cols_finding)
    grouped_view_finding = _detect_grouped_view(tb_rows, grid, header_row, cols)
    if grouped_view_finding:
        findings.append(grouped_view_finding)
    findings.extend(_detect_suspense_accounts(tb_rows, classification_by_gl_code))
    findings.append(PERIOD_LOCKING_NOTE)

    return TallyAdapterResult(
        tb_rows=tb_rows,
        classification_by_gl_code=classification_by_gl_code,
        scope=_build_scope_metadata(metadata, grouped_view_finding),
        source_metadata=_build_source_metadata(path, sheet_name, header_row, tb_rows, classification_by_gl_code),
        findings=findings,
    )

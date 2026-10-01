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
from modes.trial_balance.pipeline.tools.input_header_detect import (
    _find_split_balance,
    is_total_row_label,
    parse_amount_with_status,
)
from modes.trial_balance.pipeline.tools.tally_primary_group_lookup import classify_primary_group
from modes.trial_balance.pipeline.tools.tally_structural_hierarchy import (
    extract_tally_structural_fields_by_identity,
)
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
_OPENING_BALANCE_DIFFERENCE_RE_LABELS = {"difference in opening balances", "differences in opening balances"}
_DUPLICATE_NAME_NOISE_CHARS = str.maketrans("", "", ".,-&'\"")


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


def _extract_group_fields(
    row: Optional[list], cols: dict, hint: Optional[GroupingHint], structural_main_head: Optional[str] = None,
) -> Dict[str, str]:
    """Row-embedded columns are the primary source (Tally commonly carries
    Head/Sub-Head directly on the TB sheet itself); a companion GroupingHint
    is only consulted per-field when the row doesn't carry it. Field-keyed
    (not positional) so a missing middle field (e.g. no Sub Head 1 column,
    but Main Head and Sub Head 2 both present) can never be mislabeled.

    Third fallback, main_head only: `structural_main_head`, from
    tally_structural_hierarchy's sum-reconciliation parser -- used when
    NEITHER the row's own columns nor the hint carry main_head at all
    (the majority native-export shape, which has no Head/Sub-Head columns
    whatsoever). Never fabricates sub_head_1/sub_head_2 this way; that
    structure genuinely isn't recoverable from a row-order-only export."""
    fields = {}
    for field_name in _GROUP_HIERARCHY_FIELDS:
        value = _row_field(row, cols, field_name)
        if not value and hint:
            value = hint.known_fields.get(field_name)
        if not value and field_name == "main_head" and structural_main_head:
            value = structural_main_head
        if value:
            fields[field_name] = value
    return fields


def _extract_group_hierarchy(row: Optional[list], cols: dict, hint: Optional[GroupingHint]) -> List[str]:
    return list(_extract_group_fields(row, cols, hint).values())


def _build_classification_evidence(
    gl_name: str, row: Optional[list], cols: dict, hint: Optional[GroupingHint],
    *, structural_main_head: Optional[str] = None, structural_erp_group_path: Optional[str] = None,
    closing_balance: Optional[float] = None,
) -> ClassificationEvidence:
    fields = _extract_group_fields(row, cols, hint, structural_main_head)
    account_classification = dict(fields)
    path = list(fields.values())
    if path:
        account_classification["erp_group"] = path[-1]
        account_classification["erp_group_path"] = " -> ".join(path)

    # main_head is the ONLY hierarchy field the structural strategy ever
    # supplies (see _extract_group_fields) -- but its own erp_group_path
    # may carry a deeper, real audit trail (e.g. nested sub-groups the
    # sum-reconciliation parser resolved) than the single-field path
    # computed just above. Never overrides main_head itself -- only
    # enriches erp_group/erp_group_path when nothing richer already came
    # from the row's own columns or a GroupingHint.
    if list(fields.keys()) == ["main_head"] and structural_erp_group_path:
        segments = structural_erp_group_path.split(" -> ")
        if len(segments) > 1:
            account_classification["erp_group_path"] = structural_erp_group_path
            account_classification["erp_group"] = segments[-1]

    # Fourth fallback tier: once main_head is known (from any of the three
    # sources above), the static Primary Group lookup can fill bs_pl/
    # account_type directly for Tally's 15 shipped groups -- never
    # overwrites a value the row/hint already supplied itself, and never
    # guesses for an org-created custom group (classify_primary_group
    # returns None there, same "never guess" contract as
    # taxonomy_repository.derive_account_type).
    main_head = account_classification.get("main_head")
    if main_head and ("bs_pl" not in account_classification or "account_type" not in account_classification):
        classification = classify_primary_group(main_head, closing_balance=closing_balance)
        if classification is not None:
            account_classification.setdefault("bs_pl", classification.bs_pl)
            if classification.account_type is not None:
                account_classification.setdefault("account_type", classification.account_type)

    return ClassificationEvidence(
        erp_type=ERPType.TALLY,
        account_identity={"gl_name": gl_name},
        account_classification=account_classification,
    )


def _detect_closing_only_no_movement(missing: List[str]) -> Optional[TallyAdapterFinding]:
    """IREL IDCOL-style: closing balance present, but opening AND
    debit/credit movement BOTH structurally absent -- its own explicit
    finding/code, distinct from the generic 'missing one or the other'
    message, so downstream consumers can tell 'no visibility into the
    year's movement at all' apart from a partial gap."""
    if set(missing) != {"opening balance", "debit/credit movement"}:
        return None
    return TallyAdapterFinding(
        severity="warning",
        code="tally_closing_only_no_opening_or_movement",
        message=(
            "This export carries a closing balance only -- no opening balance and no "
            "debit/credit movement for the year are available at all. Do not assume this "
            "means no movement occurred; confirm whether these columns were switched off "
            "in the source Tally UI or the export was deliberately closing-only."
        ),
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
    closing_only_finding = _detect_closing_only_no_movement(missing)
    if closing_only_finding is not None:
        return closing_only_finding
    return TallyAdapterFinding(
        severity="warning",
        code="tally_missing_opening_or_movement_columns",
        message=(
            f"{', '.join(missing)} column(s) not found in this export -- do not assume zero; "
            "confirm whether they were switched off in the source Tally UI."
        ),
    )


def _detect_opening_balance_difference_row(tb_rows: List[TBRow]) -> Optional[TallyAdapterFinding]:
    """Tally silently prints a 'Difference in Opening Balances' line when
    manually-keyed opening balances don't net to nil (client audit
    methodology doc, section 3.2/6.2) -- a HIGH-risk signal that the
    opening position was never balanced, so every figure resting on it is
    suspect until resolved."""
    for row in tb_rows:
        if row.gl_name.strip().lower() in _OPENING_BALANCE_DIFFERENCE_RE_LABELS:
            return TallyAdapterFinding(
                severity="audit_risk",
                code="tally_opening_balance_difference",
                message=(
                    f"'{row.gl_name}' row found (amount={row.closing:,.2f}) -- Tally's own signal that "
                    "manually-keyed opening balances did not net to nil. The opening position was never "
                    "balanced; every figure resting on it is suspect until this is resolved."
                ),
            )
    return None


def _normalize_ledger_name(name: str) -> str:
    return " ".join(name.strip().lower().translate(_DUPLICATE_NAME_NOISE_CHARS).split())


def _detect_near_duplicate_ledger_names(tb_rows: List[TBRow]) -> List[TallyAdapterFinding]:
    """Classic Tally defect (client audit doc, section 5.2): 'ABC Traders',
    'A.B.C. Traders', and 'ABC Traders ' with a trailing space are three
    separate ledgers to Tally, each with its own balance, and nothing warns
    anyone. Normalizes names (case, whitespace, common punctuation) and
    flags any group of >1 distinct RAW name sharing a normalized key."""
    names_by_normalized: Dict[str, List[str]] = {}
    for row in tb_rows:
        if not row.gl_name or is_total_row_label(row.gl_name):
            continue
        normalized = _normalize_ledger_name(row.gl_name)
        if not normalized:
            continue
        names_by_normalized.setdefault(normalized, [])
        if row.gl_name not in names_by_normalized[normalized]:
            names_by_normalized[normalized].append(row.gl_name)

    findings = []
    for normalized, raw_names in names_by_normalized.items():
        if len(raw_names) > 1:
            findings.append(TallyAdapterFinding(
                severity="warning",
                code="tally_near_duplicate_ledger_name",
                message=(
                    f"{len(raw_names)} distinct ledger names normalize to the same identity "
                    f"({', '.join(sorted(raw_names))!s}) -- likely the same account split across "
                    "near-duplicate names (trailing spaces, punctuation variants), concealing the true position."
                ),
            ))
    return findings


def _detect_dr_cr_suffix_loss(grid: list, header_row: int, cols: dict) -> Optional[TallyAdapterFinding]:
    """Tally shows a balance's side as a Dr/Cr text suffix rather than a
    minus sign, which travels badly through export -- may arrive as text
    inside the same cell, or be dropped entirely. Re-scans the closing
    column with parse_amount_with_status to detect any cell that carries
    an ambiguous/unparseable Dr/Cr-like value, flagging that sign info may
    have been lost rather than silently treating it as zero."""
    closing_col = cols.get("closing")
    if closing_col is None:
        return None
    ambiguous_count = 0
    for row in grid[header_row + 1:]:
        if closing_col >= len(row):
            continue
        _amount, parsed_cleanly = parse_amount_with_status(row[closing_col])
        if not parsed_cleanly:
            ambiguous_count += 1
    if ambiguous_count == 0:
        return None
    return TallyAdapterFinding(
        severity="warning",
        code="tally_dr_cr_suffix_loss",
        message=(
            f"{ambiguous_count} row(s) in the closing balance column could not be parsed cleanly -- "
            "sign (Dr/Cr) information may have been lost in this export. Do not treat these as zero; "
            "confirm the true balances against the source before relying on this document."
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
    source_row_indices: Optional[List[int]] = None,
) -> List[ERPSourceMetadata]:
    result = []
    for i, row in enumerate(tb_rows):
        evidence = classification_by_gl_code.get(row.gl_code)
        result.append(ERPSourceMetadata(
            source_file_id=path.name,
            source_sheet=sheet_name,
            source_row_number=(source_row_indices[i] + 1) if source_row_indices else header_row + 2 + i,
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
    source_row_indices: Optional[List[int]] = None,
) -> TallyAdapterResult:
    grouping_hints = grouping_hints or {}
    rows_by_gl_code = {}
    for row in grid[header_row + 1:]:
        gl_code_val = _row_field(row, cols, "gl_code")
        gl_name_val = _row_field(row, cols, "gl_name")
        key = gl_code_val or gl_name_val
        if key:
            rows_by_gl_code[key] = row

    # Second hierarchy strategy: sum-reconciliation structural extraction,
    # for the majority native-export shape with no Head/Sub-Head columns
    # at all. Best-effort/never-raises already, at the module level; only
    # supplies main_head, never sub_head_1/sub_head_2 (see
    # _extract_group_fields) -- erp_group_path is carried separately as
    # audit trail only.
    structural_fields_by_identity = extract_tally_structural_fields_by_identity(
        path, sheet_name, header_row, cols,
    )

    classification_by_gl_code = {
        tb_row.gl_code: _build_classification_evidence(
            tb_row.gl_name, rows_by_gl_code.get(tb_row.gl_code), cols, grouping_hints.get(tb_row.gl_code),
            structural_main_head=structural_fields_by_identity.get(tb_row.gl_code, {}).get("main_head"),
            structural_erp_group_path=structural_fields_by_identity.get(tb_row.gl_code, {}).get("erp_group_path"),
            closing_balance=tb_row.closing,
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
    opening_balance_difference_finding = _detect_opening_balance_difference_row(tb_rows)
    if opening_balance_difference_finding:
        findings.append(opening_balance_difference_finding)
    findings.extend(_detect_near_duplicate_ledger_names(tb_rows))
    dr_cr_suffix_finding = _detect_dr_cr_suffix_loss(grid, header_row, cols)
    if dr_cr_suffix_finding:
        findings.append(dr_cr_suffix_finding)
    findings.append(PERIOD_LOCKING_NOTE)

    return TallyAdapterResult(
        tb_rows=tb_rows,
        classification_by_gl_code=classification_by_gl_code,
        scope=_build_scope_metadata(metadata, grouped_view_finding),
        source_metadata=_build_source_metadata(
            path, sheet_name, header_row, tb_rows, classification_by_gl_code, source_row_indices,
        ),
        findings=findings,
    )

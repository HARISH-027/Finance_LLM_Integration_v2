"""Oracle ERP adapter (TB Multi-ERP Adaptability upgrade, Phase 4 -- see
modes/trial_balance/documentation/
TB_Multi_ERP_Adaptability_Implementation_Upgrade_Design.md, sections 5.2/22,
scope corrected against real fixtures -- see section 35's revised Item 1).

Purely additive: not wired into input_dispatch.py or any existing parser
yet. Same entry-point shape as tally_adapter.py (Phase 3) / sap_adapter.py
(Phase 5): operates on already-loaded (grid, header_row, cols, tb_rows).

Real-fixture correction: the design doc's original Phase 4 scope (multi-
segment account-combination splitting, e.g. "01-520-4310-000-000") does not
match any of the 8 real Oracle client exports received (Bhavini,
Bridge_and_Roof, CONCOR, Grid_Controller_of_India, HUDCO, IRCTC, NHPC,
Railtel) -- every GL code across all 8 is a plain numeric value or a single
lightly-segmented alphanumeric code (GCIL's "10901-1"), never a multi-
segment combination. That original scope is deferred until a genuinely
multi-segment raw export is received (not built here).

What real classified clients (Bhavini, Bridge_and_Roof, CONCOR, GCIL,
Railtel) actually have: a separate Grouping file, flat and GL-code-keyed,
with bs_pl/sub_head_2/sub_head_1/main_head/account_type columns -- already
recognized by input_header_detect.FIELD_ALIASES and already fully extracted
into GroupingHint.known_fields by the existing, unmodified
input_grouping_source.extract_structured_fields() via Scenario B. Unlike
Tally/SAP, Oracle's own TB sheet never carries this data row-embedded in
any real file seen -- so grouping_hints is the PRIMARY source here, not a
fallback (row-embedded columns are still checked first for symmetry/
future-proofing, but no real file exercises that branch today).

HUDCO and NHPC are TB-only exports (no Grouping file, no classification
columns anywhere). IRCTC's Grouping file uses merged/duplicated header
labels ("Company Code" appears twice; the GL-code+description column is a
single combined "GL Code, Text and Headings" cell) that locate_header's
alias matching does not recognize as a gl_code column at all -- confirmed
against the real file: extract_structured_fields/extract_hint_text both
return {} for it. All three shapes are handled identically by this module:
the adapter still runs and produces empty classification_by_gl_code, and
raises oracle_no_classification_data_available rather than silently
proceeding as if no groups were needed. IRCTC's underlying data does exist
in the file (unlike HUDCO/NHPC) -- fixing locate_header to recognize its
merged-header shape is a separate, real, not-yet-built follow-up (see
design doc section 35)."""

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

_CLASSIFICATION_FIELDS = ("bs_pl", "account_type", "main_head", "sub_head_1", "sub_head_2")


@dataclass(frozen=True)
class OracleAdapterFinding:
    """Same narrower, adapter-local vocabulary as TallyAdapterFinding/
    SAPAdapterFinding -- not the full section-17 Validation Severity Model,
    which is Phase 6's own scope-aware quality gate."""

    severity: FindingSeverity
    code: str
    message: str


@dataclass(frozen=True)
class OracleAdapterResult:
    tb_rows: List[TBRow]
    classification_by_gl_code: Dict[str, ClassificationEvidence]
    scope: ScopeMetadata
    source_metadata: List[ERPSourceMetadata]
    findings: List[OracleAdapterFinding]


def _row_field(row: Optional[list], cols: dict, field_name: str) -> Optional[str]:
    if row is None:
        return None
    idx = cols.get(field_name)
    if idx is None or idx >= len(row) or row[idx] in (None, ""):
        return None
    return str(row[idx]).strip()


def _build_classification_evidence(
    row: Optional[list], cols: dict, tb_row: TBRow, hint: Optional[GroupingHint],
) -> ClassificationEvidence:
    account_classification = {}
    for field_name in _CLASSIFICATION_FIELDS:
        value = _row_field(row, cols, field_name) or (hint.known_fields.get(field_name) if hint else None)
        if value:
            account_classification[field_name] = value
    return ClassificationEvidence(
        erp_type=ERPType.ORACLE,
        account_identity={"gl_code": tb_row.gl_code, "gl_name": tb_row.gl_name},
        account_classification=account_classification,
    )


def _detect_no_classification_data(
    classification_by_gl_code: Dict[str, ClassificationEvidence],
) -> Optional[OracleAdapterFinding]:
    """HUDCO/NHPC/IRCTC shape: no Grouping file, no embedded columns, or a
    Grouping file whose header shape isn't recognized -- every row's
    account_classification ends up empty. Flag explicitly rather than
    silently proceeding as if no groups were needed."""
    if not classification_by_gl_code:
        return None
    if any(evidence.account_classification for evidence in classification_by_gl_code.values()):
        return None
    return OracleAdapterFinding(
        severity="audit_risk",
        code="oracle_no_classification_data_available",
        message=(
            "No BS/PL, Main Head, Sub Head, or Account Type data found for any account in this "
            "document -- either no separate Grouping file was supplied, the Trial Balance sheet "
            "itself carries no classification columns, or the Grouping file's header shape wasn't "
            "recognized. Confirm classification data exists for this export before relying on "
            "automated taxonomy resolution."
        ),
    )


def _build_scope_metadata(metadata: Optional[DocumentMetadata]) -> ScopeMetadata:
    return ScopeMetadata(
        entity=metadata.entity_name if metadata else None,
        fiscal_year=metadata.financial_year if metadata else None,
        scope_type=ScopeType.FULL_LEDGER,
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
            source_raw_group=evidence.account_classification.get("main_head") if evidence else None,
            erp_type=ERPType.ORACLE,
        ))
    return result


def run_oracle_adapter(
    path: Path, grid: list, header_row: int, cols: dict, tb_rows: List[TBRow],
    *, grouping_hints: Optional[Dict[str, GroupingHint]] = None,
    metadata: Optional[DocumentMetadata] = None, sheet_name: str = "",
) -> OracleAdapterResult:
    grouping_hints = grouping_hints or {}
    rows_by_gl_code = {}
    for row in grid[header_row + 1:]:
        gl_code_val = _row_field(row, cols, "gl_code")
        if gl_code_val:
            rows_by_gl_code[gl_code_val] = row

    classification_by_gl_code = {
        tb_row.gl_code: _build_classification_evidence(
            rows_by_gl_code.get(tb_row.gl_code), cols, tb_row, grouping_hints.get(tb_row.gl_code),
        )
        for tb_row in tb_rows
    }

    findings: List[OracleAdapterFinding] = []
    no_classification_finding = _detect_no_classification_data(classification_by_gl_code)
    if no_classification_finding:
        findings.append(no_classification_finding)

    return OracleAdapterResult(
        tb_rows=tb_rows,
        classification_by_gl_code=classification_by_gl_code,
        scope=_build_scope_metadata(metadata),
        source_metadata=_build_source_metadata(path, sheet_name, header_row, tb_rows, classification_by_gl_code),
        findings=findings,
    )

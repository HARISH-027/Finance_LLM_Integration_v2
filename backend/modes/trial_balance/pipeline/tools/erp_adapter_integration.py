"""Wires the Tally/SAP/Oracle ERP adapters (Phases 1-10 of the TB Multi-ERP
Adaptability upgrade) into the live ingestion pipeline.

ParsedInput (input_dispatch.py's output) does not carry the raw workbook
grid/header_row/cols past parsing -- those are internal locals inside each
scenario parser. Rather than extending ParsedInput and touching
input_scenario_a/b/c/d.py, this module re-derives them for the TB-bearing
file by reloading it once more with the exact same helpers those parsers
already use internally (_load_workbook_sheets, pick_tb_sheet,
classify_workbooks). input_dispatch.py and the scenario parsers are never
called into or modified.

Automatic ERP detection resolves to UNKNOWN for almost any real uploaded
file (its temp path won't literally contain "SAP"/"Tally" as a segment --
confirmed in Phase 2), so wiring this in is a near no-op for existing
production traffic unless a caller explicitly passes an erp_type. That is
the actual safety mechanism for this integration, not a feature flag.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from modes.trial_balance.pipeline.tools.classification_evidence_layer import classification_evidence_map_to_grouping_hints
from modes.trial_balance.pipeline.tools.erp_detector import detect_erp_type
from modes.trial_balance.pipeline.tools.input_header_detect import pick_tb_sheet
from modes.trial_balance.pipeline.tools.input_scenario_b import _load_workbook_sheets, classify_workbooks
from modes.trial_balance.pipeline.tools.oracle_adapter import run_oracle_adapter
from modes.trial_balance.pipeline.tools.parsed_input import ParsedInput
from modes.trial_balance.pipeline.tools.sap_adapter import run_sap_adapter
from modes.trial_balance.pipeline.tools.scope_aware_quality_gate import run_scope_aware_quality_gate
from modes.trial_balance.pipeline.tools.tally_adapter import run_tally_adapter
from modes.trial_balance.pipeline.tools.tally_row_normalizer import normalize_tally_rows
from modes.trial_balance.pipeline.tools.tb_erp_contracts import ClassificationEvidence, ERPType, ScopeMetadata
from modes.trial_balance.pipeline.tools.tb_models import GroupingHint, TBRow

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ERPEnrichmentResult:
    erp_type: ERPType
    scope: ScopeMetadata
    classification_by_gl_code: Dict[str, ClassificationEvidence]
    enriched_grouping_hints: Dict[str, GroupingHint]
    findings: List[dict] = field(default_factory=list)  # [{severity, code, message}]
    # Tally only: tb_rows re-signed (Dr +/Cr -) with Group/Head container
    # rows separated out; None = leave parsed.tb_rows untouched.
    normalized_tb_rows: Optional[List[TBRow]] = None
    # Tally only: audit-trail records of the removed Group/Head rows.
    container_rows: List[dict] = field(default_factory=list)


def enrich_parsed_input_with_erp_adapter(
    input_files: List[Path], parsed: ParsedInput, *, explicit_erp_type: Optional[ERPType] = None,
) -> Optional[ERPEnrichmentResult]:
    """Best-effort, never raises: returns None on ANY failure, or when the
    file isn't Tally/SAP/Oracle-shaped (GENERIC/UNKNOWN skipped -- current
    behavior is unchanged in every one of those cases)."""
    try:
        if len(input_files) == 1:
            tb_path = input_files[0]
        elif len(input_files) == 2:
            tb_path, _grouping_path = classify_workbooks(input_files)
        else:
            return None

        sheets = _load_workbook_sheets(tb_path, max_row=None)
        pick = pick_tb_sheet(sheets)
        if pick is None:
            return None
        sheet_name, header_row, cols = pick
        grid = sheets[sheet_name]

        detection = detect_erp_type(tb_path, sheets, explicit_erp_type=explicit_erp_type)

        working_tb_rows = parsed.tb_rows
        normalized = None
        if detection.erp_type == ERPType.TALLY:
            # Re-sign from Dr/Cr number formats and separate Group/Head
            # container rows from Ledger rows BEFORE the adapter and
            # classification see them (fail-safe: unchanged on any doubt).
            normalized = normalize_tally_rows(tb_path, sheet_name, header_row, cols, parsed.tb_rows)
            if normalized.applied:
                working_tb_rows = normalized.ledger_rows
            adapter_result = run_tally_adapter(
                tb_path, grid, header_row, cols, working_tb_rows,
                grouping_hints=parsed.grouping_hints, metadata=parsed.metadata, sheet_name=sheet_name,
                source_row_indices=normalized.ledger_grid_indices if normalized.applied else None,
            )
        elif detection.erp_type == ERPType.SAP:
            adapter_result = run_sap_adapter(
                tb_path, grid, header_row, cols, parsed.tb_rows,
                grouping_hints=parsed.grouping_hints, metadata=parsed.metadata, sheet_name=sheet_name,
            )
        elif detection.erp_type == ERPType.ORACLE:
            adapter_result = run_oracle_adapter(
                tb_path, grid, header_row, cols, parsed.tb_rows,
                grouping_hints=parsed.grouping_hints, metadata=parsed.metadata, sheet_name=sheet_name,
            )
        else:
            return None

        enriched_hints = classification_evidence_map_to_grouping_hints(adapter_result.classification_by_gl_code)
        quality_report = run_scope_aware_quality_gate(
            working_tb_rows, adapter_result.scope, adapter_result.classification_by_gl_code,
        )
        findings = [
            {"severity": f.severity, "code": f.code, "message": f.message}
            for f in (list(adapter_result.findings) + list(quality_report.findings))
        ]

        return ERPEnrichmentResult(
            erp_type=detection.erp_type, scope=adapter_result.scope,
            classification_by_gl_code=adapter_result.classification_by_gl_code,
            enriched_grouping_hints=enriched_hints, findings=findings,
            normalized_tb_rows=working_tb_rows if (normalized is not None and normalized.applied) else None,
            container_rows=normalized.container_rows if normalized is not None else [],
        )
    except Exception:
        logger.warning("ERP adapter enrichment failed; proceeding without it", exc_info=True)
        return None

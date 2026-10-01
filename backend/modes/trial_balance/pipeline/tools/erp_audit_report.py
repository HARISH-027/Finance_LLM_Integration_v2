"""ERP-aware audit report data layer (TB Multi-ERP Adaptability upgrade,
Phase 10 -- see modes/trial_balance/documentation/
TB_Multi_ERP_Adaptability_Implementation_Upgrade_Design.md, sections
21/22/26).

Purely additive: not wired into reports.py (build_docx_report/
build_excel_report/build_report_markdown), input_dispatch.py, or any
live-wired file. reports.py operates on a polars.DataFrame of already-
canonicalized TB rows plus JSON sidecar files -- it predates ScopeMetadata/
ClassificationEvidence/ERPType entirely, and wiring this module's output
into that 2267-line production file is a real, separate follow-up.

Showing a TaxonomyResolution here is not the same decision as persisting
one (Phase 9's user-confirmed "skip tb_taxonomy_resolution" was specifically
about durable DB storage, per tb_models.py's own docstring). Displaying
resolution_method/resolution_status transiently in an in-memory report row
does not touch that decision.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from modes.trial_balance.pipeline.tools.classification_evidence_layer import extract_taxonomy_fields
from modes.trial_balance.pipeline.tools.tb_erp_contracts import ClassificationEvidence, ERPType, ScopeMetadata, ScopeType
from modes.trial_balance.pipeline.tools.tb_models import TaxonomyResolution, TBRow


@dataclass(frozen=True)
class ERPAuditRow:
    """One account's full audit-traceability view (section 21/26) -- source
    ERP identity, the raw ERP-specific evidence, the canonical taxonomy
    fields extracted from it (Phase 7), and, if supplied, transient
    resolution evidence (never persisted -- see Phase 9's own note)."""

    erp_type: ERPType
    gl_code: str
    gl_name: str
    entity: Optional[str]
    ledger: Optional[str]
    scope_type: ScopeType
    erp_classification: Dict[str, str]
    canonical_classification: Dict[str, str]
    resolution_method: Optional[str] = None
    resolution_status: Optional[str] = None


def _merge_erp_specific_fields(evidence: ClassificationEvidence) -> Dict[str, str]:
    """The raw ERP view -- everything account_classification/business_dimensions/
    financial_statement carry, ERP-specific keys included (erp_group,
    account_group, business_area, ...). Distinct from canonical_classification
    (Phase 7's extract_taxonomy_fields), which is the ERP-neutral subset."""
    merged = dict(evidence.account_classification)
    merged.update(evidence.business_dimensions)
    merged.update(evidence.financial_statement)
    return merged


def build_erp_audit_rows(
    tb_rows: List[TBRow], scope: ScopeMetadata, classification_by_gl_code: Dict[str, ClassificationEvidence],
    *, resolutions_by_gl_code: Optional[Dict[str, TaxonomyResolution]] = None,
) -> List[ERPAuditRow]:
    resolutions_by_gl_code = resolutions_by_gl_code or {}
    rows = []
    for tb_row in tb_rows:
        evidence = classification_by_gl_code.get(tb_row.gl_code)
        resolution = resolutions_by_gl_code.get(tb_row.gl_code)
        rows.append(ERPAuditRow(
            erp_type=evidence.erp_type if evidence else ERPType.UNKNOWN,
            gl_code=tb_row.gl_code, gl_name=tb_row.gl_name,
            entity=scope.entity, ledger=scope.ledger, scope_type=scope.scope_type,
            erp_classification=_merge_erp_specific_fields(evidence) if evidence else {},
            canonical_classification=extract_taxonomy_fields(evidence) if evidence else {},
            resolution_method=resolution.method if resolution else None,
            resolution_status=resolution.status if resolution else None,
        ))
    return rows


def render_erp_audit_row_markdown(row: ERPAuditRow) -> str:
    """Matches section 26's own worked-example rendering shape."""
    lines = [
        f"**ERP:** {row.erp_type.value.upper()}",
        f"**G/L:** {row.gl_code} ({row.gl_name})",
    ]
    if row.entity:
        lines.append(f"**Entity:** {row.entity}")
    if row.ledger:
        lines.append(f"**Ledger:** {row.ledger}")
    lines.append(f"**Scope:** {row.scope_type.value}")
    if row.erp_classification:
        lines.append(f"**ERP classification:** {' / '.join(row.erp_classification.values())}")
    if row.canonical_classification:
        parts = [
            row.canonical_classification.get(k)
            for k in ("main_head", "sub_head_1", "sub_head_2")
            if row.canonical_classification.get(k)
        ]
        if parts:
            lines.append(f"**Canonical taxonomy:** {' > '.join(parts)}")
    if row.resolution_method:
        lines.append(f"**Resolution:** {row.resolution_method}")
    return "\n".join(lines)


def render_erp_audit_section_markdown(rows: List[ERPAuditRow]) -> str:
    return "\n\n---\n\n".join(render_erp_audit_row_markdown(row) for row in rows)

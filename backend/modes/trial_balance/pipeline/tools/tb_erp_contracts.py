"""ERP-agnostic data contracts (TB Multi-ERP Adaptability upgrade, Phase 1
-- see modes/trial_balance/documentation/
TB_Multi_ERP_Adaptability_Implementation_Upgrade_Design.md, section 22).

Purely additive: nothing in this module is wired into any existing parser,
reasoner, or orchestrator yet. These are the target shapes future ERP
adapters (Tally/Oracle/SAP, later phases) will produce and downstream
validation/taxonomy/persistence will consume, once built.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Optional


class ERPType(str, Enum):
    """Which ERP produced this document. GENERIC covers the "In_house_or_other"
    real-file category (design doc sections 1.6.3/34) -- structurally similar
    to SAP/Oracle's standardized shape, deliberately NOT given its own
    adapter, but still a real, encountered source that needs a name."""

    TALLY = "tally"
    ORACLE = "oracle"
    SAP = "sap"
    GENERIC = "generic"
    UNKNOWN = "unknown"


class ScenarioType(str, Enum):
    """File-structure scenario (design doc sections 6.2/19) -- orthogonal to
    ERPType. Scenario and ERP vary independently (e.g. Scenario B + SAP,
    Scenario B + Tally are both valid); never conflate the two."""

    A = "A"  # normalized template
    B = "B"  # TB + Grouping files
    C = "C"  # combined workbook
    D = "D"  # multi-year / multiple documents


class ScopeType(str, Enum):
    """What slice of the ledger a TB document represents (design doc
    sections 7.5/18) -- a non-zero grand total or an unbalanced document is
    only a real finding once scope is known; a PROFIT_CENTER_SLICE is
    expected to look "incomplete" as a full ledger."""

    FULL_LEDGER = "full_ledger"
    ENTITY_SLICE = "entity_slice"
    ACCOUNT_SLICE = "account_slice"
    PROFIT_CENTER_SLICE = "profit_center_slice"
    SEGMENT_SLICE = "segment_slice"
    MULTI_ENTITY = "multi_entity"
    CONSOLIDATED = "consolidated"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class ScopeMetadata:
    """First-class scope object (design doc section 18) -- distinct from
    DocumentMetadata (document/entity identity + framework). Covers what
    slice of the ledger this document represents: which entity/company/
    ledger, what period, and how much of the chart of accounts. All fields
    optional since scope is inferred with varying confidence per ERP/export."""

    entity: Optional[str] = None
    company_code: Optional[str] = None
    ledger: Optional[str] = None
    fiscal_year: Optional[str] = None
    period_from: Optional[int] = None
    period_to: Optional[int] = None
    currency: Optional[str] = None
    account_range: Optional[str] = None
    business_scope: Optional[str] = None
    profit_center: Optional[str] = None
    segment: Optional[str] = None
    scope_type: ScopeType = ScopeType.UNKNOWN


@dataclass(frozen=True)
class ClassificationEvidence:
    """Generic classification structure (design doc section 13) every ERP
    adapter produces and the taxonomy engine consumes, without the engine
    ever inspecting raw Tally group columns / Oracle segments / SAP FS
    Version columns directly. Sub-groups are plain str->str dicts, not rigid
    nested dataclasses, because the actual keys present vary by ERP (SAP:
    account_group/account_type/bs_pl; Tally: erp_group/erp_group_path;
    Oracle: natural_account/report_definition) -- see section 13's JSON
    examples."""

    erp_type: ERPType
    account_identity: Dict[str, str] = field(default_factory=dict)
    account_classification: Dict[str, str] = field(default_factory=dict)
    financial_statement: Dict[str, str] = field(default_factory=dict)
    business_dimensions: Dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class ERPSourceMetadata:
    """Per-row audit traceability (design doc section 21) -- lets an
    auditor's "why was this account mapped this way" question be answered
    back to the exact source file/sheet/row, independent of TBRow itself
    (which stays engine-facing, not audit-facing)."""

    source_file_id: Optional[str] = None
    source_sheet: Optional[str] = None
    source_row_number: Optional[int] = None
    source_account_identifier: Optional[str] = None
    source_raw_account_name: Optional[str] = None
    source_raw_group: Optional[str] = None
    erp_type: ERPType = ERPType.UNKNOWN


@dataclass(frozen=True)
class CanonicalTBRow:
    """Target row contract (design doc section 12 "TBLine") that future ERP
    adapters normalize into -- a superset of today's engine-facing TBRow
    (tb_models.py), adding source-neutral identity fields (source_account_id/
    code/name) alongside the existing gl_code/gl_name so ERP-specific
    identity is never silently discarded during canonicalization (section
    7.2). Not wired into any existing parser/reasoner in this phase -- TBRow
    remains the shape every current call site uses; adapters built in later
    phases will be the first producers of this type."""

    gl_code: str
    gl_name: str
    opening: float
    debit: float
    credit: float
    closing: float
    source_account_id: Optional[str] = None
    source_account_code: Optional[str] = None
    source_account_name: Optional[str] = None
    account_type: Optional[str] = None
    bs_pl: Optional[str] = None

"""Classification Evidence Layer (TB Multi-ERP Adaptability upgrade,
Phases 7-8 -- see modes/trial_balance/documentation/
TB_Multi_ERP_Adaptability_Implementation_Upgrade_Design.md, sections
13/14/22).

Phase 7 (extract_taxonomy_fields): the unifying accessor a taxonomy-facing
consumer reads from, regardless of which ERP adapter produced the evidence.

Phase 8 (classification_evidence_to_grouping_hint /
classification_evidence_map_to_grouping_hints): adapts ClassificationEvidence
to taxonomy_resolver.py's EXISTING input contract (GroupingHint.known_fields/
hint_text) -- per the design doc's own section 1.5.2 correction, Phase 8 is
NOT "build a new resolution ladder" (EXACT -> ALIAS -> CANDIDATE_AUTO ->
DETERMINISTIC_RULE already exists and is unchanged), it is this conversion.

Purely additive: not wired into taxonomy_resolver.py, input_grouping_source.py,
input_dispatch.py, or any existing parser/adapter/live pipeline path. Both
tally_adapter.py's and sap_adapter.py's classification_by_gl_code output can
be passed to classification_evidence_map_to_grouping_hints and the result
handed straight to taxonomy_resolver.resolve_grouping_hints -- neither
function needs to change.
"""

from __future__ import annotations

from typing import Dict

from modes.trial_balance.pipeline.tools.tb_erp_contracts import ClassificationEvidence
from modes.trial_balance.pipeline.tools.tb_models import GroupingHint

# The taxonomy engine's existing input vocabulary (matches
# GroupingHint.known_fields / input_grouping_source.py's STRUCTURED_FIELDS)
# -- the canonical fields every ERP adapter is expected to populate into
# account_classification whenever its underlying data supports them.
_TAXONOMY_FIELDS = ("main_head", "sub_head_1", "sub_head_2", "bs_pl", "account_type")


def extract_taxonomy_fields(evidence: ClassificationEvidence) -> Dict[str, str]:
    """The one place a taxonomy-facing consumer should read classification
    from, regardless of which ERP produced the evidence (section 13) --
    never inspect account_classification's ERP-specific keys (erp_group,
    erp_group_path, account_group, ...) directly. Returns only whichever of
    the canonical taxonomy fields are actually present; an adapter that
    hasn't populated a field (e.g. no Sub Head 2 column in the source) is
    silently absent here, never fabricated."""
    return {
        field_name: evidence.account_classification[field_name]
        for field_name in _TAXONOMY_FIELDS
        if field_name in evidence.account_classification
    }


def _build_hint_text(evidence: ClassificationEvidence) -> str:
    """Free-text candidate evidence for _try_alias/_try_candidate_auto --
    deliberately broader than known_fields: includes ERP-specific
    enrichment (erp_group_path, business_area, account_group, ...) since
    hint_text has always been "a candidate to test, never trusted output"
    (taxonomy_resolver.py's own module docstring)."""
    parts = list(evidence.account_classification.values())
    parts.extend(evidence.business_dimensions.values())
    parts.extend(evidence.financial_statement.values())
    return " ".join(str(p) for p in parts if p)


def classification_evidence_to_grouping_hint(gl_code: str, evidence: ClassificationEvidence) -> GroupingHint:
    """The actual "connect the evidence layer to the existing resolver"
    step (section 22 Phase 8) -- builds the exact input shape
    resolve_grouping_hints/resolve_grouping_evidence already expect, so
    neither function needs to change to accept adapter output."""
    return GroupingHint(
        gl_code=gl_code,
        known_fields=extract_taxonomy_fields(evidence),
        hint_text=_build_hint_text(evidence),
    )


def classification_evidence_map_to_grouping_hints(
    classification_by_gl_code: Dict[str, ClassificationEvidence],
) -> Dict[str, GroupingHint]:
    """Batch form, matching the shape taxonomy_resolver.resolve_grouping_hints
    expects as its `grouping_hints` argument -- both tally_adapter.py's and
    sap_adapter.py's `classification_by_gl_code` output can be passed here
    directly."""
    return {
        gl_code: classification_evidence_to_grouping_hint(gl_code, evidence)
        for gl_code, evidence in classification_by_gl_code.items()
    }

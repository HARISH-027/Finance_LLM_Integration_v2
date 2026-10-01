"""The single entry point every caller of the native classification engine
should use: quality gate -> data-quality tier -> confirmation gate ->
Case-driven routing (verbatim zero-touch for Case 1 / Case 2's complete
rows, full resolver+LLM pipeline for everything else) -> metrics + sanity
checks recomputed over the final row set.

Ported from TB_normalization_v1's modes.trial_balance.pipeline.py::_ingest_parsed, minus the DB
write (this module only classifies; callers decide what to do with the
result -- write to LIVE, write to a canonical_tb.parquet, etc).

This exists so ingest_tb_to_live -- the single entry point for both live
template submissions and uploaded documents -- has ONE implementation of
the tiering/confirmation/routing decision, rather than the two
independently-maintained copies (one per input path) that used to exist
before this module was introduced, which let the confirmation gate exist
on one path and not the other.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from modes.trial_balance.pipeline.tools.normalize import SanityFlags, normalize
from modes.trial_balance.pipeline.tools.normalize_metrics import RunMetrics, compute_metrics
from modes.trial_balance.pipeline.tools.quality_gate import (
    ConfirmationRequirement,
    QualityGateReport,
    QualityTierAssessment,
    check_confirmation_requirement,
    classify_quality_tier,
    hint_is_complete,
    run_input_quality_gate,
)
from modes.trial_balance.pipeline.tools.resolution_metadata import attach_resolution_metadata
from modes.trial_balance.pipeline.tools.sanity_checks import find_account_type_keyword_conflicts, find_inconsistent_same_name_groups
from modes.trial_balance.pipeline.tools.tally_main_head_classifier import classify_tally_main_head
from modes.trial_balance.pipeline.tools.taxonomy_repository import derive_account_type
from modes.trial_balance.pipeline.tools.taxonomy_resolver import resolve_grouping_hints
from modes.trial_balance.pipeline.tools.verbatim import build_verbatim_rows


class QualityGateFailedError(Exception):
    """input_quality_gate reported FAIL (currently only: zero TB rows).
    WARN-level issues never raise -- they're carried on the result for the
    caller to inspect."""


class DataQualityConfirmationRequiredError(Exception):
    """A CASE_2 document over the incomplete-row threshold, or any CASE_3
    document, needs accept_data_quality_risk=True before the resolver/
    normalize/LLM pipeline may run at all. Raised before any DB/embedding/
    LLM cost is spent."""

    def __init__(self, tier: str, requirement: ConfirmationRequirement):
        self.tier = tier
        self.requirement = requirement
        super().__init__(
            f"Classification blocked pending confirmation ({tier}): {requirement.reason} "
            "Pass accept_data_quality_risk=True to proceed, or improve the source data."
        )


@dataclass(frozen=True)
class ClassifyDocumentResult:
    rows: list
    quality_gate_report: QualityGateReport
    quality_tier: QualityTierAssessment
    metrics: RunMetrics
    sanity_flags: SanityFlags


def classify_tb_document(
    tb_rows: list,
    grouping_hints: dict,
    grouping_shape,
    standard: str,
    llm_client,
    *,
    accept_data_quality_risk: bool = False,
    use_candidates: bool = True,
    use_candidate_id_contract: bool = False,
    use_adaptive_candidates: bool = True,
    use_hierarchical_candidate_context: bool = True,
    classification_depth: str = "full",
) -> ClassifyDocumentResult:
    """Raises QualityGateFailedError / DataQualityConfirmationRequiredError
    rather than returning a sentinel -- callers translate those into
    whatever response shape their own tool contract needs (see
    ingest_tb_to_live for the pattern)."""
    quality_gate_report = run_input_quality_gate(tb_rows, grouping_hints)
    if quality_gate_report.status == "FAIL":
        raise QualityGateFailedError(f"Input quality gate FAILED: {quality_gate_report.warnings}")

    # Data-quality tier (Case 1/2/3): computed BEFORE resolution and used to
    # actually ROUTE processing, not just narrate it afterward. Case 1's
    # whole document, and Case 2's complete-hint rows, skip the resolver/
    # LLM entirely -- see verbatim.py.
    quality_tier = classify_quality_tier(tb_rows, grouping_hints, grouping_shape)

    # Confirmation gate: a CASE_2 document over the incomplete-row
    # threshold, or any CASE_3 document, must not spend resolver/LLM/DB
    # cost until a human explicitly accepts responsibility for the input's
    # data quality.
    confirmation_requirement = check_confirmation_requirement(quality_tier, tb_rows, grouping_hints)
    if confirmation_requirement is not None and not accept_data_quality_risk:
        raise DataQualityConfirmationRequiredError(quality_tier.tier, confirmation_requirement)

    if classification_depth == "main_head":
        # Tally: source has a group hierarchy but no sub_head_1/sub_head_2,
        # so every row settles at main_head depth (see
        # tally_main_head_classifier.py). Tiering and the confirmation gate
        # above still apply unchanged.
        rows = classify_tally_main_head(tb_rows, grouping_hints, llm_client)
        resolutions = {}
    elif quality_tier.tier == "CASE_1":
        # Whole document already has complete grouping data -- zero
        # resolver/DB/embedding/LLM touch for any row.
        rows = build_verbatim_rows(tb_rows, grouping_hints)
        resolutions: dict = {}
    elif quality_tier.tier == "CASE_2":
        # Only rows with an incomplete/missing hint need the resolver +
        # LLM layer -- rows with a complete hint get the same zero-touch
        # treatment as Case 1.
        complete_rows = [r for r in tb_rows if hint_is_complete(grouping_hints.get(r.gl_code))]
        needs_pipeline_rows = [r for r in tb_rows if not hint_is_complete(grouping_hints.get(r.gl_code))]
        verbatim_rows = build_verbatim_rows(complete_rows, grouping_hints)
        enriched_hints, resolutions = resolve_grouping_hints(needs_pipeline_rows, grouping_hints, standard)
        pipeline_result = normalize(
            needs_pipeline_rows, enriched_hints, standard, llm_client, use_candidates=use_candidates,
            use_candidate_id_contract=use_candidate_id_contract, use_adaptive_candidates=use_adaptive_candidates,
            use_hierarchical_candidate_context=use_hierarchical_candidate_context,
        )
        order_index = {row.gl_code: i for i, row in enumerate(tb_rows)}
        rows = sorted(verbatim_rows + pipeline_result.rows, key=lambda r: order_index[r.gl_code])
    else:  # CASE_3 -- no safely-skippable subset, full pipeline over every row
        enriched_hints, resolutions = resolve_grouping_hints(tb_rows, grouping_hints, standard)
        pipeline_result = normalize(
            tb_rows, enriched_hints, standard, llm_client, use_candidates=use_candidates,
            use_candidate_id_contract=use_candidate_id_contract, use_adaptive_candidates=use_adaptive_candidates,
            use_hierarchical_candidate_context=use_hierarchical_candidate_context,
        )
        rows = pipeline_result.rows

    # Threads the resolver's own per-GL audit data (method/confidence/evidence/
    # taxonomy_node_id) onto each row's final trace -- see resolution_metadata.py's
    # own docstring for the precedence rule (resolver wins only when it actually
    # RESOLVED the row; resolved_at_step is authoritative otherwise).
    rows = attach_resolution_metadata(rows, resolutions)

    # build_verbatim_rows (Case 1, and Case 2's complete-hint subset) is a
    # true zero-touch path -- it never derives account_type, since
    # TB_normalization_v1 doesn't need it to for its own "trust complete
    # client data verbatim" philosophy. TB-v2-git's downstream analytics
    # (risk/fsli/reports/materiality) structurally require account_type on
    # every MAPPED row, so it's backfilled here from the now-final
    # bs_pl/main_head -- after the classification decision itself (which
    # stays untouched/zero-DB-cost for these rows), not as part of it.
    rows = [
        replace(row, account_type=derive_account_type(row.bs_pl, row.main_head, standard))
        if row.mapped_status == "MAPPED" and not row.account_type and row.bs_pl and row.main_head
        else row
        for row in rows
    ]

    # Recomputed from the final (possibly merged) row set rather than reused
    # from NormalizeResult, since Case 1/2 may never call normalize() at
    # all, or only over a subset.
    metrics = compute_metrics(rows)
    sanity_flags = SanityFlags(
        keyword_conflicts=find_account_type_keyword_conflicts(rows),
        inconsistent_name_groups=find_inconsistent_same_name_groups(rows),
    )

    return ClassifyDocumentResult(
        rows=rows, quality_gate_report=quality_gate_report, quality_tier=quality_tier,
        metrics=metrics, sanity_flags=sanity_flags,
    )

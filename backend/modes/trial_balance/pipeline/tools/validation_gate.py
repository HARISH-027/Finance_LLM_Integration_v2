"""The Validation Gate -- the safety-critical, net-new piece of this engine.

Applies to EVERY row that reaches provisional MAPPED status, from any
source (Step 2 direct resolution or Step 3/Step 4 LLM output) -- no
exceptions. This is the sole place in the whole engine that can produce
mapped_status="MAPPED": NormalizedRow is only ever constructed here (via
NormalizedRow.from_tb_row), never in normalize.py, classification.py, or
staged_narrowing.py directly. This is the direct fix for the known gap in
the old ingest_tb_to_live, where a template-path row could reach a live
table with an internal "matched but never validated" status because that
code path never called the shared taxonomy-snap+fallback machinery other
paths did -- here there is no second code path that can construct a
NormalizedRow at all.

Terminal failure here always produces UNMATCHED, never UNMAPPED -- every
row reaching this gate came from direct_rows or llm_provisional in
normalize.py, both of which require a grouping hint to have existed
(UNMAPPED is reserved for normalize.py's no_match_rows, which never reach
this gate at all).

Ported from TB_normalization_v1's core/validation_gate.py.
"""

from __future__ import annotations

from modes.trial_balance.pipeline.tools._shared import Standard
from modes.trial_balance.pipeline.tools.classification import classify_all
from modes.trial_balance.pipeline.tools.taxonomy_repository import derive_account_type, snap_to_taxonomy
from modes.trial_balance.pipeline.tools.tb_models import MAPPED, UNMATCHED, ClassifyCandidate, NormalizedRow, ProvisionalMappedRow, RowTrace


def _build_repair_hint(row: ProvisionalMappedRow) -> str:
    """The row's own now-invalidated field values, folded into hint text
    for the single repair attempt."""
    parts = [p for p in (row.main_head, row.sub_head_1, row.sub_head_2) if p]
    return " / ".join(dict.fromkeys(parts))


def run_validation_gate(
    provisional_rows: list, standard: Standard, llm_client,
    *, use_candidates: bool = True, use_candidate_id_contract: bool = False,
    use_adaptive_candidates: bool = True, use_hierarchical_candidate_context: bool = True,
    _already_repaired: bool = False,
) -> list:
    """Every row in `provisional_rows` is checked via snap_to_taxonomy here.

    Match -> mapped_status = MAPPED. For Step-2-direct rows with valid
    client-supplied taxonomy text (row.client_fields), the client's own
    wording is displayed in bs_pl/main_head/sub_head_1/sub_head_2 --
    wording_source="client_original" -- since that's what the client's own
    published financial statements use; the canonical taxonomy node text
    is still captured separately in canonical_bs_pl/canonical_main_head/
    canonical_sub_head_1/canonical_sub_head_2 and used unconditionally for
    account_type derivation. Every other row still gets all 4 display
    fields rewritten to the taxonomy's canonical strings (auto-repairs
    wording drift) -- wording_source="system_canonical".

    No match -> no third status is invented. Exactly one repair attempt is
    made: the row re-enters classification.classify_all (single-shot only,
    never staged narrowing again) using its own now-invalidated field
    values as hint text. The repair result re-enters THIS SAME function
    exactly once -- enforced by `_already_repaired`, which this function
    refuses to let go around twice.

    Repair failure -> mapped_status = UNMATCHED. account_type is
    (re-)derived after every snap/rewrite -- there is no parameter or code
    path by which a caller-supplied account_type could reach a
    NormalizedRow."""
    resolved: list = []
    needs_repair: list = []

    for row in provisional_rows:
        # main_head is passed so snap_to_taxonomy disambiguates the rare
        # case where (sub_head_1, sub_head_2) legitimately exists under
        # more than one main_head (Schedule III's Non-current/Current dual
        # classification) -- every ProvisionalMappedRow construction site
        # (normalize.py Step 2, classification.py Step 3, staged_narrowing.py
        # Step 3) establishes main_head via its OWN validated
        # snap_to_taxonomy call before this function ever sees it, so
        # passing it here cannot turn a genuinely-wrong classification into
        # a false MAPPED -- it can only let an already-correct one survive
        # its own re-validation.
        snapped = snap_to_taxonomy(row.bs_pl, row.main_head, row.sub_head_1, row.sub_head_2, standard)
        if snapped is not None:
            # account_type is ALWAYS derived from the canonical node
            # (`snapped`), never from client wording.
            account_type = derive_account_type(snapped["bs_pl"], snapped["main_head"], standard)

            if row.client_fields:
                display_bs_pl = row.client_fields.get("bs_pl") or snapped["bs_pl"]
                display_main_head = row.client_fields.get("main_head") or snapped["main_head"]
                display_sub_head_1 = row.client_fields.get("sub_head_1") or snapped["sub_head_1"]
                display_sub_head_2 = row.client_fields.get("sub_head_2") or snapped["sub_head_2"]
                wording_source = "client_original"
            else:
                display_bs_pl = snapped["bs_pl"]
                display_main_head = snapped["main_head"]
                display_sub_head_1 = snapped["sub_head_1"]
                display_sub_head_2 = snapped["sub_head_2"]
                wording_source = "system_canonical"

            resolved.append(
                NormalizedRow.from_tb_row(
                    row.tb_row,
                    bs_pl=display_bs_pl, main_head=display_main_head, sub_head_1=display_sub_head_1,
                    sub_head_2=display_sub_head_2, account_type=account_type, mapped_status=MAPPED,
                    trace=RowTrace(
                        resolved_at_step=row.resolved_at_step,
                        staged_narrowing_triggered=row.staged_narrowing_triggered,
                        demoted_from_provisional_mapped=_already_repaired,
                        repair_attempted=_already_repaired,
                        repair_succeeded=True if _already_repaired else None,
                        retrieval_confidence=row.retrieval_confidence,
                    ),
                    canonical_bs_pl=snapped["bs_pl"], canonical_main_head=snapped["main_head"],
                    canonical_sub_head_1=snapped["sub_head_1"], canonical_sub_head_2=snapped["sub_head_2"],
                    wording_source=wording_source,
                )
            )
        else:
            needs_repair.append(row)

    if not needs_repair:
        return resolved

    if _already_repaired:
        # Second pass already exhausted its one repair attempt -- terminal
        # UNMATCHED for every row still unresolved.
        for row in needs_repair:
            resolved.append(
                NormalizedRow.from_tb_row(
                    row.tb_row,
                    bs_pl=None, main_head=None, sub_head_1=None, sub_head_2=None, account_type=None,
                    mapped_status=UNMATCHED,
                    trace=RowTrace(
                        resolved_at_step="unresolved", staged_narrowing_triggered=row.staged_narrowing_triggered,
                        demoted_from_provisional_mapped=True, repair_attempted=True, repair_succeeded=False,
                    ),
                )
            )
        return resolved

    # Exactly one repair attempt, single-shot mechanism only.
    candidates = [
        ClassifyCandidate(row.tb_row.gl_code, row.tb_row.gl_name, _build_repair_hint(row))
        for row in needs_repair
    ]
    repair_matches = classify_all(
        candidates, standard, llm_client, use_candidates=use_candidates,
        use_candidate_id_contract=use_candidate_id_contract, use_adaptive_candidates=use_adaptive_candidates,
        use_hierarchical_candidate_context=use_hierarchical_candidate_context,
    )

    repaired_provisional: list = []
    unrepaired: list = []
    for row in needs_repair:
        match = repair_matches.get(row.tb_row.gl_code)
        if match is None:
            unrepaired.append(row)
            continue
        repaired_provisional.append(
            ProvisionalMappedRow(
                tb_row=row.tb_row, bs_pl=match.bs_pl, main_head=match.main_head, sub_head_1=match.sub_head_1,
                sub_head_2=match.sub_head_2, resolved_at_step="repair",
                staged_narrowing_triggered=row.staged_narrowing_triggered, retrieval_confidence=match.retrieval_confidence,
            )
        )

    if repaired_provisional:
        resolved.extend(
            run_validation_gate(
                repaired_provisional, standard, llm_client, use_candidates=use_candidates,
                use_candidate_id_contract=use_candidate_id_contract, use_adaptive_candidates=use_adaptive_candidates,
                use_hierarchical_candidate_context=use_hierarchical_candidate_context, _already_repaired=True,
            )
        )

    for row in unrepaired:
        resolved.append(
            NormalizedRow.from_tb_row(
                row.tb_row,
                bs_pl=None, main_head=None, sub_head_1=None, sub_head_2=None, account_type=None,
                mapped_status=UNMATCHED,
                trace=RowTrace(
                    resolved_at_step="unresolved", staged_narrowing_triggered=row.staged_narrowing_triggered,
                    demoted_from_provisional_mapped=True, repair_attempted=True, repair_succeeded=False,
                ),
            )
        )

    return resolved


_MAIN_HEAD_ONLY_ACCOUNT_TYPES = {
    "BS": {"Asset", "Liability", "Equity", "unclassified"},
    "PL": {"Income", "Expense"},
}


def build_main_head_only_row(
    tb_row, *, bs_pl, main_head, account_type, resolution_method: str, evidence: str, confidence=None,
):
    """Validation Gate for ERPs whose source supplies a group hierarchy but
    NO sub_head_1/sub_head_2 (Tally). Returns MAPPED with sub_head_1/2 left
    None -- deliberately not a full taxonomy leaf -- only when bs_pl,
    main_head and account_type are present and mutually consistent;
    otherwise UNMATCHED. Like run_validation_gate, this is the only place
    such a MAPPED row can be constructed, and account_type is validated
    here, never trusted from a caller unchecked."""
    valid = (
        bool(main_head and str(main_head).strip())
        and bs_pl in _MAIN_HEAD_ONLY_ACCOUNT_TYPES
        and account_type in _MAIN_HEAD_ONLY_ACCOUNT_TYPES[bs_pl]
    )
    if not valid:
        return NormalizedRow.from_tb_row(
            tb_row, bs_pl=None, main_head=None, sub_head_1=None, sub_head_2=None, account_type=None,
            mapped_status=UNMATCHED, trace=RowTrace(resolved_at_step="unresolved", evidence=evidence),
        )
    return NormalizedRow.from_tb_row(
        tb_row, bs_pl=bs_pl, main_head=main_head, sub_head_1=None, sub_head_2=None, account_type=account_type,
        mapped_status=MAPPED,
        trace=RowTrace(
            resolved_at_step="tally_main_head", resolution_method=resolution_method, evidence=evidence,
            confidence=confidence,
        ),
        canonical_bs_pl=bs_pl, canonical_main_head=main_head, wording_source="client_original",
    )


__all__ = ["run_validation_gate", "build_main_head_only_row"]

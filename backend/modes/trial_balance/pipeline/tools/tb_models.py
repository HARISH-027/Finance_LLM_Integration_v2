"""Core data shapes for the native classification engine: TB row + Grouping
hint -> Normalized row. A pure-function shape -- the three concerns (TB
values, grouping hint, in-progress classification state) are three separate
types, and NormalizedRow is only ever constructed by validation_gate.py (see
that module's docstring for why).

Ported from TB_normalization_v1's core/models.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, NamedTuple, Optional

# Three persisted values. UNMATCHED is distinct from UNMAPPED: "this GL code
# has no grouping data at all" (UNMAPPED, an input-completeness problem) vs.
# "this GL code had grouping data but nothing -- deterministic, embedding,
# or LLM -- could resolve it to a real taxonomy leaf" (UNMATCHED, a
# classification/taxonomy-coverage problem).
MappedStatus = Literal["MAPPED", "UNMAPPED", "UNMATCHED"]

MAPPED: MappedStatus = "MAPPED"
UNMAPPED: MappedStatus = "UNMAPPED"
UNMATCHED: MappedStatus = "UNMATCHED"

# Where a row's classification was ultimately settled -- belongs in a
# trace/log, not in the persisted mapped_status field.
ResolvedAtStep = Literal[
    "step2_direct",       # grouping data supplied all 5 fields directly, validated first try
    "step3_single_shot",  # single-shot LLM batch classification
    "step3_staged",       # 3-stage ambiguous-vocabulary narrowing
    "repair",             # resolved only after the Validation Gate demoted and repaired it
    "tally_main_head",    # Tally: settled at main_head depth by design -- no sub_head_1/2 exists in source
    "no_grouping_match",  # GL code not found in grouping data at all -- terminal UNMAPPED
    "unresolved",         # grouping data existed but never resolved -- terminal UNMATCHED
]

ResolutionMethod = Literal[
    "EXACT", "ALIAS", "CANDIDATE_AUTO", "DETERMINISTIC_RULE", "LLM_CANDIDATE", "UNMAPPED",
]


@dataclass(frozen=True)
class TBRow:
    """One Trial Balance row, exactly as parsed. Frozen so TB cell values
    are never modified downstream."""

    gl_code: str
    gl_name: str
    opening: float
    debit: float
    credit: float
    closing: float


@dataclass(frozen=True)
class GroupingHint:
    """Whatever a company's own grouping data supplies for one GL code --
    hints and candidates, never trusted output. `known_fields` holds
    whichever of bs_pl/main_head/sub_head_1/sub_head_2 the data's own
    structured columns directly supplied (may be empty); `hint_text` is a
    free-text section label (may be empty)."""

    gl_code: str
    known_fields: dict = field(default_factory=dict)
    hint_text: str = ""
    # Snapshot of `known_fields` exactly as the client's own file supplied
    # it, taken before taxonomy_resolver.py::resolve_grouping_hints()
    # overwrites known_fields with the resolved node's own canonical text.
    # None means the resolver never touched this hint.
    original_known_fields: Optional[dict] = None


@dataclass(frozen=True)
class TaxonomyResolution:
    """The outcome of resolving one GL row's grouping evidence against the
    controlled taxonomy via taxonomy_resolver.py -- an in-memory audit-trail
    value, never a column on the Canonical TB row itself (see canonical_schema.
    py's CANONICAL_TB_COLUMNS, which stays untouched by this). Its own open
    decision (design doc section 35) has been made: threaded onto RowTrace by
    resolution_metadata.py, then persisted to its own tb_taxonomy_resolution
    table (taxonomy_resolution_persistence.py, migration 0005) -- not
    reintroduced as a canonical row column, which would ripple into every
    consumer that selects CANONICAL_TB_COLUMNS."""

    status: Literal["RESOLVED", "UNRESOLVED"]
    method: ResolutionMethod
    evidence: str
    confidence: Optional[float] = None
    taxonomy_node_id: Optional[int] = None
    taxonomy_version_id: Optional[int] = None


@dataclass(frozen=True)
class RowTrace:
    """Everything about how a row was resolved that must NOT live in
    mapped_status. Retrievable for debugging/audit, never read by anything
    that decides whether a row is MAPPED."""

    resolved_at_step: ResolvedAtStep
    staged_narrowing_triggered: bool = False
    demoted_from_provisional_mapped: bool = False
    repair_attempted: bool = False
    repair_succeeded: Optional[bool] = None
    resolution_method: Optional[str] = None
    confidence: Optional[float] = None
    evidence: Optional[str] = None
    # RetrievalConfidence.value from the row's own candidate retrieval at
    # classify time -- only set for Step 3/repair rows classified with
    # use_candidates=True; None for Step 2 direct rows and staged-narrowing
    # rows (no candidate retrieval there).
    retrieval_confidence: Optional[str] = None
    # Populated only when taxonomy_resolver.py's own resolver actually
    # resolved this row (see resolution_metadata.py::attach_resolution_metadata) --
    # never set for a row resolved by Step 2 direct/Step 3 LLM/repair, since
    # those paths don't go through the resolver at all.
    taxonomy_node_id: Optional[int] = None
    taxonomy_version_id: Optional[int] = None


@dataclass(frozen=True)
class NormalizedRow:
    """The engine's output: one TB row, classified. gl_code/gl_name and the
    four numeric fields are always echoed verbatim from the input TBRow.

    Construction is intentionally confined to validation_gate.py so
    mapped_status="MAPPED" can only ever be produced by code that has
    actually run the Validation Gate -- there is no other constructor call
    site anywhere in this engine."""

    gl_code: str
    gl_name: str
    opening: float
    debit: float
    credit: float
    closing: float
    bs_pl: Optional[str]
    main_head: Optional[str]
    sub_head_1: Optional[str]
    sub_head_2: Optional[str]
    account_type: Optional[str]
    mapped_status: MappedStatus
    trace: RowTrace
    # canonical_* are always the resolved taxonomy node's own strings
    # whenever a node was established -- kept separately so account_type
    # derivation always has the canonical form available, even on rows
    # where bs_pl/main_head/sub_head_1/sub_head_2 above display the
    # client's own original wording instead.
    canonical_bs_pl: Optional[str] = None
    canonical_main_head: Optional[str] = None
    canonical_sub_head_1: Optional[str] = None
    canonical_sub_head_2: Optional[str] = None
    # "client_original": bs_pl/main_head/sub_head_1/sub_head_2 above show
    # the client's own supplied wording (validated against the taxonomy,
    # but not replaced by it). "system_canonical": those fields show the
    # taxonomy node's own text.
    wording_source: str = "system_canonical"

    @staticmethod
    def from_tb_row(
        tb_row: TBRow,
        *,
        bs_pl, main_head, sub_head_1, sub_head_2, account_type, mapped_status, trace,
        canonical_bs_pl=None, canonical_main_head=None, canonical_sub_head_1=None, canonical_sub_head_2=None,
        wording_source="system_canonical",
    ) -> "NormalizedRow":
        return NormalizedRow(
            gl_code=tb_row.gl_code, gl_name=tb_row.gl_name, opening=tb_row.opening, debit=tb_row.debit,
            credit=tb_row.credit, closing=tb_row.closing, bs_pl=bs_pl, main_head=main_head,
            sub_head_1=sub_head_1, sub_head_2=sub_head_2, account_type=account_type,
            mapped_status=mapped_status, trace=trace, canonical_bs_pl=canonical_bs_pl,
            canonical_main_head=canonical_main_head, canonical_sub_head_1=canonical_sub_head_1,
            canonical_sub_head_2=canonical_sub_head_2, wording_source=wording_source,
        )


@dataclass(frozen=True)
class ProvisionalMappedRow:
    """A row that has resolved all 5 taxonomy fields from Step 2 or Step 3,
    but has NOT yet passed the Validation Gate -- it has no mapped_status
    at all, so no code holding one of these can accidentally treat it as
    final output."""

    tb_row: TBRow
    bs_pl: str
    main_head: str
    sub_head_1: str
    sub_head_2: str
    resolved_at_step: ResolvedAtStep
    staged_narrowing_triggered: bool = False
    # Forwarded from the MatchResult that resolved this row (Step 3 only).
    retrieval_confidence: Optional[str] = None
    # The client's own original taxonomy wording for this GL code,
    # populated ONLY by the Step 2 direct-resolution path.
    client_fields: Optional[dict] = None


class ClassifyCandidate(NamedTuple):
    """The minimal shape classification.py/staged_narrowing.py need from a
    caller. known_main_head/known_bs_pl: whatever the grouping hint's own
    structured fields supplied, used ONLY as a soft ranking boost in
    candidate retrieval, never as a hard filter or generic-bucket-injection
    gate -- a grouping file's own known_fields can themselves be wrong."""

    gl_code: str
    gl_name: str
    hint: str = ""
    known_main_head: str = ""
    known_bs_pl: str = ""


@dataclass(frozen=True)
class MatchResult:
    """A single validated (bs_pl, main_head, sub_head_1, sub_head_2,
    account_type) selection, already snapped to the taxonomy and with
    account_type already derived."""

    bs_pl: str
    main_head: str
    sub_head_1: str
    sub_head_2: str
    account_type: str
    retrieval_confidence: Optional[str] = None

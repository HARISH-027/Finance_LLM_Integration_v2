"""Persistence for TaxonomyResolution/RowTrace's audit-trail data (tb_taxonomy_
resolution table, migration 0005) -- resolves the open decision migration 0003
explicitly deferred (see erp_persistence.py's own docstring and this table's
migration file for the full history).

Client-review defect registry: a mapped account's confidence was already
computed during classification and simply discarded before persistence -- an
auditor had no way to tell an exact/alias match from a shaky LLM guess that
barely cleared the bar. This module gives that data a real home, mirroring
erp_persistence.py's exact upsert-by-tb_doc_id pattern, without touching
CANONICAL_TB_COLUMNS or any of its consumers.

mapping_confidence reconciles the two different confidence representations
RowTrace can carry into one High/Medium/Low vocabulary (NULL for UNMAPPED/
UNMATCHED, where mapped_status already conveys the answer):
  - EXACT / ALIAS / DETERMINISTIC_RULE / DIRECT (step2_direct): all
    deterministic, non-fuzzy resolution mechanisms -> High, regardless of the
    resolver's own float (that float was only ever a fixed per-method marker
    -- 1.0/0.95/0.85 -- not a real gradient; see taxonomy_resolver.py).
  - CANDIDATE_AUTO: the one path with a genuinely continuous score
    (confidence_score). >=0.80 -> High, else (including the ILIKE fallback,
    which carries no score at all) -> Medium.
  - LLM_SINGLE_SHOT / LLM_REPAIR: use_candidates=True at classify time, so
    RowTrace.retrieval_confidence (HIGH/MEDIUM/LOW/NONE) is populated --
    passed through directly.
  - LLM_STAGED: the 3-stage ambiguous-vocabulary narrowing path never gets a
    retrieval_confidence (per tb_models.py's own RowTrace docstring: "None
    for Step 2 direct rows and staged-narrowing rows") and has no resolver
    float either -- defaults to Medium (LLM-resolved, but no finer signal
    exists to distinguish it further) rather than leaving it NULL, which
    would read as a data gap rather than an honest "no finer signal here".
"""

from __future__ import annotations

from typing import Dict, List, Optional

from modes.trial_balance.pipeline.db import db_cursor

_HIGH_CONFIDENCE_METHODS = ("EXACT", "ALIAS", "DETERMINISTIC_RULE", "DIRECT")
_CANDIDATE_AUTO_HIGH_FLOOR = 0.80
_RETRIEVAL_CONFIDENCE_METHODS = ("LLM_SINGLE_SHOT", "LLM_REPAIR")
_UNRESOLVED_METHODS = ("UNMAPPED", "UNMATCHED")


def bucket_mapping_confidence(trace) -> Optional[str]:
    """Reconcile RowTrace's resolution_method/confidence/retrieval_confidence
    into the client spec's High/Medium/Low vocabulary. See module docstring
    for the full reasoning per method."""
    method = trace.resolution_method

    if method in _HIGH_CONFIDENCE_METHODS:
        return "High"

    if method == "CANDIDATE_AUTO":
        if trace.confidence is not None and trace.confidence >= _CANDIDATE_AUTO_HIGH_FLOOR:
            return "High"
        return "Medium"

    if method in _UNRESOLVED_METHODS:
        return None

    if method in _RETRIEVAL_CONFIDENCE_METHODS:
        rc = (trace.retrieval_confidence or "").upper()
        if rc == "HIGH":
            return "High"
        if rc == "MEDIUM":
            return "Medium"
        if rc == "LOW":
            return "Low"
        return None

    if method == "LLM_STAGED":
        return "Medium"

    return None


def upsert_tb_taxonomy_resolution(tb_doc_id: str, rows: List) -> int:
    """Delete-then-insert by tb_doc_id -- same pattern
    upsert_tb_erp_classification already uses (re-ingesting a document
    replaces its resolution rows, never accumulates duplicates). `rows` is
    the NormalizedRow list classify_tb_document() returns; each row's
    .trace carries the resolution audit data this table persists."""
    with db_cursor(dict_rows=False) as cur:
        cur.execute("DELETE FROM tb_taxonomy_resolution WHERE tb_doc_id = %s", (tb_doc_id,))
        for row in rows:
            trace = row.trace
            cur.execute(
                """
                INSERT INTO tb_taxonomy_resolution
                    (tb_doc_id, gl_code, mapped_status, resolution_method, mapping_confidence,
                     confidence_score, evidence, taxonomy_node_id, taxonomy_version_id)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (tb_doc_id, row.gl_code, row.mapped_status, trace.resolution_method,
                 bucket_mapping_confidence(trace), trace.confidence, trace.evidence,
                 trace.taxonomy_node_id, trace.taxonomy_version_id),
            )
    return len(rows)


def fetch_tb_taxonomy_resolution(tb_doc_id: str) -> Dict[str, dict]:
    with db_cursor() as cur:
        cur.execute("SELECT * FROM tb_taxonomy_resolution WHERE tb_doc_id = %s", (tb_doc_id,))
        return {row["gl_code"]: dict(row) for row in cur.fetchall()}

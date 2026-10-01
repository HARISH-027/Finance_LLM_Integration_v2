"""Tests for taxonomy_resolution_persistence.py (migration 0005) -- the open
decision migration 0003 deferred: per-account mapping confidence, computed
during classification and previously discarded before persistence.

bucket_mapping_confidence is pure and tested without a DB. The upsert/fetch
round trip is DB-gated via the existing db_available fixture (same pattern
as test_erp_persistence.py); skipped cleanly when no Postgres is reachable.
"""

from dataclasses import dataclass
from typing import Optional

import pytest

from modes.trial_balance.pipeline.db import db_cursor
from modes.trial_balance.pipeline.tools.classify_document import classify_tb_document
from modes.trial_balance.pipeline.tools.input_grouping_shape import GroupingShape
from modes.trial_balance.pipeline.tools.taxonomy_resolution_persistence import (
    bucket_mapping_confidence,
    fetch_tb_taxonomy_resolution,
    upsert_tb_taxonomy_resolution,
)
from modes.trial_balance.pipeline.tools.tb_models import GroupingHint, TBRow


@dataclass
class _FakeTrace:
    resolution_method: Optional[str] = None
    confidence: Optional[float] = None
    retrieval_confidence: Optional[str] = None
    evidence: Optional[str] = None
    taxonomy_node_id: Optional[int] = None
    taxonomy_version_id: Optional[int] = None


@dataclass
class _FakeRow:
    gl_code: str
    mapped_status: str
    trace: _FakeTrace


class TestBucketMappingConfidence:
    @pytest.mark.parametrize(
        "method,confidence,retrieval_confidence,expected",
        [
            ("EXACT", None, None, "High"),
            ("ALIAS", None, None, "High"),
            ("DETERMINISTIC_RULE", None, None, "High"),
            ("DIRECT", None, None, "High"),
            ("CANDIDATE_AUTO", 0.92, None, "High"),
            ("CANDIDATE_AUTO", 0.80, None, "High"),
            ("CANDIDATE_AUTO", 0.65, None, "Medium"),
            ("CANDIDATE_AUTO", None, None, "Medium"),  # ILIKE fallback -- no score at all
            ("LLM_SINGLE_SHOT", None, "HIGH", "High"),
            ("LLM_SINGLE_SHOT", None, "MEDIUM", "Medium"),
            ("LLM_SINGLE_SHOT", None, "LOW", "Low"),
            ("LLM_SINGLE_SHOT", None, "NONE", None),
            ("LLM_SINGLE_SHOT", None, None, None),
            ("LLM_REPAIR", None, "HIGH", "High"),
            ("LLM_STAGED", None, None, "Medium"),
            ("UNMAPPED", None, None, None),
            ("UNMATCHED", None, None, None),
        ],
    )
    def test_bucketing_rule_per_resolution_method(self, method, confidence, retrieval_confidence, expected):
        trace = _FakeTrace(resolution_method=method, confidence=confidence, retrieval_confidence=retrieval_confidence)
        assert bucket_mapping_confidence(trace) == expected


@pytest.fixture
def doc_id(db_available):
    if not db_available:
        pytest.skip("No reachable Postgres DB for this test run.")
    tb_doc_id = "PYTEST_TAXONOMY_RESOLUTION_DOC"
    yield tb_doc_id
    with db_cursor(dict_rows=False) as cur:
        cur.execute("DELETE FROM tb_taxonomy_resolution WHERE tb_doc_id = %s", (tb_doc_id,))


class TestUpsertFetchRoundTrip:
    def test_round_trips_mapped_and_unmapped_rows(self, doc_id):
        rows = [
            _FakeRow("10110000", "MAPPED", _FakeTrace(resolution_method="EXACT")),
            _FakeRow("20950021", "MAPPED", _FakeTrace(resolution_method="CANDIDATE_AUTO", confidence=0.72)),
            _FakeRow("90100057", "UNMAPPED", _FakeTrace(resolution_method="UNMAPPED")),
        ]
        n = upsert_tb_taxonomy_resolution(doc_id, rows)
        assert n == 3

        fetched = fetch_tb_taxonomy_resolution(doc_id)
        assert set(fetched.keys()) == {"10110000", "20950021", "90100057"}
        assert fetched["10110000"]["mapping_confidence"] == "High"
        assert fetched["20950021"]["mapping_confidence"] == "Medium"
        assert fetched["20950021"]["confidence_score"] == 0.72
        assert fetched["90100057"]["mapping_confidence"] is None
        assert fetched["90100057"]["mapped_status"] == "UNMAPPED"

    def test_re_upsert_replaces_not_accumulates(self, doc_id):
        upsert_tb_taxonomy_resolution(doc_id, [_FakeRow("1001", "MAPPED", _FakeTrace(resolution_method="EXACT"))])
        upsert_tb_taxonomy_resolution(doc_id, [_FakeRow("2001", "MAPPED", _FakeTrace(resolution_method="ALIAS"))])
        fetched = fetch_tb_taxonomy_resolution(doc_id)
        assert set(fetched.keys()) == {"2001"}
        with db_cursor() as cur:
            cur.execute("SELECT COUNT(*) AS n FROM tb_taxonomy_resolution WHERE tb_doc_id = %s", (doc_id,))
            assert cur.fetchone()["n"] == 1

    def test_fetch_returns_empty_dict_for_unknown_doc(self, db_available):
        if not db_available:
            pytest.skip("No reachable Postgres DB for this test run.")
        assert fetch_tb_taxonomy_resolution("PYTEST_NO_SUCH_DOC") == {}


class _FakeLLMClient:
    def generate(self, *args, **kwargs):
        raise AssertionError("LLM should not be called for an EXACT-resolved hint")


def test_real_classification_output_survives_full_persistence_round_trip(doc_id):
    """Runs the actual classification engine (not hand-built fake rows) so the
    real NormalizedRow.trace shape this module reads is exercised end to end,
    matching test_classify_document.py's own resolver-audit-trail setup."""
    tb_rows = [TBRow(gl_code="1001", gl_name="Freehold Land", opening=0.0, debit=1000.0, credit=0.0, closing=1000.0)]
    grouping_hints = {
        "1001": GroupingHint(
            gl_code="1001",
            known_fields={"sub_head_1": "Property, Plant and Equipment", "sub_head_2": "Land"},
        ),
    }

    result = classify_tb_document(
        tb_rows, grouping_hints, GroupingShape.HIERARCHICAL_TRAIL, "IND_AS", _FakeLLMClient(),
        accept_data_quality_risk=True,
    )
    assert result.rows[0].trace.resolution_method == "EXACT"

    upsert_tb_taxonomy_resolution(doc_id, result.rows)
    fetched = fetch_tb_taxonomy_resolution(doc_id)

    assert fetched["1001"]["mapped_status"] == "MAPPED"
    assert fetched["1001"]["resolution_method"] == "EXACT"
    assert fetched["1001"]["mapping_confidence"] == "High"
    assert fetched["1001"]["taxonomy_node_id"] is not None

"""Tests for backend/tools/erp_persistence.py -- Phase 9's persistence of
ScopeMetadata (tb_scope) and ClassificationEvidence (tb_erp_classification).
DB-gated via the existing db_available fixture (same pattern as
test_live_repository.py); skipped cleanly when no Postgres is reachable."""

from pathlib import Path

import pytest

from modes.trial_balance.pipeline.db import add_findings, db_cursor
from modes.trial_balance.pipeline.tools.erp_persistence import (
    fetch_tb_erp_classification,
    fetch_tb_scope,
    findings_to_pipeline_finding_dicts,
    upsert_tb_erp_classification,
    upsert_tb_scope,
)
from modes.trial_balance.pipeline.tools.input_header_detect import build_tb_rows_from_grid, locate_header
from modes.trial_balance.pipeline.tools.input_scenario_b import _load_workbook_sheets
from modes.trial_balance.pipeline.tools.sap_adapter import run_sap_adapter
from modes.trial_balance.pipeline.tools.tb_erp_contracts import ClassificationEvidence, ERPType, ScopeMetadata, ScopeType

_SAP_TB_FILE = (
    Path(__file__).parents[2] / "documentation" / "ERP_system_standardized_files" / "SAP"
    / "Oil_and_Gas" / "Terra_Clean"
    / "Terra_Clean_Master_Trial_Balance_and_GL_Grouping__F.Y_24-25__1607165b.xlsx"
)


@pytest.fixture
def doc_id(db_available):
    if not db_available:
        pytest.skip("No reachable Postgres DB for this test run.")
    tb_doc_id = "PYTEST_ERP_PERSISTENCE_DOC"
    yield tb_doc_id
    with db_cursor(dict_rows=False) as cur:
        cur.execute("DELETE FROM tb_scope WHERE tb_doc_id = %s", (tb_doc_id,))
        cur.execute("DELETE FROM tb_erp_classification WHERE tb_doc_id = %s", (tb_doc_id,))
        cur.execute("DELETE FROM pipeline_findings WHERE session_id = %s", (tb_doc_id,))


class TestTbScopeRoundTrip:
    def test_round_trips_every_field(self, doc_id):
        scope = ScopeMetadata(
            entity="ACME Ltd", company_code="1000", ledger="0L", fiscal_year="2025-2026",
            period_from=1, period_to=12, currency="INR", account_range="ALL", business_scope="ALL",
            profit_center="PC01", segment="SEG01", scope_type=ScopeType.FULL_LEDGER,
        )
        upsert_tb_scope(doc_id, scope)
        fetched = fetch_tb_scope(doc_id)
        assert fetched == scope

    def test_upsert_replaces_not_accumulates(self, doc_id):
        upsert_tb_scope(doc_id, ScopeMetadata(entity="First", scope_type=ScopeType.FULL_LEDGER))
        upsert_tb_scope(doc_id, ScopeMetadata(entity="Second", scope_type=ScopeType.MULTI_ENTITY))
        fetched = fetch_tb_scope(doc_id)
        assert fetched.entity == "Second"
        assert fetched.scope_type == ScopeType.MULTI_ENTITY
        with db_cursor() as cur:
            cur.execute("SELECT COUNT(*) AS n FROM tb_scope WHERE tb_doc_id = %s", (doc_id,))
            assert cur.fetchone()["n"] == 1

    def test_fetch_returns_none_for_unknown_doc(self, db_available):
        if not db_available:
            pytest.skip("No reachable Postgres DB for this test run.")
        assert fetch_tb_scope("PYTEST_NO_SUCH_DOC") is None


class TestTbErpClassificationRoundTrip:
    def test_round_trips_multi_row_map_including_nested_dicts(self, doc_id):
        classification_by_gl_code = {
            "1001": ClassificationEvidence(
                erp_type=ERPType.SAP,
                account_identity={"gl_code": "1001", "gl_name": "Freehold Land"},
                account_classification={"bs_pl": "BS", "main_head": "Non-current assets"},
                business_dimensions={"business_area": "DL01"},
            ),
            "1002": ClassificationEvidence(erp_type=ERPType.SAP, account_identity={"gl_code": "1002", "gl_name": "Cash"}),
        }
        n = upsert_tb_erp_classification(doc_id, classification_by_gl_code)
        assert n == 2

        fetched = fetch_tb_erp_classification(doc_id)
        assert set(fetched.keys()) == {"1001", "1002"}
        assert fetched["1001"] == classification_by_gl_code["1001"]
        assert fetched["1002"] == classification_by_gl_code["1002"]

    def test_re_upsert_replaces_not_accumulates(self, doc_id):
        upsert_tb_erp_classification(doc_id, {"1001": ClassificationEvidence(erp_type=ERPType.SAP)})
        upsert_tb_erp_classification(doc_id, {"2001": ClassificationEvidence(erp_type=ERPType.TALLY)})
        fetched = fetch_tb_erp_classification(doc_id)
        assert set(fetched.keys()) == {"2001"}


class TestFindingsToPipelineFindingDicts:
    def test_produces_dicts_add_findings_accepts(self, doc_id):
        class _FakeFinding:
            def __init__(self, severity, code, message):
                self.severity, self.code, self.message = severity, code, message

        findings = [_FakeFinding("audit_risk", "grand_total_imbalance", "Total Debit != Total Credit")]
        dicts = findings_to_pipeline_finding_dicts(findings)

        n = add_findings(doc_id, dicts)
        assert n == 1

        with db_cursor() as cur:
            cur.execute("SELECT * FROM pipeline_findings WHERE session_id = %s", (doc_id,))
            rows = [dict(r) for r in cur.fetchall()]
        assert len(rows) == 1
        assert rows[0]["statement"] == "Total Debit != Total Credit"
        assert rows[0]["category"] == "erp_adaptability"


@pytest.mark.skipif(not _SAP_TB_FILE.exists(), reason="Terra_Clean real-file fixture not present")
def test_real_sap_terra_clean_evidence_survives_full_persistence_round_trip(doc_id):
    sheets = _load_workbook_sheets(_SAP_TB_FILE)
    grid = sheets["F.Y_24-25"]
    header_row, cols = locate_header(grid)
    tb_rows = build_tb_rows_from_grid(grid, header_row, cols)
    result = run_sap_adapter(_SAP_TB_FILE, grid, header_row, cols, tb_rows, sheet_name="F.Y_24-25")

    upsert_tb_scope(doc_id, result.scope)
    upsert_tb_erp_classification(doc_id, result.classification_by_gl_code)

    fetched_scope = fetch_tb_scope(doc_id)
    fetched_classification = fetch_tb_erp_classification(doc_id)

    assert fetched_scope == result.scope
    assert fetched_classification["1111100000"].account_classification["main_head"] == "Equity Share Capital"

"""End-to-end integration test for the rewired ingest_tb_to_live: real
synthetic workbook -> input_dispatch auto-detection -> the native
classification engine (against the real shared taxonomy DB) -> a real
write into live_document_table/live_tb_table. Gated on `db_available`,
same as every other LIVE-staging test in this package.
"""

from pathlib import Path
from unittest.mock import patch

import openpyxl
import pytest

from modes.trial_balance.pipeline.db import db_cursor, fetch_live_document
from modes.trial_balance.pipeline.tools import delete_db_document, ingest_tb_to_live
from modes.trial_balance.pipeline.tools.erp_persistence import fetch_tb_erp_classification, fetch_tb_scope

_GURUGRAM_DIR = (
    Path(__file__).parents[3] / "documentation" / "ERP_system_standardized_files" / "Tally"
    / "Gurugram_Metro_Rail_Limited"
)
_GURUGRAM_TB = _GURUGRAM_DIR / "Gurugram_Metro_Rail_Limited_Master_Trial_Balance_Head_Sub_CAG__TB25-26__be041db1.xlsx"
_GURUGRAM_GROUPING = _GURUGRAM_DIR / "Gurugram_Metro_Rail_Limited_Master_Trial_Balance_Head_Sub_CAG__Grouping__5e8045cb.xlsx"

_BHAVINI_DIR = (
    Path(__file__).parents[3] / "documentation" / "ERP_system_standardized_files" / "Oracle"
    / "Bhavini" / "Multi_FY_2024-25_to_2025-26"
)
_BHAVINI_TB = _BHAVINI_DIR / "Bhavini_Multi_FY_2024-25_to_2025-26_Trial_Balance_and_GL_Grouping__TB_24-25__e2452013.xlsx"
_BHAVINI_GROUPING = (
    _BHAVINI_DIR / "Bhavini_Multi_FY_2024-25_to_2025-26_Trial_Balance_and_GL_Grouping__Grouping_24-25__6aa740fb.xlsx"
)


class _FakeResponse:
    def __init__(self, content):
        self.content = content


class _FakeLLMClient:
    """Only Step 1 (no hint) and Step 2 (complete, valid hint) rows are
    exercised by this fixture's data, so the LLM should never actually be
    called -- if it is, something upstream regressed."""

    def generate(self, *args, **kwargs):
        raise AssertionError("LLM should not be called for a fully-mapped template")


class _FakeAgent:
    llm_client = _FakeLLMClient()


def _write_template(path, tb_doc_name="ACME_FY2025-26.xlsx", with_grand_total_row=False):
    wb = openpyxl.Workbook()
    wb.remove(wb.active)
    company = wb.create_sheet("COMPANY_DETAILS")
    for row in [
        ["Company Details", None],
        ["COMPANY_NAME", "ACME End To End Ltd"],
        ["FINANCIAL_YEAR", "FY2025-26"],
        ["FY_START", 2025],
        ["FY_END", 2026],
    ]:
        company.append(row)

    tb = wb.create_sheet("TB&GROUPING-TEMPLATE")
    tb.append(["GL Code", "GL Name", "Opening Balance", "Debit", "Credit", "Closing Balance",
               "BS/PL", "Main Head", "Sub Head 1", "Sub Head 2"])
    tb.append(["E2E-1001", "Freehold Land (E2E test)", 0, 1000, 0, 1000,
               "BS", "Non-current assets", "Property, Plant and Equipment", "Land"])
    if with_grand_total_row:
        # Gap 1: a Grand Total row in the TB workbook itself must never reach
        # canonical_tb.parquet as a real GL account.
        tb.append(["E2E-9999", "Grand Total", 0, 1000, 0, 1000, None, None, None, None])
    wb.save(path)


def test_template_ingests_end_to_end_into_live_staging(db_available, tmp_path):
    if not db_available:
        pytest.skip("No reachable Postgres DB (settings.DB_*) -- skipping.")

    path = tmp_path / "template.xlsx"
    _write_template(path)

    with patch("modes.trial_balance.pipeline.agent.get_agent", return_value=_FakeAgent()):
        result = ingest_tb_to_live(
            tb_grouping_template_path=str(path),
            output_dir=str(tmp_path / "out"),
        )

    try:
        assert result["execution_status"] == "SUCCESS", result
        assert result["pipeline_status"] in ("SUCCESS", "WARNING"), result
        assert result["quality_tier"] == "CASE_1"
        assert result["run_metrics"]["mapped_count"] == 1
        assert (tmp_path / "out" / "canonical_tb.parquet").exists()

        import polars as pl

        out_df = pl.read_parquet(tmp_path / "out" / "canonical_tb.parquet")
        row = out_df.to_dicts()[0]
        assert row["gl_code"] == "E2E-1001"
        assert row["mapped_status"] == "MAPPED"
        assert row["sub_head_2"] == "Land"
        assert row["account_type"] == "Asset"

        # Blank overrides (the default -- no Company Details fields filled) leave
        # the workbook's own COMPANY_DETAILS metadata exactly as auto-parsed.
        live_doc = fetch_live_document(result["tb_doc_id"])
        assert live_doc["company_name"] == "ACME End To End Ltd"
        assert live_doc["financial_year"] == "FY2025-26"
    finally:
        delete_db_document(tb_doc_id=result["tb_doc_id"])


def test_company_details_override_wins_over_parsed_metadata(db_available, tmp_path):
    """A user-supplied Company Details field (company_name/cin/financial_year)
    must win over the workbook's own COMPANY_DETAILS sheet -- and the generated
    tb_doc_id must reflect the correction too, not just the display fields,
    since it's derived from company_name/financial_year (see build_tb_doc_id)."""
    if not db_available:
        pytest.skip("No reachable Postgres DB (settings.DB_*) -- skipping.")

    path = tmp_path / "template.xlsx"
    _write_template(path)

    with patch("modes.trial_balance.pipeline.agent.get_agent", return_value=_FakeAgent()):
        result = ingest_tb_to_live(
            tb_grouping_template_path=str(path),
            output_dir=str(tmp_path / "out"),
            company_name="Override Corp Ltd",
            cin="U00000MH2000GOI000001",
            financial_year="2030-31",
        )

    try:
        assert result["execution_status"] == "SUCCESS", result
        live_doc = fetch_live_document(result["tb_doc_id"])
        assert live_doc["company_name"] == "Override Corp Ltd"
        assert live_doc["cin"] == "U00000MH2000GOI000001"
        assert live_doc["financial_year"] == "2030-31"
    finally:
        delete_db_document(tb_doc_id=result["tb_doc_id"])


def test_company_details_override_applies_per_field_independently(db_available, tmp_path):
    """Only company_name is overridden here -- cin/financial_year must stay
    exactly as the workbook's own COMPANY_DETAILS sheet produced them."""
    if not db_available:
        pytest.skip("No reachable Postgres DB (settings.DB_*) -- skipping.")

    path = tmp_path / "template.xlsx"
    _write_template(path)

    with patch("modes.trial_balance.pipeline.agent.get_agent", return_value=_FakeAgent()):
        result = ingest_tb_to_live(
            tb_grouping_template_path=str(path),
            output_dir=str(tmp_path / "out"),
            company_name="Override Corp Ltd",
        )

    try:
        assert result["execution_status"] == "SUCCESS", result
        live_doc = fetch_live_document(result["tb_doc_id"])
        assert live_doc["company_name"] == "Override Corp Ltd"
        assert live_doc["financial_year"] == "FY2025-26"  # unchanged, no override given
    finally:
        delete_db_document(tb_doc_id=result["tb_doc_id"])


def test_persist_to_live_false_never_writes_to_live(db_available, tmp_path):
    """The query-analysis staging path (a future feature builds on this): the
    exact same classify/quality-gate chain runs and canonical_tb.parquet is
    still written, but nothing reaches live_document_table/live_tb_table --
    this is the direct behavioral contract that feature depends on."""
    if not db_available:
        pytest.skip("No reachable Postgres DB (settings.DB_*) -- skipping.")

    path = tmp_path / "template.xlsx"
    _write_template(path)

    with patch("modes.trial_balance.pipeline.agent.get_agent", return_value=_FakeAgent()):
        result = ingest_tb_to_live(
            tb_grouping_template_path=str(path),
            output_dir=str(tmp_path / "out"),
            persist_to_live=False,
        )

    assert result["execution_status"] == "SUCCESS", result
    assert result["persisted_to_live"] is False
    assert result["db_refs"] is None
    assert result["replaced_existing_document"] is None
    assert (tmp_path / "out" / "canonical_tb.parquet").exists()

    # The defining assertion -- nothing written to LIVE for this tb_doc_id.
    assert fetch_live_document(result["tb_doc_id"]) is None


def test_grand_total_row_never_reaches_canonical_tb(db_available, tmp_path):
    if not db_available:
        pytest.skip("No reachable Postgres DB (settings.DB_*) -- skipping.")

    path = tmp_path / "template.xlsx"
    _write_template(path, with_grand_total_row=True)

    with patch("modes.trial_balance.pipeline.agent.get_agent", return_value=_FakeAgent()):
        result = ingest_tb_to_live(
            tb_grouping_template_path=str(path),
            output_dir=str(tmp_path / "out"),
        )

    try:
        assert result["execution_status"] == "SUCCESS", result
        # Still exactly 1 real GL row -- the Grand Total row was excluded at
        # parse time, before the engine (and CASE_1's zero-LLM-touch routing)
        # ever saw it, so this stays CASE_1 rather than falling to CASE_2/3.
        assert result["quality_tier"] == "CASE_1"
        assert result["run_metrics"]["mapped_count"] == 1

        import polars as pl

        out_df = pl.read_parquet(tmp_path / "out" / "canonical_tb.parquet")
        assert out_df.height == 1
        assert out_df.to_dicts()[0]["gl_code"] == "E2E-1001"
    finally:
        delete_db_document(tb_doc_id=result["tb_doc_id"])


@pytest.mark.skipif(not _GURUGRAM_TB.exists() or not _GURUGRAM_GROUPING.exists(), reason="Gurugram real-file fixtures not present")
def test_erp_type_tally_wiring_persists_scope_and_classification(db_available, tmp_path):
    """TB Multi-ERP Adaptability upgrade, wired end-to-end: real Gurugram
    Tally pair -> ingest_tb_to_live(erp_type="tally") -> LIVE write ->
    tb_scope/tb_erp_classification actually persisted and readable back.
    The real proof this wiring works, not just that the pieces compose in
    isolation (already covered by test_erp_adapter_integration.py)."""
    if not db_available:
        pytest.skip("No reachable Postgres DB (settings.DB_*) -- skipping.")

    with patch("modes.trial_balance.pipeline.agent.get_agent", return_value=_FakeAgent()):
        result = ingest_tb_to_live(
            tb_grouping_template_path=str(_GURUGRAM_TB),
            grouping_file_path=str(_GURUGRAM_GROUPING),
            output_dir=str(tmp_path / "out"),
            standard="IND_AS",
            erp_type="tally",
            accept_data_quality_risk=True,
        )

    try:
        assert result["execution_status"] == "SUCCESS", result
        assert result["erp_type"] == "tally"
        # Real Phase 3 finding: this file genuinely has no Opening Balance column.
        assert any(f["code"] == "tally_missing_opening_or_movement_columns" for f in result["erp_findings"])

        tb_doc_id = result["tb_doc_id"]
        scope = fetch_tb_scope(tb_doc_id)
        assert scope is not None

        classification = fetch_tb_erp_classification(tb_doc_id)
        assert len(classification) > 0
        any_has_sub_head_1 = any(
            evidence.account_classification.get("sub_head_1") for evidence in classification.values()
        )
        assert any_has_sub_head_1
    finally:
        delete_db_document(tb_doc_id=result["tb_doc_id"])
        with db_cursor(dict_rows=False) as cur:
            cur.execute("DELETE FROM tb_scope WHERE tb_doc_id = %s", (result["tb_doc_id"],))
            cur.execute("DELETE FROM tb_erp_classification WHERE tb_doc_id = %s", (result["tb_doc_id"],))


@pytest.mark.skipif(not _BHAVINI_TB.exists() or not _BHAVINI_GROUPING.exists(), reason="Bhavini real-file fixtures not present")
def test_erp_type_oracle_wiring_persists_scope_and_classification(db_available, tmp_path):
    """Phase 4 (Oracle adapter), wired end-to-end: real Bhavini pair ->
    ingest_tb_to_live(erp_type="oracle") -> LIVE write -> tb_scope/
    tb_erp_classification actually persisted and readable back. Bhavini's
    classification comes entirely from its separate Grouping file (Oracle's
    TB sheet never carries it row-embedded, unlike Tally/SAP)."""
    if not db_available:
        pytest.skip("No reachable Postgres DB (settings.DB_*) -- skipping.")

    with patch("modes.trial_balance.pipeline.agent.get_agent", return_value=_FakeAgent()):
        result = ingest_tb_to_live(
            tb_grouping_template_path=str(_BHAVINI_TB),
            grouping_file_path=str(_BHAVINI_GROUPING),
            output_dir=str(tmp_path / "out"),
            standard="IND_AS",
            erp_type="oracle",
            accept_data_quality_risk=True,
        )

    try:
        assert result["execution_status"] == "SUCCESS", result
        assert result["erp_type"] == "oracle"
        # Real, verified-against-raw-cells finding: this file's grand total
        # genuinely doesn't balance (257/628 rows don't satisfy
        # opening+debit-credit=closing even in the raw file).
        assert any(f["code"] == "grand_total_imbalance" for f in result["erp_findings"])

        tb_doc_id = result["tb_doc_id"]
        scope = fetch_tb_scope(tb_doc_id)
        assert scope is not None

        classification = fetch_tb_erp_classification(tb_doc_id)
        assert len(classification) > 0
        any_has_main_head = any(
            evidence.account_classification.get("main_head") for evidence in classification.values()
        )
        assert any_has_main_head
    finally:
        delete_db_document(tb_doc_id=result["tb_doc_id"])
        with db_cursor(dict_rows=False) as cur:
            cur.execute("DELETE FROM tb_scope WHERE tb_doc_id = %s", (result["tb_doc_id"],))
            cur.execute("DELETE FROM tb_erp_classification WHERE tb_doc_id = %s", (result["tb_doc_id"],))

"""Tests for backend/tools/db_bridge/load_tb_from_db.py's custom_field_1/2/3
passthrough fix -- previously this tool dropped custom_field_1/2/3 via
`.select(CANONICAL_TB_COLUMNS)` even though repository.fetch_main_lines()
already fetches the full CANONICAL_TB_ALL_COLUMNS set from tb_table."""

import polars as pl
import pytest

from modes.trial_balance.pipeline.tools import CANONICAL_TB_ALL_COLUMNS, load_tb_from_db

_TEST_DOC_ID = "PYTEST_LOAD_TB_FROM_DB_DOC"


@pytest.fixture
def seeded_main_document(db_available):
    """Seeds one document_table row + two tb_table rows directly via SQL
    (test-only setup -- the application code itself never writes to MAIN),
    with the line-level custom_field_1 populated, then cleans up afterwards."""
    if not db_available:
        pytest.skip("No reachable Postgres DB (settings.DB_*) -- skipping.")

    from modes.trial_balance.pipeline.db import db_cursor

    with db_cursor(dict_rows=False) as cur:
        cur.execute(
            """INSERT INTO document_table
               (entity_id, entity_name, tb_doc_id, tb_doc_name, financial_year, erp_system)
               VALUES (%s, %s, %s, %s, %s, %s)""",
            ("PYTEST_ENTITY", "Pytest Entity", _TEST_DOC_ID, "Pytest TB", "FY2025-26", "SAP"),
        )
        cur.execute(
            """INSERT INTO tb_table
               (tb_doc_id, gl_code, gl_name, opening_balance, debit, credit, closing_balance,
                mapped_status, custom_field_1)
               VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)""",
            (_TEST_DOC_ID, "1001", "Cash in Hand", 1000, 500, 0, 1500, "MAPPED", "gl-custom-1"),
        )
        cur.execute(
            """INSERT INTO tb_table
               (tb_doc_id, gl_code, gl_name, opening_balance, debit, credit, closing_balance,
                mapped_status)
               VALUES (%s, %s, %s, %s, %s, %s, %s, %s)""",
            (_TEST_DOC_ID, "2001", "Trade Payable", -1000, 0, 500, -1500, "MAPPED"),
        )

    yield _TEST_DOC_ID

    with db_cursor(dict_rows=False) as cur:
        cur.execute("DELETE FROM tb_table WHERE tb_doc_id = %s", (_TEST_DOC_ID,))
        cur.execute("DELETE FROM document_table WHERE tb_doc_id = %s", (_TEST_DOC_ID,))


def test_load_tb_from_db_preserves_all_canonical_columns(seeded_main_document, tmp_path):
    result = load_tb_from_db(tb_doc_id=seeded_main_document, output_dir=str(tmp_path))
    assert result["execution_status"] == "SUCCESS"

    canonical_path = result["artifacts"][0]
    df = pl.read_parquet(canonical_path)

    assert set(df.columns) == set(CANONICAL_TB_ALL_COLUMNS)


def test_load_tb_from_db_custom_field_survives_when_present(seeded_main_document, tmp_path):
    result = load_tb_from_db(tb_doc_id=seeded_main_document, output_dir=str(tmp_path))
    df = pl.read_parquet(result["artifacts"][0])

    row = df.filter(pl.col("gl_code") == "1001").to_dicts()[0]
    assert row["custom_field_1"] == "gl-custom-1"


def test_load_tb_from_db_custom_field_null_when_absent(seeded_main_document, tmp_path):
    result = load_tb_from_db(tb_doc_id=seeded_main_document, output_dir=str(tmp_path))
    df = pl.read_parquet(result["artifacts"][0])

    row = df.filter(pl.col("gl_code") == "2001").to_dicts()[0]
    assert row["custom_field_1"] is None
    assert row["custom_field_2"] is None
    assert row["custom_field_3"] is None

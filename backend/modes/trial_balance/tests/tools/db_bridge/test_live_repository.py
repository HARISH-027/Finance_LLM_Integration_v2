"""Integration tests calling backend/db.py's upsert_live_document and
insert_live_lines_batch DIRECTLY, against the real Postgres LIVE staging tables.
Previously these were only exercised indirectly through persist_canonical_tb_to_live's
own tests -- this isolates each function's own behavior (delete-then-insert
idempotency, and insert_live_lines_batch's silent None-default for a row missing a
column). Skipped automatically if no DB is reachable (see conftest.py's db_available
fixture)."""

import pytest

from modes.trial_balance.pipeline.db import fetch_main_lines, insert_live_lines_batch, upsert_live_document


def _cleanup(tb_doc_id):
    from modes.trial_balance.pipeline.db import db_cursor

    with db_cursor(dict_rows=False) as cur:
        cur.execute("DELETE FROM live_tb_table WHERE tb_doc_id = %s", (tb_doc_id,))
        cur.execute("DELETE FROM live_document_table WHERE tb_doc_id = %s", (tb_doc_id,))


def _fetch_live_document(tb_doc_id):
    from modes.trial_balance.pipeline.db import db_cursor

    with db_cursor() as cur:
        cur.execute("SELECT * FROM live_document_table WHERE tb_doc_id = %s", (tb_doc_id,))
        row = cur.fetchone()
        return dict(row) if row else None


def _fetch_live_lines(tb_doc_id):
    from modes.trial_balance.pipeline.db import db_cursor
    from modes.trial_balance.pipeline.tools import CANONICAL_TB_ALL_COLUMNS

    with db_cursor() as cur:
        cur.execute(
            f"SELECT {', '.join(CANONICAL_TB_ALL_COLUMNS)} FROM live_tb_table WHERE tb_doc_id = %s ORDER BY gl_code",
            (tb_doc_id,),
        )
        return [dict(r) for r in cur.fetchall()]


@pytest.fixture
def doc_id(db_available):
    if not db_available:
        pytest.skip("No reachable Postgres DB for this test run.")
    tb_doc_id = "PYTEST_LIVE_REPO_DOC"
    yield tb_doc_id
    _cleanup(tb_doc_id)


def _doc_row(tb_doc_id, **overrides):
    row = {
        "entity_id": "E1", "entity_name": "Test Entity", "cin": "U00000TEST",
        "company_name": "Test Co", "fy_period_start": "2025-04-01", "fy_period_end": "2026-03-31",
        "tb_doc_id": tb_doc_id, "tb_doc_name": "test.xlsx", "statement_type": "SFS",
        "financial_year": "2025-2026", "has_grouping": True, "grouping_doc_id": tb_doc_id,
        "grouping_doc_name": "test.xlsx", "document_version": 1, "modification_dump": None,
    }
    row.update(overrides)
    return row


class TestUpsertLiveDocument:
    def test_writes_a_new_row(self, doc_id):
        upsert_live_document(_doc_row(doc_id))
        row = _fetch_live_document(doc_id)
        assert row is not None
        assert row["company_name"] == "Test Co"
        assert row["document_version"] == 1

    def test_is_delete_then_insert_not_accumulate_on_repeat_calls(self, doc_id):
        upsert_live_document(_doc_row(doc_id))
        upsert_live_document(_doc_row(doc_id, company_name="Test Co Renamed", document_version=2))
        from modes.trial_balance.pipeline.db import db_cursor

        with db_cursor() as cur:
            cur.execute("SELECT COUNT(*) AS n FROM live_document_table WHERE tb_doc_id = %s", (doc_id,))
            count = cur.fetchone()["n"]
        assert count == 1
        row = _fetch_live_document(doc_id)
        assert row["company_name"] == "Test Co Renamed"
        assert row["document_version"] == 2

    def test_custom_fields_default_to_null_when_not_supplied(self, doc_id):
        upsert_live_document(_doc_row(doc_id))
        row = _fetch_live_document(doc_id)
        assert row["erp_system"] is None
        assert row["currency"] is None
        assert row["custom_field_3"] is None

    def test_custom_fields_pass_through_when_supplied(self, doc_id):
        upsert_live_document(_doc_row(doc_id, erp_system="SAP"))
        row = _fetch_live_document(doc_id)
        assert row["erp_system"] == "SAP"

    def test_user_id_defaults_to_null_when_not_supplied(self, doc_id):
        upsert_live_document(_doc_row(doc_id))
        row = _fetch_live_document(doc_id)
        assert row["user_id"] is None

    def test_user_id_passes_through_when_supplied(self, doc_id):
        upsert_live_document(_doc_row(doc_id, user_id="44444444-4444-4444-4444-444444444444"))
        row = _fetch_live_document(doc_id)
        assert row["user_id"] == "44444444-4444-4444-4444-444444444444"


class TestFetchLiveDocument:
    """fetch_live_document() (backend/db.py's own function, not this file's local
    _fetch_live_document duplicate) is what router.py's _verify_document_access /
    delete_document use to check ownership -- worth its own direct coverage."""

    def test_returns_the_row_including_user_id(self, doc_id):
        from modes.trial_balance.pipeline.db import fetch_live_document

        upsert_live_document(_doc_row(doc_id, user_id="55555555-5555-5555-5555-555555555555"))
        row = fetch_live_document(doc_id)
        assert row is not None
        assert row["user_id"] == "55555555-5555-5555-5555-555555555555"

    def test_returns_none_for_an_unknown_doc_id(self, db_available):
        if not db_available:
            pytest.skip("No reachable Postgres DB for this test run.")
        from modes.trial_balance.pipeline.db import fetch_live_document

        assert fetch_live_document("PYTEST_NO_SUCH_DOC") is None


class TestInsertLiveLinesBatch:
    def test_writes_the_supplied_rows(self, doc_id):
        upsert_live_document(_doc_row(doc_id))
        rows = [
            {"tb_doc_id": doc_id, "gl_code": "1001", "gl_name": "Cash", "opening_balance": 100.0,
             "debit": 50.0, "credit": 0.0, "closing_balance": 150.0, "bs_pl": "BS",
             "sub_head_2": None, "sub_head_1": "Current Assets", "main_head": "Assets",
             "account_type": "Asset", "mapped_status": "MAPPED"},
            {"tb_doc_id": doc_id, "gl_code": "2001", "gl_name": "Trade Payables", "opening_balance": -200.0,
             "debit": 0.0, "credit": 30.0, "closing_balance": -230.0, "bs_pl": "BS",
             "sub_head_2": None, "sub_head_1": "Current Liabilities", "main_head": "Liabilities",
             "account_type": "Liability", "mapped_status": "MAPPED"},
        ]
        n = insert_live_lines_batch(doc_id, rows)
        assert n == 2
        stored = _fetch_live_lines(doc_id)
        assert [r["gl_code"] for r in stored] == ["1001", "2001"]
        assert stored[0]["gl_name"] == "Cash"

    def test_is_delete_then_insert_not_accumulate_on_repeat_calls(self, doc_id):
        upsert_live_document(_doc_row(doc_id))
        row = {"tb_doc_id": doc_id, "gl_code": "1001", "gl_name": "Cash", "opening_balance": 0.0,
               "debit": 0.0, "credit": 0.0, "closing_balance": 0.0, "bs_pl": "BS",
               "sub_head_2": None, "sub_head_1": "Current Assets", "main_head": "Assets",
               "account_type": "Asset", "mapped_status": "MAPPED"}
        insert_live_lines_batch(doc_id, [row])
        insert_live_lines_batch(doc_id, [row])
        assert len(_fetch_live_lines(doc_id)) == 1

    def test_empty_rows_list_writes_nothing_and_returns_zero(self, doc_id):
        upsert_live_document(_doc_row(doc_id))
        assert insert_live_lines_batch(doc_id, []) == 0
        assert _fetch_live_lines(doc_id) == []

    def test_a_row_missing_a_column_silently_defaults_that_column_to_null(self, doc_id):
        """Documents the real, non-obvious behavior found during the 2026-09 benchmark:
        insert_live_lines_batch builds its VALUES tuple via `r.get(c) for c in cols`, so a
        row dict missing an expected key does not raise -- it silently inserts NULL for
        that column instead. A schema-drift or typo in an upstream dict key would produce
        NULLs, not an error; this test exists so that behavior can't regress unnoticed."""
        incomplete_row = {"tb_doc_id": doc_id, "gl_code": "9999", "gl_name": "Incomplete Row"}
        insert_live_lines_batch(doc_id, [incomplete_row])
        stored = _fetch_live_lines(doc_id)
        assert len(stored) == 1
        assert stored[0]["gl_code"] == "9999"
        assert stored[0]["opening_balance"] is None
        assert stored[0]["closing_balance"] is None


class TestFetchMainLinesColumnSelection:
    def test_selects_exactly_canonical_tb_all_columns(self, doc_id):
        """fetch_main_lines reads from tb_table (MAIN, not LIVE) -- this only verifies the
        SQL executes without error for an unknown doc_id (empty result), since MAIN is
        read-only from this codebase and pytest has no fixture data to seed it with."""
        rows = fetch_main_lines("NO-SUCH-MAIN-DOC-ID")
        assert rows == []

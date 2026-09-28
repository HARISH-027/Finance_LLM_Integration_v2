"""Tests for backend/tools/tally_adapter.py -- the Tally ERP adapter
introduced in Phase 3 of the TB Multi-ERP Adaptability upgrade. Synthetic
tests exercise each helper in isolation; the real-fixture test confirms
behavior against the actual Gurugram Metro Rail Limited file, whose real
shape (Head/Sub-Head embedded on the TB sheet itself, no Opening Balance
column at all) is what motivated this module's design."""

from pathlib import Path

import pytest

from modes.trial_balance.pipeline.tools.input_header_detect import build_tb_rows_from_grid, locate_header
from modes.trial_balance.pipeline.tools.input_scenario_b import _load_workbook_sheets
from modes.trial_balance.pipeline.tools.tally_adapter import (
    PERIOD_LOCKING_NOTE,
    _detect_grouped_view,
    _detect_missing_opening_or_movement_columns,
    _detect_suspense_accounts,
    _extract_group_hierarchy,
    run_tally_adapter,
)
from modes.trial_balance.pipeline.tools.tb_erp_contracts import ClassificationEvidence, ERPType, ScopeType
from modes.trial_balance.pipeline.tools.tb_models import GroupingHint, TBRow

_TB_FILE = (
    Path(__file__).parents[2] / "documentation" / "ERP_system_standardized_files" / "Tally"
    / "Gurugram_Metro_Rail_Limited"
    / "Gurugram_Metro_Rail_Limited_Master_Trial_Balance_Head_Sub_CAG__TB25-26__be041db1.xlsx"
)


class TestExtractGroupHierarchy:
    def test_reads_row_embedded_columns(self):
        cols = {"main_head": 0, "sub_head_1": 1}
        row = ["Expenses", "Indirect Expenses"]
        assert _extract_group_hierarchy(row, cols, None) == ["Expenses", "Indirect Expenses"]

    def test_falls_back_to_hint_when_row_missing_field(self):
        cols = {"main_head": 0}
        row = ["Expenses"]
        hint = GroupingHint(gl_code="1001", known_fields={"sub_head_1": "Indirect Expenses"})
        assert _extract_group_hierarchy(row, cols, hint) == ["Expenses", "Indirect Expenses"]

    def test_no_columns_and_no_hint_returns_empty(self):
        assert _extract_group_hierarchy(None, {}, None) == []


class TestMissingOpeningOrMovementColumns:
    def test_flags_when_opening_and_movement_both_absent(self):
        grid = [["Particulars", "Closing Balance"], ["Cash", 100]]
        cols = {"gl_name": 0, "closing": 1}
        finding = _detect_missing_opening_or_movement_columns(grid, 0, cols)
        assert finding is not None
        assert finding.code == "tally_missing_opening_or_movement_columns"
        assert "opening balance" in finding.message
        assert "debit/credit movement" in finding.message

    def test_no_finding_when_opening_and_movement_present(self):
        grid = [["GL Code", "Opening Balance", "Debit", "Credit", "Closing Balance"], ["1001", 0, 100, 0, 100]]
        cols = {"gl_code": 0, "opening": 1, "debit": 2, "credit": 3, "closing": 4}
        assert _detect_missing_opening_or_movement_columns(grid, 0, cols) is None


class TestGroupedViewDetection:
    def test_flags_high_overlap_between_ledger_names_and_group_labels(self):
        grid = [
            ["Particulars", "Main Head"],
            ["Expenses", "Expenses"],
            ["Assets", "Assets"],
            ["Liabilities", "Liabilities"],
        ]
        cols = {"gl_name": 0, "main_head": 1}
        tb_rows = [
            TBRow(gl_code="Expenses", gl_name="Expenses", opening=0, debit=0, credit=0, closing=0),
            TBRow(gl_code="Assets", gl_name="Assets", opening=0, debit=0, credit=0, closing=0),
        ]
        finding = _detect_grouped_view(tb_rows, grid, 0, cols)
        assert finding is not None
        assert finding.code == "tally_possible_grouped_view"

    def test_no_finding_for_normal_ledger_wise_rows(self):
        grid = [
            ["Particulars", "Main Head"],
            ["Salaries and Wages", "Indirect Expenses"],
            ["Cash in Hand", "Current Assets"],
        ]
        cols = {"gl_name": 0, "main_head": 1}
        tb_rows = [
            TBRow(gl_code="Salaries and Wages", gl_name="Salaries and Wages", opening=0, debit=0, credit=0, closing=0),
            TBRow(gl_code="Cash in Hand", gl_name="Cash in Hand", opening=0, debit=0, credit=0, closing=0),
        ]
        assert _detect_grouped_view(tb_rows, grid, 0, cols) is None


class TestSuspenseAccountDetection:
    def test_flags_suspense_named_account(self):
        tb_rows = [TBRow(gl_code="9999", gl_name="Suspense Account", opening=0, debit=0, credit=0, closing=500)]
        classification = {"9999": ClassificationEvidence(erp_type=ERPType.TALLY, account_identity={"gl_name": "Suspense Account"})}
        findings = _detect_suspense_accounts(tb_rows, classification)
        assert len(findings) == 1
        assert findings[0].code == "tally_suspense_or_misc_account"
        assert findings[0].severity == "audit_risk"

    def test_no_flag_for_ordinary_account(self):
        tb_rows = [TBRow(gl_code="1001", gl_name="Freehold Land", opening=0, debit=1000, credit=0, closing=1000)]
        classification = {"1001": ClassificationEvidence(erp_type=ERPType.TALLY, account_identity={"gl_name": "Freehold Land"})}
        assert _detect_suspense_accounts(tb_rows, classification) == []


@pytest.mark.skipif(not _TB_FILE.exists(), reason="Gurugram real-file fixture not present")
def test_gurugram_tb_file_group_hierarchy_and_missing_opening_finding():
    sheets = _load_workbook_sheets(_TB_FILE)
    grid = sheets["TB25-26"]
    header_row, cols = locate_header(grid)
    tb_rows = build_tb_rows_from_grid(grid, header_row, cols)

    result = run_tally_adapter(_TB_FILE, grid, header_row, cols, tb_rows, sheet_name="TB25-26")

    # Group hierarchy read directly from the TB sheet's own Head/Sub-Head columns.
    row = next(r for r in tb_rows if r.gl_name == "Equity Share Capital-GoI")
    evidence = result.classification_by_gl_code[row.gl_code]
    assert evidence.account_classification["erp_group"]
    # Phase 7: canonical taxonomy field names, not just the Tally-specific
    # erp_group/erp_group_path keys.
    assert evidence.account_classification["sub_head_1"]

    # This real file genuinely has no Opening Balance column at all --
    # the missing-column finding must fire (section 1.5.3).
    assert any(f.code == "tally_missing_opening_or_movement_columns" for f in result.findings)

    # Real file is genuinely ledger-wise -- no false grouped-view flag.
    assert not any(f.code == "tally_possible_grouped_view" for f in result.findings)

    # Period-locking note always present.
    assert PERIOD_LOCKING_NOTE in result.findings

    assert len(result.source_metadata) == len(tb_rows)
    assert result.scope.scope_type == ScopeType.FULL_LEDGER

"""Tests for backend/tools/tally_adapter.py -- the Tally ERP adapter
introduced in Phase 3 of the TB Multi-ERP Adaptability upgrade. Synthetic
tests exercise each helper in isolation; the real-fixture test confirms
behavior against the actual Gurugram Metro Rail Limited file, whose real
shape (Head/Sub-Head embedded on the TB sheet itself, no Opening Balance
column at all) is what motivated this module's design."""

from pathlib import Path

import openpyxl
import pytest

from modes.trial_balance.pipeline.tools.input_header_detect import build_tb_rows_from_grid, locate_header
from modes.trial_balance.pipeline.tools.input_scenario_b import _load_workbook_sheets
from modes.trial_balance.pipeline.tools.tally_adapter import (
    PERIOD_LOCKING_NOTE,
    _detect_closing_only_no_movement,
    _detect_dr_cr_suffix_loss,
    _detect_grouped_view,
    _detect_missing_opening_or_movement_columns,
    _detect_near_duplicate_ledger_names,
    _detect_opening_balance_difference_row,
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

# Real client fixtures (docs/TBS-from-ERP/Tally/), reachable in this
# checkout -- unlike _TB_FILE above (documentation/ERP_system_standardized_
# files/ doesn't exist anywhere in the repo, so that whole convention is
# always skipped). These exercise the native-export shape (no Head/Sub-Head
# columns, no gl_code) that motivated tally_structural_hierarchy.py.
_MUMBAI_AVIATION_TB = (
    Path(__file__).parents[2] / "docs" / "TBS-from-ERP" / "Tally"
    / "Mumbai Aviation Fuel Farm Facility Private Limited. Trial Balance 21 to 26.xlsx"
)
_PETRONET_VK_TB = (
    Path(__file__).parents[2] / "docs" / "TBS-from-ERP" / "Tally" / "Petronet VK"
    / "Petronet VK  TrialBal FY 24-25.xlsx"
)


def _run_adapter_on_real_file(path: Path, sheet_name: str):
    import openpyxl as _openpyxl

    wb = _openpyxl.load_workbook(path, read_only=True, data_only=True)
    ws = wb[sheet_name]
    grid = [[c.value for c in row] for row in ws.iter_rows()]
    wb.close()
    header_row, cols = locate_header(grid, max_scan=15)
    tb_rows = build_tb_rows_from_grid(grid, header_row, cols)
    result = run_tally_adapter(path, grid, header_row, cols, tb_rows, sheet_name=sheet_name)
    return tb_rows, result


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
        # Both opening AND debit/credit movement missing at once now
        # delegates to the more specific tally_closing_only_no_opening_or_
        # movement finding (see TestClosingOnlyNoMovement below) rather
        # than the generic combined message.
        grid = [["Particulars", "Closing Balance"], ["Cash", 100]]
        cols = {"gl_name": 0, "closing": 1}
        finding = _detect_missing_opening_or_movement_columns(grid, 0, cols)
        assert finding is not None
        assert finding.code == "tally_closing_only_no_opening_or_movement"

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


class TestClosingOnlyNoMovement:
    def test_flags_when_both_opening_and_movement_absent(self):
        finding = _detect_closing_only_no_movement(["opening balance", "debit/credit movement"])
        assert finding is not None
        assert finding.code == "tally_closing_only_no_opening_or_movement"

    def test_no_finding_when_only_one_field_missing(self):
        assert _detect_closing_only_no_movement(["opening balance"]) is None
        assert _detect_closing_only_no_movement(["debit/credit movement"]) is None

    def test_no_finding_when_nothing_missing(self):
        assert _detect_closing_only_no_movement([]) is None


class TestOpeningBalanceDifferenceDetection:
    def test_flags_difference_in_opening_balances_row(self):
        tb_rows = [
            TBRow(gl_code="Cash", gl_name="Cash", opening=100, debit=0, credit=0, closing=100),
            TBRow(
                gl_code="Difference in Opening Balances", gl_name="Difference in Opening Balances",
                opening=0, debit=0, credit=0, closing=250.0,
            ),
        ]
        finding = _detect_opening_balance_difference_row(tb_rows)
        assert finding is not None
        assert finding.severity == "audit_risk"
        assert finding.code == "tally_opening_balance_difference"

    def test_no_finding_when_absent(self):
        tb_rows = [TBRow(gl_code="Cash", gl_name="Cash", opening=100, debit=0, credit=0, closing=100)]
        assert _detect_opening_balance_difference_row(tb_rows) is None


class TestNearDuplicateLedgerNames:
    def test_flags_whitespace_and_punctuation_variants(self):
        tb_rows = [
            TBRow(gl_code="ABC Traders", gl_name="ABC Traders", opening=0, debit=0, credit=0, closing=100),
            TBRow(gl_code="ABC Traders ", gl_name="ABC Traders ", opening=0, debit=0, credit=0, closing=50),
            TBRow(gl_code="A.B.C. Traders", gl_name="A.B.C. Traders", opening=0, debit=0, credit=0, closing=25),
        ]
        findings = _detect_near_duplicate_ledger_names(tb_rows)
        assert len(findings) == 1
        assert findings[0].code == "tally_near_duplicate_ledger_name"
        assert findings[0].severity == "warning"

    def test_no_finding_for_distinct_names(self):
        tb_rows = [
            TBRow(gl_code="Cash", gl_name="Cash", opening=0, debit=0, credit=0, closing=100),
            TBRow(gl_code="Bank", gl_name="Bank", opening=0, debit=0, credit=0, closing=200),
        ]
        assert _detect_near_duplicate_ledger_names(tb_rows) == []


class TestDrCrSuffixLossDetection:
    def test_flags_unparseable_closing_cell(self):
        grid = [["Particulars", "Closing Balance"], ["Cash", "Cr"], ["Bank", 100.0]]
        cols = {"gl_name": 0, "closing": 1}
        finding = _detect_dr_cr_suffix_loss(grid, 0, cols)
        assert finding is not None
        assert finding.code == "tally_dr_cr_suffix_loss"

    def test_no_finding_when_all_cells_parse_cleanly(self):
        grid = [["Particulars", "Closing Balance"], ["Cash", "1,000 Dr"], ["Bank", 100.0]]
        cols = {"gl_name": 0, "closing": 1}
        assert _detect_dr_cr_suffix_loss(grid, 0, cols) is None

    def test_no_closing_column_no_finding(self):
        grid = [["Particulars"], ["Cash"]]
        cols = {"gl_name": 0}
        assert _detect_dr_cr_suffix_loss(grid, 0, cols) is None


class TestRunTallyAdapterStructuralFallback:
    """End-to-end: a synthetic bold-formatted native-shape workbook (no
    Head/Sub-Head columns at all) should still get main_head/bs_pl/
    account_type via the structural extractor + primary-group lookup."""

    def test_structural_main_head_and_primary_group_lookup_populate_classification(self, tmp_path):
        path = tmp_path / "synthetic_tally_native.xlsx"
        wb = openpyxl.Workbook()
        ws = wb.active
        ws.title = "TB"
        ws.append(["Particulars", "Opening", "Debit", "Credit", "Closing"])
        ws.append(["Sundry Debtors", 100.0, 60.0, 0.0, 160.0])
        ws["A2"].font = openpyxl.styles.Font(bold=True)
        ws.append(["ABC Traders", 40.0, 60.0, 0.0, 100.0])
        ws.append(["XYZ Traders", 60.0, 0.0, 0.0, 60.0])
        wb.save(path)

        grid = [[c.value for c in row] for row in wb["TB"].iter_rows()]
        header_row, cols = locate_header(grid)
        tb_rows = build_tb_rows_from_grid(grid, header_row, cols)

        result = run_tally_adapter(path, grid, header_row, cols, tb_rows, sheet_name="TB")

        abc = next(r for r in tb_rows if r.gl_name == "ABC Traders")
        evidence = result.classification_by_gl_code[abc.gl_code]
        assert evidence.account_classification["main_head"] == "Sundry Debtors"
        # Sundry Debtors is not one of Tally's 15 shipped PRIMARY groups
        # (it's a shipped SUB-group) -- classify_primary_group correctly
        # returns None for it, so bs_pl/account_type are absent here,
        # left for the general taxonomy resolver/LLM path downstream.
        assert "bs_pl" not in evidence.account_classification


@pytest.mark.skipif(not _MUMBAI_AVIATION_TB.exists(), reason="Mumbai Aviation Fuel Farm real fixture not present")
class TestStructuralHierarchyRealFixtureMumbaiAviation:
    """Native Tally export, no Head/Sub-Head columns, no gl_code, 3-row
    split header (Particulars / Opening-Transactions-Closing / Balance-
    Debit-Credit-Balance) -- exercises input_header_detect.py's recovery
    fixes together with tally_structural_hierarchy.py's parser."""

    def test_capital_account_reconciles_reserves_and_shareholders(self):
        tb_rows, result = _run_adapter_on_real_file(
            _MUMBAI_AVIATION_TB, "Trial Balance 21-22",
        )
        pl_account = next(r for r in tb_rows if r.gl_name == "Profit & Loss Account (Retained Earnings)")
        evidence = result.classification_by_gl_code[pl_account.gl_code]
        # Reserves & Surplus resolves as a nested subtotal via the known
        # sub-group vocabulary (not bold in the real file) -- P&L Account's
        # main_head is Capital Account (the TOP-level ancestor), never the
        # intermediate Reserves & Surplus.
        assert evidence.account_classification["main_head"] == "Capital Account"
        assert evidence.account_classification["erp_group_path"] == "Capital Account -> Reserves & Surplus"

        bpcl = next(r for r in tb_rows if r.gl_name == "BPCL - Shareholder")
        bpcl_evidence = result.classification_by_gl_code[bpcl.gl_code]
        assert bpcl_evidence.account_classification["main_head"] == "Capital Account"
        # Flat leaf sibling, not nested under anything else despite
        # sharing an identical value with HPCL/IOCL/MIAL.
        assert bpcl_evidence.account_classification["erp_group_path"] == "Capital Account"

    def test_resolution_rate_is_meaningfully_above_zero(self):
        tb_rows, result = _run_adapter_on_real_file(
            _MUMBAI_AVIATION_TB, "Trial Balance 21-22",
        )
        with_main_head = sum(
            1 for e in result.classification_by_gl_code.values() if e.account_classification.get("main_head")
        )
        # Real ceiling is well below 100% for this file -- some contra
        # ledgers (e.g. Unamortised Ancillary Cost) carry no sign
        # information in this export, so their parent group's total
        # doesn't reconcile against a naive sum. Asserting a floor, not an
        # exact figure, so this doesn't become brittle against minor
        # future algorithm refinements.
        assert with_main_head / len(tb_rows) > 0.30


@pytest.mark.skipif(not _PETRONET_VK_TB.exists(), reason="Petronet VK real fixture not present")
class TestStructuralHierarchyRealFixturePetronetVK:
    def test_non_current_borrowings_group_resolved(self):
        tb_rows, result = _run_adapter_on_real_file(_PETRONET_VK_TB, "Trial Balance")
        secured_loans = next(r for r in tb_rows if r.gl_name == "IOCL Long Term Loan for OTS")
        evidence = result.classification_by_gl_code[secured_loans.gl_code]
        assert evidence.account_classification["main_head"] == "Non Current Borrowings"

    def test_resolution_rate_is_meaningfully_above_zero(self):
        tb_rows, result = _run_adapter_on_real_file(_PETRONET_VK_TB, "Trial Balance")
        with_main_head = sum(
            1 for e in result.classification_by_gl_code.values() if e.account_classification.get("main_head")
        )
        assert with_main_head / len(tb_rows) > 0.40


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

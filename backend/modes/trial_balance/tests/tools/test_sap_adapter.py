"""Tests for backend/tools/sap_adapter.py -- the SAP ERP adapter introduced
in Phase 5 of the TB Multi-ERP Adaptability upgrade. Synthetic tests
exercise each helper in isolation; real-fixture tests confirm behavior
against both observed real shapes: classification embedded directly on the
TB sheet (Terra_Clean) and a TB-only sheet with no embedded classification
at all (IOC_Global)."""

from pathlib import Path

import pytest

from modes.trial_balance.pipeline.tools.input_header_detect import build_tb_rows_from_grid, locate_header
from modes.trial_balance.pipeline.tools.input_scenario_b import _load_workbook_sheets
from modes.trial_balance.pipeline.tools.sap_adapter import (
    _detect_extended_fiscal_periods,
    _detect_multiple_company_codes,
    _detect_not_assigned_accounts,
    _scan_sap_field_columns,
    run_sap_adapter,
)
from modes.trial_balance.pipeline.tools.tb_erp_contracts import ClassificationEvidence, ERPType, ScopeType
from modes.trial_balance.pipeline.tools.tb_models import TBRow

_FIXTURES_ROOT = Path(__file__).parents[2] / "documentation" / "ERP_system_standardized_files" / "SAP"
_TERRA_CLEAN_TB = (
    _FIXTURES_ROOT / "Oil_and_Gas" / "Terra_Clean"
    / "Terra_Clean_Master_Trial_Balance_and_GL_Grouping__F.Y_24-25__1607165b.xlsx"
)
_IOC_GLOBAL_TB = (
    _FIXTURES_ROOT / "IOC_Global"
    / "IOC_Global_Master_Trial_Balance_capital_management__2024-25__dfcbfb5d.xlsx"
)


class TestScanSapFieldColumns:
    def test_detects_all_known_sap_fields_when_present(self):
        header = [
            "Company Code", "Business Area", "Ledger", "Account Group", "Profit Centre",
            "Segment", "Financial Statement Version", "Period", "Currency",
        ]
        found = _scan_sap_field_columns(header)
        assert found == {
            "company_code": 0, "business_area": 1, "ledger": 2, "account_group": 3,
            "profit_center": 4, "segment": 5, "financial_statement_version": 6,
            "period": 7, "currency": 8,
        }

    def test_returns_empty_when_none_present(self):
        assert _scan_sap_field_columns(["GL Code", "GL Name", "Closing Balance"]) == {}


class TestNotAssignedAccountDetection:
    def test_flags_not_assigned_account(self):
        tb_rows = [TBRow(gl_code="9999", gl_name="Not Assigned", opening=0, debit=0, credit=0, closing=100)]
        classification = {"9999": ClassificationEvidence(erp_type=ERPType.SAP, account_identity={"gl_code": "9999", "gl_name": "Not Assigned"})}
        findings = _detect_not_assigned_accounts(tb_rows, classification)
        assert len(findings) == 1
        assert findings[0].code == "sap_not_assigned_account"
        assert findings[0].severity == "audit_risk"

    def test_no_flag_for_ordinary_account(self):
        tb_rows = [TBRow(gl_code="1001", gl_name="Cash in Hand", opening=0, debit=100, credit=0, closing=100)]
        classification = {"1001": ClassificationEvidence(erp_type=ERPType.SAP, account_identity={"gl_code": "1001", "gl_name": "Cash in Hand"})}
        assert _detect_not_assigned_accounts(tb_rows, classification) == []


class TestExtendedFiscalPeriodDetection:
    def test_flags_when_period_14_present(self):
        grid = [["GL Code", "Period"], ["1001", "12"], ["1002", "14"]]
        sap_cols = {"period": 1}
        finding = _detect_extended_fiscal_periods(grid, 0, sap_cols)
        assert finding is not None
        assert finding.code == "sap_extended_fiscal_periods_present"

    def test_no_finding_when_only_periods_1_to_12_present(self):
        grid = [["GL Code", "Period"], ["1001", "1"], ["1002", "12"]]
        sap_cols = {"period": 1}
        assert _detect_extended_fiscal_periods(grid, 0, sap_cols) is None

    def test_no_finding_when_no_period_column(self):
        grid = [["GL Code", "Closing Balance"], ["1001", 100]]
        assert _detect_extended_fiscal_periods(grid, 0, {}) is None


class TestMultipleCompanyCodeDetection:
    def test_flags_two_distinct_company_codes(self):
        grid = [["Company Code", "GL Code"], ["1000", "1001"], ["2000", "1001"]]
        finding = _detect_multiple_company_codes(grid, 0, {"company_code": 0})
        assert finding is not None
        assert finding.code == "sap_multiple_company_codes"

    def test_no_finding_for_single_company_code(self):
        grid = [["Company Code", "GL Code"], ["1000", "1001"], ["1000", "1002"]]
        assert _detect_multiple_company_codes(grid, 0, {"company_code": 0}) is None

    def test_no_finding_when_no_company_code_column(self):
        grid = [["GL Code"], ["1001"]]
        assert _detect_multiple_company_codes(grid, 0, {}) is None


@pytest.mark.skipif(not _TERRA_CLEAN_TB.exists(), reason="Terra_Clean real-file fixture not present")
def test_terra_clean_embedded_classification_read_directly_from_tb_sheet():
    sheets = _load_workbook_sheets(_TERRA_CLEAN_TB)
    sheet_name = "F.Y_24-25"
    grid = sheets[sheet_name]
    header_row, cols = locate_header(grid)
    tb_rows = build_tb_rows_from_grid(grid, header_row, cols)

    result = run_sap_adapter(_TERRA_CLEAN_TB, grid, header_row, cols, tb_rows, sheet_name=sheet_name)

    evidence = result.classification_by_gl_code["1111100000"]
    assert evidence.account_classification["bs_pl"] == "BS"
    # Phase 7: main_head/sub_head_1/sub_head_2 now captured too, not just bs_pl/account_type.
    assert evidence.account_classification["main_head"] == "Equity Share Capital"
    assert evidence.business_dimensions["business_area"] == "DL01"
    assert result.scope.company_code == "1000"
    assert result.scope.scope_type == ScopeType.FULL_LEDGER
    assert len(result.source_metadata) == len(tb_rows)


@pytest.mark.skipif(not _IOC_GLOBAL_TB.exists(), reason="IOC_Global real-file fixture not present")
def test_iocglobal_tb_only_file_has_partial_embedded_classification():
    """This TB-only sheet carries a Business Area column (a real, present
    business dimension) but no BS/PL, Account Type, Account Group, or FS
    Version -- those live only in the companion Grouping file, which this
    test does not supply. Confirms the adapter reports exactly what's
    actually on the sheet, not more and not less."""
    sheets = _load_workbook_sheets(_IOC_GLOBAL_TB)
    sheet_name = "2024-25"
    grid = sheets[sheet_name]
    header_row, cols = locate_header(grid)
    tb_rows = build_tb_rows_from_grid(grid, header_row, cols)

    result = run_sap_adapter(_IOC_GLOBAL_TB, grid, header_row, cols, tb_rows, sheet_name=sheet_name)

    assert len(tb_rows) > 0
    assert result.scope.company_code is None
    for evidence in result.classification_by_gl_code.values():
        assert evidence.account_classification == {}
        assert evidence.financial_statement == {}
        assert set(evidence.business_dimensions.keys()) <= {"business_area"}

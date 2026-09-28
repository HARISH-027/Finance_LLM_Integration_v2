"""Tests for backend/tools/scope_aware_quality_gate.py -- the section-17
Validation Severity Model, actually implemented in Phase 6 of the TB
Multi-ERP Adaptability upgrade. Synthetic tests exercise each check branch
in isolation; real-fixture tests confirm behavior against real SAP data
already produced by sap_adapter.py (Phase 5) -- no new parsing logic
needed to compose them."""

from pathlib import Path

import pytest

from modes.trial_balance.pipeline.tools.input_header_detect import build_tb_rows_from_grid, locate_header
from modes.trial_balance.pipeline.tools.input_scenario_b import _load_workbook_sheets
from modes.trial_balance.pipeline.tools.sap_adapter import run_sap_adapter
from modes.trial_balance.pipeline.tools.scope_aware_quality_gate import (
    check_opening_against_prior_year,
    check_pl_opening,
    run_scope_aware_quality_gate,
)
from modes.trial_balance.pipeline.tools.tb_erp_contracts import ClassificationEvidence, ERPType, ScopeMetadata, ScopeType
from modes.trial_balance.pipeline.tools.tb_models import TBRow

_FIXTURES_ROOT = Path(__file__).parents[2] / "documentation" / "ERP_system_standardized_files" / "SAP"
_TERRA_CLEAN_TB = (
    _FIXTURES_ROOT / "Oil_and_Gas" / "Terra_Clean"
    / "Terra_Clean_Master_Trial_Balance_and_GL_Grouping__F.Y_24-25__1607165b.xlsx"
)
_IOC_GLOBAL_DIR = _FIXTURES_ROOT / "IOC_Global"
_IOC_GLOBAL_2024 = _IOC_GLOBAL_DIR / "IOC_Global_Master_Trial_Balance_capital_management__2024-25__dfcbfb5d.xlsx"
_IOC_GLOBAL_2025 = _IOC_GLOBAL_DIR / "IOC_Global_Master_Trial_Balance_capital_management__2025-26__66fd8eba.xlsx"


def _row(gl_code, opening=0.0, debit=0.0, credit=0.0, closing=0.0, gl_name="Test"):
    return TBRow(gl_code=gl_code, gl_name=gl_name, opening=opening, debit=debit, credit=credit, closing=closing)


class TestGrandTotalEvaluation:
    def test_full_ledger_imbalance_is_audit_risk(self):
        tb_rows = [_row("1", debit=1000, credit=0), _row("2", debit=0, credit=500)]
        scope = ScopeMetadata(scope_type=ScopeType.FULL_LEDGER)
        report = run_scope_aware_quality_gate(tb_rows, scope, {})
        codes = [f.code for f in report.findings]
        assert "grand_total_imbalance" in codes
        assert next(f for f in report.findings if f.code == "grand_total_imbalance").severity == "AUDIT_RISK"

    def test_unknown_scope_imbalance_is_confirmation_required(self):
        tb_rows = [_row("1", debit=1000, credit=0), _row("2", debit=0, credit=500)]
        scope = ScopeMetadata(scope_type=ScopeType.UNKNOWN)
        report = run_scope_aware_quality_gate(tb_rows, scope, {})
        finding = next(f for f in report.findings if f.code == "grand_total_unknown_scope")
        assert finding.severity == "CONFIRMATION_REQUIRED"

    def test_slice_scope_imbalance_produces_no_finding(self):
        tb_rows = [_row("1", debit=1000, credit=0), _row("2", debit=0, credit=500)]
        scope = ScopeMetadata(scope_type=ScopeType.PROFIT_CENTER_SLICE)
        report = run_scope_aware_quality_gate(tb_rows, scope, {})
        assert not any(f.code.startswith("grand_total") for f in report.findings)

    def test_within_tolerance_produces_no_finding(self):
        tb_rows = [_row("1", debit=1000.001, credit=1000.0)]
        scope = ScopeMetadata(scope_type=ScopeType.FULL_LEDGER)
        report = run_scope_aware_quality_gate(tb_rows, scope, {})
        assert not any(f.code.startswith("grand_total") for f in report.findings)


class TestPlOpeningCheck:
    def test_flags_nonzero_opening_pl_account(self):
        tb_rows = [_row("1", opening=500.0, gl_name="Salaries")]
        classification = {"1": ClassificationEvidence(erp_type=ERPType.SAP, account_classification={"bs_pl": "PL"})}
        findings = check_pl_opening(tb_rows, classification)
        assert len(findings) == 1
        assert findings[0].code == "pl_account_nonzero_opening"
        assert findings[0].severity == "WARNING"

    def test_no_flag_for_bs_account(self):
        tb_rows = [_row("1", opening=500.0, gl_name="Cash")]
        classification = {"1": ClassificationEvidence(erp_type=ERPType.SAP, account_classification={"bs_pl": "BS"})}
        assert check_pl_opening(tb_rows, classification) == []

    def test_no_flag_when_no_classification_available(self):
        tb_rows = [_row("1", opening=500.0)]
        assert check_pl_opening(tb_rows, {}) == []


class TestOpeningAgainstPriorYear:
    def test_flags_real_mismatch(self):
        prior = [_row("1", closing=1000.0)]
        current = [_row("1", opening=900.0)]
        findings = check_opening_against_prior_year(current, prior)
        assert len(findings) == 1
        assert findings[0].code == "opening_prior_year_closing_mismatch"
        assert findings[0].severity == "AUDIT_RISK"

    def test_no_flag_when_matching(self):
        prior = [_row("1", closing=1000.0)]
        current = [_row("1", opening=1000.0)]
        assert check_opening_against_prior_year(current, prior) == []

    def test_no_flag_when_gl_code_only_in_one_year(self):
        prior = [_row("1", closing=1000.0)]
        current = [_row("2", opening=500.0)]
        assert check_opening_against_prior_year(current, prior) == []


def _parse_real_sap_fixture(path: Path, sheet_name: str):
    sheets = _load_workbook_sheets(path)
    grid = sheets[sheet_name]
    header_row, cols = locate_header(grid)
    tb_rows = build_tb_rows_from_grid(grid, header_row, cols)
    return grid, header_row, cols, tb_rows


@pytest.mark.skipif(not _IOC_GLOBAL_2024.exists() or not _IOC_GLOBAL_2025.exists(), reason="IOC_Global real-file fixtures not present")
def test_iocglobal_two_years_opening_continuity_all_match():
    _, _, _, prior_rows = _parse_real_sap_fixture(_IOC_GLOBAL_2024, "2024-25")
    _, _, _, current_rows = _parse_real_sap_fixture(_IOC_GLOBAL_2025, "2025-26")

    findings = check_opening_against_prior_year(current_rows, prior_rows)
    assert findings == []


@pytest.mark.skipif(not _TERRA_CLEAN_TB.exists(), reason="Terra_Clean real-file fixture not present")
def test_terra_clean_pl_accounts_have_zero_opening_no_false_positive():
    grid, header_row, cols, tb_rows = _parse_real_sap_fixture(_TERRA_CLEAN_TB, "F.Y_24-25")
    result = run_sap_adapter(_TERRA_CLEAN_TB, grid, header_row, cols, tb_rows, sheet_name="F.Y_24-25")

    findings = check_pl_opening(tb_rows, result.classification_by_gl_code)
    assert findings == []


@pytest.mark.skipif(not _TERRA_CLEAN_TB.exists(), reason="Terra_Clean real-file fixture not present")
def test_terra_clean_grand_total_balances_as_full_ledger_scope():
    grid, header_row, cols, tb_rows = _parse_real_sap_fixture(_TERRA_CLEAN_TB, "F.Y_24-25")
    result = run_sap_adapter(_TERRA_CLEAN_TB, grid, header_row, cols, tb_rows, sheet_name="F.Y_24-25")

    assert result.scope.scope_type == ScopeType.FULL_LEDGER
    report = run_scope_aware_quality_gate(tb_rows, result.scope, result.classification_by_gl_code)
    assert not any(f.code.startswith("grand_total") for f in report.findings)

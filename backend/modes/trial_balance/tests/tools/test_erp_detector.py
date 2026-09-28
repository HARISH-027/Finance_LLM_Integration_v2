"""Tests for backend/tools/erp_detector.py -- deterministic ERP
identification introduced in Phase 2 of the TB Multi-ERP Adaptability
upgrade. Synthetic tests exercise each tier in isolation; real-fixture
tests confirm the honest, documented outcome for the current batch (see
design doc section 1.6.1): pre-standardized real files carry no raw ERP
fingerprints, so only the path/filename-convention tier actually fires."""

from pathlib import Path

import pytest

from modes.trial_balance.pipeline.tools.erp_detector import detect_erp_type
from modes.trial_balance.pipeline.tools.input_scenario_b import _load_workbook_sheets
from modes.trial_balance.pipeline.tools.tb_erp_contracts import ERPType

_FIXTURES_ROOT = Path(__file__).parents[2] / "documentation" / "ERP_system_standardized_files"

_SAP_IOC_GLOBAL_TB = (
    _FIXTURES_ROOT / "SAP" / "IOC_Global"
    / "IOC_Global_Master_Trial_Balance_capital_management__2024-25__dfcbfb5d.xlsx"
)
_TALLY_GURUGRAM_TB = (
    _FIXTURES_ROOT / "Tally" / "Gurugram_Metro_Rail_Limited"
    / "Gurugram_Metro_Rail_Limited_Master_Trial_Balance_Head_Sub_CAG__TB25-26__be041db1.xlsx"
)
_CEIL_TB = (
    _FIXTURES_ROOT / "In_house_or_other" / "CEIL"
    / "CEIL_Master_Trial_Balance_and_GL_Grouping__2024-25__1c9b1573.xlsx"
)


class TestExplicitOverride:
    def test_explicit_selection_wins_over_content(self):
        sheets = {"Sheet1": [["BUKRS", "HKONT"], ["1000", "431020"]]}
        result = detect_erp_type(Path("anything.xlsx"), sheets, explicit_erp_type=ERPType.TALLY)
        assert result.erp_type == ERPType.TALLY
        assert result.confidence == 1.0
        assert result.evidence == ["explicit_selection: tally"]


class TestStructuralFingerprintTier:
    def test_sap_fingerprints_detected(self):
        sheets = {"Sheet1": [["BUKRS", "HKONT", "GJAHR"], ["1000", "431020", "2025"]]}
        result = detect_erp_type(Path("export.xlsx"), sheets)
        assert result.erp_type == ERPType.SAP
        assert result.confidence == 0.95

    def test_oracle_fingerprints_detected(self):
        sheets = {"Sheet1": [["Natural Account", "Amount"], ["4310", 1000]]}
        result = detect_erp_type(Path("export.xlsx"), sheets)
        assert result.erp_type == ERPType.ORACLE
        assert result.confidence == 0.95

    def test_tally_fingerprints_detected(self):
        sheets = {"Sheet1": [["Ledger", "Primary Group"], ["Cash", "Current Assets"]]}
        result = detect_erp_type(Path("export.xlsx"), sheets)
        assert result.erp_type == ERPType.TALLY
        assert result.confidence == 0.95

    def test_fingerprint_tier_wins_over_alias_tier(self):
        sheets = {"Sheet1": [["BUKRS", "Company Code"], ["1000", "1000"]]}
        result = detect_erp_type(Path("export.xlsx"), sheets)
        assert result.erp_type == ERPType.SAP
        assert result.confidence == 0.95


class TestColumnAliasTier:
    def test_sap_alias_signals_detected_without_fingerprints(self):
        sheets = {"Sheet1": [["Company Code", "Profit Centre", "GL Code"], ["1000", "PC01", "431020"]]}
        result = detect_erp_type(Path("export.xlsx"), sheets)
        assert result.erp_type == ERPType.SAP
        assert result.confidence == 0.6

    def test_oracle_alias_signals_detected_without_fingerprints(self):
        sheets = {"Sheet1": [["Cost Centre", "GL Code"], ["520", "4310"]]}
        result = detect_erp_type(Path("export.xlsx"), sheets)
        assert result.erp_type == ERPType.ORACLE
        assert result.confidence == 0.6

    def test_tally_alias_signals_detected_without_fingerprints(self):
        sheets = {"Sheet1": [["Ledger Name", "Group"], ["Cash", "Current Assets"]]}
        result = detect_erp_type(Path("export.xlsx"), sheets)
        assert result.erp_type == ERPType.TALLY
        assert result.confidence == 0.6


class TestPathTokenTier:
    def test_sap_path_segment_detected_with_no_content_signal(self):
        sheets = {"Sheet1": [["Foo", "Bar"], [1, 2]]}
        path = Path("uploads") / "SAP" / "IOC_Global" / "trial_balance.xlsx"
        result = detect_erp_type(path, sheets)
        assert result.erp_type == ERPType.SAP
        assert result.confidence == 0.4

    def test_generic_path_segment_detected_for_in_house_folder(self):
        sheets = {"Sheet1": [["Foo", "Bar"], [1, 2]]}
        path = Path("uploads") / "In_house_or_other" / "CEIL" / "trial_balance.xlsx"
        result = detect_erp_type(path, sheets)
        assert result.erp_type == ERPType.GENERIC
        assert result.confidence == 0.4

    def test_no_signal_anywhere_reports_unknown(self):
        sheets = {"Sheet1": [["Foo", "Bar"], [1, 2]]}
        path = Path("uploads") / "acme_ltd" / "trial_balance.xlsx"
        result = detect_erp_type(path, sheets)
        assert result.erp_type == ERPType.UNKNOWN
        assert result.confidence == 0.0
        assert result.evidence == ["no_signal_matched"]


@pytest.mark.skipif(not _SAP_IOC_GLOBAL_TB.exists(), reason="SAP IOC_Global real-file fixture not present")
def test_real_sap_iocglobal_file_has_no_content_signal_and_relies_on_path():
    """Confirms section 1.6.1's finding directly: this real, client-
    pre-standardized SAP file carries no raw SAP fingerprints or even
    alias-tier signals -- only the "SAP" path segment identifies it."""
    sheets = _load_workbook_sheets(_SAP_IOC_GLOBAL_TB, max_row=30)
    result = detect_erp_type(_SAP_IOC_GLOBAL_TB, sheets)
    assert result.erp_type == ERPType.SAP
    assert result.confidence == 0.4
    assert result.evidence[0].startswith("path_token:")


@pytest.mark.skipif(not _TALLY_GURUGRAM_TB.exists(), reason="Gurugram real-file fixture not present")
def test_real_gurugram_tally_tb_file_has_no_content_signal_and_relies_on_path():
    sheets = _load_workbook_sheets(_TALLY_GURUGRAM_TB, max_row=30)
    result = detect_erp_type(_TALLY_GURUGRAM_TB, sheets)
    assert result.erp_type == ERPType.TALLY
    assert result.confidence == 0.4
    assert result.evidence[0].startswith("path_token:")


@pytest.mark.skipif(not _CEIL_TB.exists(), reason="CEIL real-file fixture not present")
def test_real_ceil_file_resolves_to_generic_via_path_convention():
    sheets = _load_workbook_sheets(_CEIL_TB, max_row=30)
    result = detect_erp_type(_CEIL_TB, sheets)
    assert result.erp_type == ERPType.GENERIC
    assert result.confidence == 0.4

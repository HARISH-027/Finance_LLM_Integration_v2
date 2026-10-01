"""Tests for backend/tools/oracle_adapter.py -- the Oracle ERP adapter
introduced in Phase 4 of the TB Multi-ERP Adaptability upgrade. Real-fixture
tests confirm the corrected finding (design doc section 35's revised Item
1): 5 of 8 real Oracle clients ship a separate, GL-code-keyed Grouping file
that the existing input_grouping_source.py already extracts into
grouping_hints -- no upstream code changes were needed for those besides two
narrow FIELD_ALIASES additions (CONCOR's "GL ACC"/"GL Code Description"/
"Beginning Balance" headers). HUDCO, NHPC (no classification data anywhere)
and IRCTC (a Grouping file whose merged-header shape locate_header doesn't
recognize) all correctly raise oracle_no_classification_data_available."""

from pathlib import Path

import pytest

from modes.trial_balance.pipeline.tools.input_dispatch import parse_tb_input
from modes.trial_balance.pipeline.tools.input_header_detect import build_tb_rows_from_grid, locate_header
from modes.trial_balance.pipeline.tools.input_scenario_b import _load_workbook_sheets
from modes.trial_balance.pipeline.tools.oracle_adapter import (
    _detect_no_classification_data,
    run_oracle_adapter,
)
from modes.trial_balance.pipeline.tools.tb_erp_contracts import ClassificationEvidence, ERPType
from modes.trial_balance.pipeline.tools.tb_models import TBRow

_ORACLE_ROOT = Path(__file__).parents[2] / "documentation" / "ERP_system_standardized_files" / "Oracle"

_BHAVINI_DIR = _ORACLE_ROOT / "Bhavini" / "Multi_FY_2024-25_to_2025-26"
_BHAVINI_TB = _BHAVINI_DIR / "Bhavini_Multi_FY_2024-25_to_2025-26_Trial_Balance_and_GL_Grouping__TB_24-25__e2452013.xlsx"
_BHAVINI_GRP = (
    _BHAVINI_DIR / "Bhavini_Multi_FY_2024-25_to_2025-26_Trial_Balance_and_GL_Grouping__Grouping_24-25__6aa740fb.xlsx"
)

_CONCOR_DIR = _ORACLE_ROOT / "Container_Corporation_of_India_Limited" / "Multi_FY_2024-25_to_2025-26"
_CONCOR_TB = (
    _CONCOR_DIR / "Container_Corporation_of_India_Limited_Multi_FY_2024-25_to_2025-26_Trial_Balance_and_GL_"
    "Grouping_CONCOR__Trial_Bal_as_31.03.2025__aa054b06.xlsx"
)
_CONCOR_GRP = (
    _CONCOR_DIR / "Container_Corporation_of_India_Limited_Multi_FY_2024-25_to_2025-26_Trial_Balance_and_GL_"
    "Grouping_CONCOR__Grouping_Mar_26__1e6e655d.xlsx"
)

_GCIL_DIR = _ORACLE_ROOT / "Grid_Controller_of_India" / "FY2024-25"
_GCIL_TB = _GCIL_DIR / "Grid_Controller_of_India_FY2024-25_Trial_Balance_and_GL_Grouping_GCIL__TB__677ec1c6.xlsx"
_GCIL_GRP = _GCIL_DIR / "Grid_Controller_of_India_FY2024-25_Trial_Balance_and_GL_Grouping_GCIL__Grouping__1ee7f4e7.xlsx"

_IRCTC_DIR = (
    _ORACLE_ROOT / "Indian_Railway_Catering_and_Tourism_Corporation_Limited" / "Multi_FY_2024-25_to_2025-26"
)
_IRCTC_TB = (
    _IRCTC_DIR / "Indian_Railway_Catering_and_Tourism_Corporation_Limited_Multi_FY_2024-25_to_2025-26_Trial_"
    "Balance_and_GL_Grouping_IRCTC__TB_2024-25__f3742410.xlsx"
)
_IRCTC_GRP = (
    _IRCTC_DIR / "Indian_Railway_Catering_and_Tourism_Corporation_Limited_Multi_FY_2024-25_to_2025-26_Trial_"
    "Balance_and_GL_Grouping_IRCTC__Grp_2024-25__819d45b6.xlsx"
)

_HUDCO_TB = _ORACLE_ROOT / "HUDCO" / "HUDCO_Master_Trial_Balance__2024-25__59d8575c.xlsx"

_NHPC_TB = (
    _ORACLE_ROOT / "NHPC" / "Multi_FY_2024_to_2026" / "NHPC_Multi_FY_2024_to_2026_Trial_Balance__31.03.2025__781922c2.xlsx"
)


class TestDetectNoClassificationData:
    def test_none_when_dict_empty(self):
        assert _detect_no_classification_data({}) is None

    def test_none_when_at_least_one_row_has_classification(self):
        classification = {
            "1": ClassificationEvidence(erp_type=ERPType.ORACLE, account_classification={"main_head": "Assets"}),
            "2": ClassificationEvidence(erp_type=ERPType.ORACLE, account_classification={}),
        }
        assert _detect_no_classification_data(classification) is None

    def test_finding_when_every_row_has_empty_classification(self):
        classification = {
            "1": ClassificationEvidence(erp_type=ERPType.ORACLE, account_classification={}),
            "2": ClassificationEvidence(erp_type=ERPType.ORACLE, account_classification={}),
        }
        finding = _detect_no_classification_data(classification)
        assert finding is not None
        assert finding.code == "oracle_no_classification_data_available"
        assert finding.severity == "audit_risk"


def _tb_rows_via_parse_tb_input(input_files):
    parsed_inputs = parse_tb_input(input_files, framework="IND_AS")
    return parsed_inputs[0]


@pytest.mark.skipif(not _BHAVINI_TB.exists() or not _BHAVINI_GRP.exists(), reason="Bhavini real-file fixtures not present")
def test_bhavini_classification_resolves_via_separate_grouping_file():
    parsed = _tb_rows_via_parse_tb_input([_BHAVINI_TB, _BHAVINI_GRP])
    sheets = _load_workbook_sheets(_BHAVINI_TB)
    sheet_name = next(iter(sheets))
    grid = sheets[sheet_name]
    header_row, cols = locate_header(grid)
    tb_rows = build_tb_rows_from_grid(grid, header_row, cols)

    result = run_oracle_adapter(
        _BHAVINI_TB, grid, header_row, cols, tb_rows, grouping_hints=parsed.grouping_hints, sheet_name=sheet_name,
    )

    evidence = result.classification_by_gl_code["10103"]
    assert evidence.account_classification["bs_pl"] == "BS"
    assert evidence.account_classification["main_head"] == "Property Plant and Equipment"
    assert evidence.account_classification["account_type"] == "ASSETS"
    assert result.findings == []


@pytest.mark.skipif(not _CONCOR_TB.exists() or not _CONCOR_GRP.exists(), reason="CONCOR real-file fixtures not present")
def test_concor_gl_acc_and_sub_head1_aliases_resolve_classification():
    """Proves the new FIELD_ALIASES additions this phase needed: CONCOR's
    TB header is "GL ACC" (not "GL Code") and its Grouping file's header is
    "SUB HEAD1" (no space before the digit) -- both previously unrecognized."""
    parsed = _tb_rows_via_parse_tb_input([_CONCOR_TB, _CONCOR_GRP])
    sheets = _load_workbook_sheets(_CONCOR_TB)
    sheet_name = next(iter(sheets))
    grid = sheets[sheet_name]
    header_row, cols = locate_header(grid)
    tb_rows = build_tb_rows_from_grid(grid, header_row, cols)
    assert len(tb_rows) > 0

    result = run_oracle_adapter(
        _CONCOR_TB, grid, header_row, cols, tb_rows, grouping_hints=parsed.grouping_hints, sheet_name=sheet_name,
    )

    evidence = result.classification_by_gl_code["100001"]
    assert evidence.account_classification["main_head"] == "Property, plant and equipment"
    assert evidence.account_classification["sub_head_1"] == "Carrying Value - Freehold Land"
    assert result.findings == []


@pytest.mark.skipif(not _GCIL_TB.exists() or not _GCIL_GRP.exists(), reason="GCIL real-file fixtures not present")
def test_gcil_segmented_alphanumeric_gl_code_joins_correctly():
    """Proves GCIL's dash-segmented codes ("10901-1") join TB<->Grouping
    correctly -- the only real Oracle client whose GL code isn't plain
    numeric."""
    parsed = _tb_rows_via_parse_tb_input([_GCIL_TB, _GCIL_GRP])
    sheets = _load_workbook_sheets(_GCIL_TB)
    sheet_name = next(iter(sheets))
    grid = sheets[sheet_name]
    header_row, cols = locate_header(grid)
    tb_rows = build_tb_rows_from_grid(grid, header_row, cols)

    result = run_oracle_adapter(
        _GCIL_TB, grid, header_row, cols, tb_rows, grouping_hints=parsed.grouping_hints, sheet_name=sheet_name,
    )

    evidence = result.classification_by_gl_code["10901-1"]
    assert evidence.account_classification["bs_pl"] == "PL"
    assert evidence.account_classification["main_head"] == "Revenue from operations"
    assert evidence.account_classification["account_type"] == "CR"


@pytest.mark.skipif(not _IRCTC_TB.exists() or not _IRCTC_GRP.exists(), reason="IRCTC real-file fixtures not present")
def test_irctc_hierarchical_grouping_file_yields_no_classification():
    """IRCTC's real Grouping file has classification data, but its merged/
    duplicated header labels ("Company Code" appears twice; the GL code is
    embedded in a combined "GL Code, Text and Headings" cell) aren't
    recognized by locate_header's alias matching -- confirmed directly:
    extract_structured_fields/extract_hint_text both return {} for this
    file. The adapter must still run safely and flag this rather than
    silently reporting "fully classified" or crashing (design doc section
    35 logs fixing locate_header for this shape as a separate follow-up)."""
    parsed = _tb_rows_via_parse_tb_input([_IRCTC_TB, _IRCTC_GRP])
    sheets = _load_workbook_sheets(_IRCTC_TB)
    sheet_name = next(iter(sheets))
    grid = sheets[sheet_name]
    header_row, cols = locate_header(grid)
    tb_rows = build_tb_rows_from_grid(grid, header_row, cols)
    assert len(tb_rows) > 0

    result = run_oracle_adapter(
        _IRCTC_TB, grid, header_row, cols, tb_rows, grouping_hints=parsed.grouping_hints, sheet_name=sheet_name,
    )

    assert any(f.code == "oracle_no_classification_data_available" for f in result.findings)


@pytest.mark.skipif(not _HUDCO_TB.exists(), reason="HUDCO real-file fixture not present")
def test_hudco_tb_only_no_grouping_file_raises_finding():
    sheets = _load_workbook_sheets(_HUDCO_TB)
    sheet_name = next(iter(sheets))
    grid = sheets[sheet_name]
    header_row, cols = locate_header(grid)
    tb_rows = build_tb_rows_from_grid(grid, header_row, cols)
    assert len(tb_rows) > 0

    result = run_oracle_adapter(_HUDCO_TB, grid, header_row, cols, tb_rows, sheet_name=sheet_name)

    assert any(f.code == "oracle_no_classification_data_available" for f in result.findings)
    for evidence in result.classification_by_gl_code.values():
        assert evidence.account_classification == {}


@pytest.mark.skipif(not _NHPC_TB.exists(), reason="NHPC real-file fixture not present")
def test_nhpc_tb_only_no_classification_columns_raises_finding():
    sheets = _load_workbook_sheets(_NHPC_TB)
    sheet_name = next(iter(sheets))
    grid = sheets[sheet_name]
    header_row, cols = locate_header(grid)
    tb_rows = build_tb_rows_from_grid(grid, header_row, cols)
    assert len(tb_rows) > 0

    result = run_oracle_adapter(_NHPC_TB, grid, header_row, cols, tb_rows, sheet_name=sheet_name)

    assert any(f.code == "oracle_no_classification_data_available" for f in result.findings)


def test_synthetic_row_embedded_columns_take_priority_over_hint():
    """Symmetry with Tally/SAP: if a future Oracle file DOES carry
    classification directly on the TB row (none does today), that value
    wins over a companion GroupingHint for the same field."""
    from modes.trial_balance.pipeline.tools.tb_models import GroupingHint

    grid = [
        ["GL Code", "GL Name", "Opening Balance", "Debit", "Credit", "Closing Balance", "Main Head"],
        ["1001", "Freehold Land", 0, 100, 0, 100, "Row-Embedded Head"],
    ]
    header_row, cols = locate_header(grid)
    tb_rows = build_tb_rows_from_grid(grid, header_row, cols)
    hints = {"1001": GroupingHint(gl_code="1001", known_fields={"main_head": "Hint-Sourced Head"}, hint_text="")}

    result = run_oracle_adapter(Path("synthetic.xlsx"), grid, header_row, cols, tb_rows, grouping_hints=hints)

    assert result.classification_by_gl_code["1001"].account_classification["main_head"] == "Row-Embedded Head"

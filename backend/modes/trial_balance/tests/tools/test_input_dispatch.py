"""End-to-end tests for the ported input-parsing layer (Scenario A/B/C
auto-detection) against real, synthetic openpyxl workbooks -- not just
import-checking. Scenario D (multi-year) is covered separately since it
needs multi-sheet fixtures with year-bearing names.
"""

from pathlib import Path

import openpyxl
import pytest

from modes.trial_balance.pipeline.tools.input_dispatch import parse_tb_input, parse_tb_input_single_year
from modes.trial_balance.pipeline.tools.input_scenario_a import COMPANY_DETAILS_SHEET, TB_GROUPING_SHEET
from modes.trial_balance.pipeline.tools.input_scenario_b import classify_workbooks, parse_tb_grouping_pair

_GURUGRAM_DIR = (
    Path(__file__).parents[2]
    / "documentation" / "ERP_system_standardized_files" / "Tally" / "Gurugram_Metro_Rail_Limited"
)
_GURUGRAM_TB_FILE = _GURUGRAM_DIR / "Gurugram_Metro_Rail_Limited_Master_Trial_Balance_Head_Sub_CAG__TB25-26__be041db1.xlsx"
_GURUGRAM_GROUPING_FILE = _GURUGRAM_DIR / "Gurugram_Metro_Rail_Limited_Master_Trial_Balance_Head_Sub_CAG__Grouping__5e8045cb.xlsx"


def _write_workbook(path, sheets: dict):
    wb = openpyxl.Workbook()
    wb.remove(wb.active)
    for name, rows in sheets.items():
        ws = wb.create_sheet(name)
        for row in rows:
            ws.append(row)
    wb.save(path)


def test_scenario_a_template_parses_tb_rows_and_grouping_hints(tmp_path):
    path = tmp_path / "template.xlsx"
    _write_workbook(path, {
        COMPANY_DETAILS_SHEET: [
            ["Company Details", None],
            ["COMPANY_NAME", "ACME Ltd"],
            ["FINANCIAL_YEAR", "FY2024-25"],
        ],
        TB_GROUPING_SHEET: [
            ["GL Code", "GL Name", "Opening Balance", "Debit", "Credit", "Closing Balance",
             "BS/PL", "Main Head", "Sub Head 1", "Sub Head 2"],
            ["1001", "Freehold Land", 0, 1000, 0, 1000,
             "BS", "Non-current assets", "Property, Plant and Equipment", "Land"],
            ["1002", "Cash in Hand", 0, 500, 0, 500, None, None, None, None],
        ],
    })

    results = parse_tb_input([path])
    assert len(results) == 1
    parsed = results[0]
    assert parsed.metadata.company_name == "ACME Ltd"
    assert len(parsed.tb_rows) == 2
    assert parsed.grouping_hints["1001"].known_fields["sub_head_2"] == "Land"
    assert "1002" not in parsed.grouping_hints


@pytest.mark.parametrize("total_label", ["Total", "Grand Total", "Sub Total", "Sub-Total", "GRAND TOTAL"])
def test_total_and_grand_total_rows_are_excluded_from_tb_rows(tmp_path, total_label):
    """Gap 1 (TB-v2-git-specific regression re-add): a Total/Grand-Total/
    Sub-Total row in the TB workbook itself must never be classified as a
    real GL account -- build_canonical_tb used to guarantee this before it
    was retired; the unified pipeline must guarantee it too."""
    path = tmp_path / "template.xlsx"
    _write_workbook(path, {
        COMPANY_DETAILS_SHEET: [
            ["Company Details", None],
            ["COMPANY_NAME", "ACME Ltd"],
            ["FINANCIAL_YEAR", "FY2024-25"],
        ],
        TB_GROUPING_SHEET: [
            ["GL Code", "GL Name", "Opening Balance", "Debit", "Credit", "Closing Balance",
             "BS/PL", "Main Head", "Sub Head 1", "Sub Head 2"],
            ["1001", "Freehold Land", 0, 1000, 0, 1000,
             "BS", "Non-current assets", "Property, Plant and Equipment", "Land"],
            ["9999", total_label, 0, 1000, 0, 1000, None, None, None, None],
        ],
    })

    parsed = parse_tb_input([path])[0]
    assert len(parsed.tb_rows) == 1
    assert parsed.tb_rows[0].gl_code == "1001"


def test_scenario_b_two_files_auto_classified_content_based(tmp_path):
    tb_path = tmp_path / "trial_balance_export.xlsx"
    grouping_path = tmp_path / "grouping_export.xlsx"
    _write_workbook(tb_path, {
        "Sheet1": [
            ["GL Code", "GL Name", "Opening Balance", "Debit", "Credit", "Closing Balance"],
            ["2001", "Freehold Land", 0, 2000, 0, 2000],
        ],
    })
    _write_workbook(grouping_path, {
        "Sheet1": [
            ["GL Code", "Main Head", "Sub Head 1", "Sub Head 2", "BS/PL"],
            ["2001", "Non-current assets", "Property, Plant and Equipment", "Land", "BS"],
        ],
    })

    results = parse_tb_input([tb_path, grouping_path])
    assert len(results) == 1
    parsed = results[0]
    assert len(parsed.tb_rows) == 1
    assert parsed.tb_rows[0].gl_code == "2001"
    assert parsed.grouping_hints["2001"].known_fields["sub_head_2"] == "Land"


def test_scenario_b_works_regardless_of_which_file_is_listed_first(tmp_path):
    """Content-based classification, never filename- or order-based."""
    tb_path = tmp_path / "a_grouping_looking_name.xlsx"
    grouping_path = tmp_path / "b_tb_looking_name.xlsx"
    _write_workbook(tb_path, {
        "Sheet1": [
            ["GL Code", "GL Name", "Opening Balance", "Debit", "Credit", "Closing Balance"],
            ["3001", "Cash", 0, 300, 0, 300],
        ],
    })
    _write_workbook(grouping_path, {
        "Sheet1": [
            ["GL Code", "Main Head", "Sub Head 1", "Sub Head 2", "BS/PL"],
            ["3001", "Current assets", "Financial Assets - Cash and cash equivalents", "Cash on hand", "BS"],
        ],
    })

    results = parse_tb_input([grouping_path, tb_path])  # grouping listed first
    parsed = results[0]
    assert parsed.tb_rows[0].gl_code == "3001"
    assert parsed.grouping_hints["3001"].known_fields["sub_head_2"] == "Cash on hand"


def test_scenario_c_combined_single_sheet_flat_category(tmp_path):
    path = tmp_path / "combined.xlsx"
    _write_workbook(path, {
        "TB": [
            ["GL Code", "GL Name", "Opening Balance", "Debit", "Credit", "Closing Balance", "Grouping"],
            ["4001", "Freehold Land", 0, 4000, 0, 4000, "Property plant and Equipment"],
        ],
    })

    results = parse_tb_input([path])
    parsed = results[0]
    assert len(parsed.tb_rows) == 1
    assert parsed.grouping_hints["4001"].hint_text == "Property plant and Equipment"


def test_missing_gl_code_column_raises_unsupported_or_structure_error(tmp_path):
    path = tmp_path / "not_a_tb.xlsx"
    _write_workbook(path, {"Sheet1": [["Foo", "Bar"], [1, 2]]})

    with pytest.raises(Exception):
        parse_tb_input_single_year([path])


@pytest.mark.skipif(not _GURUGRAM_TB_FILE.exists(), reason="Gurugram real-file fixture not present")
def test_gurugram_tally_pair_classified_correctly_order1():
    """Regression for a real misclassification: the Grouping file's `GL
    CODE` column is all-placeholder ("NA"), and the genuine TB file
    (Particulars/Head/Sub-Head/Closing Balance, no gl_code at all) must
    still win as the TB, not the Grouping file."""
    tb_path, grouping_path = classify_workbooks([_GURUGRAM_TB_FILE, _GURUGRAM_GROUPING_FILE])
    assert tb_path == _GURUGRAM_TB_FILE
    assert grouping_path == _GURUGRAM_GROUPING_FILE


@pytest.mark.skipif(not _GURUGRAM_TB_FILE.exists(), reason="Gurugram real-file fixture not present")
def test_gurugram_tally_pair_classified_correctly_order2():
    """Content-based classification, never order-based."""
    tb_path, grouping_path = classify_workbooks([_GURUGRAM_GROUPING_FILE, _GURUGRAM_TB_FILE])
    assert tb_path == _GURUGRAM_TB_FILE
    assert grouping_path == _GURUGRAM_GROUPING_FILE


@pytest.mark.skipif(not _GURUGRAM_TB_FILE.exists(), reason="Gurugram real-file fixture not present")
def test_gurugram_tb_rows_nonempty_and_split_closing_balance_correct():
    """Once classification is fixed, rows must not be silently dropped for
    having no gl_code (Tally has none) -- and the two-row Debit/Credit
    sub-header under "Closing Balance" must be read as a signed split
    balance, not misread as period debit/credit movement."""
    parsed = parse_tb_grouping_pair(_GURUGRAM_TB_FILE, _GURUGRAM_GROUPING_FILE)
    assert len(parsed.tb_rows) > 0

    row = next((r for r in parsed.tb_rows if r.gl_name == "Equity Share Capital-GoI"), None)
    assert row is not None, "expected ledger row not found in parsed TB rows"
    assert row.closing == -1_100_000_000  # pure-credit row: Debit=0, Credit=1,100,000,000

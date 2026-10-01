"""Tests for tally_row_normalizer.py: Dr/Cr re-signing + Group/Ledger split."""
from pathlib import Path

import openpyxl
import pytest
from openpyxl.styles import Font

from modes.trial_balance.pipeline.tools.input_header_detect import build_tb_rows_from_grid, pick_tb_sheet
from modes.trial_balance.pipeline.tools.input_scenario_b import _load_workbook_sheets
from modes.trial_balance.pipeline.tools.tally_row_normalizer import normalize_tally_rows

_KTPO = Path(__file__).parents[2] / "docs" / "TBS-from-ERP" / "Tally" / "KTPO" / "Trial Balance FY 2025-26.xlsx"
_DR, _CR = '0.00" Dr"', '0.00" Cr"'


def _workbook(tmp_path):
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Trial Balance"
    ws.append(["Particulars", "Opening", "Debit", "Credit", "Closing"])
    rows = [
        ("CAPITAL", 100, _CR, True), ("Share Capital", 100, _CR, False),
        ("ASSETS", 100, _DR, True), ("Cash", 60, _DR, False), ("Bank", 40, _DR, False),
    ]
    for label, value, fmt, bold in rows:
        ws.append([label, value, None, None, value])
        for col in (2, 5):
            ws.cell(ws.max_row, col).number_format = fmt
        ws.cell(ws.max_row, 1).font = Font(bold=bold)
    path = tmp_path / "tb.xlsx"
    wb.save(path)
    return path


def _parse(path):
    sheets = _load_workbook_sheets(path, max_row=None)
    name, header_row, cols = pick_tb_sheet(sheets)
    return name, header_row, cols, build_tb_rows_from_grid(sheets[name], header_row, cols)


class TestNormalizeTallyRows:
    def test_signs_and_container_split(self, tmp_path):
        path = _workbook(tmp_path)
        name, header_row, cols, rows = _parse(path)
        result = normalize_tally_rows(path, name, header_row, cols, rows)
        assert result.applied and result.signed
        assert [r.gl_name for r in result.ledger_rows] == ["Share Capital", "Cash", "Bank"]
        assert [c["gl_name"] for c in result.container_rows] == ["CAPITAL", "ASSETS"]
        assert result.ledger_rows[0].closing == -100 and result.ledger_rows[1].closing == 60
        assert sum(r.closing for r in result.ledger_rows) == 0

    def test_misaligned_rows_left_unchanged(self, tmp_path):
        path = _workbook(tmp_path)
        name, header_row, cols, rows = _parse(path)
        result = normalize_tally_rows(path, name, header_row, cols, rows[:-1])
        assert result.applied is False and len(result.ledger_rows) == len(rows) - 1

    @pytest.mark.skipif(not _KTPO.exists(), reason="KTPO real fixture not present")
    def test_real_ktpo_nets_to_zero_and_removes_groups(self):
        name, header_row, cols, rows = _parse(_KTPO)
        result = normalize_tally_rows(_KTPO, name, header_row, cols, rows)
        assert len(result.container_rows) == 42 and len(result.ledger_rows) == 202
        assert abs(sum(r.closing for r in result.ledger_rows)) < 0.01

"""Unit tests for input_header_detect.py. No dedicated test file existed
for this module before; added alongside the Tally 3-row-split-header
recovery fixes (_recover_gl_name_from_earlier_row,
_recover_two_row_amount_labels) and the Dr/Cr text-suffix amount parser
(parse_amount / parse_amount_with_status), both of which are otherwise
untested at the unit level -- prior coverage was only indirect, via other
ERP adapters' real-fixture tests.
"""

from modes.trial_balance.pipeline.tools.input_header_detect import (
    locate_header,
    parse_amount,
    parse_amount_with_status,
)


class TestLocateHeaderThreeRowSplitHeader:
    """Real Tally native-export shape (Mumbai Aviation Fuel Farm, Petronet
    VK): a Particulars row, then an Opening/Transactions/Closing parent
    row, then a Balance/Debit/Credit/Balance sub-row directly beneath it."""

    def _grid(self):
        return [
            ["Some Company Ltd", None, None, None, None],
            ["Trial Balance", None, None, None, None],
            ["Particulars", "1-Apr-24 to 31-Mar-25", None, None, None],
            ["", "Opening", "Transactions", None, "Closing"],
            ["", "Balance", "Debit", "Credit", "Balance"],
            ["Capital Account", 100.0, 0.0, 10.0, 110.0],
        ]

    def test_recovers_gl_name_and_amount_columns_from_earlier_rows(self):
        header_row, cols = locate_header(self._grid(), max_scan=15)
        assert header_row == 4
        assert cols == {"debit": 2, "credit": 3, "gl_name": 0, "opening": 1, "closing": 4}

    def test_does_not_overwrite_a_field_header_row_already_found(self):
        # header_row itself already carries debit/credit -- the recovery
        # steps must never override those with something from another row.
        header_row, cols = locate_header(self._grid(), max_scan=15)
        assert cols["debit"] == 2
        assert cols["credit"] == 3


class TestLocateHeaderSingleRowHeaderUnaffected:
    """The common, simpler case (one row carries every alias) must behave
    exactly as before -- the recovery steps are no-ops here."""

    def test_single_row_header_unchanged(self):
        grid = [
            ["GL Code", "GL Name", "Opening Balance", "Debit", "Credit", "Closing Balance"],
            ["1001", "Cash", 100.0, 50.0, 20.0, 130.0],
        ]
        header_row, cols = locate_header(grid, max_scan=15)
        assert header_row == 0
        assert cols["gl_code"] == 0
        assert cols["gl_name"] == 1
        assert cols["opening"] == 2
        assert cols["closing"] == 5

    def test_no_gl_name_anywhere_stays_absent(self):
        grid = [
            ["Debit", "Credit"],
            [10.0, 20.0],
        ]
        header_row, cols = locate_header(grid, max_scan=15)
        assert "gl_name" not in cols


class TestParseAmountDrCrSuffix:
    def test_plain_number_unaffected(self):
        assert parse_amount(1234.5) == 1234.5

    def test_dr_suffix_is_positive(self):
        assert parse_amount("50,00,000 Dr") == 5000000.0

    def test_cr_suffix_negates(self):
        assert parse_amount("50,00,000 Cr") == -5000000.0

    def test_dr_suffix_lowercase_and_no_space(self):
        assert parse_amount("1500dr") == 1500.0

    def test_cr_suffix_lowercase_and_no_space(self):
        assert parse_amount("1500cr") == -1500.0

    def test_existing_unparseable_contract_preserved(self):
        # parse_amount() itself must still silently coerce to 0.0 for
        # anything it can't parse (backward compatibility for every
        # existing non-Tally call site) -- ambiguity visibility is
        # parse_amount_with_status()'s job, not this function's.
        assert parse_amount("not a number") == 0.0

    def test_dash_placeholder_still_zero(self):
        assert parse_amount("-") == 0.0


class TestParseAmountWithStatus:
    def test_clean_plain_number(self):
        amount, clean = parse_amount_with_status(1234.5)
        assert amount == 1234.5
        assert clean is True

    def test_clean_dr_suffix(self):
        amount, clean = parse_amount_with_status("1,000 Dr")
        assert amount == 1000.0
        assert clean is True

    def test_clean_cr_suffix(self):
        amount, clean = parse_amount_with_status("1,000 Cr")
        assert amount == -1000.0
        assert clean is True

    def test_ambiguous_bare_suffix_no_number_is_not_clean(self):
        amount, clean = parse_amount_with_status("Cr")
        assert clean is False
        assert amount == 0.0

    def test_ambiguous_unparseable_text_is_not_clean(self):
        amount, clean = parse_amount_with_status("N/A")
        assert clean is False

    def test_dash_placeholder_is_clean_zero(self):
        # A dash is a well-understood zero-balance convention, not
        # ambiguous -- distinct from a genuinely unparseable cell.
        amount, clean = parse_amount_with_status("-")
        assert amount == 0.0
        assert clean is True

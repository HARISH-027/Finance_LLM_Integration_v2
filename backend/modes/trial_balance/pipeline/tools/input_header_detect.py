"""Content-based header row/column locator for messy ERP exports (SAP/Tally
G/L dumps). Real-world TB and grouping workbooks routinely have a title
block above the header (company name, report title, period, blank rows)
and the header itself can start at any column, not just A. Never assume
row 1 / col A -- scan for the row whose cells best match known field-name
aliases instead.

Ported from TB_normalization_v1's input/header_detect.py (itself ported
from TB_ingestion/scripts/common/header_detect.py) -- pure parsing logic,
battle-tested against real SAP/Oracle/Tally exports across many companies.
"""

from __future__ import annotations

import re
from typing import Optional

# TB-v2-git-specific addition, NOT present in TB_normalization_v1's own
# header_detect.py (which never filters Total rows out of the TB row list --
# only input_grouping_source.py's Total handling, for the GROUPING file's own
# section trail, was ported faithfully). The now-retired build_canonical_tb
# used to strip GL rows named "Total"/"Grand Total"/"Sub Total" via
# _TOTAL_PREFIXES before classification; re-added here, at the single shared
# TB-row builder every Scenario A/B/C/D parser calls, so a Total row in the TB
# workbook itself doesn't get classified as a real GL account on either the
# Live or Upload path. Same regex shape as input_grouping_source.py's
# _TOTAL_RE, for consistency across the input layer.
_TOTAL_ROW_RE = re.compile(r"^(total\b|grand\s*total\b|sub[- ]?total\b|net\s*total\b)", re.IGNORECASE)


def is_total_row_label(gl_name: str) -> bool:
    """True when `gl_name` reads as a Total/Grand-Total/Sub-Total/Net-Total
    label -- shared by every TB-row construction site (build_tb_rows_from_grid
    below, and input_scenario_a.py's own inline loop, which doesn't go through
    that shared builder)."""
    return bool(gl_name) and bool(_TOTAL_ROW_RE.match(gl_name))

# canonical_field -> set of lowercased, whitespace-stripped aliases seen
# across real client exports.
FIELD_ALIASES: dict = {
    "gl_code": {
        "gl code", "gl codes", "g/l acct", "g/l account", "account code", "account number",
        "acct code", "g/l code", "gl account", "a/c code", "gl acc",
    },
    "gl_name": {
        "gl name", "short text", "text (name of the gl)", "account description",
        "g/l acct long text", "gl description", "acct code description", "name of gl",
        "g/l account text", "text (name of gl)", "description", "particulars",
        "gl code description", "name of the gl",
    },
    "opening": {
        "opening balance", "balance carryforward", "opening", "beg. balance", "opening balances",
        "beginning balance",
    },
    "debit": {
        "debit", "debit rept.period", "debit during the year", "dr.", "accum. value - d",
        "transaction (dr.)",
    },
    "credit": {
        "credit", "credit report per.", "credit during the year", "cr.", "accum. value - c",
        "transaction (cr.)",
    },
    "closing": {
        "closing balance", "accumulated balance", "closing", "cl. bal.", "closing balances",
        "closing balnce",
    },
    "bs_pl": {"bs/pl"},
    "sub_head_2": {"sub head 2"},
    "sub_head_1": {"sub head 1", "sub note name", "sub-head", "sub head", "sub head1"},
    "main_head": {"main head", "note name", "head"},
    "account_type": {"account type"},
    "group_label": {"grouping", "group"},
}

_ALL_ALIASES = {alias for aliases in FIELD_ALIASES.values() for alias in aliases}

# debit/credit headers routinely carry a trailing period range (e.g. "Debit
# 1- 16") an exact-alias set can't enumerate -- scoped narrowly to just
# these two fields rather than broadening matching generally.
_PREFIX_FIELDS = ("debit", "credit")


def _norm(value) -> str:
    if value is None:
        return ""
    # Underscore-separated headers ("GL_CODE") are otherwise invisible to
    # every space-separated alias -- folding "_" into a space is a pure
    # win, not a collision risk (no existing alias contains an underscore).
    return re.sub(r"[_\s]+", " ", str(value).strip().lower())


def _matches(norm: str, field: str) -> bool:
    if norm in FIELD_ALIASES[field]:
        return True
    return field in _PREFIX_FIELDS and norm.startswith(field)


# Lightweight header-confidence scoring: augments the lexical alias-match
# count with two cheap, high-value data-shape signals so locate_header()
# can disambiguate rows that tie (or nearly tie) on alias-match count
# alone. Never overrides a clearly-stronger lexical match: the alias-match
# count still dominates the blended score.
_AMOUNT_FIELDS = ("opening", "debit", "credit", "closing")
_CONFIDENCE_SAMPLE_ROWS = 20

_ZERO_PLACEHOLDERS = {"-", "--", "—", "–", "", "nan", "none"}


def _looks_numeric(value) -> bool:
    """Stricter than parse_amount() on purpose -- parse_amount() always
    returns a float (0.0 for anything it can't parse), which would make
    every column look "100% numeric" if reused directly for a confidence
    signal. This checks whether the RAW cell actually looks numeric."""
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return True
    text = str(value).strip()
    if not text:
        return False
    if text.lower() in _ZERO_PLACEHOLDERS:
        return True
    try:
        float(text.replace(",", ""))
        return True
    except ValueError:
        return False


def _sample_column(grid: list, header_row: int, col_idx: int, sample_rows: int) -> list:
    values = []
    for row in grid[header_row + 1: header_row + 1 + sample_rows]:
        if col_idx < len(row) and row[col_idx] not in (None, ""):
            values.append(row[col_idx])
    return values


def _column_numeric_ratio(grid: list, header_row: int, col_idx: int) -> float:
    values = _sample_column(grid, header_row, col_idx, _CONFIDENCE_SAMPLE_ROWS)
    if not values:
        return 0.0
    return sum(1 for v in values if _looks_numeric(v)) / len(values)


def _column_identifier_ratio(grid: list, header_row: int, col_idx: int) -> float:
    """GL-code columns should be mostly-unique, short values -- distinct
    from a free-text description column, which tends to repeat less
    uniquely but run much longer."""
    values = [str(v).strip() for v in _sample_column(grid, header_row, col_idx, _CONFIDENCE_SAMPLE_ROWS)]
    if not values:
        return 0.0
    uniqueness = len(set(values)) / len(values)
    avg_len = sum(len(v) for v in values) / len(values)
    return uniqueness if avg_len <= 20 else uniqueness * 0.5


def header_confidence(grid: list, header_row: int, cols: dict) -> float:
    """Blended confidence score for one candidate header row: lexical
    alias-match count (dominant term) plus a numeric-data-shape signal for
    amount columns and an identifier-shape signal for gl_code."""
    if not cols:
        return 0.0
    lexical_score = float(len(cols))

    numeric_fields = [f for f in _AMOUNT_FIELDS if f in cols]
    numeric_signal = (
        sum(_column_numeric_ratio(grid, header_row, cols[f]) for f in numeric_fields) / len(numeric_fields)
        if numeric_fields else 0.0
    )
    identifier_signal = _column_identifier_ratio(grid, header_row, cols["gl_code"]) if "gl_code" in cols else 0.0

    return lexical_score + numeric_signal + identifier_signal


def locate_header(grid: list, max_scan: int = 15) -> tuple:
    """Scan the first `max_scan` rows of a raw grid for the row that best
    matches known field-name aliases, using header_confidence() to rank
    candidates. Returns (header_row_index, {canonical_field: column_index}).
    Falls back to row 0 with whatever direct alias matches exist (possibly
    none) if nothing scores above zero."""
    best_row, best_score, best_cols = 0, -1.0, {}

    for r in range(min(max_scan, len(grid))):
        row = grid[r]
        cols: dict = {}
        for c, cell in enumerate(row):
            norm = _norm(cell)
            if not norm:
                continue
            is_alias = norm in _ALL_ALIASES or any(norm.startswith(f) for f in _PREFIX_FIELDS)
            if not is_alias:
                continue
            for field in FIELD_ALIASES:
                if field not in cols and _matches(norm, field):
                    cols[field] = c
        if not cols:
            continue
        score = header_confidence(grid, r, cols)
        if score > best_score:
            best_row, best_score, best_cols = r, score, cols

    return best_row, best_cols


def parse_amount(value) -> float:
    """Coerce a TB cell to a float. Dash/blank placeholders (a common
    zero-balance convention in TB exports) normalize to 0.0."""
    if value is None:
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if text.lower() in _ZERO_PLACEHOLDERS:
        return 0.0
    text = text.replace(",", "")
    try:
        return float(text)
    except ValueError:
        return 0.0


REQUIRED_TB_FIELDS = {"gl_code", "closing"}

_PLACEHOLDER_GL_CODES = {"na", "n/a", "-", "--", "", "none", "nil", "null"}


def gl_code_column_is_real(grid: list, header_row: int, col_idx: int) -> bool:
    """A gl_code column is only a legitimate per-row identity if it actually
    carries codes -- some grouping exports carry a "GL CODE" header with
    every cell set to a literal placeholder ("NA") purely to satisfy a
    template's column list, not to identify rows."""
    values = [str(v).strip().lower() for v in _sample_column(grid, header_row, col_idx, _CONFIDENCE_SAMPLE_ROWS)]
    values = [v for v in values if v]
    if not values:
        return False
    return not all(v in _PLACEHOLDER_GL_CODES for v in values)


def qualifies_as_tb_sheet(grid: list, header_row: int, cols: dict) -> bool:
    """A TB-shaped sheet needs a per-row account identity (a real gl_code
    column, or a gl_name column for engines like Tally that have no formal
    GL code) plus a closing balance -- which can instead be reported as a
    debit-side/credit-side split pair (same-row literal aliases, or a
    two-row parent/child header) with no plain "Closing" column at all."""
    has_gl_code = "gl_code" in cols and gl_code_column_is_real(grid, header_row, cols["gl_code"])
    if not (has_gl_code or "gl_name" in cols):
        return False
    if "closing" in cols:
        return True
    return "closing" in _find_split_balance(grid, header_row, cols)


def pick_tb_sheet(sheets: dict):
    """Given {sheet_name: raw_grid}, pick the sheet that actually looks
    like a TB (has a GL code + a balance column and a reasonable number of
    data rows) rather than a derived pivot/summary sheet. Returns
    (sheet_name, header_row, cols) or None if nothing qualifies."""
    best = None
    best_score = -1
    for name, grid in sheets.items():
        if not grid:
            continue
        header_row, cols = locate_header(grid, max_scan=min(15, len(grid)))
        if not qualifies_as_tb_sheet(grid, header_row, cols):
            continue
        data_rows = len(grid) - header_row - 1
        score = len(cols) * 100 + min(data_rows, 999)
        if score > best_score:
            best_score = score
            best = (name, header_row, cols)
    return best


def parse_gl_code(value) -> str:
    """Normalize a GL code cell to a plain string key (drops trailing .0
    from floats, strips whitespace) so codes can be compared across
    engines."""
    if value is None:
        return ""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value).strip()


def _cell(row: list, cols: dict, field_name: str):
    idx = cols.get(field_name)
    return row[idx] if idx is not None and idx < len(row) else None


# Some real SAP exports report opening/closing balance as two separate
# debit-side/credit-side columns instead of one net figure. Kept entirely
# separate from FIELD_ALIASES/locate_header()'s cols dict -- only
# build_tb_rows_from_grid needs to combine a debit-side/credit-side pair
# into one signed value per row.
_SPLIT_BALANCE_DEBIT_ALIASES = {
    "opening": {"opening debit balance"},
    "closing": {"closing debit balance", "closing bal (dr.)"},
}
_SPLIT_BALANCE_CREDIT_ALIASES = {
    "opening": {"opening credit balance"},
    "closing": {"closing credit balance", "closing bal (cr.)"},
}

# Some real SAP exports report the year's movement as ONE signed net
# debit/credit column instead of separate debit and credit columns.
_NET_MOVEMENT_ALIASES = {"net debit/ credit", "net debit/credit"}


def _find_split_balance_columns(grid: list, header_row: int) -> dict:
    if header_row >= len(grid):
        return {}
    row = grid[header_row]
    debit_cols: dict = {}
    credit_cols: dict = {}
    for c, cell in enumerate(row):
        norm = _norm(cell)
        if not norm:
            continue
        for field, aliases in _SPLIT_BALANCE_DEBIT_ALIASES.items():
            if field not in debit_cols and norm in aliases:
                debit_cols[field] = c
        for field, aliases in _SPLIT_BALANCE_CREDIT_ALIASES.items():
            if field not in credit_cols and norm in aliases:
                credit_cols[field] = c
    return {field: (debit_cols[field], credit_cols[field]) for field in debit_cols if field in credit_cols}


def _find_subrow_split_columns(grid: list, header_row: int, cols: dict) -> dict:
    """Some exports (e.g. Tally) use a two-row header: a parent amount
    label ("Closing Balance") in header_row itself, with bare "Debit"/
    "Credit" sub-labels in the row directly below. Distinct from
    _find_split_balance_columns, which only recognizes single-row literal
    compound aliases (e.g. "closing debit balance")."""
    sub_row_idx = header_row + 1
    if sub_row_idx >= len(grid):
        return {}
    sub_row = grid[sub_row_idx]
    debit_col = credit_col = None
    for c, cell in enumerate(sub_row):
        norm = _norm(cell)
        if norm in ("debit", "dr", "dr.") and debit_col is None:
            debit_col = c
        elif norm in ("credit", "cr", "cr.") and credit_col is None:
            credit_col = c
    if debit_col is None or credit_col is None:
        return {}
    for field in _AMOUNT_FIELDS:
        if cols.get(field) in (debit_col, debit_col - 1):
            return {field: (debit_col, credit_col)}
    return {}


def _find_split_balance(grid: list, header_row: int, cols: dict) -> dict:
    """Single-row literal-alias split first, then the two-row parent/child
    convention as a fallback."""
    found = dict(_find_split_balance_columns(grid, header_row))
    if any(f not in found and f in cols for f in _AMOUNT_FIELDS):
        found.update(_find_subrow_split_columns(grid, header_row, cols))
    return found


_SUBROW_DEBIT_LABELS = ("debit", "dr", "dr.", "dr amount")
_SUBROW_CREDIT_LABELS = ("credit", "cr", "cr.", "cr amount")


def _find_subrow_debit_credit_direct(grid: list, header_row: int, cols: dict) -> Optional[tuple]:
    """A real Oracle export (Bhavini) reports Debit/Credit themselves as a
    two-row split under a generic parent label ("Current Year") rather than
    a named field ("Debit"/"Credit") in header_row itself -- distinct from
    _find_subrow_split_columns, which only fires when the parent label IS
    an already-recognized field's own name (e.g. Tally's Closing-balance
    split). Only used when neither "debit" nor "credit" is already in
    cols, so this can never override an ordinary single-row debit/credit
    column."""
    if "debit" in cols or "credit" in cols:
        return None
    sub_row_idx = header_row + 1
    if sub_row_idx >= len(grid):
        return None
    sub_row = grid[sub_row_idx]
    debit_col = credit_col = None
    for c, cell in enumerate(sub_row):
        norm = _norm(cell)
        if norm in _SUBROW_DEBIT_LABELS and debit_col is None:
            debit_col = c
        elif norm in _SUBROW_CREDIT_LABELS and credit_col is None:
            credit_col = c
    if debit_col is None or credit_col is None:
        return None
    return debit_col, credit_col


def _find_net_movement_column(grid: list, header_row: int):
    if header_row >= len(grid):
        return None
    for c, cell in enumerate(grid[header_row]):
        if _norm(cell) in _NET_MOVEMENT_ALIASES:
            return c
    return None


def build_tb_rows_from_grid(grid: list, header_row: int, cols: dict) -> list:
    """Shared TB-row builder given an ALREADY-KNOWN sheet/header/cols --
    unlike pick_tb_sheet, this never re-discovers which sheet to use, so
    it's safe to call against one specific sheet of a workbook that has
    multiple TB-qualifying sheets (e.g. a multi-year-in-one-file
    workbook). Falls back to the split-balance-column and net-movement-
    column conventions ONLY when the plain single-column field is
    genuinely absent."""
    from modes.trial_balance.pipeline.tools.tb_models import TBRow

    split_balance = {} if ("opening" in cols and "closing" in cols) else _find_split_balance(grid, header_row, cols)
    net_movement_col = None
    subrow_debit_credit = None
    if "debit" not in cols and "credit" not in cols:
        subrow_debit_credit = _find_subrow_debit_credit_direct(grid, header_row, cols)
        if subrow_debit_credit is None:
            net_movement_col = _find_net_movement_column(grid, header_row)

    def _amount(row: list, field: str) -> float:
        # split_balance is checked first: for a two-row parent/child header
        # (_find_subrow_split_columns), split_balance[field] and cols[field]
        # can legitimately point at the same column -- the plain column
        # index alone would read the wrong (Debit-only) value, so the
        # debit/credit pair must win whenever it's present.
        if field in split_balance:
            debit_col, credit_col = split_balance[field]
            debit_val = parse_amount(row[debit_col]) if debit_col < len(row) else 0.0
            credit_val = parse_amount(row[credit_col]) if credit_col < len(row) else 0.0
            return debit_val - credit_val
        if field in cols:
            return parse_amount(_cell(row, cols, field))
        return 0.0

    def _debit_credit(row: list) -> tuple:
        if "debit" in cols or "credit" in cols:
            return parse_amount(_cell(row, cols, "debit")), parse_amount(_cell(row, cols, "credit"))
        if subrow_debit_credit is not None:
            debit_col, credit_col = subrow_debit_credit
            debit_val = parse_amount(row[debit_col]) if debit_col < len(row) else 0.0
            credit_val = parse_amount(row[credit_col]) if credit_col < len(row) else 0.0
            return debit_val, credit_val
        if net_movement_col is not None and net_movement_col < len(row):
            net = parse_amount(row[net_movement_col])
            return (net, 0.0) if net >= 0 else (0.0, -net)
        return 0.0, 0.0

    rows = []
    for row in grid[header_row + 1:]:
        gl_code_raw = _cell(row, cols, "gl_code")
        gl_name_value = _cell(row, cols, "gl_name")
        gl_name = str(gl_name_value).strip() if gl_name_value not in (None, "") else ""
        if gl_code_raw in (None, "") and not gl_name:
            continue  # no identity at all -- nothing to key this row by
        # Engines with no formal GL code (e.g. Tally) key rows by ledger
        # name instead -- TBRow.gl_code is a plain str, so this stays
        # schema-compatible while preserving per-row identity.
        gl_code = parse_gl_code(gl_code_raw) if gl_code_raw not in (None, "") else gl_name
        if is_total_row_label(gl_name):
            continue
        debit, credit = _debit_credit(row)
        rows.append(TBRow(
            gl_code=gl_code, gl_name=gl_name, opening=_amount(row, "opening"),
            debit=debit, credit=credit, closing=_amount(row, "closing"),
        ))
    return rows

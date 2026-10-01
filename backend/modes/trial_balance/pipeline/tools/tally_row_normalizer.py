"""Tally-only row normalization applied to the shared parse's tb_rows
BEFORE the Tally adapter and classification see them.

Two defects of the shared, ERP-agnostic row builder are corrected here,
scoped to Tally so no other ERP's behavior changes:

1. Sign loss: a Tally native export stores every balance as a positive
   magnitude and encodes the side only in the cell number format
   (`0.00" Dr"` / `0.00" Cr"`). Opening/closing are re-signed
   (Dr = +, Cr = -) so the canonical document nets to zero.
2. Head vs Ledger: Group/Head container rows (SHAREHOLDER'S FUNDS,
   CURRENT LIABILITIES, ...) share the Particulars column with real
   Ledger accounts. Rows the structural extractor proved to be containers
   are separated out of the ledger list -- kept as audit-trail
   `container_rows`, never classified, never double-counted.

Best-effort and fail-safe: any misalignment or failed safety check returns
the input rows unchanged with `applied=False` and a note.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Dict, List, Optional

from modes.trial_balance.pipeline.tools.input_header_detect import is_total_row_label
from modes.trial_balance.pipeline.tools.tally_structural_hierarchy import (
    build_hierarchy_with_ambiguity,
    extract_structural_rows,
    load_bold_aware_grid,
    resolve_main_head_per_row,
    sign_from_number_format,
)
from modes.trial_balance.pipeline.tools.tb_models import TBRow

# Absolute currency tolerance for the ledger-sum vs top-level-sum safety check.
_SUM_CHECK_TOLERANCE = 1.0


@dataclass(frozen=True)
class TallyNormalizedRows:
    ledger_rows: List[TBRow]
    # Source-sheet index (0-based grid row) of each ledger row, same order.
    ledger_grid_indices: List[int]
    # Audit-trail records of removed Group/Head rows.
    container_rows: List[dict] = field(default_factory=list)
    applied: bool = False
    signed: bool = False
    notes: List[str] = field(default_factory=list)


def _unchanged(tb_rows: List[TBRow], note: str) -> TallyNormalizedRows:
    return TallyNormalizedRows(
        ledger_rows=list(tb_rows), ledger_grid_indices=[], container_rows=[], applied=False, signed=False, notes=[note],
    )


def normalize_tally_rows(
    path: Path, sheet_name: str, header_row: int, cols: dict, tb_rows: List[TBRow],
) -> TallyNormalizedRows:
    """Re-signs opening/closing from Dr/Cr number formats and splits Group
    rows from Ledger rows. Never raises."""
    try:
        return _normalize(path, sheet_name, header_row, cols, tb_rows)
    except Exception as exc:  # fail-safe: never block ingestion on this step
        return _unchanged(tb_rows, f"tally_row_normalization_skipped: {type(exc).__name__}: {exc}")


def _normalize(path: Path, sheet_name: str, header_row: int, cols: dict, tb_rows: List[TBRow]) -> TallyNormalizedRows:
    label_col = cols.get("gl_name")
    if label_col is None or not tb_rows:
        return _unchanged(tb_rows, "no gl_name column or no rows")

    grid = load_bold_aware_grid(path, sheet_name)

    # Replay build_tb_rows_from_grid's own keep/skip rule to recover each
    # tb_row's source grid index; verify by name so any drift aborts safely.
    aligned: List[int] = []
    for idx in range(header_row + 1, len(grid)):
        row = grid[idx]
        code_cell = row[cols["gl_code"]].value if "gl_code" in cols and cols["gl_code"] < len(row) else None
        name_raw = row[label_col].value if label_col < len(row) else None
        name = str(name_raw).strip() if name_raw not in (None, "") else ""
        if code_cell in (None, "") and not name:
            continue
        if is_total_row_label(name):
            continue
        aligned.append(idx)
    if len(aligned) != len(tb_rows) or any(
        str(grid[i][label_col].value or "").strip() != r.gl_name for i, r in zip(aligned, tb_rows)
    ):
        return _unchanged(tb_rows, "tb_rows do not align with source grid; left unchanged")

    def _side_sign(idx: int, field_name: str) -> Optional[int]:
        col = cols.get(field_name)
        if col is None or col >= len(grid[idx]):
            return None
        fmt = grid[idx][col].number_format or ""
        return sign_from_number_format(fmt) if ("dr" in fmt.lower() or "cr" in fmt.lower()) else None

    signed_rows: Dict[int, TBRow] = {}
    any_signed = False
    for idx, row in zip(aligned, tb_rows):
        opening_sign = _side_sign(idx, "opening")
        closing_sign = _side_sign(idx, "closing")
        any_signed = any_signed or opening_sign is not None or closing_sign is not None
        signed_rows[idx] = replace(
            row,
            opening=row.opening * (opening_sign or 1),
            closing=row.closing * (closing_sign or 1),
        )

    # Structural split (uses the same Dr/Cr signs via extract_structural_rows).
    structural_rows = extract_structural_rows(path, sheet_name, header_row, cols)
    notes: List[str] = []
    group_indices: set = set()
    container_rows: List[dict] = []
    if structural_rows:
        nodes, ambiguous = build_hierarchy_with_ambiguity(structural_rows)
        result = resolve_main_head_per_row(nodes, ambiguous_row_indices=ambiguous)
        group_indices = set(result.group_row_indices)

        # Safety: ledgers must sum to what the top-level nodes claim, else
        # a row was wrongly classed as a container and money would be lost.
        ledger_sum = sum(signed_rows[i].closing for i in aligned if i not in group_indices and i in signed_rows)
        top_sum = sum(n.value for n in nodes)
        if abs(ledger_sum - top_sum) > _SUM_CHECK_TOLERANCE:
            notes.append(
                f"container_split_skipped: ledger sum {ledger_sum:,.2f} != top-level sum {top_sum:,.2f}"
            )
            group_indices = set()
        else:
            label_of = {r.row_index: r.label for r in structural_rows}
            ancestor_path = result.erp_group_path_by_row
            for i in aligned:
                if i in group_indices:
                    container_rows.append({
                        "gl_name": signed_rows[i].gl_name,
                        "grid_row_index": i,
                        "opening": signed_rows[i].opening,
                        "closing": signed_rows[i].closing,
                        "label": label_of.get(i, signed_rows[i].gl_name),
                    })
            _ = ancestor_path  # group ancestry is carried by leaf erp_group_path
    else:
        notes.append("no structural rows; container split not attempted")

    ledger_idx = [i for i in aligned if i not in group_indices]
    return TallyNormalizedRows(
        ledger_rows=[signed_rows[i] for i in ledger_idx],
        ledger_grid_indices=ledger_idx,
        container_rows=container_rows,
        applied=True,
        signed=any_signed,
        notes=notes,
    )

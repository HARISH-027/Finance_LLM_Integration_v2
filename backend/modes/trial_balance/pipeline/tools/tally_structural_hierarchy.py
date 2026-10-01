"""Tally native-export structural hierarchy extractor (TB Multi-ERP
Adaptability upgrade -- see docs/TBS-from-ERP/Tally/ real-fixture analysis).

Tally's native TB export has no GL code and no Main Head/Sub Head 1/Sub
Head 2 columns -- group vs ledger is encoded purely structurally, in a
single Particulars column holding both Group-label rows and Ledger rows
with nothing marking which is which. Confirmed against real client files
(Mumbai Aviation Fuel Farm, Petronet VK): no leading-whitespace
indentation, no literal "Total"/"Grand Total" rows closing a section, and
the GROUP ROW APPEARS FIRST, with its children listed below it (pre-order,
not a trailing subtotal) -- but a Group row's own closing balance always
exactly equals the sum of the closing balances of the sibling subtrees
that follow it (verified arithmetically, e.g. Capital Account = Reserves &
Surplus + 4x Shareholder ledgers, and recursively Reserves & Surplus =
its own single child, Profit & Loss Account).

Because a group appears BEFORE its children, reconciling a candidate row
against "the sum of the following N raw rows" is wrong whenever there is
more than one level of nesting -- a nested subtotal's own descendant rows
would be double-counted (once via the subtotal's own row value, again via
each descendant's row value). This module instead parses with proper
recursive descent: each sibling subtree is resolved FIRST (consuming
however many raw rows it needs, one or many), and a candidate row is a
group only if the cumulative value of successive, already-resolved SIBLING
SUBTREES (not raw rows) reconciles to its own value.

A second, real ambiguity: sum-reconciliation alone cannot distinguish a
genuine single-child nested group (e.g. "Reserves & Surplus" containing
only "Profit & Loss Account", both valued identically since it's the
group's only child) from two flat LEAF siblings that simply happen to
carry equal balances (verified in real data: all 4 shareholder loan
ledgers -- BPCL/HPCL/IOCL/MIAL -- carry the exact same value). A match
against exactly one following subtree is therefore only accepted as a
genuine group when corroborated by an independent signal: the candidate's
label is one of Tally's 13 shipped sub-group names (from the client's own
audit-methodology doc, section 5.4), or the candidate's label cell is
bold (confirmed reliable at the top level of the hierarchy on every real
fixture inspected). A match against two or more following subtrees needs
no such corroboration -- it's implausible for several distinct ledgers to
coincidentally sum to an unrelated total by chance.

Distinct from input_grouping_source.build_hint_map(): that module's row-
type discriminator is gl_code-plausibility + literal Total/Sub-total text
matching -- both structurally absent from Tally's native export. Do not
extend that function for this case; it is designed for OTHER ERPs'
separate grouping files.

Only ever claims a `main_head` (the TOP-LEVEL reconciled ancestor) for a
ledger row -- never sub_head_1/sub_head_2. Forcing a fixed 2-level split
onto Tally's arbitrary-depth nesting would fabricate structure that
genuinely isn't in the source file.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import openpyxl

from modes.trial_balance.pipeline.tools.input_header_detect import is_total_row_label, parse_amount
from modes.trial_balance.pipeline.tools.input_long_path import to_long_path

# Mirrors quality_gate.DEFAULT_ARITHMETIC_TOLERANCE / scope_aware_quality_gate.DEFAULT_TOLERANCE
# (both 0.01) -- kept as its own local constant rather than importing either,
# since this module has no other dependency on either file and a shared
# import would be a needless coupling for one float.
_RECONCILE_TOLERANCE = 0.01

_NOISE_LABELS = {"", "-", "--"}

# Tally's 13 shipped sub-groups (client audit-methodology doc, section 5.4)
# -- used only as a corroborating signal for an otherwise-ambiguous
# single-child reconciliation match (see module docstring). NOT a
# classification lookup (that's tally_primary_group_lookup.py); an org's
# own custom sub-groups are legitimately absent from this list, which is
# fine -- they still qualify as groups via the k>=2 or is_bold signals.
_KNOWN_TALLY_SUB_GROUPS = {
    "reserves & surplus",
    "bank od a/c",
    "secured loans",
    "unsecured loans",
    "duties & taxes",
    "provisions",
    "sundry creditors",
    "bank accounts",
    "cash-in-hand",
    "deposits (asset)",
    "loans & advances (asset)",
    "stock-in-hand",
    "sundry debtors",
}


def sign_from_number_format(number_format: Optional[str]) -> int:
    """Tally native exports store the balance side ONLY in the cell number
    format (`0.00" Dr"` / `0.00" Cr"`) -- the stored value is always a
    positive magnitude. Dr -> +1, Cr -> -1 (matching the debit-minus-credit
    convention build_tb_rows_from_grid already uses); a format with neither
    suffix leaves the value untouched (+1)."""
    fmt = (number_format or "").lower()
    if '" cr"' in fmt or fmt.rstrip("\"; ").endswith("cr"):
        return -1
    return 1


def _normalize_label(label: str) -> str:
    return " ".join(label.strip().lower().split())


@dataclass(frozen=True)
class StructuralRow:
    """One physical ledger-candidate row on the sheet, as seen by this
    module only. `row_index` is the absolute row index in the sheet (for
    audit trail)."""

    row_index: int
    label: str
    value: float
    is_bold: bool


@dataclass(frozen=True)
class HierarchyNode:
    label: str
    value: float
    row_index: int
    is_group: bool
    children: Tuple["HierarchyNode", ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class StructuralHierarchyResult:
    # Keyed by row_index. A ledger leaf row maps to its top-level ancestor's
    # label, or None if it was never captured inside any reconciled group
    # (flat/ungrouped -- genuinely unresolved, not a bug to paper over).
    main_head_by_row: Dict[int, Optional[str]] = field(default_factory=dict)
    # " -> ".join(ancestor labels top-to-bottom), "" if the row has no ancestors.
    erp_group_path_by_row: Dict[int, str] = field(default_factory=dict)
    unresolved_row_indices: List[int] = field(default_factory=list)
    # Row index of every row that reconciled as a Group/Head CONTAINER
    # (not an account) -- kept so callers can separate Head rows from
    # Ledger rows instead of discarding the distinction.
    group_row_indices: List[int] = field(default_factory=list)
    # Row index of any GROUP whose acceptance relied on a single-child
    # match corroborated only by the vocabulary/bold signal (never a bare
    # arithmetic coincidence) -- disclosed for audit visibility, never
    # silently accepted and hidden.
    ambiguous_row_indices: List[int] = field(default_factory=list)
    resolution_rate: float = 0.0


def load_bold_aware_grid(path: Path, sheet_name: str) -> list:
    """Loads the workbook WITHOUT read_only=True (unlike every other grid
    loader in this codebase, e.g. input_scenario_b._load_workbook_sheets /
    input_grouping_source.load_raw_grid), because read_only mode strips
    cell.font -- confirmed against real Mumbai Aviation Fuel Farm/Petronet
    VK fixtures, where read_only cells carry no style information at all.
    Returns a list of tuples of openpyxl Cell OBJECTS (not raw values, and
    not the values_only=True convention used elsewhere), since callers here
    need both `.value` and `.font.bold` per cell. Scoped only to this
    Tally structural-extraction path -- non-read-only load is slower and
    more memory-hungry on very large workbooks, an acceptable, deliberate,
    isolated cost here, never the default TB parse."""
    wb = openpyxl.load_workbook(to_long_path(Path(path)), data_only=True)
    try:
        ws = wb[sheet_name]
        return [tuple(row) for row in ws.iter_rows()]
    finally:
        wb.close()


def extract_structural_rows(path: Path, sheet_name: str, header_row: int, cols: dict) -> List[StructuralRow]:
    """Re-opens the workbook via load_bold_aware_grid (the primary parse
    already happened via a read_only loader elsewhere in the pipeline;
    duplicating the load is the deliberate, documented cost of recovering
    font info) and walks rows after header_row, pairing each row's
    gl_name-column text with the closing-column value and that same
    label cell's bold flag. Filters empty/noise labels and any literal
    Total/Grand Total row (via the shared is_total_row_label helper) --
    neither is a ledger-candidate row. Returns [] if the sheet doesn't
    carry both a gl_name and a closing column (nothing to reconcile against)."""
    label_col = cols.get("gl_name")
    value_col = cols.get("closing")
    if label_col is None or value_col is None:
        return []

    ws_rows = load_bold_aware_grid(path, sheet_name)
    result: List[StructuralRow] = []
    for row_index in range(header_row + 1, len(ws_rows)):
        row = ws_rows[row_index]
        if label_col >= len(row) or value_col >= len(row):
            continue
        label_cell = row[label_col]
        label_raw = label_cell.value
        if label_raw is None:
            continue
        label = str(label_raw).strip()
        if not label or label in _NOISE_LABELS or is_total_row_label(label):
            continue
        value_cell = row[value_col]
        value = parse_amount(value_cell.value) * sign_from_number_format(value_cell.number_format)
        bold = bool(label_cell.font.bold) if label_cell.font is not None else False
        result.append(StructuralRow(row_index=row_index, label=label, value=value, is_bold=bold))
    return result


@dataclass(frozen=True)
class _Option:
    """One candidate parse of a row: the node it becomes, the row index
    just past its subtree, and how many unproven ('soft') groups the
    subtree relies on (fewer is preferred when several parses reconcile)."""

    node: HierarchyNode
    next_index: int
    soft: int


# Cap on distinct (cumulative-sum) states kept per row position while
# searching for a reconciling child chain -- bounds worst-case work.
_MAX_STATES_PER_POSITION = 32


class _Parser:
    """Group-before-children (pre-order) parse, right to left, WITHOUT
    native recursion (a naive recursive version hit Python's limit on a
    real 560-row fixture). rows[i]'s alternatives depend only on rows
    after i, so every row is resolved once in decreasing index order.

    Each row keeps up to two alternatives:
      * STRICT group -- children reconcile AND the row is corroborated
        (>=2 non-zero children, bold, or a known Tally sub-group name).
        Only this alternative is kept.
      * SOFT group -- children reconcile but only via a single non-zero
        child (e.g. `Integrated GST = INPUT (IGST)`). Locally
        indistinguishable from a coincidentally equal sibling, so BOTH the
        leaf and the group alternative are kept; an ancestor's own
        reconciliation decides which one is consistent (a single
        coincidence cannot satisfy every ancestor's sum too).
    """

    def __init__(self, rows: List[StructuralRow], tolerance: float = _RECONCILE_TOLERANCE):
        self._rows = rows
        self._tolerance = tolerance
        self.ambiguous_row_indices: List[int] = []
        self._memo: Dict[int, List[_Option]] = {}

    def parse_siblings(self, start: int, end: int) -> List[HierarchyNode]:
        for i in range(end - 1, start - 1, -1):
            if i not in self._memo:
                self._memo[i] = self._resolve_options(i, end)

        # Top level has no target sum to disambiguate against, so the
        # first (most conservative) alternative is taken.
        nodes: List[HierarchyNode] = []
        i = start
        while i < end:
            option = self._memo[i][0]
            nodes.append(option.node)
            i = option.next_index
        return nodes

    def _resolve_options(self, i: int, end: int) -> List[_Option]:
        row = self._rows[i]
        leaf = _Option(
            HierarchyNode(label=row.label, value=row.value, row_index=row.row_index, is_group=False), i + 1, 0,
        )
        if abs(row.value) <= self._tolerance:
            return [leaf]  # zero rows are never group candidates
        match = self._match_children(i + 1, end, row.value)
        if match is None:
            return [leaf]
        children, next_i, was_ambiguous, soft_inside = match
        # Zero-valued children carry no reconciliation evidence, so they
        # must not by themselves satisfy the >=2-children gate.
        nonzero_children = sum(1 for c in children if abs(c.value) > self._tolerance)
        strict = nonzero_children >= 2 or row.is_bold or _normalize_label(row.label) in _KNOWN_TALLY_SUB_GROUPS
        group = _Option(
            HierarchyNode(
                label=row.label, value=row.value, row_index=row.row_index,
                is_group=True, children=tuple(children),
            ),
            next_i, soft_inside + (0 if strict else 1),
        )
        if was_ambiguous:
            self.ambiguous_row_indices.append(row.row_index)
        return [group] if strict else [leaf, group]

    def _match_children(
        self, start: int, end: int, target_value: float,
    ) -> Optional[Tuple[List[HierarchyNode], int, bool, int]]:
        """Forward search over sibling-subtree alternatives for a chain
        whose values sum to target_value. States are merged by cumulative
        sum (so a position never fans out beyond distinct sums). Returns
        the LONGEST reconciling chain (fewest soft groups on ties), whether
        more than one chain length reconciled, and its soft-group count."""
        scale = 1.0 / self._tolerance
        # states[pos][key] = (cumulative, soft, prev_state, node)
        states: Dict[int, Dict[int, tuple]] = {start: {0: (0.0, 0, None, None)}}
        matches: List[Tuple[int, int, tuple]] = []  # (next_index, soft, state)
        for pos in range(start, end):
            here = states.pop(pos, None)
            if not here:
                continue
            for state in here.values():
                cum, soft, _, _ = state
                for option in self._memo[pos]:
                    q = option.next_index
                    ncum = cum + option.node.value
                    nsoft = soft + option.soft
                    nstate = (ncum, nsoft, state, option.node)
                    bucket = states.setdefault(q, {})
                    key = int(round(ncum * scale))
                    if key not in bucket or bucket[key][1] > nsoft:
                        if len(bucket) < _MAX_STATES_PER_POSITION or key in bucket:
                            bucket[key] = nstate
                    if abs(ncum - target_value) <= self._tolerance:
                        matches.append((q, nsoft, nstate))
        if not matches:
            return None
        best_q = max(m[0] for m in matches)
        best = min((m for m in matches if m[0] == best_q), key=lambda m: m[1])
        chain: List[HierarchyNode] = []
        state = best[2]
        while state is not None and state[3] is not None:
            chain.append(state[3])
            state = state[2]
        chain.reverse()
        distinct_ends = len({m[0] for m in matches})
        return chain, best_q, distinct_ends > 1, best[1]


def build_hierarchy(rows: List[StructuralRow]) -> List[HierarchyNode]:
    """Public entry point: parses the full flat `rows` sequence into a
    forest of top-level sibling nodes (typically one node per Tally
    Primary Group, since real exports rarely reduce to a single top-level
    total). Pure function -- no I/O, directly unit-testable against
    hand-built StructuralRow lists."""
    parser = _Parser(rows)
    return parser.parse_siblings(0, len(rows))


def build_hierarchy_with_ambiguity(rows: List[StructuralRow]) -> Tuple[List[HierarchyNode], List[int]]:
    """Same as build_hierarchy, but also returns the row indices of any
    group whose acceptance relied on a single-child match corroborated
    only by the vocabulary/bold signal, for callers that need audit
    visibility into ambiguous resolutions."""
    parser = _Parser(rows)
    nodes = parser.parse_siblings(0, len(rows))
    # parse_one_node is not memoized -- the same row can legitimately be
    # recomputed via more than one call path (e.g. tested once as a
    # sibling-subtree candidate for two different ancestors), so dedupe
    # while preserving first-seen order rather than reporting a row twice.
    return nodes, list(dict.fromkeys(parser.ambiguous_row_indices))


def resolve_main_head_per_row(
    nodes: List[HierarchyNode], ambiguous_row_indices: Optional[List[int]] = None,
) -> StructuralHierarchyResult:
    """Flattens the tree: every LEDGER LEAF row's main_head is its
    TOP-LEVEL ancestor's label only (never an intermediate ancestor --
    main_head names the top of the hierarchy specifically). erp_group_path
    is the full ancestor chain top-to-bottom, for audit trail. A row that
    resolved as a Group itself is a container, not an account -- excluded
    from main_head_by_row and from resolution_rate's denominator, so it is
    never double-counted as an unresolved ledger. A leaf with NO ancestors
    at all (never captured inside any reconciled group) gets
    main_head_by_row[row_index] = None and is listed in
    unresolved_row_indices."""
    main_head_by_row: Dict[int, Optional[str]] = {}
    erp_group_path_by_row: Dict[int, str] = {}
    unresolved: List[int] = []
    group_rows: List[int] = []

    def walk(node_list: List[HierarchyNode], ancestors: List[str]) -> None:
        for node in node_list:
            if node.is_group:
                group_rows.append(node.row_index)
                walk(list(node.children), ancestors + [node.label])
            else:
                main_head_by_row[node.row_index] = ancestors[0] if ancestors else None
                erp_group_path_by_row[node.row_index] = " -> ".join(ancestors)
                if not ancestors:
                    unresolved.append(node.row_index)

    walk(nodes, [])
    total = len(main_head_by_row)
    resolved = sum(1 for v in main_head_by_row.values() if v is not None)
    resolution_rate = (resolved / total) if total else 0.0

    return StructuralHierarchyResult(
        main_head_by_row=main_head_by_row,
        erp_group_path_by_row=erp_group_path_by_row,
        unresolved_row_indices=unresolved,
        group_row_indices=group_rows,
        ambiguous_row_indices=list(ambiguous_row_indices or []),
        resolution_rate=resolution_rate,
    )


def extract_tally_structural_hierarchy(
    path: Path, sheet_name: str, header_row: int, cols: dict,
) -> StructuralHierarchyResult:
    """Single public entry point for callers that want the full result
    (row-index-keyed, with resolution_rate/ambiguity metadata). Never
    raises: any internal failure (corrupt bold read, header misdetection,
    missing columns, etc.) is caught and converted into an all-unresolved,
    resolution_rate=0.0 result -- mirrors erp_adapter_integration.
    enrich_parsed_input_with_erp_adapter's own 'best-effort, never raises'
    contract at a smaller grain."""
    try:
        rows = extract_structural_rows(path, sheet_name, header_row, cols)
        if not rows:
            return StructuralHierarchyResult()
        nodes, ambiguous = build_hierarchy_with_ambiguity(rows)
        return resolve_main_head_per_row(nodes, ambiguous_row_indices=ambiguous)
    except Exception:
        return StructuralHierarchyResult()


def extract_tally_structural_fields_by_identity(
    path: Path, sheet_name: str, header_row: int, cols: dict,
) -> Dict[str, Dict[str, str]]:
    """Same result as extract_tally_structural_hierarchy, but re-keyed by
    each row's own label text (the gl_name column's value) instead of
    absolute row_index -- the join key tally_adapter.py's
    classification_by_gl_code dict already uses (gl_code, or gl_name when
    no gl_code exists -- the dominant case this extractor targets, where
    gl_code IS gl_name, so this key matches exactly).

    Returns {identity: {"main_head": ..., "erp_group_path": ...}} -- only
    for rows with a resolved main_head; a row with none is omitted
    entirely (never an empty/None entry), so callers can safely
    `.get(identity, {})`. erp_group_path is the FULL ancestor chain (audit
    trail), which may be deeper than main_head alone when nested groups
    were resolved (e.g. "Capital Account -> Reserves & Surplus") -- kept
    separate from main_head (which always names only the top-level
    ancestor) so a caller can choose how much of the chain to surface.

    Never raises; returns {} on any failure, same contract as
    extract_tally_structural_hierarchy."""
    try:
        rows = extract_structural_rows(path, sheet_name, header_row, cols)
        if not rows:
            return {}
        nodes, ambiguous = build_hierarchy_with_ambiguity(rows)
        result = resolve_main_head_per_row(nodes, ambiguous_row_indices=ambiguous)
        label_by_row_index = {r.row_index: r.label for r in rows}
        fields_by_identity: Dict[str, Dict[str, str]] = {}
        for row_index, main_head in result.main_head_by_row.items():
            if main_head is None or row_index not in label_by_row_index:
                continue
            identity = label_by_row_index[row_index]
            fields_by_identity[identity] = {
                "main_head": main_head,
                "erp_group_path": result.erp_group_path_by_row.get(row_index, main_head),
            }
        return fields_by_identity
    except Exception:
        return {}

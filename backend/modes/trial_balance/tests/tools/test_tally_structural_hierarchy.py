"""Unit tests for tally_structural_hierarchy.py's recursive-descent
sum-reconciliation parser. Synthetic StructuralRow lists only -- no real
fixtures needed here (see a real-fixture sibling test file for real-file
coverage). Mirrors the existing ERP-adapter test convention: class-grouped,
plain test_... methods, no parametrize, no shared grid fixtures.
"""

from modes.trial_balance.pipeline.tools.tally_structural_hierarchy import (
    HierarchyNode,
    StructuralRow,
    build_hierarchy,
    build_hierarchy_with_ambiguity,
    resolve_main_head_per_row,
)


def _row(row_index: int, label: str, value: float, is_bold: bool = False) -> StructuralRow:
    return StructuralRow(row_index=row_index, label=label, value=value, is_bold=is_bold)


class TestBuildHierarchyRealShapeReconstruction:
    """Real Mumbai Aviation Fuel Farm shape: group row FIRST, children
    listed below it (pre-order), values scaled down for readability."""

    def _capital_account_rows(self):
        return [
            _row(0, "Capital Account", 2552.85, is_bold=True),
            _row(1, "Reserves & Surplus", 436.10),
            _row(2, "Profit & Loss Account (Retained Earnings)", 436.10),
            _row(3, "BPCL - Shareholder", 529.1875),
            _row(4, "HPCL - Shareholder", 529.1875),
            _row(5, "IOCL - Shareholder", 529.1875),
            _row(6, "MIAL - Shareholder", 529.1875),
        ]

    def test_three_level_nesting_recovers_full_tree(self):
        nodes = build_hierarchy(self._capital_account_rows())
        assert len(nodes) == 1
        capital = nodes[0]
        assert capital.is_group is True
        assert capital.label == "Capital Account"
        # Reserves & Surplus (nested subtotal) + 4 flat ledger leaves = 5 children
        assert len(capital.children) == 5
        reserves, bpcl, hpcl, iocl, mial = capital.children
        assert reserves.is_group is True
        assert reserves.label == "Reserves & Surplus"
        assert len(reserves.children) == 1
        assert reserves.children[0].is_group is False
        assert reserves.children[0].label == "Profit & Loss Account (Retained Earnings)"
        for leaf in (bpcl, hpcl, iocl, mial):
            assert leaf.is_group is False

    def test_coincidentally_equal_leaf_siblings_are_not_falsely_nested(self):
        # The critical false-positive guard: BPCL/HPCL/IOCL/MIAL all carry
        # the exact same value (true in the real fixture too). A naive
        # single-child sum match would misread HPCL as a "child" of BPCL,
        # IOCL as a "child" of HPCL, etc. None of these labels are in the
        # known sub-group vocabulary and none are bold -- so a k=1 match
        # must be rejected for every one of them.
        nodes = build_hierarchy(self._capital_account_rows())
        capital = nodes[0]
        _reserves, bpcl, hpcl, iocl, mial = capital.children
        for leaf in (bpcl, hpcl, iocl, mial):
            assert leaf.is_group is False
            assert leaf.children == ()

    def test_reserves_and_surplus_accepted_via_known_subgroup_vocabulary(self):
        # Source row is not bold in the real data (see _capital_account_rows) --
        # acceptance must come from the vocabulary match, not the bold signal.
        nodes = build_hierarchy(self._capital_account_rows())
        reserves = nodes[0].children[0]
        assert reserves.label == "Reserves & Surplus"
        assert reserves.is_group is True


class TestBuildHierarchySingleChildAcceptanceGate:
    def test_unknown_label_single_child_bold_is_accepted(self):
        # An org-created custom sub-group (not in the shipped vocabulary)
        # with exactly one child is still accepted as a group when the
        # candidate row is bold -- the second corroborating signal.
        rows = [
            _row(0, "Custom Consolidation Adjustment", 50.0, is_bold=True),
            _row(1, "Inter-Unit Clearing", 50.0),
        ]
        nodes = build_hierarchy(rows)
        assert len(nodes) == 1
        assert nodes[0].is_group is True
        assert len(nodes[0].children) == 1

    def test_unknown_label_single_child_not_bold_is_rejected(self):
        # Same shape, no corroborating signal at all -- must NOT be
        # treated as a group; both rows are flat leaf siblings.
        rows = [
            _row(0, "Some Ledger A", 50.0),
            _row(1, "Some Ledger B", 50.0),
        ]
        nodes = build_hierarchy(rows)
        assert len(nodes) == 2
        assert all(not n.is_group for n in nodes)

    def test_two_or_more_children_accepted_without_corroboration(self):
        # k>=2 needs no vocabulary/bold corroboration at all -- implausible
        # for two distinct, differently-valued ledgers to coincidentally
        # sum to an unrelated total by chance.
        rows = [
            _row(0, "Sundry Debtors", 100.0),
            _row(1, "ABC Traders", 60.0),
            _row(2, "XYZ Traders", 40.0),
        ]
        nodes = build_hierarchy(rows)
        assert len(nodes) == 1
        assert nodes[0].is_group is True
        assert len(nodes[0].children) == 2


class TestBuildHierarchyEdgeCases:
    def test_zero_valued_candidate_never_treated_as_group(self):
        # A zero group-total is indistinguishable from a zero-balance
        # ledger reconciling against coincidentally zero-valued siblings.
        rows = [
            _row(0, "Suspense A/c", 0.0),
            _row(1, "Some Zero Ledger A", 0.0),
            _row(2, "Some Zero Ledger B", 0.0),
        ]
        nodes = build_hierarchy(rows)
        assert len(nodes) == 3
        assert all(not n.is_group for n in nodes)

    def test_no_reconciliation_possible_returns_all_leaves(self):
        rows = [
            _row(0, "Sales - Power", 100.0),
            _row(1, "Unrelated Ledger", 40.0),
            _row(2, "Another Ledger", 20.0),
        ]
        nodes = build_hierarchy(rows)
        assert len(nodes) == 3
        assert all(not n.is_group for n in nodes)

    def test_flat_ungrouped_rows_all_become_leaves(self):
        rows = [
            _row(0, "Ledger A", 10.0),
            _row(1, "Ledger B", 20.0),
            _row(2, "Ledger C", 30.0),
        ]
        nodes = build_hierarchy(rows)
        assert all(not n.is_group for n in nodes)
        assert len(nodes) == 3

    def test_empty_input_returns_empty(self):
        assert build_hierarchy([]) == []

    def test_last_row_has_no_following_rows_is_a_leaf(self):
        rows = [_row(0, "Only Row", 100.0)]
        nodes = build_hierarchy(rows)
        assert len(nodes) == 1
        assert nodes[0].is_group is False


class TestAmbiguityDisclosure:
    def test_ambiguous_multiple_prefix_lengths_prefers_longest_and_flags(self):
        # rows[1] alone matches the target at k=1 (coincidentally), AND the
        # full run [1,2] also matches at k=2 (rows[2].value == 0.0) --
        # longest must win, and the match is disclosed as ambiguous.
        rows = [
            _row(0, "Sundry Creditors", 100.0),
            _row(1, "ABC Traders", 100.0),
            _row(2, "Zero Balance Ledger", 0.0),
        ]
        nodes, ambiguous = build_hierarchy_with_ambiguity(rows)
        assert len(nodes) == 1
        assert nodes[0].is_group is True
        assert len(nodes[0].children) == 2
        assert ambiguous == [0]

    def test_unambiguous_match_is_not_flagged(self):
        rows = [
            _row(0, "Sundry Debtors", 100.0),
            _row(1, "ABC Traders", 60.0),
            _row(2, "XYZ Traders", 40.0),
        ]
        _nodes, ambiguous = build_hierarchy_with_ambiguity(rows)
        assert ambiguous == []


class TestResolveMainHeadPerRow:
    def test_leaf_main_head_is_top_level_ancestor_only(self):
        nested = HierarchyNode(
            label="Reserves & Surplus", value=436.10, row_index=1, is_group=True,
            children=(HierarchyNode(label="Profit & Loss Account", value=436.10, row_index=2, is_group=False),),
        )
        leaf = HierarchyNode(label="BPCL - Shareholder", value=529.1875, row_index=3, is_group=False)
        capital = HierarchyNode(
            label="Capital Account", value=2552.85, row_index=0, is_group=True,
            children=(nested, leaf),
        )
        result = resolve_main_head_per_row([capital])
        # Both leaves' main_head is "Capital Account" (top level), never
        # the intermediate "Reserves & Surplus".
        assert result.main_head_by_row[2] == "Capital Account"
        assert result.erp_group_path_by_row[2] == "Capital Account -> Reserves & Surplus"
        assert result.main_head_by_row[3] == "Capital Account"
        assert result.erp_group_path_by_row[3] == "Capital Account"
        # Group container rows themselves are never in main_head_by_row.
        assert 1 not in result.main_head_by_row
        assert 0 not in result.main_head_by_row
        assert result.resolution_rate == 1.0

    def test_row_with_no_ancestors_is_unresolved(self):
        leaf = HierarchyNode(label="Flat Ledger", value=10.0, row_index=5, is_group=False)
        result = resolve_main_head_per_row([leaf])
        assert result.main_head_by_row[5] is None
        assert 5 in result.unresolved_row_indices
        assert result.resolution_rate == 0.0

    def test_ambiguous_row_indices_passed_through(self):
        leaf = HierarchyNode(label="Flat Ledger", value=10.0, row_index=5, is_group=False)
        result = resolve_main_head_per_row([leaf], ambiguous_row_indices=[0])
        assert result.ambiguous_row_indices == [0]

    def test_empty_input_yields_zero_resolution_rate_no_crash(self):
        result = resolve_main_head_per_row([])
        assert result.resolution_rate == 0.0
        assert result.main_head_by_row == {}


class TestSoftGroupAndSignedReconciliation:
    def test_single_child_group_needed_by_ancestor_is_kept(self):
        # KTPO: Integrated GST = its one non-zero child; the ancestor only reconciles if it is a group.
        rows = [
            _row(0, "GST", -310.0),
            _row(1, "Central", -110.0), _row(2, "Integrated", -50.0), _row(3, "IGST", 0.0),
            _row(4, "INPUT IGST", -50.0), _row(5, "State", -150.0),
        ]
        nodes = build_hierarchy(rows)
        assert len(nodes) == 1 and nodes[0].label == "GST"
        assert [c.label for c in nodes[0].children] == ["Central", "Integrated", "State"]
        assert nodes[0].children[1].is_group is True

    def test_equal_siblings_with_zero_rows_are_not_falsely_nested(self):
        rows = [
            _row(0, "GST TDS", -36.0),
            _row(1, "CGST TDS", -18.0), _row(2, "SGST TDS", -18.0),
            _row(3, "ITC Reconciliation", 0.0), _row(4, "ITC Claimed", 0.0),
        ]
        nodes = build_hierarchy(rows)
        group = nodes[0]
        assert group.is_group and [c.label for c in group.children][:2] == ["CGST TDS", "SGST TDS"]
        assert group.children[0].is_group is False

    def test_top_level_signed_values_sum_to_zero(self):
        rows = [_row(0, "Capital", -100.0, is_bold=True), _row(1, "Equity", -100.0),
                _row(2, "Assets", 100.0, is_bold=True), _row(3, "Cash", 100.0)]
        assert sum(n.value for n in build_hierarchy(rows)) == 0.0


class TestSignFromNumberFormat:
    def test_cr_is_negative_dr_is_positive_plain_untouched(self):
        from modes.trial_balance.pipeline.tools.tally_structural_hierarchy import sign_from_number_format
        assert sign_from_number_format('0.00" Cr"') == -1
        assert sign_from_number_format('""0.00" Dr"') == 1
        assert sign_from_number_format('""0.00') == 1
        assert sign_from_number_format(None) == 1

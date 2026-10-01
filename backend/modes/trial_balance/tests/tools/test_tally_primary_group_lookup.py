"""Unit tests for tally_primary_group_lookup.py's static Primary Group ->
(bs_pl, account_type) lookup."""

from modes.trial_balance.pipeline.tools.tally_primary_group_lookup import classify_primary_group


class TestKnownPrimaryGroups:
    def test_all_thirteen_fixed_nature_groups_classify_correctly(self):
        expected = {
            "Capital Account": ("BS", "Equity"),
            "Loans (Liability)": ("BS", "Liability"),
            "Current Liabilities": ("BS", "Liability"),
            "Fixed Assets": ("BS", "Asset"),
            "Investments": ("BS", "Asset"),
            "Current Assets": ("BS", "Asset"),
            "Misc. Expenses (ASSET)": ("BS", "Asset"),
            "Sales Accounts": ("PL", "Income"),
            "Direct Incomes": ("PL", "Income"),
            "Indirect Incomes": ("PL", "Income"),
            "Purchase Accounts": ("PL", "Expense"),
            "Direct Expenses": ("PL", "Expense"),
            "Indirect Expenses": ("PL", "Expense"),
        }
        for main_head, (bs_pl, account_type) in expected.items():
            result = classify_primary_group(main_head)
            assert result is not None, f"{main_head} should classify"
            assert result.bs_pl == bs_pl
            assert result.account_type == account_type

    def test_case_and_whitespace_insensitive(self):
        result = classify_primary_group("  capital account  ")
        assert result is not None
        assert result.bs_pl == "BS"
        assert result.account_type == "Equity"


class TestSuspenseSpecialCase:
    def test_suspense_returns_unclassified_not_a_guess(self):
        result = classify_primary_group("Suspense A/c")
        assert result is not None
        assert result.bs_pl == "BS"
        assert result.account_type == "unclassified"


class TestBranchDivisionsSpecialCase:
    def test_debit_balance_classifies_as_asset(self):
        result = classify_primary_group("Branch/Divisions", closing_balance=100.0)
        assert result is not None
        assert result.bs_pl == "BS"
        assert result.account_type == "Asset"

    def test_credit_balance_classifies_as_liability(self):
        result = classify_primary_group("Branch/Divisions", closing_balance=-100.0)
        assert result is not None
        assert result.bs_pl == "BS"
        assert result.account_type == "Liability"

    def test_no_closing_balance_supplied_returns_none(self):
        # Must not silently pick a sign -- force the caller to fall back.
        assert classify_primary_group("Branch/Divisions") is None


class TestUnknownCustomPrimaryGroup:
    def test_unrecognized_group_returns_none(self):
        assert classify_primary_group("Some Org-Created Custom Group") is None

    def test_none_main_head_returns_none(self):
        assert classify_primary_group(None) is None

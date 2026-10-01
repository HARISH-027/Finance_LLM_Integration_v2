"""Static lookup: Tally's 15 shipped Primary Groups -> (bs_pl, account_type).

Derived from the client's own audit-methodology doc ("Reading TALLY Trial
Balance v3.0"), section 5.4's table of the fifteen primary groups and where
each one lands (Balance Sheet vs Profit & Loss). Org-created custom primary
groups (anything not in this dict) are NOT guessed here -- they fall
through to the existing taxonomy resolver/LLM classification path exactly
like any other unrecognized main_head, mirroring
taxonomy_repository.derive_account_type's own "never guess, return empty"
philosophy.
"""

from __future__ import annotations

from typing import NamedTuple, Optional


class PrimaryGroupClassification(NamedTuple):
    bs_pl: str                      # "BS" | "PL"
    account_type: Optional[str]     # None only for the sign-dependent special case


_SUSPENSE_PRIMARY_GROUP = "Suspense A/c"
_BRANCH_DIVISION_PRIMARY_GROUP = "Branch/Divisions"

# The 13 shipped primary groups with a fixed, unambiguous nature. Suspense
# A/c and Branch/Divisions are deliberately NOT here -- both are special-
# cased in classify_primary_group() below rather than given a guessed
# fixed type.
_TALLY_PRIMARY_GROUP_LOOKUP: dict = {
    "Capital Account": PrimaryGroupClassification("BS", "Equity"),
    "Loans (Liability)": PrimaryGroupClassification("BS", "Liability"),
    "Current Liabilities": PrimaryGroupClassification("BS", "Liability"),
    "Fixed Assets": PrimaryGroupClassification("BS", "Asset"),
    "Investments": PrimaryGroupClassification("BS", "Asset"),
    "Current Assets": PrimaryGroupClassification("BS", "Asset"),
    # Despite the name, this is a Balance Sheet group per the client's own
    # doc -- deferred revenue expenditure, never reaches the P&L directly.
    "Misc. Expenses (ASSET)": PrimaryGroupClassification("BS", "Asset"),
    "Sales Accounts": PrimaryGroupClassification("PL", "Income"),
    "Direct Incomes": PrimaryGroupClassification("PL", "Income"),
    "Indirect Incomes": PrimaryGroupClassification("PL", "Income"),
    "Purchase Accounts": PrimaryGroupClassification("PL", "Expense"),
    "Direct Expenses": PrimaryGroupClassification("PL", "Expense"),
    "Indirect Expenses": PrimaryGroupClassification("PL", "Expense"),
}


def _normalize(main_head: str) -> str:
    return " ".join(main_head.strip().lower().split())


_NORMALIZED_LOOKUP = {_normalize(k): v for k, v in _TALLY_PRIMARY_GROUP_LOOKUP.items()}
_NORMALIZED_SUSPENSE = _normalize(_SUSPENSE_PRIMARY_GROUP)
_NORMALIZED_BRANCH_DIVISION = _normalize(_BRANCH_DIVISION_PRIMARY_GROUP)


def classify_primary_group(
    main_head: str, *, closing_balance: Optional[float] = None,
) -> Optional[PrimaryGroupClassification]:
    """Returns None for anything not one of Tally's 15 shipped groups --
    caller must route to the taxonomy resolver/LLM classification path in
    that case, never fabricate a guess (mirrors derive_account_type's own
    contract).

    Special case 1 (Suspense A/c): returns an explicit
    PrimaryGroupClassification(bs_pl="BS", account_type="unclassified") --
    an honest label, not a guessed real account_type. Per the client's own
    audit-methodology doc, a balance here is unallocated by definition and
    must be itemised, never silently bucketed as an asset/liability/etc.

    Special case 2 (Branch/Divisions): sign-dependent -- a debit balance
    reads as an asset-like (receivable) position, a credit balance as a
    liability-like (payable) position. Requires `closing_balance` to
    disambiguate; returns None (forcing the caller to fall back rather than
    silently pick a sign) if not supplied."""
    if main_head is None:
        return None
    normalized = _normalize(main_head)

    if normalized == _NORMALIZED_SUSPENSE:
        return PrimaryGroupClassification("BS", "unclassified")

    if normalized == _NORMALIZED_BRANCH_DIVISION:
        if closing_balance is None:
            return None
        account_type = "Asset" if closing_balance >= 0 else "Liability"
        return PrimaryGroupClassification("BS", account_type)

    return _NORMALIZED_LOOKUP.get(normalized)

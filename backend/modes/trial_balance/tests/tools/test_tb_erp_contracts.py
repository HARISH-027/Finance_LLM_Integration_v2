"""Tests for backend/tools/tb_erp_contracts.py -- the ERP-agnostic data
contracts introduced in Phase 1 of the TB Multi-ERP Adaptability upgrade.
Purely additive/structural: these dataclasses are not wired into any parser
or reasoner yet, so tests only cover construction, defaults, immutability,
and enum values."""

import dataclasses

import pytest

from modes.trial_balance.pipeline.tools.tb_erp_contracts import (
    CanonicalTBRow,
    ClassificationEvidence,
    ERPSourceMetadata,
    ERPType,
    ScenarioType,
    ScopeMetadata,
    ScopeType,
)


class TestERPType:
    def test_values_are_lowercase_strings(self):
        assert ERPType.TALLY.value == "tally"
        assert ERPType.ORACLE.value == "oracle"
        assert ERPType.SAP.value == "sap"
        assert ERPType.GENERIC.value == "generic"
        assert ERPType.UNKNOWN.value == "unknown"


class TestScenarioType:
    def test_values_match_existing_scenario_letters(self):
        assert ScenarioType.A.value == "A"
        assert ScenarioType.B.value == "B"
        assert ScenarioType.C.value == "C"
        assert ScenarioType.D.value == "D"


class TestScopeType:
    def test_values_are_lowercase_strings(self):
        assert ScopeType.FULL_LEDGER.value == "full_ledger"
        assert ScopeType.PROFIT_CENTER_SLICE.value == "profit_center_slice"
        assert ScopeType.UNKNOWN.value == "unknown"


class TestScopeMetadata:
    def test_defaults_are_all_none_with_unknown_scope_type(self):
        scope = ScopeMetadata()
        assert scope.entity is None
        assert scope.company_code is None
        assert scope.ledger is None
        assert scope.fiscal_year is None
        assert scope.period_from is None
        assert scope.period_to is None
        assert scope.currency is None
        assert scope.account_range is None
        assert scope.business_scope is None
        assert scope.profit_center is None
        assert scope.segment is None
        assert scope.scope_type == ScopeType.UNKNOWN

    def test_is_frozen(self):
        scope = ScopeMetadata(entity="ACME Ltd")
        with pytest.raises(dataclasses.FrozenInstanceError):
            scope.entity = "Other Ltd"


class TestClassificationEvidence:
    def test_defaults_to_empty_dicts_per_subgroup(self):
        evidence = ClassificationEvidence(erp_type=ERPType.SAP)
        assert evidence.erp_type == ERPType.SAP
        assert evidence.account_identity == {}
        assert evidence.account_classification == {}
        assert evidence.financial_statement == {}
        assert evidence.business_dimensions == {}

    def test_default_dicts_are_independent_per_instance(self):
        """A shared mutable default would leak state across instances --
        dataclasses.field(default_factory=dict) prevents that, but this
        test exists so the fix can't silently regress to a bare `= {}`."""
        first = ClassificationEvidence(erp_type=ERPType.TALLY)
        second = ClassificationEvidence(erp_type=ERPType.ORACLE)

        object.__setattr__(first, "account_identity", {**first.account_identity, "gl_name": "Salaries"})

        assert first.account_identity == {"gl_name": "Salaries"}
        assert second.account_identity == {}

    def test_is_frozen(self):
        evidence = ClassificationEvidence(erp_type=ERPType.SAP)
        with pytest.raises(dataclasses.FrozenInstanceError):
            evidence.erp_type = ERPType.ORACLE


class TestERPSourceMetadata:
    def test_defaults_are_all_none_with_unknown_erp_type(self):
        source = ERPSourceMetadata()
        assert source.source_file_id is None
        assert source.source_sheet is None
        assert source.source_row_number is None
        assert source.source_account_identifier is None
        assert source.source_raw_account_name is None
        assert source.source_raw_group is None
        assert source.erp_type == ERPType.UNKNOWN

    def test_is_frozen(self):
        source = ERPSourceMetadata(source_sheet="TB25-26")
        with pytest.raises(dataclasses.FrozenInstanceError):
            source.source_sheet = "Other"


class TestCanonicalTBRow:
    def test_construction_with_only_required_fields(self):
        row = CanonicalTBRow(
            gl_code="1001", gl_name="Freehold Land", opening=0.0, debit=1000.0, credit=0.0, closing=1000.0,
        )
        assert row.gl_code == "1001"
        assert row.gl_name == "Freehold Land"
        assert row.opening == 0.0
        assert row.debit == 1000.0
        assert row.credit == 0.0
        assert row.closing == 1000.0
        assert row.source_account_id is None
        assert row.source_account_code is None
        assert row.source_account_name is None
        assert row.account_type is None
        assert row.bs_pl is None

    def test_is_frozen(self):
        row = CanonicalTBRow(gl_code="1001", gl_name="Cash", opening=0.0, debit=0.0, credit=0.0, closing=0.0)
        with pytest.raises(dataclasses.FrozenInstanceError):
            row.gl_code = "2001"

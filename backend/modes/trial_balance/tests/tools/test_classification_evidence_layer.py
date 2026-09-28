"""Tests for backend/tools/classification_evidence_layer.py -- Phase 7's
unifying accessor and Phase 8's conversion to taxonomy_resolver.py's
existing GroupingHint input contract (TB Multi-ERP Adaptability upgrade).
The real point of "unify" is proven by the cross-ERP test: the same
accessor, with the same key names, works on evidence produced by two
different adapters (Tally and SAP). The real point of Phase 8 is proven by
handing real adapter-derived evidence straight to the EXISTING, unmodified
taxonomy_resolver.resolve_grouping_evidence."""

from pathlib import Path

import pytest

from modes.trial_balance.pipeline.tools import taxonomy_resolver
from modes.trial_balance.pipeline.tools.classification_evidence_layer import (
    classification_evidence_map_to_grouping_hints,
    classification_evidence_to_grouping_hint,
    extract_taxonomy_fields,
)
from modes.trial_balance.pipeline.tools.input_header_detect import build_tb_rows_from_grid, locate_header
from modes.trial_balance.pipeline.tools.input_scenario_b import _load_workbook_sheets
from modes.trial_balance.pipeline.tools.sap_adapter import run_sap_adapter
from modes.trial_balance.pipeline.tools.tally_adapter import run_tally_adapter
from modes.trial_balance.pipeline.tools.tb_erp_contracts import ClassificationEvidence, ERPType

_TALLY_TB_FILE = (
    Path(__file__).parents[2] / "documentation" / "ERP_system_standardized_files" / "Tally"
    / "Gurugram_Metro_Rail_Limited"
    / "Gurugram_Metro_Rail_Limited_Master_Trial_Balance_Head_Sub_CAG__TB25-26__be041db1.xlsx"
)
_SAP_TB_FILE = (
    Path(__file__).parents[2] / "documentation" / "ERP_system_standardized_files" / "SAP"
    / "Oil_and_Gas" / "Terra_Clean"
    / "Terra_Clean_Master_Trial_Balance_and_GL_Grouping__F.Y_24-25__1607165b.xlsx"
)


class TestExtractTaxonomyFields:
    def test_returns_only_canonical_fields_ignoring_erp_specific_keys(self):
        evidence = ClassificationEvidence(
            erp_type=ERPType.TALLY,
            account_classification={
                "main_head": "Expenses", "sub_head_1": "Indirect Expenses",
                "erp_group": "Indirect Expenses", "erp_group_path": "Expenses -> Indirect Expenses",
            },
        )
        assert extract_taxonomy_fields(evidence) == {"main_head": "Expenses", "sub_head_1": "Indirect Expenses"}

    def test_empty_account_classification_returns_empty_dict(self):
        evidence = ClassificationEvidence(erp_type=ERPType.SAP)
        assert extract_taxonomy_fields(evidence) == {}

    def test_ignores_account_group_which_is_sap_specific(self):
        evidence = ClassificationEvidence(
            erp_type=ERPType.SAP, account_classification={"account_group": "Expenses", "bs_pl": "PL"},
        )
        assert extract_taxonomy_fields(evidence) == {"bs_pl": "PL"}


@pytest.mark.skipif(not _TALLY_TB_FILE.exists() or not _SAP_TB_FILE.exists(), reason="real fixtures not present")
def test_same_accessor_works_uniformly_across_tally_and_sap_evidence():
    """The actual point of Phase 7: the same function, same key names,
    reads classification correctly from two different ERP adapters'
    output without needing to know which ERP produced it."""
    tally_sheets = _load_workbook_sheets(_TALLY_TB_FILE)
    tally_grid = tally_sheets["TB25-26"]
    tally_header_row, tally_cols = locate_header(tally_grid)
    tally_tb_rows = build_tb_rows_from_grid(tally_grid, tally_header_row, tally_cols)
    tally_result = run_tally_adapter(_TALLY_TB_FILE, tally_grid, tally_header_row, tally_cols, tally_tb_rows, sheet_name="TB25-26")
    tally_row = next(r for r in tally_tb_rows if r.gl_name == "Equity Share Capital-GoI")
    tally_evidence = tally_result.classification_by_gl_code[tally_row.gl_code]

    sap_sheets = _load_workbook_sheets(_SAP_TB_FILE)
    sap_grid = sap_sheets["F.Y_24-25"]
    sap_header_row, sap_cols = locate_header(sap_grid)
    sap_tb_rows = build_tb_rows_from_grid(sap_grid, sap_header_row, sap_cols)
    sap_result = run_sap_adapter(_SAP_TB_FILE, sap_grid, sap_header_row, sap_cols, sap_tb_rows, sheet_name="F.Y_24-25")
    sap_evidence = sap_result.classification_by_gl_code["1111100000"]

    tally_fields = extract_taxonomy_fields(tally_evidence)
    sap_fields = extract_taxonomy_fields(sap_evidence)

    assert "main_head" in tally_fields
    assert "main_head" in sap_fields


class TestClassificationEvidenceToGroupingHint:
    def test_known_fields_matches_extract_taxonomy_fields(self):
        evidence = ClassificationEvidence(
            erp_type=ERPType.TALLY,
            account_classification={
                "main_head": "Expenses", "sub_head_1": "Indirect Expenses",
                "erp_group": "Indirect Expenses", "erp_group_path": "Expenses -> Indirect Expenses",
            },
        )
        hint = classification_evidence_to_grouping_hint("1001", evidence)
        assert hint.gl_code == "1001"
        assert hint.known_fields == extract_taxonomy_fields(evidence)

    def test_hint_text_includes_erp_specific_values_excluded_from_known_fields(self):
        evidence = ClassificationEvidence(
            erp_type=ERPType.TALLY,
            account_classification={"main_head": "Expenses", "erp_group_path": "Expenses -> Indirect Expenses"},
        )
        hint = classification_evidence_to_grouping_hint("1001", evidence)
        assert "erp_group_path" not in hint.known_fields
        assert "Expenses -> Indirect Expenses" in hint.hint_text

    def test_hint_text_includes_business_dimensions_and_financial_statement(self):
        evidence = ClassificationEvidence(
            erp_type=ERPType.SAP,
            business_dimensions={"business_area": "DL01"},
            financial_statement={"financial_statement_version": "Employee Benefits Expense"},
        )
        hint = classification_evidence_to_grouping_hint("1001", evidence)
        assert "DL01" in hint.hint_text
        assert "Employee Benefits Expense" in hint.hint_text


class TestClassificationEvidenceMapToGroupingHints:
    def test_produces_one_hint_per_gl_code_correctly_keyed(self):
        classification_by_gl_code = {
            "1001": ClassificationEvidence(erp_type=ERPType.SAP, account_classification={"bs_pl": "BS"}),
            "1002": ClassificationEvidence(erp_type=ERPType.SAP, account_classification={"bs_pl": "PL"}),
        }
        hints = classification_evidence_map_to_grouping_hints(classification_by_gl_code)
        assert set(hints.keys()) == {"1001", "1002"}
        assert hints["1001"].known_fields["bs_pl"] == "BS"
        assert hints["1002"].known_fields["bs_pl"] == "PL"


@pytest.mark.skipif(not _SAP_TB_FILE.exists(), reason="Terra_Clean real-file fixture not present")
def test_sap_terra_clean_evidence_resolves_via_existing_resolver_exact_step(monkeypatch):
    """Real Terra_Clean classification evidence -> Phase 8's converter ->
    the EXISTING, unmodified taxonomy_resolver.resolve_grouping_evidence.
    Real evidence has both sub_head_1 and sub_head_2 (confirmed in Phase 7),
    so this exercises the EXACT step with zero resolver changes. Mocks the
    DB-backed taxonomy_repository functions, same pattern as
    test_taxonomy_resolver.py's own tests -- never requires a populated
    taxonomy DB."""
    sap_sheets = _load_workbook_sheets(_SAP_TB_FILE)
    sap_grid = sap_sheets["F.Y_24-25"]
    sap_header_row, sap_cols = locate_header(sap_grid)
    sap_tb_rows = build_tb_rows_from_grid(sap_grid, sap_header_row, sap_cols)
    sap_result = run_sap_adapter(_SAP_TB_FILE, sap_grid, sap_header_row, sap_cols, sap_tb_rows, sheet_name="F.Y_24-25")

    tb_row = next(r for r in sap_tb_rows if r.gl_code == "1111100000")
    evidence = sap_result.classification_by_gl_code["1111100000"]
    grouping_hint = classification_evidence_to_grouping_hint("1111100000", evidence)

    assert grouping_hint.known_fields["sub_head_1"] and grouping_hint.known_fields["sub_head_2"]

    fake_node = {
        "id": 1,
        "sub_head_1": grouping_hint.known_fields["sub_head_1"],
        "sub_head_2": grouping_hint.known_fields["sub_head_2"],
    }
    monkeypatch.setattr(taxonomy_resolver.repo, "snap_to_taxonomy", lambda *a, **k: fake_node)
    monkeypatch.setattr(taxonomy_resolver.repo, "resolve_active_scope", lambda *a, **k: 1)

    resolution = taxonomy_resolver.resolve_grouping_evidence(tb_row, grouping_hint, "IND_AS")

    assert resolution.status == "RESOLVED"
    assert resolution.method == "EXACT"
    assert resolution.taxonomy_node_id == 1

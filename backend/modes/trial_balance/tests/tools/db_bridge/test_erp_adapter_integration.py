"""Tests for backend/tools/erp_adapter_integration.py -- wires the Tally/SAP
ERP adapters into the live ingestion pipeline (db_bridge.py). Synthetic
tests prove the never-raises/graceful-degradation contract; real-fixture
tests prove it against the actual Gurugram/Terra_Clean pairs already used
throughout Phases 3/5/7."""

from pathlib import Path

import pytest

from modes.trial_balance.pipeline.tools.document_metadata import blank_metadata
from modes.trial_balance.pipeline.tools.erp_adapter_integration import enrich_parsed_input_with_erp_adapter
from modes.trial_balance.pipeline.tools.parsed_input import ParsedInput
from modes.trial_balance.pipeline.tools.tb_erp_contracts import ERPType
from modes.trial_balance.pipeline.tools.tb_models import TBRow

_GURUGRAM_DIR = (
    Path(__file__).parents[3] / "documentation" / "ERP_system_standardized_files" / "Tally"
    / "Gurugram_Metro_Rail_Limited"
)
_GURUGRAM_TB = _GURUGRAM_DIR / "Gurugram_Metro_Rail_Limited_Master_Trial_Balance_Head_Sub_CAG__TB25-26__be041db1.xlsx"
_GURUGRAM_GROUPING = _GURUGRAM_DIR / "Gurugram_Metro_Rail_Limited_Master_Trial_Balance_Head_Sub_CAG__Grouping__5e8045cb.xlsx"

_TERRA_CLEAN_TB = (
    Path(__file__).parents[3] / "documentation" / "ERP_system_standardized_files" / "SAP"
    / "Oil_and_Gas" / "Terra_Clean"
    / "Terra_Clean_Master_Trial_Balance_and_GL_Grouping__F.Y_24-25__1607165b.xlsx"
)

_BHAVINI_DIR = (
    Path(__file__).parents[3] / "documentation" / "ERP_system_standardized_files" / "Oracle"
    / "Bhavini" / "Multi_FY_2024-25_to_2025-26"
)
_BHAVINI_TB = _BHAVINI_DIR / "Bhavini_Multi_FY_2024-25_to_2025-26_Trial_Balance_and_GL_Grouping__TB_24-25__e2452013.xlsx"
_BHAVINI_GRP = (
    _BHAVINI_DIR / "Bhavini_Multi_FY_2024-25_to_2025-26_Trial_Balance_and_GL_Grouping__Grouping_24-25__6aa740fb.xlsx"
)


def _blank_parsed(tb_rows):
    return ParsedInput(metadata=blank_metadata(), tb_rows=tb_rows, grouping_hints={})


class TestNeverRaisesContract:
    def test_returns_none_for_more_than_two_input_files(self):
        parsed = _blank_parsed([TBRow(gl_code="1", gl_name="X", opening=0, debit=0, credit=0, closing=0)])
        result = enrich_parsed_input_with_erp_adapter(
            [Path("a.xlsx"), Path("b.xlsx"), Path("c.xlsx")], parsed,
        )
        assert result is None

    def test_returns_none_when_file_does_not_exist(self):
        """A nonexistent path raises inside _load_workbook_sheets/openpyxl --
        must degrade to None, never propagate."""
        parsed = _blank_parsed([TBRow(gl_code="1", gl_name="X", opening=0, debit=0, credit=0, closing=0)])
        result = enrich_parsed_input_with_erp_adapter([Path("/no/such/file.xlsx")], parsed)
        assert result is None


@pytest.mark.skipif(not _GURUGRAM_TB.exists() or not _GURUGRAM_GROUPING.exists(), reason="Gurugram real-file fixtures not present")
class TestGurugramRealFixture:
    def test_auto_detects_tally_and_enriches_grouping_hints(self):
        from modes.trial_balance.pipeline.tools.input_dispatch import parse_tb_input

        input_files = [_GURUGRAM_TB, _GURUGRAM_GROUPING]
        parsed_inputs = parse_tb_input(input_files, framework="IND_AS")
        parsed = parsed_inputs[0]

        result = enrich_parsed_input_with_erp_adapter(input_files, parsed)

        assert result is not None
        assert result.erp_type == ERPType.TALLY
        # Phase 7's fix: canonical taxonomy fields, not just erp_group/erp_group_path.
        any_hint_has_sub_head_1 = any(
            hint.known_fields.get("sub_head_1") for hint in result.enriched_grouping_hints.values()
        )
        assert any_hint_has_sub_head_1
        # Phase 3's real finding: this file genuinely has no Opening Balance column.
        assert any(f["code"] == "tally_missing_opening_or_movement_columns" for f in result.findings)

    def test_explicit_erp_type_override_is_honored(self):
        from modes.trial_balance.pipeline.tools.input_dispatch import parse_tb_input

        input_files = [_GURUGRAM_TB, _GURUGRAM_GROUPING]
        parsed_inputs = parse_tb_input(input_files, framework="IND_AS")
        parsed = parsed_inputs[0]

        result = enrich_parsed_input_with_erp_adapter(input_files, parsed, explicit_erp_type=ERPType.SAP)

        # Forced SAP despite this being a real Tally file -- proves the
        # override path is actually honored, not just auto-detection.
        assert result.erp_type == ERPType.SAP


@pytest.mark.skipif(not _TERRA_CLEAN_TB.exists(), reason="Terra_Clean real-file fixture not present")
def test_terra_clean_single_file_auto_detects_sap():
    from modes.trial_balance.pipeline.tools.input_scenario_c import parse_combined_workbook

    parsed = parse_combined_workbook(_TERRA_CLEAN_TB, metadata=blank_metadata())

    result = enrich_parsed_input_with_erp_adapter([_TERRA_CLEAN_TB], parsed)

    assert result is not None
    assert result.erp_type == ERPType.SAP
    any_hint_has_main_head = any(
        hint.known_fields.get("main_head") for hint in result.enriched_grouping_hints.values()
    )
    assert any_hint_has_main_head


@pytest.mark.skipif(not _BHAVINI_TB.exists() or not _BHAVINI_GRP.exists(), reason="Bhavini real-file fixtures not present")
class TestBhaviniRealFixture:
    def test_auto_detects_oracle_and_enriches_grouping_hints(self):
        from modes.trial_balance.pipeline.tools.input_dispatch import parse_tb_input

        input_files = [_BHAVINI_TB, _BHAVINI_GRP]
        parsed_inputs = parse_tb_input(input_files, framework="IND_AS")
        parsed = parsed_inputs[0]

        result = enrich_parsed_input_with_erp_adapter(input_files, parsed)

        assert result is not None
        assert result.erp_type == ERPType.ORACLE
        any_hint_has_main_head = any(
            hint.known_fields.get("main_head") for hint in result.enriched_grouping_hints.values()
        )
        assert any_hint_has_main_head
        # This real file has a genuine grand-total imbalance (257/628 rows
        # don't satisfy opening+debit-credit=closing even in the raw cells,
        # verified directly) -- Phase 6's quality gate correctly surfaces it
        # as a real audit-risk finding rather than hiding it.
        assert any(f["code"] == "grand_total_imbalance" for f in result.findings)

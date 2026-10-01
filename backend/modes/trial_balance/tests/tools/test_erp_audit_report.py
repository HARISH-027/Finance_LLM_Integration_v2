"""Tests for backend/tools/erp_audit_report.py -- Phase 10's ERP-aware audit
report data layer. Synthetic tests exercise each degradation path in
isolation; the real-fixture test composes entirely from already-built
Phase 5/8 outputs (sap_adapter.py + taxonomy_resolver.py, mocked the same
way test_classification_evidence_layer.py's own resolver test does)."""

from pathlib import Path

import pytest

from modes.trial_balance.pipeline.tools import taxonomy_resolver
from modes.trial_balance.pipeline.tools.classification_evidence_layer import classification_evidence_to_grouping_hint
from modes.trial_balance.pipeline.tools.erp_audit_report import (
    build_erp_audit_rows,
    render_erp_audit_row_markdown,
    render_erp_audit_section_markdown,
)
from modes.trial_balance.pipeline.tools.input_header_detect import build_tb_rows_from_grid, locate_header
from modes.trial_balance.pipeline.tools.input_scenario_b import _load_workbook_sheets
from modes.trial_balance.pipeline.tools.sap_adapter import run_sap_adapter
from modes.trial_balance.pipeline.tools.tb_erp_contracts import ClassificationEvidence, ERPType, ScopeMetadata, ScopeType
from modes.trial_balance.pipeline.tools.tb_models import TaxonomyResolution, TBRow

_SAP_TB_FILE = (
    Path(__file__).parents[2] / "documentation" / "ERP_system_standardized_files" / "SAP"
    / "Oil_and_Gas" / "Terra_Clean"
    / "Terra_Clean_Master_Trial_Balance_and_GL_Grouping__F.Y_24-25__1607165b.xlsx"
)


class TestBuildErpAuditRows:
    def test_full_evidence_row_merges_scope_and_classification(self):
        tb_rows = [TBRow(gl_code="1001", gl_name="Freehold Land", opening=0, debit=1000, credit=0, closing=1000)]
        scope = ScopeMetadata(entity="ACME Ltd", ledger="0L", scope_type=ScopeType.FULL_LEDGER)
        classification = {
            "1001": ClassificationEvidence(
                erp_type=ERPType.SAP,
                account_classification={"bs_pl": "BS", "main_head": "Non-current assets"},
                business_dimensions={"business_area": "DL01"},
            ),
        }
        rows = build_erp_audit_rows(tb_rows, scope, classification)
        assert len(rows) == 1
        row = rows[0]
        assert row.erp_type == ERPType.SAP
        assert row.entity == "ACME Ltd"
        assert row.ledger == "0L"
        assert row.scope_type == ScopeType.FULL_LEDGER
        assert row.erp_classification == {"bs_pl": "BS", "main_head": "Non-current assets", "business_area": "DL01"}
        assert row.canonical_classification == {"bs_pl": "BS", "main_head": "Non-current assets"}
        assert row.resolution_method is None

    def test_row_with_no_classification_degrades_gracefully(self):
        tb_rows = [TBRow(gl_code="9999", gl_name="Unknown", opening=0, debit=0, credit=0, closing=0)]
        scope = ScopeMetadata()
        rows = build_erp_audit_rows(tb_rows, scope, {})
        row = rows[0]
        assert row.erp_type == ERPType.UNKNOWN
        assert row.erp_classification == {}
        assert row.canonical_classification == {}

    def test_resolution_populates_only_when_supplied(self):
        tb_rows = [TBRow(gl_code="1001", gl_name="Freehold Land", opening=0, debit=1000, credit=0, closing=1000)]
        scope = ScopeMetadata()
        classification = {"1001": ClassificationEvidence(erp_type=ERPType.SAP)}
        resolution = TaxonomyResolution(status="RESOLVED", method="EXACT", evidence="Non-current assets Land")
        rows = build_erp_audit_rows(tb_rows, scope, classification, resolutions_by_gl_code={"1001": resolution})
        assert rows[0].resolution_method == "EXACT"
        assert rows[0].resolution_status == "RESOLVED"

        rows_no_resolution = build_erp_audit_rows(tb_rows, scope, classification)
        assert rows_no_resolution[0].resolution_method is None


class TestRenderErpAuditRowMarkdown:
    def _row(self, **overrides):
        from modes.trial_balance.pipeline.tools.erp_audit_report import ERPAuditRow
        defaults = dict(
            erp_type=ERPType.SAP, gl_code="1001", gl_name="Freehold Land", entity=None, ledger=None,
            scope_type=ScopeType.FULL_LEDGER, erp_classification={}, canonical_classification={},
        )
        defaults.update(overrides)
        return ERPAuditRow(**defaults)

    def test_omits_ledger_line_when_none(self):
        markdown = render_erp_audit_row_markdown(self._row(ledger=None))
        assert "**Ledger:**" not in markdown

    def test_includes_ledger_line_when_present(self):
        markdown = render_erp_audit_row_markdown(self._row(ledger="0L"))
        assert "**Ledger:** 0L" in markdown

    def test_omits_resolution_line_when_not_supplied(self):
        markdown = render_erp_audit_row_markdown(self._row())
        assert "**Resolution:**" not in markdown

    def test_includes_resolution_line_when_supplied(self):
        markdown = render_erp_audit_row_markdown(self._row(resolution_method="EXACT"))
        assert "**Resolution:** EXACT" in markdown

    def test_section_renderer_joins_multiple_rows(self):
        section = render_erp_audit_section_markdown([self._row(gl_code="1001"), self._row(gl_code="1002")])
        assert "1001" in section and "1002" in section
        assert "---" in section


@pytest.mark.skipif(not _SAP_TB_FILE.exists(), reason="Terra_Clean real-file fixture not present")
def test_terra_clean_audit_row_matches_real_classification_and_resolves_via_existing_resolver(monkeypatch):
    sheets = _load_workbook_sheets(_SAP_TB_FILE)
    grid = sheets["F.Y_24-25"]
    header_row, cols = locate_header(grid)
    tb_rows = build_tb_rows_from_grid(grid, header_row, cols)
    result = run_sap_adapter(_SAP_TB_FILE, grid, header_row, cols, tb_rows, sheet_name="F.Y_24-25")

    tb_row = next(r for r in tb_rows if r.gl_code == "1111100000")
    evidence = result.classification_by_gl_code["1111100000"]
    grouping_hint = classification_evidence_to_grouping_hint("1111100000", evidence)

    fake_node = {
        "id": 1,
        "sub_head_1": grouping_hint.known_fields["sub_head_1"],
        "sub_head_2": grouping_hint.known_fields["sub_head_2"],
    }
    monkeypatch.setattr(taxonomy_resolver.repo, "snap_to_taxonomy", lambda *a, **k: fake_node)
    monkeypatch.setattr(taxonomy_resolver.repo, "resolve_active_scope", lambda *a, **k: 1)
    resolution = taxonomy_resolver.resolve_grouping_evidence(tb_row, grouping_hint, "IND_AS")

    rows = build_erp_audit_rows(
        tb_rows, result.scope, result.classification_by_gl_code,
        resolutions_by_gl_code={"1111100000": resolution},
    )
    row = next(r for r in rows if r.gl_code == "1111100000")

    assert row.canonical_classification["main_head"] == "Equity Share Capital"
    assert row.resolution_method == "EXACT"

    markdown = render_erp_audit_row_markdown(row)
    assert "Equity Share Capital" in markdown
    assert "**Resolution:** EXACT" in markdown

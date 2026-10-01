"""Tests for the Tally main_head-only classification path."""
import json

from modes.trial_balance.pipeline.tools.tally_main_head_classifier import classify_tally_main_head
from modes.trial_balance.pipeline.tools.tb_models import GroupingHint, TBRow
from modes.trial_balance.pipeline.tools.validation_gate import build_main_head_only_row


class _Resp:
    def __init__(self, text):
        self.choices = [type("C", (), {"message": type("M", (), {"content": text})()})()]
        self.content = text


class _FakeLLM:
    def __init__(self, answers):
        self.answers, self.calls = list(answers), 0

    def generate(self, messages, **_):
        self.calls += 1
        return _Resp(self.answers.pop(0))


def _row(code, closing=1.0):
    return TBRow(gl_code=code, gl_name=code, opening=0.0, debit=0.0, credit=0.0, closing=closing)


def _hint(code, head):
    return GroupingHint(gl_code=code, known_fields={"main_head": head} if head else {})


class TestBuildMainHeadOnlyRow:
    def test_valid_is_mapped_without_sub_heads(self):
        row = build_main_head_only_row(_row("a"), bs_pl="BS", main_head="X", account_type="Asset",
                                       resolution_method="TALLY_STRUCTURAL", evidence="e")
        assert row.mapped_status == "MAPPED" and row.sub_head_1 is None and row.sub_head_2 is None

    def test_inconsistent_pair_is_unmatched(self):
        row = build_main_head_only_row(_row("a"), bs_pl="PL", main_head="X", account_type="Asset",
                                       resolution_method="m", evidence="e")
        assert row.mapped_status == "UNMATCHED"


class TestClassifyTallyMainHead:
    def test_known_primary_group_needs_no_llm(self):
        llm = _FakeLLM([])
        rows = classify_tally_main_head([_row("cash")], {"cash": _hint("cash", "Current Assets")}, llm)
        assert rows[0].account_type == "Asset" and rows[0].bs_pl == "BS" and llm.calls == 0
        assert rows[0].sub_head_1 is None and rows[0].sub_head_2 is None

    def test_custom_group_nature_asked_once_per_group(self):
        llm = _FakeLLM([json.dumps({"bs_pl": "BS", "account_type": "Equity"})])
        rows = classify_tally_main_head(
            [_row("a", -1), _row("b", -2)],
            {"a": _hint("a", "SHAREHOLDER'S FUNDS"), "b": _hint("b", "SHAREHOLDER'S FUNDS")}, llm,
        )
        assert llm.calls == 1 and all(r.account_type == "Equity" for r in rows)

    def test_unplaced_row_narrowed_only_to_existing_heads(self):
        llm = _FakeLLM([json.dumps([{"gl_code": "x", "main_head": "Current Assets"}])])
        rows = classify_tally_main_head(
            [_row("cash"), _row("x")], {"cash": _hint("cash", "Current Assets")}, llm,
        )
        assert rows[1].main_head == "Current Assets" and rows[1].trace.resolution_method == "TALLY_LLM_MAIN_HEAD"

    def test_off_list_answer_leaves_row_unmatched(self):
        llm = _FakeLLM([json.dumps([{"gl_code": "x", "main_head": "Invented"}])])
        rows = classify_tally_main_head([_row("cash"), _row("x")], {"cash": _hint("cash", "Current Assets")}, llm)
        assert rows[1].mapped_status == "UNMATCHED"

    def test_llm_failure_never_raises(self):
        class Boom:
            def generate(self, *a, **k):
                raise RuntimeError("down")
        rows = classify_tally_main_head([_row("a")], {"a": _hint("a", "Weird Group")}, Boom())
        assert rows[0].mapped_status == "UNMATCHED"

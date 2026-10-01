"""Tally-only classification that stops at main_head.

Tally exports carry a group hierarchy but no sub_head_1/sub_head_2, so
sending Tally rows through the shared Step-3 engine would make the LLM
invent a full Schedule III leaf the source never stated. This path
instead:

* takes main_head from the structural hierarchy (already on the row's
  grouping hint) -- provenance `structural`;
* derives bs_pl + account_type ONCE PER main_head (a closed 5-way choice),
  by the shipped-Primary-Group lookup first, then a single group-level LLM
  call for org-custom group names (e.g. "SHAREHOLDER'S FUNDS");
* for the few rows the structure could not place, lets the LLM choose ONLY
  among the document's own existing main_heads (candidate narrowing over a
  closed list, never the full taxonomy) -- provenance `llm_inferred`;
* leaves sub_head_1/sub_head_2 NULL, always.

Every MAPPED row is constructed by validation_gate.build_main_head_only_row.
Never raises: any LLM failure leaves affected rows UNMATCHED.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from modes.trial_balance.pipeline.tools.llm_call import read_content
from modes.trial_balance.pipeline.tools.tally_primary_group_lookup import classify_primary_group
from modes.trial_balance.pipeline.tools.tb_models import TBRow
from modes.trial_balance.pipeline.tools.validation_gate import build_main_head_only_row

logger = logging.getLogger(__name__)

_ATTRIBUTE_OPTIONS = {
    "BS": ["Asset", "Liability", "Equity"],
    "PL": ["Income", "Expense"],
}
_ROW_BATCH_SIZE = 25
_MEMBER_SAMPLE = 12


def _parse_json(response) -> object:
    cleaned = read_content(response).strip()
    for fence in ("```json", "```"):
        if cleaned.startswith(fence):
            cleaned = cleaned[len(fence):]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    return json.loads(cleaned.strip())


def _group_attributes_via_llm(
    main_head: str, members: List[TBRow], net_closing: float, llm_client,
) -> Optional[Tuple[str, str]]:
    """One closed-set call: which of BS/Asset, BS/Liability, BS/Equity,
    PL/Income, PL/Expense does this Tally group belong to?"""
    sample = ", ".join(m.gl_name for m in members[:_MEMBER_SAMPLE])
    side = "debit (Dr)" if net_closing >= 0 else "credit (Cr)"
    messages = [
        {"role": "system", "content": (
            "You classify a Tally primary group into exactly one financial-statement nature. "
            "Reply with JSON only: {\"bs_pl\": \"BS\"|\"PL\", \"account_type\": "
            "\"Asset\"|\"Liability\"|\"Equity\"|\"Income\"|\"Expense\"}. "
            "BS pairs with Asset/Liability/Equity; PL pairs with Income/Expense."
        )},
        {"role": "user", "content": (
            f"Group: {main_head}\nNet closing balance side: {side}\nSample ledgers: {sample}"
        )},
    ]
    try:
        parsed = _parse_json(llm_client.generate(messages, max_tokens=200, temperature=0))
        bs_pl, account_type = parsed.get("bs_pl"), parsed.get("account_type")
    except Exception:
        return None
    if bs_pl in _ATTRIBUTE_OPTIONS and account_type in _ATTRIBUTE_OPTIONS[bs_pl]:
        return bs_pl, account_type
    return None


def _assign_heads_via_llm(rows: List[TBRow], heads: List[str], llm_client) -> Dict[str, str]:
    """Narrowing over the document's OWN main_heads only. Off-list or null
    answers are dropped (row stays unresolved), never coerced."""
    assigned: Dict[str, str] = {}
    for start in range(0, len(rows), _ROW_BATCH_SIZE):
        batch = rows[start:start + _ROW_BATCH_SIZE]
        messages = [
            {"role": "system", "content": (
                "Assign each Tally ledger to exactly one group from the allowed list, or null if none clearly "
                "fits. Reply with JSON only: [{\"gl_code\": ..., \"main_head\": <allowed group or null>}]."
            )},
            {"role": "user", "content": (
                "Allowed groups: " + json.dumps(heads) + "\nLedgers: "
                + json.dumps([{"gl_code": r.gl_code, "gl_name": r.gl_name} for r in batch])
            )},
        ]
        try:
            parsed = _parse_json(llm_client.generate(messages, max_tokens=max(600, 60 * len(batch)), temperature=0))
        except Exception:
            continue
        if not isinstance(parsed, list):
            continue
        for item in parsed:
            if isinstance(item, dict) and item.get("gl_code") and item.get("main_head") in heads:
                assigned[item["gl_code"]] = item["main_head"]
    return assigned


def classify_tally_main_head(tb_rows: List[TBRow], grouping_hints: dict, llm_client) -> list:
    """Returns one NormalizedRow per input row, in input order."""
    head_by_gl: Dict[str, str] = {}
    source_by_gl: Dict[str, str] = {}
    for row in tb_rows:
        hint = grouping_hints.get(row.gl_code)
        main_head = (hint.known_fields or {}).get("main_head") if hint else None
        if main_head and str(main_head).strip():
            head_by_gl[row.gl_code] = str(main_head).strip()
            source_by_gl[row.gl_code] = "structural"

    heads = sorted(set(head_by_gl.values()))
    unplaced = [r for r in tb_rows if r.gl_code not in head_by_gl]
    if unplaced and heads and llm_client is not None:
        for gl_code, head in _assign_heads_via_llm(unplaced, heads, llm_client).items():
            head_by_gl[gl_code] = head
            source_by_gl[gl_code] = "llm_inferred"

    members_by_head: Dict[str, List[TBRow]] = defaultdict(list)
    for row in tb_rows:
        if row.gl_code in head_by_gl:
            members_by_head[head_by_gl[row.gl_code]].append(row)

    attributes: Dict[str, Optional[Tuple[str, str]]] = {}
    attribute_source: Dict[str, str] = {}
    for head, members in members_by_head.items():
        net = sum(m.closing for m in members)
        known = classify_primary_group(head, closing_balance=net)
        if known is not None and known.account_type is not None:
            attributes[head] = (known.bs_pl, known.account_type)
            attribute_source[head] = "primary_group_lookup"
        elif llm_client is not None:
            attributes[head] = _group_attributes_via_llm(head, members, net, llm_client)
            attribute_source[head] = "llm_group_nature"
        else:
            attributes[head] = None

    out = []
    for row in tb_rows:
        head = head_by_gl.get(row.gl_code)
        attrs = attributes.get(head) if head else None
        if head and attrs:
            method = "TALLY_STRUCTURAL" if source_by_gl[row.gl_code] == "structural" else "TALLY_LLM_MAIN_HEAD"
            evidence = f"main_head={source_by_gl[row.gl_code]}; bs_pl/account_type={attribute_source[head]}"
            out.append(build_main_head_only_row(
                row, bs_pl=attrs[0], main_head=head, account_type=attrs[1],
                resolution_method=method, evidence=evidence,
            ))
        else:
            out.append(build_main_head_only_row(
                row, bs_pl=None, main_head=None, account_type=None, resolution_method="UNMATCHED",
                evidence="no main_head placement" if not head else "group nature unresolved",
            ))
    return out

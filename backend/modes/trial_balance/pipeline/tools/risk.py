import datetime
import difflib
import json
import math
import os
import random
import re
from pathlib import Path

import polars as pl

from modes.trial_balance.pipeline.tools._shared import *  # noqa: F401,F403
from modes.trial_balance.pipeline.tools.canonical_schema import *  # noqa: F401,F403

# build_sample_selection (moved here from fsli.py, see below) reuses materiality's
# priority-band ratios for its stratified-sampling bands.
from modes.trial_balance.pipeline.tools.materiality import (
    PRIORITY_CRITICAL_RATIO,
    PRIORITY_HIGH_RATIO,
    PRIORITY_LOW_RATIO,
    PRIORITY_MEDIUM_RATIO,
)
from modes.trial_balance.pipeline.tools.pipeline_tool import *  # noqa: F401,F403

_CONTRA_MARKERS = (
    "accumulated depreciation", "accumulated amortisation", "accumulated amortization",
    "provision for doubtful", "provision for bad", "allowance for", "impairment loss on",
    "less:", "contra", "accumulated impairment",
)

def _is_contra(gl_name: str) -> bool:
    n = (gl_name or "").lower()
    return any(m in n for m in _CONTRA_MARKERS)

@pipeline_tool("build_abnormal_sign_screen", domain="risk")
def build_abnormal_sign_screen(
    canonical_tb_file: str,
    materiality_file: str = None,
    output_dir: str = None,
    **kwargs,
) -> dict:
    """Name every account carrying a closing balance on the opposite side to its
    account class's normal side, with the legitimate explanations for that class and
    the record that would resolve it. Contra accounts are reported separately, not
    as findings. Writes abnormal_sign_screen.json.

    materiality_file is optional -- without it every abnormal sign is reported
    rather than only those above the clearly-trivial threshold."""
    out_dir = resolve_output_dir(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tb_path = Path(canonical_tb_file)
    if not tb_path.exists():
        raise PipelineFileError(str(tb_path), "Canonical TB (supply canonical_tb_file)")

    df = load_canonical_tb(tb_path)
    if df is None or df.is_empty():
        return {
            "execution_status": "SUCCESS", "pipeline_status": "WARNING", "can_continue": True,
            "message": "Canonical TB is empty -- no abnormal-sign screening performed.",
            "artifacts": [], "errors": [],
        }

    warnings = []
    mat_path = Path(materiality_file) if materiality_file else out_dir / "materiality.json"
    mat = safe_load_json(mat_path) if mat_path.exists() else {}
    thresholds = mat.get("thresholds") or {}
    trivial = float(thresholds.get("clearly_trivial") or 0.0)
    perf = float(thresholds.get("performance") or 0.0)
    if not trivial:
        warnings.append(
            "materiality.json not available -- every abnormal sign is reported rather than "
            "only those above the clearly-trivial threshold. Run build_materiality first."
        )

    anchors = load_pack("anchors")
    interpretations = anchors["abnormal_sign_interpretations"]
    debit_heads, credit_heads = normal_debit_heads(), normal_credit_heads()
    assertions_pack = load_pack("assertions")

    findings, contra_accounts, unclassified = [], [], 0

    for row in df.iter_rows(named=True):
        closing = float(row.get("closing_balance") or 0.0)
        if closing == 0.0:
            continue  # a nil balance has no side

        head = str(row.get("report_head") or "").strip()
        head_lower = head.lower()
        if head_lower in debit_heads:
            expected = "Debit"
        elif head_lower in credit_heads:
            expected = "Credit"
        else:
            unclassified += 1
            continue  # unmapped -- an information request, not a sign finding

        actual = "Debit" if closing > 0 else "Credit"
        if actual == expected:
            continue

        gl_code = str(row.get("gl_code") or "")
        gl_name = str(row.get("gl_name") or "")
        account = f"{gl_code} - {gl_name}"
        magnitude = abs(closing)

        if _is_contra(gl_name):
            contra_accounts.append({
                "account": account, "gl_code": gl_code, "gl_name": gl_name,
                "closing_balance": round(closing, 2), "report_head": head,
                "note": "Contra account -- opposite side is structurally expected, not an anomaly.",
            })
            continue

        if trivial and magnitude < trivial:
            continue

        # Evidence comes from whichever account area this row belongs to, so a
        # credit receivable asks for the customer ledger while a debit payable asks
        # for the vendor ledger -- sec 13.1's "exact record", not a generic request.
        haystack = f"{gl_name} {row.get('main_head') or ''} {row.get('sub_head_1') or ''}".lower()
        area = assertions_pack["areas"]["unclassified"]
        area_name = "unclassified"
        for name, spec in assertions_pack["areas"].items():
            if name == "unclassified":
                continue
            if any(kw in haystack for kw in spec.get("match", [])):
                area, area_name = spec, name
                break

        rating = "high" if (perf and magnitude >= perf) else "medium"

        findings.append(make_record(
            source_screen="build_abnormal_sign_screen",
            account=account,
            fsli=str(row.get("sub_head_1") or row.get("main_head") or head),
            amount=round(closing, 2),
            normal_balance_expectation=expected,
            source_row_id=gl_code,
            observation=(
                f"'{gl_name}' is classified under {head} but carries a {actual.lower()} "
                f"closing balance of {magnitude:,.2f}."
            ),
            expectation=f"{head} accounts ordinarily carry a {expected.lower()} balance.",
            gap=f"Balance sits on the {actual.lower()} side, opposite the {expected.lower()} side expected for {head}.",
            assertions=area.get("assertions", ["Classification"]),
            risk_basis=["nature", "value"] if rating == "high" else ["nature"],
            risk_rating=rating,
            data_sufficiency="medium" if trivial else "low",
            proposed_response=(
                "Establish which of the listed explanations applies by inspecting the "
                "party-wise ledger, then confirm whether reclassification is required."
            ),
            evidence_requested=area.get("evidence", [])[:4] + [
                "Party-wise ledger for this account showing the movements that created the balance"
            ],
            valid_reasons=interpretations.get(head, []),
            extra={"report_head": head, "actual_balance_side": actual, "account_area": area_name},
        ))

    findings.sort(key=lambda f: abs(float(f["amount"] or 0)), reverse=True)

    payload = {
        "methodology": (
            "Each account's closing side is compared with the normal side for its account "
            "class (from the anchors knowledge pack). Contra accounts are separated out "
            "rather than flagged; unmapped accounts are excluded because their normal side "
            "is unknown. sec 5.3's class-specific explanations accompany every finding -- an "
            "abnormal sign is an indicator, never a misstatement."
        ),
        "knowledge_pack_version": anchors["_meta"]["version"],
        "generated_at": datetime.datetime.now().isoformat(),
        "data_sufficiency": "medium" if trivial else "low",
        "summary": {
            "accounts_screened": df.height,
            "abnormal_signs": len(findings),
            "contra_accounts_excluded": len(contra_accounts),
            "unmapped_accounts_skipped": unclassified,
            "clearly_trivial_threshold_applied": trivial or None,
        },
        "contra_accounts": contra_accounts,
        "finding_records": findings,
        "safe_limitation": SAFE_WORDING_DISCLAIMER,
    }

    out_path = out_dir / "abnormal_sign_screen.json"
    with atomic_write(out_path) as tmp:
        Path(tmp).write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    return {
        "execution_status": "SUCCESS",
        "pipeline_status": "SUCCESS",
        "can_continue": True,
        "message": (
            f"Abnormal-sign screen: {len(findings)} account(s) on the opposite side to their "
            f"class, {len(contra_accounts)} contra account(s) excluded, "
            f"{unclassified} unmapped account(s) skipped."
        ),
        "artifacts": [str(out_path.resolve())],
        "errors": [],
        "warnings": warnings,
    }


_BENFORD_EXPECTED_PCT = {d: math.log10(1 + 1 / d) * 100 for d in range(1, 10)}

_BENFORD_DEVIATION_THRESHOLD = 10.0  # percentage points

_ROUND_MIN_MAGNITUDE = 10000.0

_ROUND_MODULUS = 1000.0

_ROUND_PCT_THRESHOLD = 25.0

_DUP_DESC_SIMILARITY_THRESHOLD = 0.9

_DUP_DESC_MAX_ACCOUNTS = 3000  # O(n^2) guard -- skip on very large charts of accounts

_YEAR_RE = re.compile(r"(19\d{2}|20\d{2})")

_YEAR_GAP_THRESHOLD = 3

_SEMANTIC_MISMATCH_BATCH_SIZE = 30

_SEMANTIC_MISMATCH_PROMPT_TEMPLATE = """You are reviewing a Trial Balance's account classification for internal \
consistency. Below is a list of (index, account name, assigned classification) triples.

{{pairs_payload}}

For each entry whose account name strongly suggests a DIFFERENT classification than assigned \
(e.g. an account named "Office Rent Paid" classified under "Fixed Assets"), return its index and a \
one-sentence reason. Do not flag anything you are not confident about -- only clear mismatches.

Respond with ONLY a JSON array, no markdown fences, no commentary:
[{"index": 0, "reason": "<why the name suggests a different classification>"}]
If nothing looks mismatched, respond with an empty array: []
"""

def _leading_digit(value: float):
    v = abs(value)
    if v == 0:
        return None
    while v < 1:
        v *= 10
    while v >= 10:
        v /= 10
    return int(v)

def _check_benford(df: pl.DataFrame) -> dict:
    digits = [_leading_digit(v) for v in df["closing_balance"].to_list()]
    digits = [d for d in digits if d is not None]
    total = len(digits)
    if total < 30:
        return {"status": "SKIPPED", "reason": "Fewer than 30 nonzero balances -- not enough data for a Benford comparison."}

    observed_pct = {d: (digits.count(d) / total) * 100 for d in range(1, 10)}
    max_deviation = max(abs(observed_pct[d] - _BENFORD_EXPECTED_PCT[d]) for d in range(1, 10))
    flagged = max_deviation > _BENFORD_DEVIATION_THRESHOLD

    return {
        "status": "FLAGGED" if flagged else "PASS",
        "observed_pct": {str(d): round(observed_pct[d], 2) for d in range(1, 10)},
        "expected_pct": {str(d): round(_BENFORD_EXPECTED_PCT[d], 2) for d in range(1, 10)},
        "max_deviation_pct_points": round(max_deviation, 2),
    }

def _check_round_numbers(df: pl.DataFrame) -> dict:
    balances = [v for v in df["closing_balance"].to_list() if v]
    if not balances:
        return {"status": "SKIPPED", "reason": "No nonzero closing balances."}

    round_rows = df.filter(
        (pl.col("closing_balance").abs() >= _ROUND_MIN_MAGNITUDE)
        & ((pl.col("closing_balance").abs() % _ROUND_MODULUS) == 0)
    )
    round_pct = (round_rows.height / len(balances)) * 100
    return {
        "status": "FLAGGED" if round_pct > _ROUND_PCT_THRESHOLD else "PASS",
        "round_figure_pct": round(round_pct, 2),
        "round_figure_count": round_rows.height,
        "rows": round_rows.select(["gl_code", "gl_name", "closing_balance"]).to_dicts(),
    }

def _check_duplicate_descriptions(df: pl.DataFrame) -> dict:
    unique_accounts = df.select(["gl_code", "gl_name"]).unique(subset=["gl_code"]).to_dicts()
    if len(unique_accounts) > _DUP_DESC_MAX_ACCOUNTS:
        return {
            "status": "SKIPPED",
            "reason": f"{len(unique_accounts)} accounts exceeds the {_DUP_DESC_MAX_ACCOUNTS}-account "
            "guard for this O(n^2) fuzzy-match check.",
        }

    pairs = []
    for i in range(len(unique_accounts)):
        name_i = str(unique_accounts[i].get("gl_name") or "").strip().lower()
        if not name_i:
            continue
        for j in range(i + 1, len(unique_accounts)):
            name_j = str(unique_accounts[j].get("gl_name") or "").strip().lower()
            if not name_j:
                continue
            ratio = difflib.SequenceMatcher(None, name_i, name_j).ratio()
            if ratio >= _DUP_DESC_SIMILARITY_THRESHOLD:
                pairs.append(
                    {
                        "gl_code_a": unique_accounts[i]["gl_code"],
                        "gl_name_a": unique_accounts[i]["gl_name"],
                        "gl_code_b": unique_accounts[j]["gl_code"],
                        "gl_name_b": unique_accounts[j]["gl_name"],
                        "similarity": round(ratio, 3),
                    }
                )

    return {"status": "FLAGGED" if pairs else "PASS", "pairs": pairs}

def _check_year_consistency(df: pl.DataFrame, tb_year: int) -> dict:
    if not tb_year:
        return {"status": "SKIPPED", "reason": "No tb_year supplied -- cannot compare against GL description years."}

    flagged = []
    for row in df.iter_rows(named=True):
        name = str(row.get("gl_name") or "")
        m = _YEAR_RE.search(name)
        if not m:
            continue
        year = int(m.group(1))
        if abs(tb_year - year) > _YEAR_GAP_THRESHOLD:
            flagged.append({"gl_code": row.get("gl_code"), "gl_name": row.get("gl_name"), "year_found": year, "gap_years": abs(tb_year - year)})

    return {"status": "FLAGGED" if flagged else "PASS", "rows": flagged}

def _check_semantic_mismatch(df: pl.DataFrame, llm_client) -> dict:
    if llm_client is None:
        return {"status": "SKIPPED", "reason": "No llm_client supplied -- this sub-check requires an LLM call."}

    candidates = [
        {"gl_name": row.get("gl_name"), "main_head": row.get("main_head")}
        for row in df.select(["gl_name", "main_head"]).unique().to_dicts()
        if row.get("gl_name") and row.get("main_head")
    ][:_SEMANTIC_MISMATCH_BATCH_SIZE]

    if not candidates:
        return {"status": "SKIPPED", "reason": "No (gl_name, main_head) pairs available -- canonical TB may be unmapped."}

    pairs_payload = "\n".join(f"{i}: \"{c['gl_name']}\" classified as \"{c['main_head']}\"" for i, c in enumerate(candidates))
    prompt = _SEMANTIC_MISMATCH_PROMPT_TEMPLATE.replace("{{pairs_payload}}", pairs_payload)

    try:
        response = llm_client.generate([{"role": "user", "content": prompt}], max_tokens=1024)
        content = response.get("content", "[]") if isinstance(response, dict) else getattr(response, "content", "[]")
        cleaned = content.strip()
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        if cleaned.startswith("```"):
            cleaned = cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        mismatches = json.loads(cleaned.strip())
    except Exception as e:
        return {"status": "SKIPPED", "reason": f"LLM call failed: {e}"}

    flagged = []
    for m in mismatches:
        idx = m.get("index")
        if isinstance(idx, int) and 0 <= idx < len(candidates):
            flagged.append({**candidates[idx], "reason": m.get("reason", "")})

    return {"status": "FLAGGED" if flagged else "PASS", "rows": flagged}

@pipeline_tool("build_anomaly_scanner", domain="risk")
def build_anomaly_scanner(canonical_tb_file: str, output_dir: str = None, llm_client=None, tb_year: int = None, **kwargs) -> dict:
    """Run statistical/pattern anomaly checks (Benford's Law, round-numbers, duplicate descriptions) on a canonical TB."""
    tb_path = Path(canonical_tb_file)
    if not tb_path.exists():
        raise PipelineFileError(str(tb_path), "Canonical Trial Balance (supply canonical_tb_file from load_tb_from_db/ingest_tb_to_live)")

    df = pl.read_parquet(tb_path)
    if df.is_empty():
        return {
            "execution_status": "SUCCESS",
            "pipeline_status": "SUCCESS",
            "message": "Canonical TB is empty -- no anomaly checks run.",
            "artifacts": [],
            "errors": [],
        }

    df = df.with_columns(pl.col("closing_balance").cast(pl.Float64, strict=False).fill_null(0.0))

    checks = {
        "TB-021": _check_benford(df),
        "TB-023": _check_round_numbers(df),
        "TB-030": _check_duplicate_descriptions(df),
        "TB-033": _check_year_consistency(df, tb_year),
        "TB-022": _check_semantic_mismatch(df, llm_client),
    }

    findings = []

    if checks["TB-021"]["status"] == "FLAGGED":
        findings.append({
            "rule_id": "TB-021",
            "account": "SYSTEM_ANOMALY",
            "severity": "Info",
            "score": 10.0,
            "description": (
                f"Leading-digit distribution deviates {checks['TB-021']['max_deviation_pct_points']} "
                "percentage points from Benford's Law expectation -- unusual but not conclusive on its own."
            ),
            "trigger_value": checks["TB-021"]["max_deviation_pct_points"],
        })

    if checks["TB-023"]["status"] == "FLAGGED":
        for row in checks["TB-023"]["rows"]:
            findings.append({
                "rule_id": "TB-023",
                "account": str(row.get("gl_code", "")),
                "severity": "Info",
                "score": 8.0,
                "description": f"Round-figure closing balance {row.get('closing_balance')} -- request supporting workings.",
                "trigger_value": row.get("closing_balance"),
            })

    if checks["TB-030"]["status"] == "FLAGGED":
        for pair in checks["TB-030"]["pairs"]:
            findings.append({
                "rule_id": "TB-030",
                "account": str(pair["gl_code_a"]),
                "severity": "Info",
                "score": 8.0,
                "description": (
                    f"GL '{pair['gl_code_a']}' ({pair['gl_name_a']}) and GL '{pair['gl_code_b']}' "
                    f"({pair['gl_name_b']}) are {pair['similarity']:.0%} similar -- possible duplicate account setup."
                ),
                "trigger_value": pair["similarity"],
            })

    if checks["TB-033"]["status"] == "FLAGGED":
        for row in checks["TB-033"]["rows"]:
            findings.append({
                "rule_id": "TB-033",
                "account": str(row.get("gl_code", "")),
                "severity": "Info",
                "score": 8.0,
                "description": (
                    f"GL description references {row.get('year_found')}, {row.get('gap_years')} year(s) "
                    "from the TB's own period -- flagged for write-off/recoverability review."
                ),
                "trigger_value": row.get("year_found"),
            })

    if checks["TB-022"]["status"] == "FLAGGED":
        for row in checks["TB-022"]["rows"]:
            findings.append({
                "rule_id": "TB-022",
                "account": str(row.get("gl_name", "")),
                "severity": "Warning",
                "score": 12.0,
                "description": f"'{row.get('gl_name')}' classified as '{row.get('main_head')}' -- {row.get('reason')}",
                "trigger_value": row.get("main_head"),
            })

    out_dir = resolve_output_dir(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "anomaly_findings.json"
    payload = {
        "methodology": "Deterministic statistical/pattern anomaly scan (TB-021/023/030/033) plus an "
        "optional LLM semantic-mismatch check (TB-022).",
        "checks": {k: {kk: vv for kk, vv in v.items() if kk not in ("rows", "pairs")} for k, v in checks.items()},
        "findings": findings,
    }
    write_json_atomic(payload, out_path, indent=4, default=str)

    return {
        "execution_status": "SUCCESS",
        "pipeline_status": "SUCCESS",
        "can_continue": True,
        "message": f"Anomaly scan complete: {len(findings)} finding(s) across {sum(1 for c in checks.values() if c['status'] == 'FLAGGED')} flagged check(s).",
        "artifacts": [str(out_path)],
        "errors": [],
    }


def _match_accounts(df, keywords) -> list:
    """Accounts whose name or grouping text contains any indicator keyword."""
    if not keywords:
        return []
    hits = []
    for row in df.iter_rows(named=True):
        closing = float(row.get("closing_balance") or 0.0)
        text = " ".join(str(row.get(c) or "") for c in
                        ("gl_name", "main_head", "sub_head_1", "sub_head_2")).lower()
        matched = [k for k in keywords if k in text]
        if not matched:
            continue
        hits.append({
            "account": f"{row.get('gl_code')} - {row.get('gl_name')}",
            "gl_code": str(row.get("gl_code") or ""),
            "gl_name": str(row.get("gl_name") or ""),
            "closing_balance": round(closing, 2),
            "matched_terms": matched[:4],
        })
    hits.sort(key=lambda h: abs(h["closing_balance"]), reverse=True)
    return hits

@pipeline_tool("build_caro_indicators", domain="risk")
def build_caro_indicators(
    canonical_tb_file: str,
    engagement_context_file: str = None,
    output_dir: str = None,
    **kwargs,
) -> dict:
    """Flag trial-balance indicators relevant to CARO 2020 areas and Companies Act
    compliance heads (PPE and title deeds, inventory and bank stock statements, loans
    and guarantees, deposits, statutory dues, borrowing defaults, CSR, related
    parties, share capital, dividend, managerial remuneration, audit trail), each with
    the register, return or working that would let the audit team form a view.

    Gated on confirmed CARO applicability: when unconfirmed, every indicator is
    reported as an information request rather than a reporting matter, since
    applicability cannot be derived from a trial balance. Never concludes a CARO
    reporting matter. Writes caro_indicators.json."""
    out_dir = resolve_output_dir(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tb_path = Path(canonical_tb_file)
    if not tb_path.exists():
        raise PipelineFileError(str(tb_path), "Canonical TB (supply canonical_tb_file)")

    df = load_canonical_tb(tb_path)
    if df is None or df.is_empty():
        return {
            "execution_status": "SUCCESS", "pipeline_status": "WARNING", "can_continue": True,
            "message": "Canonical TB is empty -- no CARO indicator screening performed.",
            "artifacts": [], "errors": [],
        }

    warnings = []
    ctx_path = (Path(engagement_context_file) if engagement_context_file
                else out_dir / "engagement_context.json")
    ctx = safe_load_json(ctx_path) if ctx_path.exists() else {}
    gate = ((ctx.get("applicability_gates") or {}).get("caro") or {})
    caro_applicable, gate_basis = gate.get("applicable"), gate.get("basis", "unconfirmed")

    confirmed = caro_applicable is True
    if not confirmed:
        warnings.append(
            "CARO applicability is unconfirmed -- indicators are reported as information "
            "requests, not reporting matters. Supply caro_applicable to build_engagement_context "
            "once the engagement team has confirmed it (sec 10.2)."
        )

    pack = load_pack("compliance")
    findings, areas_reported = [], []

    def screen(area, kind):
        keywords = tuple(area.get("tb_indicators") or [])
        # The audit-trail head is a metadata question, not a keyword match -- it always
        # applies when an export lacks ERP provenance, so it is handled by
        # build_normalisation_note / build_run_log rather than raised here.
        if not keywords:
            return
        hits = _match_accounts(df, keywords)
        if not hits:
            return
        total = round(sum(abs(h["closing_balance"]) for h in hits), 2)
        areas_reported.append({
            "id": area["id"], "area": area["area"], "kind": kind,
            "clause_hint": area.get("clause_hint"),
            "accounts_matched": len(hits), "total_balance": total,
            "accounts": hits[:15],
        })

        clause = f" (clause {area['clause_hint']})" if area.get("clause_hint") else ""
        findings.append(make_record(
            source_screen="build_caro_indicators",
            fsli=area["area"],
            amount=total,
            observation=(
                f"{len(hits)} account(s) totalling {total:,.2f} carry terms associated with "
                f"{area['area']}{clause}: "
                + ", ".join(sorted({t for h in hits[:5] for t in h['matched_terms']})[:5]) + "."
            ),
            expectation=(
                f"Where CARO applies, {area['area'].lower()} requires the supporting registers, "
                "approvals and workings named below before any reporting view is formed."
            ),
            gap=(
                "The trial balance shows the balances but carries none of the underlying records. "
                + ("" if confirmed else
                   "CARO applicability is also unconfirmed, so this is an information request "
                   "rather than a reporting matter.")
            ),
            assertions=["Classification", "Presentation"],
            risk_basis=["context", "nature"],
            # Unconfirmed applicability caps the rating -- sec 10.2's safe rule.
            risk_rating="medium" if confirmed else "information_request",
            regularity_flag=True,
            data_sufficiency="low",
            proposed_response=(
                "Obtain the registers and workings listed and evaluate them against the clause "
                "requirements. No CARO reporting matter is concluded from the trial balance."
            ),
            evidence_requested=area.get("evidence_to_request", []),
            valid_reasons=[
                "The entity may hold all required records; the TB simply cannot show them",
                "CARO may not apply to this entity at all",
                "The matched terms may describe an account whose substance differs from its name",
            ],
            extra={
                "caro_area_id": area["id"],
                "clause_hint": area.get("clause_hint"),
                "kind": kind,
                "applicability_confirmed": confirmed,
                "caution": area.get("caution"),
            },
        ))

    for area in pack["caro_2020_areas"]:
        screen(area, "caro_2020")
    for area in pack["companies_act_heads"]:
        screen(area, "companies_act")

    # sec 10.3 disclosure cues -- request prompts, not findings.
    disclosure_cues = []
    for cue in pack.get("schedule_iii_disclosures", []):
        disclosure_cues.append({"cue": cue["cue"], "request": cue["request"]})

    payload = {
        "safe_rule": pack["_meta"]["safe_rule"],
        "applicability_gate": {
            "caro_applicable": caro_applicable,
            "basis": gate_basis,
            "effect": (
                "Confirmed -- indicators reported at medium rating."
                if confirmed else
                "Unconfirmed -- every indicator downgraded to information_request. CARO "
                "applicability is not derivable from a trial balance (sec 10.2)."
            ),
        },
        "knowledge_pack_version": pack["_meta"]["version"],
        "generated_at": datetime.datetime.now().isoformat(),
        "data_sufficiency": "low",
        "summary": {
            "areas_with_indicators": len(areas_reported),
            "caro_areas_defined": len(pack["caro_2020_areas"]),
            "companies_act_heads_defined": len(pack["companies_act_heads"]),
            "findings": len(findings),
        },
        "areas": areas_reported,
        "schedule_iii_disclosure_cues": disclosure_cues,
        "finding_records": findings,
        "safe_limitation": SAFE_WORDING_DISCLAIMER,
        # Wave 2 Fix 2b: the statutory-dues-relevant areas here (PF/ESI, TDS, GST clauses)
        # are one of five independent statutory-dues populations across this codebase.
        "statutory_dues_population_basis": (
            "compliance-pack (CARO 2020/Companies Act) keyword screen; see also "
            "build_statutory_screen, build_sensitive_detector, build_going_concern_screen for "
            "independently-computed statutory-dues populations"
        ),
    }

    out_path = out_dir / "caro_indicators.json"
    with atomic_write(out_path) as tmp:
        Path(tmp).write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    return {
        "execution_status": "SUCCESS",
        "pipeline_status": "SUCCESS",
        "can_continue": True,
        "message": (
            f"CARO/Companies Act indicators: {len(areas_reported)} area(s) with trial-balance "
            f"indicators, applicability {gate_basis}"
            + ("." if confirmed else " -- reported as information requests only.")
        ),
        "artifacts": [str(out_path.resolve())],
        "errors": [],
        "warnings": warnings,
    }


_COUNTERPART_PRESENCE_RATIO = 0.001

def _resolve_head(fsli_df, tb_df, keywords) -> dict:
    """Locate a head by keyword, preferring the FSLI hierarchy and falling back to
    the canonical TB. Returns {"balance", "source", "detail"}.

    The fallback matters: on a thinly-mapped or unmapped TB the FSLI hierarchy may
    have no matching node at all, and reporting "counterpart missing" purely because
    the GROUPING is missing would be a false positive about the accounts.
    """
    node = find_fsli_component(fsli_df, keywords)
    if node and node["balance"] > 0:
        return {"balance": node["balance"], "source": "fsli", "detail": node["node_name"]}

    gl = gl_total_by_keywords(tb_df, keywords)
    if gl["balance"] > 0:
        # Name the accounts when there are few enough to be useful -- an audit party
        # cannot act on "3 account(s) matched by ledger name", but can act on the
        # ledger names themselves.
        names = [a["gl_name"] for a in gl["accounts"][:3] if a["gl_name"]]
        if names and gl["account_count"] <= 3:
            detail = ", ".join(names)
        elif names:
            detail = f"{', '.join(names)} and {gl['account_count'] - len(names)} other account(s)"
        else:
            detail = f"{gl['account_count']} account(s) matched by ledger name"
        return {
            "balance": gl["balance"],
            "source": "gl_name",
            "detail": detail,
            "accounts": gl["accounts"],
        }
    return {"balance": 0.0, "source": None, "detail": "no matching head found"}

@pipeline_tool("build_counterpart_screen", domain="risk")
def build_counterpart_screen(
    canonical_tb_file: str,
    fsli_summary_file: str = None,
    materiality_file: str = None,
    output_dir: str = None,
    **kwargs,
) -> dict:
    """Flag expected account pairs where the source head is material but its
    counterpart is missing or immaterial (PPE without depreciation, borrowings
    without finance cost, revenue without GST output, payroll without statutory
    dues, and so on). Writes counterpart_screen.json.

    fsli_summary_file and materiality_file are optional: without the FSLI hierarchy
    the screen matches on ledger names instead, and without materiality it reports
    every gap rather than only material ones, marking data sufficiency lower."""
    out_dir = resolve_output_dir(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tb_path = Path(canonical_tb_file)
    if not tb_path.exists():
        raise PipelineFileError(str(tb_path), "Canonical TB (supply canonical_tb_file)")

    tb_df = load_canonical_tb(tb_path)
    if tb_df is None or tb_df.is_empty():
        return {
            "execution_status": "SUCCESS",
            "pipeline_status": "WARNING",
            "can_continue": True,
            "message": "Canonical TB is empty -- no counterpart screening performed.",
            "artifacts": [],
            "errors": [],
        }

    warnings = []
    fsli_path = Path(fsli_summary_file) if fsli_summary_file else out_dir / "fsli_summary.parquet"
    fsli_df = safe_load_parquet(fsli_path) if fsli_path.exists() else None
    if fsli_df is None or fsli_df.is_empty():
        warnings.append(
            "fsli_summary.parquet not available -- heads matched on ledger names only, "
            "which is weaker than hierarchy matching. Run build_fsli_summary first."
        )

    mat_path = Path(materiality_file) if materiality_file else out_dir / "materiality.json"
    mat = safe_load_json(mat_path) if mat_path.exists() else {}
    perf_materiality = float((mat.get("thresholds") or {}).get("performance") or 0.0)
    if not perf_materiality:
        warnings.append(
            "materiality.json not available -- every counterpart gap is reported rather "
            "than only material ones. Run build_materiality for a prioritised screen."
        )
    data_sufficiency = "medium" if (perf_materiality and fsli_df is not None) else "low"

    pack = load_pack("relationships")
    findings = []
    checked = []

    for rel in pack["relationships"]:
        if rel.get("direction") != "target_expected_when_source_present":
            continue  # context_pair relationships are build_relationship_expectations' job

        source = _resolve_head(fsli_df, tb_df, rel["source_keywords"])
        target = _resolve_head(fsli_df, tb_df, rel["target_keywords"])
        checked.append({
            "id": rel["id"], "name": rel["name"],
            "source_balance": round(source["balance"], 2),
            "target_balance": round(target["balance"], 2),
        })

        if source["balance"] <= 0:
            continue  # source head absent -- the relationship simply does not apply
        if perf_materiality and source["balance"] < perf_materiality:
            continue  # source head immaterial -- not worth an audit question

        threshold = source["balance"] * _COUNTERPART_PRESENCE_RATIO
        if target["balance"] > threshold:
            continue  # counterpart present at a plausible magnitude

        ratio = (target["balance"] / source["balance"]) if source["balance"] else 0.0
        absent = target["balance"] == 0.0
        gap = (
            f"No {rel['name'].split(' vs ')[-1]} head is present in the trial balance."
            if absent else
            f"Counterpart is {target['balance']:,.2f} against a source head of "
            f"{source['balance']:,.2f} ({ratio:.4%}) -- materially smaller than the "
            f"relationship would ordinarily produce."
        )

        findings.append(make_record(
            source_screen="build_counterpart_screen",
            fsli=source["detail"],
            amount=round(source["balance"], 2),
            observation=(
                f"{rel['name']}: source head '{source['detail']}' carries "
                f"{source['balance']:,.2f}, but the expected counterpart is "
                + ("absent." if absent else f"only {target['balance']:,.2f}.")
            ),
            expectation=rel["expectation"],
            gap=gap,
            assertions=rel.get("assertions", []),
            risk_basis=["relationship", "value"] + (["context"] if rel.get("regularity_flag") else []),
            risk_rating=rel.get("default_severity", "medium"),
            regularity_flag=bool(rel.get("regularity_flag")),
            data_sufficiency=data_sufficiency,
            proposed_response=(
                "Confirm whether one of the legitimate explanations applies before treating "
                "this as a completeness issue, then obtain the records listed."
            ),
            evidence_requested=rel.get("resolving_evidence", []),
            valid_reasons=rel.get("valid_reasons", []),
            extra={
                "relationship_id": rel["id"],
                "source_head": source["detail"],
                "source_balance": round(source["balance"], 2),
                "target_balance": round(target["balance"], 2),
                "counterpart_ratio": round(ratio, 6),
                "match_basis": source["source"],
                "risk_if_inconsistent": rel.get("risk_if_inconsistent"),
            },
        ))

    payload = {
        "methodology": (
            "For each expected pair in the relationships knowledge pack, the source head is "
            "located in the FSLI hierarchy (falling back to ledger-name matching), and a gap "
            "is raised only where the source is material and the counterpart is absent or "
            f"below {_COUNTERPART_PRESENCE_RATIO:.1%} of it. Presence of both heads is NOT a "
            "finding -- the audit signal is the missing counterpart (spec sec 6)."
        ),
        "knowledge_pack_version": pack["_meta"]["version"],
        "generated_at": datetime.datetime.now().isoformat(),
        "performance_materiality_applied": perf_materiality or None,
        "data_sufficiency": data_sufficiency,
        "relationships_checked": checked,
        "summary": {
            "relationships_evaluated": len(checked),
            "gaps_found": len(findings),
            "high": sum(1 for f in findings if f["risk_rating"] == "high"),
        },
        "finding_records": findings,
        "safe_limitation": SAFE_WORDING_DISCLAIMER,
    }

    out_path = out_dir / "counterpart_screen.json"
    with atomic_write(out_path) as tmp:
        Path(tmp).write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    return {
        "execution_status": "SUCCESS",
        "pipeline_status": "SUCCESS",
        "can_continue": True,
        "message": (
            f"Counterpart screen: {len(findings)} missing-counterpart gap(s) across "
            f"{len(checked)} expected relationship(s)."
        ),
        "artifacts": [str(out_path.resolve())],
        "errors": [],
        "warnings": warnings,
    }


@pipeline_tool("build_estimation_exposure", domain="risk")
def build_estimation_exposure(canonical_tb_file: str, output_dir: str = None, **kwargs) -> dict:
    """Flags every account whose carrying value rests on management estimate/judgment
    (acquisition cost, CWIP, exploration, impairment, goodwill), ranked by absolute size
    regardless of in-year movement. Writes estimation_exposure.json/.parquet."""
    errors = []
    artifacts = []

    out_dir = resolve_output_dir(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tb_path = Path(canonical_tb_file)
    if not tb_path.exists():
        raise PipelineFileError(str(tb_path), "canonical_tb.parquet")

    df = load_canonical_tb(tb_path)
    if df is None or df.is_empty():
        errors.append({"type": "EmptyDataError", "message": "canonical_tb.parquet loaded but is empty."})
        return {"execution_status": "FAILED", "errors": errors, "message": "Canonical TB is empty."}
    df = mapped_only(df)

    rows = [
        {
            "gl_code": r.get("gl_code", ""),
            "gl_name": r.get("gl_name", ""),
            "fs_head": r.get("report_head", ""),
            "closing_balance": float(r.get("closing_balance", 0.0)),
            "moved_in_year": bool(float(r.get("debit", 0.0)) != 0 or float(r.get("credit", 0.0)) != 0),
        }
        for r in df.iter_rows(named=True)
        if is_estimation_risk_account(r.get("gl_name"))
    ]
    rows.sort(key=lambda r: abs(r["closing_balance"]), reverse=True)

    total_exposure = sum(abs(r["closing_balance"]) for r in rows)

    output_data = {
        "methodology": (
            "Deterministic keyword screen over the canonical TB for accounts whose carrying "
            "value rests on management estimate/judgment (acquisition cost, CWIP, exploration, "
            "impairment, goodwill) rather than a transactable price. Ranked by absolute size, "
            "independent of whether the account moved in the year -- movement or its absence "
            "does not change the estimation risk."
        ),
        "summary": {
            "accounts_flagged": len(rows),
            "total_exposure": round(total_exposure, 2),
            "dormant_within_flagged": len([r for r in rows if not r["moved_in_year"]]),
        },
        "accounts": rows,
        "generated_at": datetime.datetime.now().isoformat(),
    }

    out_file_json = out_dir / "estimation_exposure.json"
    write_json_atomic(output_data, out_file_json, indent=4)
    artifacts.append(str(out_file_json.resolve()))

    out_file_pq = out_dir / "estimation_exposure.parquet"
    write_parquet_atomic(
        pl.DataFrame(rows if rows else {"gl_code": [], "gl_name": [], "fs_head": [], "closing_balance": [], "moved_in_year": []}),
        out_file_pq,
    )
    artifacts.append(str(out_file_pq.resolve()))

    return {
        "execution_status": "SUCCESS",
        "pipeline_status": "SUCCESS",
        "message": f"Flagged {len(rows)} estimation-risk account(s), total exposure {total_exposure:,.0f}.",
        "artifacts": artifacts,
        "errors": errors,
        "data": output_data,
    }


def _load_json(path):
    if not path:
        return {}
    p = Path(path)
    if p.exists():
        with open(p, "r") as f:
            return json.load(f)
    return {}

def _load_pq(path):
    if not path:
        return pl.DataFrame()
    p = Path(path)
    if p.exists():
        return pl.read_parquet(p)
    return pl.DataFrame()

def _account_type(row) -> str:
    """Prefer the explicit account_type column; fall back to string-matching main_head."""
    at = row.get("account_type")
    if at is not None and str(at).strip() and str(at).strip().lower() != "nan":
        s = str(at).strip().lower()
        if "asset" in s:
            return "Asset"
        if "liab" in s:
            return "Liability"
        if "equity" in s:
            return "Equity"
        if "revenue" in s or "income" in s:
            return "Revenue"
        if "expense" in s:
            return "Expense"
        return str(at).strip()
    head = str(row.get("main_head") or "")
    if "Asset" in head:
        return "Asset"
    if "Liabilit" in head:
        return "Liability"
    if "Equity" in head:
        return "Equity"
    if "Revenue" in head:
        return "Revenue"
    if "Expense" in head:
        return "Expense"
    return "Unknown"

def _normal_balance(acct_type: str) -> str:
    if acct_type in ["Asset", "Expense"]:
        return "Debit"
    if acct_type in ["Liability", "Equity", "Revenue"]:
        return "Credit"
    return "Unknown"

@pipeline_tool("build_exception_consolidator", domain="risk")
def build_exception_consolidator(
    canonical_tb_file: str,
    materiality_file: str = None,
    financial_snapshot_file: str = None,
    risk_indicators_file: str = None,
    sensitive_accounts_file: str = None,
    relationship_analytics_file: str = None,
    relationship_graph_file: str = None,
    layer1_results_file: str = None,
    variance_analysis_file: str = None,
    anomaly_findings_file: str = None,
    output_dir: str = None,
) -> dict:
    """Correlate validation/risk/sensitivity/variance/relationship/anomaly analytics into
    canonical exception entities, cluster them by business-process + FSLI, and score/rank them.

    materiality_file/financial_snapshot_file/risk_indicators_file/sensitive_accounts_file/
    relationship_analytics_file/relationship_graph_file all default to their canonical
    filename inside output_dir when not explicitly supplied (materiality.json,
    financial_snapshot.json, risk_indicators.json, sensitive_accounts.json,
    relationship_analytics.json, relationship_graph.json) -- SINGLE_TB uses one shared
    output_dir per run (see system_prompt.py's Artifact-Passing Convention), so these six
    upstream tools' outputs are always at those paths once each has run.

    variance_analysis_file is optional -- Single-TB mode doesn't run variance analysis
    (requires a PY/CY comparison basis), so this degrades gracefully to no variance
    exceptions when the file isn't supplied/doesn't exist.

    layer1_results_file is optional -- it only exists for the upload-from-Excel input path
    (validate_layer1_tb runs against the raw workbook). Canonical TBs sourced from the DB or
    from a live template submission never produced a raw-workbook validation pass, so this
    degrades gracefully to zero validation-rule exceptions when the file isn't supplied.

    anomaly_findings_file is optional -- build_anomaly_scanner's output. Absent, this
    degrades gracefully to zero anomaly-rule exceptions (matches every other optional
    analytics source above).
    """
    errors = []
    warnings = []
    artifacts = []

    out_dir = resolve_output_dir(output_dir)
    os.makedirs(out_dir, exist_ok=True)

    # resolve_artifact_path also corrects a wrong-format path (e.g. an LLM passing
    # financial_snapshot.parquet -- binary -- into a param that needs the .json sibling;
    # _load_json's json.load() on a Parquet file crashes with a raw UnicodeDecodeError
    # otherwise) back to the canonical filename in out_dir, when that file actually exists.
    materiality_file = resolve_artifact_path(materiality_file, out_dir, "materiality.json")
    financial_snapshot_file = resolve_artifact_path(financial_snapshot_file, out_dir, "financial_snapshot.json")
    risk_indicators_file = resolve_artifact_path(risk_indicators_file, out_dir, "risk_indicators.json")
    sensitive_accounts_file = resolve_artifact_path(sensitive_accounts_file, out_dir, "sensitive_accounts.json")
    relationship_analytics_file = resolve_artifact_path(relationship_analytics_file, out_dir, "relationship_analytics.json")
    relationship_graph_file = resolve_artifact_path(relationship_graph_file, out_dir, "relationship_graph.json")
    if variance_analysis_file:
        variance_analysis_file = resolve_artifact_path(variance_analysis_file, out_dir, "variance_analysis.json")

    tb_pq = _load_pq(canonical_tb_file)
    mat_data = _load_json(materiality_file)
    _load_json(financial_snapshot_file)  # loaded for parity with TB-v1 signature; not consumed further here
    var_data = _load_json(variance_analysis_file)
    risk_data = _load_json(risk_indicators_file)
    sens_data = _load_json(sensitive_accounts_file)
    _load_json(relationship_analytics_file)  # loaded for parity with TB-v1 signature; not consumed further here
    graph_data = _load_json(relationship_graph_file)
    val_data = _load_json(layer1_results_file)
    anomaly_data = _load_json(anomaly_findings_file)

    thresholds = mat_data.get("thresholds", {})
    plan_mat = thresholds.get("overall", 0.0)
    pm = thresholds.get("performance", 0.0)
    ct = thresholds.get("clearly_trivial", 0.0)

    tb_total_balance = 0.0
    fsli_totals = {}
    if not tb_pq.is_empty():
        tb_pq = tb_pq.with_columns(
            [
                pl.col(col).cast(pl.Float64, strict=False).fill_null(0.0).alias(col)
                for col in ["opening_balance", "debit", "credit", "closing_balance"]
                if col in tb_pq.columns
            ]
        )
        tb_pq = tb_pq.with_columns(
            pl.Series("_fsli", [row_fsli(r) for r in tb_pq.iter_rows(named=True)])
        )
        tb_total_balance = float(tb_pq["closing_balance"].abs().sum())
        # polars' group_by treats null keys as their own group by default -- matches
        # the pandas dropna=False the original code passed explicitly for fsli_rank
        # (row_fsli() never actually returns None, but kept consistent either way).
        # remark #9 fix: gross-then-sum (abs().sum()), not net-then-abs (sum().abs()) --
        # the latter lets offsetting dr/cr sub-accounts within one FSLI head cancel before
        # taking abs, so one account's own gross balance can exceed the head's net total
        # and read as >100% contribution_to_fsli. contribution_to_tb (tb_total_balance,
        # above) already uses the correct gross-then-sum basis -- match it here.
        fsli_totals = dict(
            tb_pq.group_by("_fsli").agg(pl.col("closing_balance").abs().sum().alias("v")).iter_rows()
        )
        tb_pq = tb_pq.with_columns(pl.col("closing_balance").abs().alias("abs_bal"))
        tb_pq = tb_pq.with_columns(
            [
                pl.col("abs_bal").rank(method="min", descending=True).fill_null(0).alias("tb_rank"),
                pl.col("abs_bal").rank(method="min", descending=True).over("_fsli").fill_null(0).alias("fsli_rank"),
            ]
        )

    exceptions_map = {}

    def get_or_create_exc(entity_id, entity_type):
        entity_id_str = str(entity_id)
        if entity_id_str not in exceptions_map:
            exceptions_map[entity_id_str] = {
                "metadata": {
                    "exception_id": f"EX-{len(exceptions_map)+1:04d}",
                    "generated_at": datetime.datetime.now().isoformat(),
                    "pipeline_version": "2.0",
                    "analytics_sources": [],
                    "deterministic": True,
                },
                "entity": {
                    "entity_id": entity_id_str,
                    "entity_name": entity_id_str,
                    "entity_type": entity_type,
                },
                "business_context": {},
                "hierarchy_context": {},
                "financial_context": {},
                "materiality_context": {
                    "planning_materiality": plan_mat,
                    "performance_materiality": pm,
                    "clearly_trivial_threshold": ct,
                    "materiality_multiple": 0.0,
                    "exceeds_planning_materiality": False,
                },
                "relationship_context": {
                    "same_fsli": [],
                    "peer_accounts": [],
                    "related_exceptions": [],
                    "same_business_process": [],
                    "same_risk_cluster": [],
                    "relationship_failures": [],
                },
                "cluster_context": {},
                "evidence_package": {
                    "validation": {},
                    "variance": {},
                    "materiality": {},
                    "risk": {},
                    "relationships": {},
                    "sensitivity": {},
                    "anomaly": {},
                    "financial_snapshot": {},
                    "supporting_metrics": {},
                },
                "triggered_rules": [],
                "audit_planning": {
                    "audit_assertions": [],
                    "required_evidence": [],
                    "expected_audit_procedures": [],
                    "primary_audit_objective": "Determine material correctness.",
                    "planning_priority": "Low",
                },
                "scoring": {
                    "composite_score": 0.0,
                    "severity": "Information Request",
                    "risk_score": 0.0,
                    "variance_score": 0.0,
                    "sensitivity_score": 0.0,
                    "validation_score": 0.0,
                    "anomaly_score": 0.0,
                },
                "classification": {
                    "exception_type": "Single",
                    "category": "General",
                    "primary_engine": "",
                },
            }
        return exceptions_map[entity_id_str]

    def enrich_from_tb(exc, entity_id, entity_type):
        if tb_pq.is_empty():
            return
        entity_id_str = str(entity_id)
        if entity_type == "GL":
            gl_code = entity_id_str.split(" - ")[0] if " - " in entity_id_str else entity_id_str
            if gl_code.startswith("GL_"):
                gl_code = gl_code[3:]

            row = tb_pq.filter(pl.col("gl_code").cast(pl.Utf8) == gl_code)
            if not row.is_empty():
                r = row.row(0, named=True)
                exc["entity"]["entity_name"] = str(r.get("gl_name", entity_id_str))

                acct_head = str(r.get("main_head", ""))
                acct_type = _account_type(r)
                norm_bal = _normal_balance(acct_type)

                exc["business_context"] = {
                    "gl_code": str(r.get("gl_code", "")),
                    "gl_name": str(r.get("gl_name", "")),
                    "fsli": str(r.get("_fsli", "")),
                    "account_type": acct_type,
                    "business_unit": "Corporate",
                    "division": "General",
                    "normal_balance": norm_bal,
                }
                h_path = hierarchy_path(r)
                exc["hierarchy_context"] = {
                    "fs_head": acct_head,
                    "line_item": str(r.get("_fsli", "")),
                    "group": str(r.get("sub_head_2", "")),
                    "sub_group": "",
                    "hierarchy_path": h_path.split(" > ") if h_path else [],
                    "hierarchy_depth": len(h_path.split(" > ")) if h_path else 0,
                }
                bal = float(r.get("closing_balance") or 0.0)
                fsli = str(r.get("_fsli", ""))
                fsli_t = float(fsli_totals.get(fsli) or 0.0)
                exc["financial_context"] = {
                    "opening_balance": float(r.get("opening_balance") or 0.0),
                    "debit": float(r.get("debit") or 0.0),
                    "credit": float(r.get("credit") or 0.0),
                    "closing_balance": bal,
                    "movement": bal - float(r.get("opening_balance") or 0.0),
                    "movement_percent": 0.0,
                    "fsli_total": fsli_t,
                    "contribution_to_fsli": round((abs(bal) / fsli_t) * 100, 2) if fsli_t != 0 else 0.0,
                    "contribution_to_tb": round((abs(bal) / tb_total_balance) * 100, 2) if tb_total_balance != 0 else 0.0,
                    "rank_within_fsli": int(r.get("fsli_rank", 0)),
                    "rank_within_tb": int(r.get("tb_rank", 0)),
                    "normal_balance": norm_bal,
                }
                exc["classification"]["category"] = acct_head
                exc["classification"]["report_group"] = acct_head
        elif entity_type == "FSLI":
            fsli_name = entity_id_str
            if fsli_name.startswith("FSLI_"):
                fsli_name = fsli_name[5:]
            rows = tb_pq.filter(pl.col("_fsli") == fsli_name)
            if not rows.is_empty():
                exc["entity"]["entity_name"] = fsli_name
                first_row = rows.row(0, named=True)
                acct_head = str(first_row.get("main_head", ""))
                acct_type = _account_type(first_row)
                norm_bal = _normal_balance(acct_type)

                exc["business_context"] = {
                    "gl_code": "",
                    "gl_name": fsli_name,
                    "fsli": fsli_name,
                    "account_type": acct_type,
                    "business_unit": "Corporate",
                    "division": "General",
                    "normal_balance": norm_bal,
                }

                h_path = hierarchy_path(first_row)
                exc["hierarchy_context"] = {
                    "fs_head": acct_head,
                    "line_item": fsli_name,
                    "group": str(first_row.get("sub_head_2", "")),
                    "sub_group": "",
                    "hierarchy_path": [acct_head, fsli_name] if acct_head else h_path.split(" > "),
                    "hierarchy_depth": 2,
                }
                bal = float(rows["closing_balance"].sum())
                exc["financial_context"] = {
                    "opening_balance": float(rows["opening_balance"].sum()),
                    "debit": float(rows["debit"].sum()),
                    "credit": float(rows["credit"].sum()),
                    "closing_balance": bal,
                    "movement": bal - float(rows["opening_balance"].sum()),
                    "movement_percent": 0.0,
                    "fsli_total": abs(bal),
                    "contribution_to_fsli": 100.0,
                    "contribution_to_tb": round((abs(bal) / tb_total_balance) * 100, 2) if tb_total_balance != 0 else 0.0,
                    "rank_within_fsli": 1,
                    "rank_within_tb": 0,
                    "normal_balance": norm_bal,
                }
                exc["classification"]["category"] = acct_head
                exc["classification"]["report_group"] = acct_head

    def _process_validation_exceptions():
        # Validation Data
        rules = val_data if isinstance(val_data, list) else val_data.get("rules", [])
        for r in rules:
            if str(r.get("status")) in ["FAILED", "WARNING"]:
                entity = str(r.get("account", "SYSTEM_VALIDATION"))
                exc = get_or_create_exc(entity, "Validation")
                if "Validation" not in exc["metadata"]["analytics_sources"]:
                    exc["metadata"]["analytics_sources"].append("Validation")
                exc["scoring"]["validation_score"] += 15.0
                exc["scoring"]["composite_score"] += 15.0
                exc["triggered_rules"].append({
                    "rule_id": str(r.get("rule_name", "VALIDATION_FAIL")),
                    "source_engine": "validate_layer1_tb",
                    "description": str(r.get("message", "Pipeline governance validation failed.")),
                    "threshold": "Validation Failure",
                    "weight": 15.0,
                    "trigger_value": str(r.get("status", "FAILED")),
                })
                exc["evidence_package"]["validation"] = {
                    "message": str(r.get("message", "")),
                    "status": str(r.get("status", "")),
                    "rule_name": str(r.get("rule_name", "")),
                }
                if not exc["classification"]["primary_engine"]:
                    exc["classification"]["primary_engine"] = "Validation"

    _process_validation_exceptions()

    def _process_risk_indicator_exceptions():
        # Risk Indicators. build_risk_indicators.py's actual output has no "indicators" key --
        # its per-account risk entries live under "critical_risks" (the top-20, Critical-severity
        # accounts by score). Reading the never-existing "indicators" key silently produced zero
        # risk-sourced exceptions on every run.
        indicators = risk_data.get("critical_risks", [])
        for row in indicators:
            score = float(row.get("score") or row.get("weight") or 15.0)
            acct = str(row.get("account", ""))
            exc = get_or_create_exc(acct, "GL")
            if "Risk" not in exc["metadata"]["analytics_sources"]:
                exc["metadata"]["analytics_sources"].append("Risk")
            exc["scoring"]["risk_score"] += score
            exc["scoring"]["composite_score"] += score
            exc["triggered_rules"].append({
                "rule_id": row.get("type", row.get("theme", "RISK_ENGINE_FLAG")),
                "source_engine": "build_risk_indicators",
                "description": row.get("description", "Structural anomaly detected in account."),
                "threshold": "Risk Condition Met",
                "weight": score,
                "trigger_value": row.get("value", "Flagged"),
            })
            exc["evidence_package"]["risk"] = row
            enrich_from_tb(exc, acct, "GL")
            exc["classification"]["primary_engine"] = "Risk"

    _process_risk_indicator_exceptions()

    def _process_sensitive_account_exceptions():
        # Sensitive Accounts. build_sensitive_detector.py's actual output has no
        # "sensitive_accounts" list key -- that name only exists as a summary *count* under
        # summary.sensitive_accounts. The per-account list is "account_sensitivity". Reading
        # the never-existing "sensitive_accounts" key silently produced zero sensitivity-sourced
        # exceptions on every run, even when Critical-priority sensitive accounts existed.
        sensitive_accs = sens_data.get("account_sensitivity", [])
        for row in sensitive_accs:
            score = float(row.get("sensitivity_score") or row.get("score") or 20.0)
            acct = str(row.get("account", ""))
            exc = get_or_create_exc(acct, "GL")
            if "Sensitivity" not in exc["metadata"]["analytics_sources"]:
                exc["metadata"]["analytics_sources"].append("Sensitivity")
            exc["scoring"]["sensitivity_score"] += score
            exc["scoring"]["composite_score"] += score
            exc["triggered_rules"].append({
                "rule_id": f"SENSITIVE_{str(row.get('category', 'CAT')).upper()}",
                "source_engine": "build_sensitive_detector",
                "description": f"Account flagged as sensitive category: {row.get('category')}.",
                "threshold": "Matches sensitive dictionary",
                "weight": score,
                "trigger_value": row.get("category", ""),
            })
            exc["evidence_package"]["sensitivity"] = row
            enrich_from_tb(exc, acct, "GL")
            if not exc["classification"]["primary_engine"]:
                exc["classification"]["primary_engine"] = "Sensitivity"

    _process_sensitive_account_exceptions()

    def _process_anomaly_exceptions():
        # Anomaly Findings (build_anomaly_scanner) -- "SYSTEM_ANOMALY" is a
        # file-level finding (e.g. Benford's Law) with no single GL account to
        # enrich against; enrich_from_tb no-ops gracefully on any account string
        # that doesn't match a gl_code, so it's safe to call unconditionally.
        anomaly_findings = anomaly_data.get("findings", []) if anomaly_data else []
        for row in anomaly_findings:
            score = float(row.get("score") or 8.0)
            acct = str(row.get("account", "SYSTEM_ANOMALY"))
            entity_type = "System" if acct == "SYSTEM_ANOMALY" else "GL"
            exc = get_or_create_exc(acct, entity_type)
            if "Anomaly" not in exc["metadata"]["analytics_sources"]:
                exc["metadata"]["analytics_sources"].append("Anomaly")
            exc["scoring"]["anomaly_score"] += score
            exc["scoring"]["composite_score"] += score
            exc["triggered_rules"].append({
                "rule_id": row.get("rule_id", "ANOMALY_FLAG"),
                "source_engine": "build_anomaly_scanner",
                "description": row.get("description", "Anomaly pattern detected."),
                "threshold": "Anomaly Condition Met",
                "weight": score,
                "trigger_value": row.get("trigger_value", "Flagged"),
            })
            exc["evidence_package"]["anomaly"] = row
            if entity_type == "GL":
                enrich_from_tb(exc, acct, "GL")
            if not exc["classification"]["primary_engine"]:
                exc["classification"]["primary_engine"] = "Anomaly"

    _process_anomaly_exceptions()

    def _process_variance_exceptions():
        # Variance Data
        variances = var_data.get("material_variances", []) if isinstance(var_data, dict) else var_data
        for v in variances:
            entity = v.get("account") or v.get("fsli_group") or v.get("line_item")
            if entity:
                ent_str = str(entity)
                ent_type = "GL" if "GL" in ent_str or " - " in ent_str else "FSLI"
                exc = get_or_create_exc(ent_str, ent_type)
                if "Variance" not in exc["metadata"]["analytics_sources"]:
                    exc["metadata"]["analytics_sources"].append("Variance")

                score = float(v.get("weight") or 25.0)
                exc["scoring"]["variance_score"] += score
                exc["scoring"]["composite_score"] += score

                mov = abs(float(v.get("movement") or 0.0))
                multiple = round(mov / pm, 2) if pm > 0 else 0.0
                exc["materiality_context"]["materiality_multiple"] = multiple
                exc["materiality_context"]["exceeds_planning_materiality"] = (mov > plan_mat)

                exc["triggered_rules"].append({
                    "rule_id": "MATERIAL_VARIANCE",
                    "source_engine": "build_variance_analysis",
                    "description": str(v.get("note", "Movement exceeded materiality thresholds.")),
                    "threshold": "Performance Materiality",
                    "weight": score,
                    "trigger_value": mov,
                })
                exc["evidence_package"]["variance"] = v
                enrich_from_tb(exc, ent_str, ent_type)
                if not exc["classification"]["primary_engine"]:
                    exc["classification"]["primary_engine"] = "Variance"

    _process_variance_exceptions()

    def _apply_relationship_graph_context():
        # Traversal of Relationship Graph
        graph_nodes = {n["id"]: n for n in graph_data.get("nodes", [])}
        graph_edges = graph_data.get("edges", [])

        for exc in exceptions_map.values():
            gl_code = exc.get("business_context", {}).get("gl_code")
            if gl_code:
                node_id = f"GL_{gl_code}"
                if node_id in graph_nodes:
                    bp = graph_nodes[node_id].get("business_process")
                    if bp:
                        exc["relationship_context"]["same_business_process"] = [bp]
                        exc["business_context"]["business_process"] = bp
            fsli = exc.get("business_context", {}).get("fsli")
            if fsli and fsli != "Unmapped":
                node_id = f"FSLI_{fsli}"
                if node_id in graph_nodes:
                    bp = graph_nodes[node_id].get("business_process")
                    if bp:
                        exc["relationship_context"]["same_business_process"] = [bp]
                        exc["business_context"]["business_process"] = bp

        for e in graph_edges:
            if e["type"] == "PEER_OF":
                s_id = e["source"]
                t_id = e["target"]
                s_code = s_id.replace("GL_", "")
                t_code = t_id.replace("GL_", "")

                for exc in exceptions_map.values():
                    c = exc.get("business_context", {}).get("gl_code")
                    if c == s_code:
                        if t_code not in exc["relationship_context"]["peer_accounts"]:
                            exc["relationship_context"]["peer_accounts"].append(t_code)
                    elif c == t_code:
                        if s_code not in exc["relationship_context"]["peer_accounts"]:
                            exc["relationship_context"]["peer_accounts"].append(s_code)

    _apply_relationship_graph_context()

    consolidated = list(exceptions_map.values())

    def _score_exception_severities():
        for ex in consolidated:
            s = ex["scoring"]["composite_score"]
            amount = abs(ex.get("financial_context", {}).get("closing_balance", 0.0))
            ex["scoring"]["severity"] = resolve_severity(s, amount, ct)

    _score_exception_severities()

    consolidated = [ex for ex in consolidated if ex["scoring"]["severity"] != "Information Request" or ex["entity"]["entity_type"] == "Validation"]

    clusters_map = {}

    def _build_exception_clusters():
        for idx, exc in enumerate(consolidated):
            assertions = set()
            req_ev = set()
            for rule in exc["triggered_rules"]:
                sname = rule["source_engine"]
                if "risk" in sname:
                    assertions.update(["Accuracy", "Valuation"])
                    req_ev.add("Ledgers")
                if "sensitive" in sname:
                    assertions.update(["Classification", "Presentation", "Rights & Obligations"])
                    req_ev.add("Contracts/Agreements")
                if "variance" in sname:
                    assertions.update(["Completeness", "Cut-off"])
                    req_ev.add("Reconciliations")
                if "validat" in sname:
                    assertions.update(["Existence"])
                    req_ev.add("Mapping Review")

            if not assertions:
                assertions.add("Accuracy")
            if not req_ev:
                req_ev.add("Ledgers")

            exc["audit_planning"] = {
                "audit_assertions": list(assertions),
                "required_evidence": list(req_ev),
                "expected_audit_procedures": [f"Review {ev}" for ev in req_ev],
                "primary_audit_objective": f"Verify {list(assertions)[0]} assertion.",
                "planning_priority": exc["scoring"]["severity"],
            }

            bps = exc.get("relationship_context", {}).get("same_business_process", [])
            bp = bps[0] if bps else "General"

            fsli = exc.get("hierarchy_context", {}).get("line_item", "")
            if not fsli:
                fsli = "Unmapped"

            cluster_key = f"{bp}::{fsli}"
            if cluster_key not in clusters_map:
                clusters_map[cluster_key] = {
                    "cluster_id": f"CL-{len(clusters_map)+1:03d}",
                    "cluster_name": f"{bp} - {fsli}",
                    "cluster_type": "Business Process",
                    "business_process": bp,
                    "fsli": fsli,
                    "risk_themes": set(),
                    "member_exceptions": [],
                    "lead_exception": None,
                    "cluster_severity": "Information Request",
                    "member_count": 0,
                    "critical_count": 0,
                    "high_count": 0,
                    "total_balance": 0.0,
                    "cluster_score": 0.0,
                }

            cluster = clusters_map[cluster_key]
            exc_id = exc["metadata"]["exception_id"]
            cluster["member_exceptions"].append(exc_id)
            cluster["member_count"] += 1

            sev = exc.get("scoring", {}).get("severity", "Low")
            if sev == "Critical":
                cluster["critical_count"] += 1
            if sev == "High":
                cluster["high_count"] += 1

            cluster["total_balance"] += abs(exc.get("financial_context", {}).get("closing_balance", 0.0))
            cluster["cluster_score"] += exc.get("scoring", {}).get("composite_score", 0.0)

            for rule in exc.get("triggered_rules", []):
                cluster["risk_themes"].add(rule.get("rule_id", "Unknown"))

            exc["cluster_context"] = {
                "cluster_id": cluster["cluster_id"],
                "cluster_name": f"{bp} - {fsli}",
                "cluster_type": "Business Process",
                "business_process": bp,
                "lead_exception": None,
                "member_count": 0,
                "cluster_score": 0.0,
                "cluster_severity": "Information Request",
            }

    _build_exception_clusters()

    final_clusters = []
    def _finalize_clusters():
        for c_key, c_val in clusters_map.items():
            c_val["risk_themes"] = list(c_val["risk_themes"])
            if c_val["critical_count"] > 0:
                c_val["cluster_severity"] = "Critical"
            elif c_val["high_count"] > 0:
                c_val["cluster_severity"] = "High"
            elif c_val["cluster_score"] > 25:
                c_val["cluster_severity"] = "Medium"
            else:
                c_val["cluster_severity"] = "Low"

            if c_val["member_exceptions"]:
                c_val["lead_exception"] = c_val["member_exceptions"][0]

            final_clusters.append(c_val)

    _finalize_clusters()

    def _link_exceptions_to_clusters():
        for exc in consolidated:
            cid = exc.get("cluster_context", {}).get("cluster_id")
            if cid:
                c_val = next((c for c in final_clusters if c["cluster_id"] == cid), None)
                if c_val:
                    exc["cluster_context"]["lead_exception"] = c_val["lead_exception"]
                    exc["cluster_context"]["cluster_severity"] = c_val["cluster_severity"]
                    exc["cluster_context"]["member_count"] = c_val["member_count"]
                    exc["cluster_context"]["cluster_score"] = c_val["cluster_score"]

    _link_exceptions_to_clusters()

    consolidated.sort(key=lambda x: x["scoring"]["composite_score"], reverse=True)

    cat_summary = {}
    def _summarize_categories():
        for x in consolidated:
            for r in x["triggered_rules"]:
                src = r["source_engine"]
                cat_summary[src] = cat_summary.get(src, 0) + 1

    _summarize_categories()

    output_data = {
        "summary": {
            "exceptions_generated": len(consolidated),
            "critical": len([x for x in consolidated if x["scoring"]["severity"] == "Critical"]),
            "high": len([x for x in consolidated if x["scoring"]["severity"] == "High"]),
            "medium": len([x for x in consolidated if x["scoring"]["severity"] == "Medium"]),
            "low": len([x for x in consolidated if x["scoring"]["severity"] == "Low"]),
            "total_clusters": len(final_clusters),
            "top_engines": cat_summary,
        },
        "clusters": final_clusters,
        "exceptions": consolidated,
    }

    pq_out = []
    for exc in consolidated:
        flat = {}
        for k, v in exc.items():
            if isinstance(v, dict):
                for sk, sv in v.items():
                    flat[f"{k}_{sk}"] = str(sv)
            elif isinstance(v, list):
                flat[k] = str(v)
            else:
                flat[k] = v
        pq_out.append(flat)

    pq_df = pl.DataFrame(pq_out)
    parquet_path = out_dir / "consolidated_exceptions.parquet"
    if not pq_df.is_empty():
        write_parquet_atomic(pq_df, parquet_path)
        artifacts.append(str(parquet_path.resolve()))
    else:
        warnings.append("No exceptions generated; consolidated_exceptions.parquet not written.")

    json_path = out_dir / "consolidated_exceptions.json"
    write_json_atomic(output_data, json_path, indent=4)
    artifacts.append(str(json_path.resolve()))

    return {
        "execution_status": "SUCCESS",
        "pipeline_status": "SUCCESS",
        "message": f"Exception Engine correlated {len(consolidated)} canonical exception entities.",
        "artifacts": artifacts,
        "errors": errors,
        "warnings": warnings,
    }


_MONETARY_KEYWORDS = (
    "bank", "cash", "loan", "debtor", "receivable", "creditor", "payable", "borrowing",
    "deposit", "advance", "ecb", "fcnr",
)

@pipeline_tool("build_fx_exposure", domain="risk")
def build_fx_exposure(canonical_tb_file: str, output_dir: str = None, entity_profile_file: str = None, **kwargs) -> dict:
    """FX/translation-risk screen: classifies FCY-tagged accounts as monetary (retranslated,
    ongoing exposure) vs. non-monetary (historical-rate, no ongoing translation exposure),
    and surfaces any translation/exchange-fluctuation reserve balance. Gated on
    entity_profile.json's has_foreign_ops flag -- skipped (not merely empty) when that flag
    is False or entity_profile.json is absent, so the report doesn't carry an irrelevant
    section for a purely domestic entity."""
    errors = []
    artifacts = []

    out_dir = resolve_output_dir(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tb_path = Path(canonical_tb_file)
    if not tb_path.exists():
        raise PipelineFileError(str(tb_path), "canonical_tb.parquet")

    profile_path = Path(entity_profile_file) if entity_profile_file else out_dir / "entity_profile.json"
    if not profile_path.exists():
        profile_path = tb_path.parent / "entity_profile.json"
    has_foreign_ops = False
    if profile_path.exists():
        try:
            with open(profile_path, "r", encoding="utf-8") as f:
                has_foreign_ops = bool(json.load(f).get("flags", {}).get("has_foreign_ops", False))
        except Exception:
            errors.append({"type": "ParseError", "file": str(profile_path)})

    out_file = out_dir / "fx_exposure.json"

    if not has_foreign_ops:
        output_data = {
            "screened": False,
            "reason": (
                "entity_profile.json's has_foreign_ops flag is not set -- no material foreign-currency "
                "signal was found in this TB, so the FX/translation-risk screen was not run."
            ),
            "generated_at": datetime.datetime.now().isoformat(),
        }
        write_json_atomic(output_data, out_file, indent=4)
        return {
            "execution_status": "SUCCESS",
            "pipeline_status": "SUCCESS",
            "message": "FX exposure screen skipped -- has_foreign_ops is False.",
            "artifacts": [str(out_file.resolve())],
            "errors": errors,
            "data": output_data,
        }

    df = load_canonical_tb(tb_path)
    if df is None or df.is_empty():
        errors.append({"type": "EmptyDataError", "message": "canonical_tb.parquet loaded but is empty."})
        return {"execution_status": "FAILED", "errors": errors, "message": "Canonical TB is empty."}
    df = mapped_only(df)

    # The FCY/translation signal can live in the grouping workbook's FSLI text (sub_head_1/
    # sub_head_2, e.g. an "Exchange rate fluctuation" FSLI bucket) rather than the
    # individual gl_name -- live testing against a real TB confirmed gl_name-only matching
    # missed genuine FCY accounts whose own abbreviated names never mentioned currency.
    gl_name_lower = pl.col("gl_name").cast(pl.Utf8).str.to_lowercase().fill_null("")
    sub_head_1_lower = pl.col("sub_head_1").cast(pl.Utf8).str.to_lowercase().fill_null("") if "sub_head_1" in df.columns else pl.lit("")
    sub_head_2_lower = pl.col("sub_head_2").cast(pl.Utf8).str.to_lowercase().fill_null("") if "sub_head_2" in df.columns else pl.lit("")
    fcy_mask = pl.lit(False)
    for kw in FCY_KEYWORDS:
        fcy_mask = (
            fcy_mask
            | gl_name_lower.str.contains(kw, literal=True)
            | sub_head_1_lower.str.contains(kw, literal=True)
            | sub_head_2_lower.str.contains(kw, literal=True)
        )
    fcy_df = df.filter(fcy_mask)

    translation_mask = pl.lit(False)
    for kw in ("translation reserve", "exchange fluctuation"):
        translation_mask = (
            translation_mask
            | gl_name_lower.str.contains(kw, literal=True)
            | sub_head_1_lower.str.contains(kw, literal=True)
            | sub_head_2_lower.str.contains(kw, literal=True)
        )
    translation_df = df.filter(translation_mask)

    monetary_rows, non_monetary_rows = [], []
    for r in fcy_df.iter_rows(named=True):
        name_lower = str(r.get("gl_name", "")).lower()
        sub_head_lower = f"{r.get('sub_head_1', '')} {r.get('sub_head_2', '')}".lower()
        row_data = {
            "gl_code": r.get("gl_code", ""),
            "gl_name": r.get("gl_name", ""),
            "fs_head": r.get("report_head", ""),
            "closing_balance": float(r.get("closing_balance", 0.0)),
        }
        if "translation reserve" in sub_head_lower or "exchange fluctuation" in sub_head_lower:
            continue  # reported separately below, not double-counted as an exposure account
        if any(kw in name_lower or kw in sub_head_lower for kw in _MONETARY_KEYWORDS):
            monetary_rows.append(row_data)
        else:
            non_monetary_rows.append(row_data)

    monetary_rows.sort(key=lambda r: abs(r["closing_balance"]), reverse=True)
    non_monetary_rows.sort(key=lambda r: abs(r["closing_balance"]), reverse=True)

    translation_rows = [
        {
            "gl_code": r.get("gl_code", ""),
            "gl_name": r.get("gl_name", ""),
            "opening_balance": float(r.get("opening_balance", 0.0)),
            "movement": float(r.get("closing_balance", 0.0)) - float(r.get("opening_balance", 0.0)),
            "closing_balance": float(r.get("closing_balance", 0.0)),
        }
        for r in translation_df.iter_rows(named=True)
    ]

    total_monetary_exposure = sum(abs(r["closing_balance"]) for r in monetary_rows)

    output_data = {
        "screened": True,
        "methodology": (
            "Deterministic keyword screen over FCY-tagged accounts, split into monetary "
            "(retranslated at closing rate each period -- ongoing translation exposure: bank, "
            "loans, receivables/payables) vs. non-monetary (carried at historical rate once "
            "recognised -- no ongoing translation exposure), per Ind AS 21. Translation/"
            "exchange-fluctuation reserve movement reported separately."
        ),
        "summary": {
            "monetary_accounts": len(monetary_rows),
            "non_monetary_accounts": len(non_monetary_rows),
            "total_monetary_exposure": round(total_monetary_exposure, 2),
            "translation_reserve_accounts": len(translation_rows),
        },
        "monetary_items": monetary_rows,
        "non_monetary_items": non_monetary_rows,
        "translation_reserve_movement": translation_rows,
        "generated_at": datetime.datetime.now().isoformat(),
    }

    write_json_atomic(output_data, out_file, indent=4)
    artifacts.append(str(out_file.resolve()))

    return {
        "execution_status": "SUCCESS",
        "pipeline_status": "SUCCESS",
        "message": (
            f"FX exposure: {len(monetary_rows)} monetary account(s) "
            f"(exposure {total_monetary_exposure:,.0f}), {len(non_monetary_rows)} non-monetary, "
            f"{len(translation_rows)} translation-reserve account(s)."
        ),
        "artifacts": artifacts,
        "errors": errors,
        "data": output_data,
    }


_NON_CONCLUSION = (
    "These are going-concern INDICATORS drawn from the trial balance only. They are not a "
    "conclusion that a material uncertainty exists, and they are not an assessment of the "
    "appropriateness of the going-concern basis. That assessment rests with the audit team "
    "on evidence the trial balance cannot contain -- management's own assessment, cash-flow "
    "forecasts, facility and covenant terms, and letters of support (spec sec 1.2, sec 12.2)."
)

_GC_EVIDENCE = [
    "Management's documented going-concern assessment covering at least twelve months",
    "Cash-flow forecast with its underlying assumptions",
    "Bank and lender confirmations including facility headroom",
    "Loan covenant compliance working and any breach or waiver correspondence",
    "Repayment schedule showing obligations falling due",
    "Letters of support or comfort from the parent or government, with their enforceability assessed",
]

def _head_total(df, head: str) -> float:
    if "report_head" not in df.columns:
        return 0.0
    block = df.filter(pl.col("report_head") == head)
    return float(block["closing_balance"].sum()) if not block.is_empty() else 0.0

@pipeline_tool("build_going_concern_screen", domain="risk")
def build_going_concern_screen(
    canonical_tb_file: str,
    materiality_file: str = None,
    netting_file: str = None,
    output_dir: str = None,
    **kwargs,
) -> dict:
    """Screen the trial balance for sec 12.2's going-concern indicators -- accumulated
    losses or debit net worth, borrowings against available cash, unpaid statutory
    dues, idle or blocked funds, and dependence on government support -- each reported
    as an indicator with the management assessment, forecast and confirmations that
    would let the audit team form a view. Draws no going-concern conclusion. Writes
    going_concern_screen.json."""
    out_dir = resolve_output_dir(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tb_path = Path(canonical_tb_file)
    if not tb_path.exists():
        raise PipelineFileError(str(tb_path), "Canonical TB (supply canonical_tb_file)")

    df = load_canonical_tb(tb_path)
    if df is None or df.is_empty():
        return {
            "execution_status": "SUCCESS", "pipeline_status": "WARNING", "can_continue": True,
            "message": "Canonical TB is empty -- no going-concern screening performed.",
            "artifacts": [], "errors": [],
        }
    df = mapped_only(df)

    warnings = []
    mat = safe_load_json(Path(materiality_file) if materiality_file else out_dir / "materiality.json")
    perf = float(((mat or {}).get("thresholds") or {}).get("performance") or 0.0)

    # closing_balance is debit-positive, so equity and reserves carry a NEGATIVE sign.
    # Net worth is therefore the negated sum -- stated in the artifact so the
    # arithmetic is checkable rather than implicit.
    equity_raw = _head_total(df, "Equity")
    net_worth = -equity_raw
    total_assets = abs(_head_total(df, "Assets"))

    borrowings = gl_total_by_keywords(df, (
        "borrowing", "term loan", "cash credit", "overdraft", "debenture",
        "secured loan", "unsecured loan",
    ))

    # Wave 2 Fix 3b: -R/-P/-M clearing pairs (e.g. GL 20950021/20950022, "SBI- MUSCAT
    # (US$) -R"/"-P") were both summed in full here, inflating this liquidity signal from
    # a true ~Rs 700cr to a reported ~Rs 52,629cr on a live EPIL run. When
    # build_netting_screen has already run for this output_dir, substitute each netted
    # GL's closing_balance with its net figure before the cash/bank keyword sum --
    # optional and additive: absent netted_balances.parquet, this degrades to the prior
    # gross behavior exactly as before.
    netting_path = Path(netting_file) if netting_file else out_dir / "netted_balances.parquet"
    cash_df = df
    if netting_path.exists():
        netted = safe_load_parquet(netting_path)
        if netted is not None and not netted.is_empty() and "gl_code" in df.columns:
            cash_df = (
                df.join(netted, on="gl_code", how="left")
                .with_columns(pl.coalesce(["net_closing_balance", "closing_balance"]).alias("closing_balance"))
                .drop("net_closing_balance")
            )
    cash = gl_total_by_keywords(cash_df, ("cash", "bank"), exclude=("overdraft", "cash credit"))
    statutory = gl_total_by_keywords(df, (
        "gst payable", "tds payable", "provident fund", "esi", "professional tax",
        "income tax payable", "duty payable", "cess payable",
    ))
    idle = gl_total_by_keywords(df, ("idle fund", "blocked fund", "unspent", "earmarked", "non-moving"))
    govt_support = gl_total_by_keywords(df, (
        "grant-in-aid", "grant in aid", "budgetary support", "government grant",
        "equity infusion", "viability gap", "subsidy",
    ))

    indicators, findings = [], []

    def raise_indicator(key, present, observation, expectation, gap, rating, extra):
        indicators.append({"indicator": key, "present": present, **extra})
        if not present:
            return
        findings.append(make_record(
            source_screen="build_going_concern_screen",
            fsli="Going concern",
            amount=extra.get("amount"),
            observation=observation,
            expectation=expectation,
            gap=gap,
            assertions=["Presentation", "Valuation"],
            risk_basis=["context", "value"],
            risk_rating=rating,
            data_sufficiency="low",
            proposed_response=(
                "Obtain management's going-concern assessment and supporting forecast, and "
                "evaluate them against the facility and covenant position. No view on the "
                "going-concern basis can be formed from the trial balance."
            ),
            evidence_requested=_GC_EVIDENCE,
            valid_reasons=[
                "Losses may be a start-up or project-phase position anticipated in the business plan",
                "Borrowings may be long-dated with substantial undrawn headroom",
                "Parent or government support may be committed and enforceable",
                "Statutory dues may be within their due dates and settled after the period end",
            ],
            extra={"indicator": key, **extra},
        ))

    # 1. Accumulated losses / debit net worth
    raise_indicator(
        "negative_or_eroded_net_worth",
        net_worth < 0,
        observation=(
            f"Net worth computed from equity and reserves is {net_worth:,.2f}"
            + (f" against total assets of {total_assets:,.2f}." if total_assets else ".")
        ),
        expectation="Equity and reserves ordinarily carry a net credit position representing positive net worth.",
        gap="Net worth is negative -- accumulated losses exceed capital and reserves.",
        rating="high",
        extra={"net_worth": round(net_worth, 2), "total_assets": round(total_assets, 2),
               "amount": round(net_worth, 2),
               "sign_convention": "closing_balance is debit-positive; net worth = -(sum of Equity head)"},
    )

    # 2. Borrowings against available cash
    cover = (cash["balance"] / borrowings["balance"]) if borrowings["balance"] > 0 else None
    raise_indicator(
        "borrowings_high_relative_to_cash",
        bool(borrowings["balance"] > 0 and cover is not None and cover < 0.05
             and (not perf or borrowings["balance"] >= perf)),
        observation=(
            f"Borrowings of {borrowings['balance']:,.2f} against cash and bank balances of "
            f"{cash['balance']:,.2f}"
            + (f" -- cash covers {cover:.1%} of borrowings." if cover is not None else ".")
        ),
        expectation="Liquid balances would ordinarily provide some cover against borrowing obligations falling due.",
        gap="Cash cover is thin relative to the borrowing position.",
        rating="high",
        extra={"borrowings": round(borrowings["balance"], 2), "cash": round(cash["balance"], 2),
               "cash_cover_ratio": round(cover, 6) if cover is not None else None,
               "amount": round(borrowings["balance"], 2)},
    )

    # 3. Large unpaid statutory dues
    raise_indicator(
        "large_unpaid_statutory_dues",
        bool(statutory["balance"] > 0 and perf and statutory["balance"] >= perf),
        observation=f"Statutory dues carried at the period end total {statutory['balance']:,.2f}.",
        expectation="Statutory dues are ordinarily remitted by their due dates and carry only a short-period balance.",
        gap="The carried balance is material, which may indicate liquidity or compliance stress.",
        rating="medium",
        extra={"statutory_dues": round(statutory["balance"], 2), "amount": round(statutory["balance"], 2)},
    )

    # 4. Idle or blocked funds
    raise_indicator(
        "idle_or_blocked_funds",
        bool(idle["balance"] > 0 and (not perf or idle["balance"] >= perf)),
        observation=f"Balances described as idle, blocked, unspent or earmarked total {idle['balance']:,.2f}.",
        expectation="Funds held for a purpose would ordinarily be applied within their utilisation window.",
        gap="Funds appear held without application, raising recoverability and operational-viability questions.",
        rating="medium",
        extra={"idle_funds": round(idle["balance"], 2), "amount": round(idle["balance"], 2),
               "accounts": idle["accounts"][:5]},
    )

    # 5. Dependence on government support
    raise_indicator(
        "dependence_on_government_support",
        bool(govt_support["balance"] > 0 and (not perf or govt_support["balance"] >= perf)),
        observation=f"Government grant, subsidy or infusion heads total {govt_support['balance']:,.2f}.",
        expectation="An entity's continuity assumption should be assessed against how far it depends on external support.",
        gap="The scale of government support raises a continuity-dependence question the TB cannot resolve.",
        rating="medium",
        extra={"government_support": round(govt_support["balance"], 2),
               "amount": round(govt_support["balance"], 2),
               "accounts": govt_support["accounts"][:5]},
    )

    if not perf:
        warnings.append(
            "materiality.json not available -- indicators that depend on a materiality threshold "
            "(statutory dues, idle funds, government support) were assessed on presence alone."
        )

    present = [i for i in indicators if i["present"]]
    payload = {
        "methodology": (
            "sec 12.2's five indicators computed from the canonical TB's own classification. "
            "Net worth is derived from the Equity head under the debit-positive sign convention, "
            "stated explicitly so the arithmetic is checkable."
        ),
        "non_conclusion": _NON_CONCLUSION,
        "generated_at": datetime.datetime.now().isoformat(),
        "data_sufficiency": "low",
        "computed": {
            "net_worth": round(net_worth, 2),
            "total_assets": round(total_assets, 2),
            "borrowings": round(borrowings["balance"], 2),
            "cash_and_bank": round(cash["balance"], 2),
            "statutory_dues": round(statutory["balance"], 2),
            "idle_or_blocked_funds": round(idle["balance"], 2),
            "government_support": round(govt_support["balance"], 2),
        },
        "summary": {"indicators_evaluated": len(indicators), "indicators_present": len(present)},
        "indicators": indicators,
        "finding_records": findings,
        "safe_limitation": SAFE_WORDING_DISCLAIMER,
        # Wave 2 Fix 2b: "statutory_dues" above is one of five independent statutory-dues
        # populations across this codebase, via this function's own inline keyword tuple.
        "statutory_dues_population_basis": (
            "inline keyword tuple (gst payable/tds payable/provident fund/esi/professional tax); "
            "see also build_statutory_screen, build_caro_indicators, build_sensitive_detector for "
            "independently-computed statutory-dues populations"
        ),
    }

    out_path = out_dir / "going_concern_screen.json"
    with atomic_write(out_path) as tmp:
        Path(tmp).write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    return {
        "execution_status": "SUCCESS",
        "pipeline_status": "SUCCESS",
        "can_continue": True,
        "message": (
            f"Going-concern screen: {len(present)} of {len(indicators)} indicator(s) present. "
            "Indicators only -- no conclusion on the going-concern basis is drawn."
        ),
        "artifacts": [str(out_path.resolve())],
        "errors": [],
        "warnings": warnings,
    }


_PLANNING_ONLY = (
    "This is a planning-stage fraud-RISK indicator for audit-team evaluation. It is not a "
    "statement that fraud or irregularity has occurred, and it is not a conclusion on "
    "management integrity. Evaluating fraud risk, and any reporting that follows, rests with "
    "the audit team under its own professional and statutory responsibilities (spec sec 1.2, "
    "sec 12.1)."
)

_OVERRIDE_EVIDENCE = [
    "Journal dump for manual and year-end entries, with user and timestamp",
    "Authorisation and maker-checker trail for the entries concerned",
    "Approval records for the underlying transactions",
    "Management's written explanation of the balance and how it arose",
    "Account-wise reconciliation with a dated clearance plan",
]

_FRAUD_SENSITIVE = (
    "cash in hand", "petty cash", "advance to director", "advance to staff",
    "advance to officer", "advance to employee", "loan to director", "imprest",
)

@pipeline_tool("build_override_indicators", domain="risk")
def build_override_indicators(
    canonical_tb_file: str,
    anomaly_findings_file: str = None,
    materiality_file: str = None,
    output_dir: str = None,
    **kwargs,
) -> dict:
    """Gather sec 12.1's six management-override and irregularity indicators under one
    planning frame: non-nil suspense and clearing heads, round-sum and Benford
    deviations (read from build_anomaly_scanner rather than recomputed), long-unadjusted
    grants deposits and advances, write-offs and waivers, and fraud-sensitive heads such
    as large cash and director advances. Every output is a planning-stage risk indicator
    seeking evidence -- never a statement that fraud occurred. Writes
    override_indicators.json."""
    out_dir = resolve_output_dir(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tb_path = Path(canonical_tb_file)
    if not tb_path.exists():
        raise PipelineFileError(str(tb_path), "Canonical TB (supply canonical_tb_file)")

    df = load_canonical_tb(tb_path)
    if df is None or df.is_empty():
        return {
            "execution_status": "SUCCESS", "pipeline_status": "WARNING", "can_continue": True,
            "message": "Canonical TB is empty -- no override screening performed.",
            "artifacts": [], "errors": [],
        }

    warnings = []
    mat = safe_load_json(Path(materiality_file) if materiality_file else out_dir / "materiality.json")
    thresholds = (mat or {}).get("thresholds") or {}
    perf = float(thresholds.get("performance") or 0.0)
    trivial = float(thresholds.get("clearly_trivial") or 0.0)

    findings, indicators = [], []

    def add(indicator, observation, gap, rating, expectation, extra, account=None, amount=None, valid_reasons=None):
        findings.append(make_record(
            source_screen="build_override_indicators",
            account=account,
            fsli="Management override risk",
            amount=amount,
            observation=observation,
            expectation=expectation,
            gap=gap,
            assertions=["Accuracy", "Classification", "Completeness"],
            risk_basis=["nature", "context"] + (["value"] if perf and abs(amount or 0) >= perf else []),
            risk_rating=rating,
            regularity_flag=True,
            data_sufficiency="low",
            proposed_response=(
                "Obtain the journal dump and approval trail for the entries concerned and "
                "evaluate them as part of the audit team's own fraud-risk assessment."
            ),
            evidence_requested=_OVERRIDE_EVIDENCE,
            valid_reasons=valid_reasons or [
                "The balance may be a routine transitional position awaiting normal clearance",
                "Round amounts may reflect genuinely round contractual sums",
                "The account may be fully supported by documentation the TB cannot show",
            ],
            extra={"indicator": indicator, "planning_only": _PLANNING_ONLY, **extra},
        ))

    def _indicator_1_nonnil_suspense():
        # 1. Non-nil suspense / clearing heads
        suspense_kws = sensitive_keywords("suspense_control")
        suspense_hits = []
        for row in df.iter_rows(named=True):
            closing = float(row.get("closing_balance") or 0.0)
            if closing == 0.0:
                continue
            haystack = f" {str(row.get('gl_name') or '').lower()} {str(row.get('gl_code') or '').lower()} "
            if any(k in haystack for k in suspense_kws):
                suspense_hits.append({
                    "account": f"{row.get('gl_code')} - {row.get('gl_name')}",
                    "gl_code": str(row.get("gl_code") or ""),
                    "closing_balance": round(closing, 2),
                })
        indicators.append({"indicator": "non_nil_suspense", "count": len(suspense_hits),
                           "total": round(sum(abs(h["closing_balance"]) for h in suspense_hits), 2)})
        for h in suspense_hits:
            if trivial and abs(h["closing_balance"]) < trivial:
                continue
            add("non_nil_suspense", account=h["account"], amount=h["closing_balance"],
                observation=f"Suspense/clearing head '{h['account']}' carries {abs(h['closing_balance']):,.2f}.",
                expectation="Suspense and clearing heads are transitional and would ordinarily clear to nil by the period end.",
                gap="A carried balance means items remain unresolved, which can mask offsetting errors larger than the net figure.",
                rating="high" if (perf and abs(h["closing_balance"]) >= perf) else "medium",
                extra={"gl_code": h["gl_code"]})

    _indicator_1_nonnil_suspense()

    def _indicator_2_3_round_sum_and_benford():
        # 2 & 3. Round-sum and Benford -- read from the anomaly scanner, not recomputed.
        anom_path = (Path(anomaly_findings_file) if anomaly_findings_file
                     else out_dir / "anomaly_findings.json")
        anom = safe_load_json(anom_path) if anom_path.exists() else {}
        anom_findings = anom.get("findings") or []
        if not anom_path.exists():
            warnings.append(
                "anomaly_findings.json not available -- round-sum (TB-023) and Benford (TB-021) "
                "signals are not reframed as override indicators. Run build_anomaly_scanner first."
            )
        round_rows = [f for f in anom_findings if f.get("rule_id") == "TB-023"]
        benford = [f for f in anom_findings if f.get("rule_id") == "TB-021"]
        indicators.append({"indicator": "round_sum_balances", "count": len(round_rows)})
        indicators.append({"indicator": "benford_deviation", "count": len(benford)})

        if round_rows:
            add("round_sum_balances", amount=None,
                observation=(
                    f"{len(round_rows)} account(s) carry conspicuously round balances "
                    "(reported by build_anomaly_scanner as TB-023)."
                ),
                expectation="Balances arising from ordinary transaction flow are rarely round to several trailing zeros.",
                gap="Round balances can indicate manual estimates, unsupported provisions or parked amounts.",
                rating="medium",
                extra={"source_rule": "TB-023", "accounts": [r.get("account") for r in round_rows[:15]]})
        for b in benford:
            add("benford_deviation", amount=None,
                observation=f"Leading-digit distribution deviates from the Benford expectation: {b.get('description')}",
                expectation="Naturally-arising financial populations approximate the Benford leading-digit distribution.",
                gap="Deviation is a population-level signal only; it identifies no specific account.",
                rating="medium",
                extra={"source_rule": "TB-021", "trigger_value": b.get("trigger_value")},
                valid_reasons=[
                    "Small populations deviate from Benford by chance alone",
                    "Regulated or contractually-fixed pricing produces non-Benford distributions",
                    "A population dominated by a few large balances is not Benford-shaped",
                ])

    _indicator_2_3_round_sum_and_benford()

    def _indicator_4_long_unadjusted_balances():
        # 4. Long-unadjusted grants, deposits and advances
        unadjusted = gl_total_by_keywords(df, (
            "advance to", "mobilisation advance", "mobilization advance", "security deposit",
            "earnest money", "emd", "retention money", "unspent", "grant-in-aid", "grant in aid",
        ))
        indicators.append({"indicator": "long_unadjusted_balances",
                           "count": unadjusted["account_count"], "total": round(unadjusted["balance"], 2)})
        if unadjusted["balance"] > 0 and (not perf or unadjusted["balance"] >= perf):
            add("long_unadjusted_balances", amount=round(unadjusted["balance"], 2),
                observation=(
                    f"{unadjusted['account_count']} grant, deposit or advance account(s) totalling "
                    f"{unadjusted['balance']:,.2f} are carried at the period end."
                ),
                expectation="Advances and deposits are ordinarily adjusted or recovered within their contractual cycle.",
                gap="The trial balance shows no ageing, so how long these have been carried cannot be determined from it.",
                rating="medium",
                extra={"accounts": unadjusted["accounts"][:10]},
                valid_reasons=[
                    "The balances may be current and well within their adjustment cycle",
                    "Deposits may be contractually held until project completion",
                    "Grant funds may sit within their permitted utilisation window",
                ])

    _indicator_4_long_unadjusted_balances()

    def _indicator_5_write_offs_and_waivers():
        # 5. Write-offs, waivers and losses
        propriety = gl_total_by_keywords(df, sensitive_keywords("propriety"))
        indicators.append({"indicator": "write_offs_and_waivers",
                           "count": propriety["account_count"], "total": round(propriety["balance"], 2)})
        if propriety["balance"] > 0:
            add("write_offs_and_waivers", amount=round(propriety["balance"], 2),
                observation=(
                    f"{propriety['account_count']} write-off, waiver, ex-gratia, loss or shortage "
                    f"account(s) totalling {propriety['balance']:,.2f}."
                ),
                expectation="Write-offs and waivers require competent sanction and evidence that recovery was pursued first.",
                gap="Sanction and recovery evidence are not visible in the trial balance.",
                rating="high",
                extra={"accounts": propriety["accounts"][:10]},
                valid_reasons=[
                    "Sanction may exist and simply not appear in the ledger",
                    "The amounts may be routine provisions rather than actual write-offs",
                ])

    _indicator_5_write_offs_and_waivers()

    def _indicator_6_fraud_sensitive_heads():
        # 6. Fraud-sensitive heads
        sensitive_heads = gl_total_by_keywords(df, _FRAUD_SENSITIVE)
        indicators.append({"indicator": "fraud_sensitive_heads",
                           "count": sensitive_heads["account_count"], "total": round(sensitive_heads["balance"], 2)})
        if sensitive_heads["balance"] > 0 and (not perf or sensitive_heads["balance"] >= perf):
            add("fraud_sensitive_heads", amount=round(sensitive_heads["balance"], 2),
                observation=(
                    f"Fraud-sensitive heads (cash in hand, imprest, director and staff advances) "
                    f"total {sensitive_heads['balance']:,.2f} across "
                    f"{sensitive_heads['account_count']} account(s)."
                ),
                expectation="These heads are inherently susceptible by nature and ordinarily carry modest balances.",
                gap="The trial balance cannot evidence physical verification, authorisation or recovery status.",
                rating="high",
                extra={"accounts": sensitive_heads["accounts"][:10]},
                valid_reasons=[
                    "Cash balances may be fully supported by a verification certificate",
                    "Advances may be recent and within policy limits",
                ])

    _indicator_6_fraud_sensitive_heads()

    present = [i for i in indicators if i.get("count")]
    payload = {
        "methodology": (
            "sec 12.1's six indicators gathered under one planning frame. Round-sum and Benford "
            "signals are READ from build_anomaly_scanner's existing output rather than "
            "recomputed, so the two tools cannot disagree; the remaining four are screened here "
            "from the canonical TB and the sensitive knowledge pack."
        ),
        "planning_only": _PLANNING_ONLY,
        "generated_at": datetime.datetime.now().isoformat(),
        "data_sufficiency": "low",
        "summary": {
            "indicators_evaluated": len(indicators),
            "indicators_present": len(present),
            "findings": len(findings),
        },
        "indicators": indicators,
        "finding_records": findings,
        "safe_limitation": SAFE_WORDING_DISCLAIMER,
    }

    out_path = out_dir / "override_indicators.json"
    with atomic_write(out_path) as tmp:
        Path(tmp).write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    return {
        "execution_status": "SUCCESS",
        "pipeline_status": "SUCCESS",
        "can_continue": True,
        "message": (
            f"Override indicators: {len(present)} of {len(indicators)} indicator(s) present, "
            f"{len(findings)} planning-stage finding(s). No statement of fraud or irregularity "
            "is made."
        ),
        "artifacts": [str(out_path.resolve())],
        "errors": [],
        "warnings": warnings,
    }


_QUESTIONS = [
    {
        "id": "PS-AUTHORISED",
        "question": "Was the transaction authorised?",
        "tags": ("grant_subsidy", "propriety"),
        "extra_keywords": ("equity infusion", "budgetary support", "ex-gratia", "ex gratia"),
        "expectation": (
            "Expenditure or retention of public money requires competent sanction, budget "
            "provision and delegation of powers before it is incurred."
        ),
        "evidence": [
            "Competent sanction order naming the sanctioning authority",
            "Delegation of powers under which the sanction was issued",
            "Budget provision or re-appropriation covering the expenditure",
            "Board or government approval",
        ],
        "assertions": ["Rights", "Classification"],
    },
    {
        "id": "PS-PURPOSE",
        "question": "Was the money used for the approved purpose?",
        "tags": ("grant_subsidy",),
        "extra_keywords": ("project bank", "unspent", "earmarked", "restricted fund"),
        "expectation": (
            "Grant and subsidy funds carry purpose restrictions; expenditure must match the "
            "sanctioned activity and unspent balances must be identifiable."
        ),
        "evidence": [
            "Sanction conditions setting out the approved purpose",
            "Utilisation certificate",
            "Fund-utilisation statement and project expenditure statement",
            "Bank trail from receipt through to application of funds",
            "Unspent funds schedule with the treatment of interest earned",
        ],
        "assertions": ["Classification", "Completeness", "Presentation"],
    },
    {
        "id": "PS-PRUDENT",
        "question": "Was the expenditure prudent and economical?",
        "tags": ("propriety",),
        "extra_keywords": (
            "idle fund", "blocked fund", "repair and maintenance", "excess advance",
            "project delay", "loss on", "shortage",
        ),
        "expectation": (
            "Canons of financial propriety require expenditure to be prudent, economical and "
            "free of undue benefit to any person."
        ),
        "evidence": [
            "Procurement file including tender and comparative statement",
            "Justification note and need analysis",
            "Cost-benefit assessment where one was required",
            "Project review recording the reason for delay or idleness",
        ],
        "assertions": ["Valuation", "Classification"],
    },
    {
        "id": "PS-RECOVERY",
        "question": "Were recoveries pursued?",
        "tags": ("advance", "deposit", "propriety"),
        "extra_keywords": (
            "recoverable", "advance to staff", "advance to officer", "advance to employee",
            "shortage", "loss", "overpayment",
        ),
        "expectation": (
            "Long-pending advances, deposits, losses and shortages should show evidence of "
            "recovery action and responsibility fixation."
        ),
        "evidence": [
            "Party-wise ageing of the advance or recoverable",
            "Recovery notices issued and responses received",
            "Adjustment records showing how the balance was settled",
            "Responsibility fixation record where recovery has not occurred",
        ],
        "assertions": ["Existence", "Valuation", "Recoverability"],
    },
]

@pipeline_tool("build_public_sector_lens", domain="risk")
def build_public_sector_lens(
    canonical_tb_file: str,
    engagement_context_file: str = None,
    materiality_file: str = None,
    output_dir: str = None,
    **kwargs,
) -> dict:
    """Run the four public-sector regularity and propriety questions -- was it
    authorised, was it used for the approved purpose, was it prudent, were recoveries
    pursued -- against grant, subsidy, write-off, waiver, ex-gratia, advance, deposit
    and idle-fund heads. These are elevated by context rather than value, so a small
    balance still raises the sanction question. Writes public_sector_lens.json.

    Reports whether Government-company applicability was confirmed; never concludes it
    from the trial balance."""
    out_dir = resolve_output_dir(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tb_path = Path(canonical_tb_file)
    if not tb_path.exists():
        raise PipelineFileError(str(tb_path), "Canonical TB (supply canonical_tb_file)")

    df = load_canonical_tb(tb_path)
    if df is None or df.is_empty():
        return {
            "execution_status": "SUCCESS", "pipeline_status": "WARNING", "can_continue": True,
            "message": "Canonical TB is empty -- no public-sector screening performed.",
            "artifacts": [], "errors": [],
        }

    warnings = []
    ctx_path = (Path(engagement_context_file) if engagement_context_file
                else out_dir / "engagement_context.json")
    ctx = safe_load_json(ctx_path) if ctx_path.exists() else {}
    gate = ((ctx.get("applicability_gates") or {}).get("public_sector_lens") or {})
    applicable, gate_basis = gate.get("applicable"), gate.get("basis", "unconfirmed")
    if applicable is None:
        warnings.append(
            "Government-company status unconfirmed -- these observations are reported as "
            "information requests pending confirmation of the audit mandate (sec 2.1)."
        )

    mat = safe_load_json(Path(materiality_file) if materiality_file else out_dir / "materiality.json")
    perf = float(((mat or {}).get("thresholds") or {}).get("performance") or 0.0)

    findings, by_question = [], {}

    for spec in _QUESTIONS:
        keywords = set(spec["extra_keywords"])
        for tag in spec["tags"]:
            try:
                keywords.update(sensitive_keywords(tag))
            except Exception:  # noqa: BLE001 -- a renamed tag must not break the screen
                warnings.append(f"Sensitive tag {tag!r} not found in the pack; {spec['id']} ran without it.")
        keywords = tuple(sorted(keywords))

        hits = []
        for row in df.iter_rows(named=True):
            closing = float(row.get("closing_balance") or 0.0)
            if closing == 0.0:
                continue  # a nil balance raises no regularity question
            haystack = f" {str(row.get('gl_name') or '').lower()} {str(row.get('gl_code') or '').lower()} "
            matched = [k for k in keywords if k in haystack]
            if not matched:
                continue
            hits.append({
                "account": f"{row.get('gl_code')} - {row.get('gl_name')}",
                "gl_code": str(row.get("gl_code") or ""),
                "gl_name": str(row.get("gl_name") or ""),
                "closing_balance": round(closing, 2),
                "matched_terms": matched[:4],
                # remark #12: needed by PS-RECOVERY below to distinguish a genuine
                # deposit paid out (asset, debit-normal) from a payable withheld from
                # someone else (liability, credit-normal) -- e.g. retention money.
                "account_type": _account_type(row),
            })

        hits.sort(key=lambda h: abs(h["closing_balance"]), reverse=True)
        by_question[spec["id"]] = {
            "question": spec["question"],
            "accounts_matched": len(hits),
            "total_balance": round(sum(abs(h["closing_balance"]) for h in hits), 2),
            "accounts": hits[:20],
        }

        for hit in hits:
            material = perf and abs(hit["closing_balance"]) >= perf
            # remark #12 fix: PS-RECOVERY's tags include "deposit", which sweeps in both
            # genuine deposits paid out (asset, e.g. EMD/security deposit) and payables
            # withheld from someone else (liability, e.g. retention money withheld from
            # a contractor) -- the latter needs payable-appropriate assertions, not
            # asset-side recoverability ones. Every other question's assertion set is
            # unaffected (none of the other 3 mix asset- and liability-side populations
            # under one question the way PS-RECOVERY's "deposit" tag does).
            assertions = spec["assertions"]
            if spec["id"] == "PS-RECOVERY" and hit.get("account_type") == "Liability":
                assertions = ["Completeness", "Classification", "Cut-off"]
            findings.append(make_record(
                source_screen="build_public_sector_lens",
                account=hit["account"],
                fsli=str(hit["gl_name"]),
                amount=hit["closing_balance"],
                source_row_id=hit["gl_code"],
                observation=(
                    f"'{hit['gl_name']}' carries {abs(hit['closing_balance']):,.2f} and matches "
                    f"public-money terms ({', '.join(hit['matched_terms'])}). "
                    f"Regularity question: {spec['question']}"
                ),
                expectation=spec["expectation"],
                gap=(
                    "The trial balance shows the balance but carries no evidence of sanction, "
                    "purpose compliance, prudence or recovery action -- none of which it can show."
                ),
                assertions=assertions,
                # sec 2.2: public money makes CONTEXT decisive; value is secondary here.
                risk_basis=["context", "nature"] + (["value"] if material else []),
                risk_rating="high" if material else "medium",
                regularity_flag=True,
                data_sufficiency="low",
                proposed_response=(
                    "Obtain the sanction and utilisation records listed and test whether authority, "
                    "purpose, economy and recovery were satisfied. No view is possible from the TB."
                ),
                evidence_requested=spec["evidence"],
                valid_reasons=[
                    "Sanction may exist and simply not be visible in the ledger",
                    "The balance may relate to a scheme still within its permitted utilisation window",
                    "Recovery action may be in progress but not reflected in the account",
                ],
                extra={
                    "question_id": spec["id"],
                    "regularity_question": spec["question"],
                    "matched_terms": hit["matched_terms"],
                    "applicability_confirmed": applicable is True,
                },
            ))

    payload = {
        "methodology": (
            "sec 12.4's four regularity questions are run against grant, propriety, advance and "
            "deposit heads identified from the sensitive knowledge pack. Accounts are raised by "
            "CONTEXT rather than value: public money makes authority and purpose decisive "
            "regardless of amount (sec 2.2), so only a nil balance is excluded."
        ),
        "generated_at": datetime.datetime.now().isoformat(),
        "applicability": {
            "government_company_confirmed": applicable,
            "basis": gate_basis,
            "note": (
                "A grant or write-off head in the ledger warrants the regularity question either "
                "way; it does NOT establish Government-company status, which no trial balance can "
                "show (sec 1.2)."
            ),
        },
        "data_sufficiency": "low",
        "summary": {
            "questions_run": len(_QUESTIONS),
            "accounts_flagged": len(findings),
            "by_question": {k: v["accounts_matched"] for k, v in by_question.items()},
        },
        "questions": by_question,
        "finding_records": findings,
        "safe_limitation": SAFE_WORDING_DISCLAIMER,
    }

    out_path = out_dir / "public_sector_lens.json"
    with atomic_write(out_path) as tmp:
        Path(tmp).write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    return {
        "execution_status": "SUCCESS",
        "pipeline_status": "SUCCESS",
        "can_continue": True,
        "message": (
            f"Public-sector lens: {len(findings)} regularity/propriety observation(s) across "
            f"{len(_QUESTIONS)} question(s); applicability {gate_basis}."
        ),
        "artifacts": [str(out_path.resolve())],
        "errors": [],
        "warnings": warnings,
    }


_TOOLS_ROOT = Path(__file__).resolve().parents[1]

def _legacy_expected_pairs() -> list:
    """The 5 (source, target) FSLI pairs this graph tool has always drawn
    RELATED_TO edges for, read from the relationships pack via their
    legacy_pair marker so there is one definition rather than two."""
    legacy = [r for r in load_pack("relationships")["relationships"] if r.get("legacy_pair")]
    # Sorted by legacy_order, not by REL id: the pack lists relationships in spec
    # sec 7's table order, but this tool's output array must keep the original
    # iteration order so the artifact stays byte-identical after extraction.
    legacy.sort(key=lambda r: r["legacy_order"])
    return [tuple(r["legacy_pair"]) for r in legacy]

def _get_bp(fsli_name, bp_mapping) -> str:
    for k, v in bp_mapping.items():
        if k.lower() in str(fsli_name).lower():
            return v
    return "General Ledger & Reporting"  # default; see business_process_mapping's _note

def _link_analytics(data_list, link_type, target_type, nodes, edges, out_relationships):
    for item in data_list:
        acct_raw = item.get("account", "")
        if not acct_raw:
            continue

        gl_code = str(acct_raw).split(" - ")[0]
        gl_id = f"GL_{gl_code}"

        theme = item.get("theme", item.get("type", item.get("category", "Anomaly")))
        theme_id = f"{target_type}_{theme}"

        if not any(n["id"] == theme_id for n in nodes):
            nodes.append({"id": theme_id, "type": target_type, "name": theme})

        edges.append({"source": gl_id, "target": theme_id, "type": link_type})
        if target_type == "Risk Theme":
            out_relationships.append({"account": gl_id, "risk": theme})
        else:
            out_relationships.append({"account": gl_id, "validation": theme})

def _screen_relationship_signals(canonical_tb_path) -> dict:
    """Screen the canonical TB for the four relationship-domain signals
    build_risk_indicators scores on.

    These four rules -- ORPHAN_NODE, SUSPENSE_CLEARING, INTERCOMPANY and
    SIGN_ANOMALY, 70 of the scoring model's weight points -- read the keys
    produced here. Before this, no tool wrote them at any point in the chain, so
    all four contributed zero on every run, and suspense/inter-company accounts
    (TB spec sec 5.4, sec 12.1: material by NATURE regardless of value) could
    never reach the Critical band through this path.

    Every entry carries `account` formatted "{gl_code} - {gl_name}", because that
    is the exact lookup key build_risk_indicators' add_risk() matches on -- a
    differently-formatted key silently scores nothing.

    Keyword vocabulary comes from the sensitive knowledge pack rather than a local
    list, so this screen and build_sensitive_detector cannot disagree about what
    counts as a suspense or inter-company account. Sign is judged against
    report_head, which load_canonical_tb() derives via classify_row().
    """
    signals = {
        "orphan_nodes": [],
        "suspense_accounts": [],
        "intercompany": [],
        "sign_anomalies": [],
    }

    df = load_canonical_tb(canonical_tb_path)
    if df is None or df.is_empty():
        return signals

    suspense_kws = sensitive_keywords("suspense_control")
    # Inter-company detection uses the structural subset of the related_party
    # vocabulary -- counterparty-relationship terms, not the governance ones
    # (director/KMP/promoter), which indicate a related party without implying an
    # inter-entity BALANCE that should mirror on a counterparty's books.
    _IC_TERMS = (
        "inter", "due from", "due to", "icd", "iut", "cash call", "holding",
        "subsidiary", "associate", "joint venture", "jv", "group co", "parent co",
        "prnt co", "holdg co",
    )
    intercompany_kws = tuple(
        k for k in sensitive_keywords("related_party") if any(t in k for t in _IC_TERMS)
    )
    debit_heads = normal_debit_heads()
    credit_heads = normal_credit_heads()

    for row in df.iter_rows(named=True):
        gl_code = str(row.get("gl_code") or "")
        gl_name = str(row.get("gl_name") or "")
        account = f"{gl_code} - {gl_name}"
        closing = float(row.get("closing_balance") or 0.0)
        # Same padded haystack build_sensitive_detector uses, so a keyword anchored
        # on spaces (e.g. " iut ") behaves identically in both tools.
        haystack = f" {gl_name.lower()} {gl_code.lower()} "
        head = str(row.get("report_head") or "").strip()
        head_lower = head.lower()

        if head_lower == "unmapped" or str(row.get("mapped_status") or "").upper() in (
            MAPPED_STATUS_UNMAPPED,
            MAPPED_STATUS_UNMATCHED,
        ):
            signals["orphan_nodes"].append({
                "account": account, "gl_code": gl_code, "gl_name": gl_name,
                "closing_balance": round(closing, 2),
                "reason": "No reliable FSLI mapping -- excluded from hierarchy analytics.",
            })

        # Suspense/clearing is only a signal when it carries a balance: a nil
        # suspense account is the normal, correctly-cleared state (sec 12.1).
        if closing != 0.0 and any(k in haystack for k in suspense_kws):
            signals["suspense_accounts"].append({
                "account": account, "gl_code": gl_code, "gl_name": gl_name,
                "closing_balance": round(closing, 2),
                "reason": "Suspense/clearing/control head carrying a non-nil balance.",
            })

        if any(k in haystack for k in intercompany_kws):
            signals["intercompany"].append({
                "account": account, "gl_code": gl_code, "gl_name": gl_name,
                "closing_balance": round(closing, 2),
                "reason": "Inter-company / inter-unit balance indicator in the ledger name.",
            })

        # closing_balance is debit-positive by construction (TB-000 sanity-checks
        # that assumption). A nil balance has no side, so is never an anomaly.
        if closing != 0.0:
            if head_lower in debit_heads:
                expected = "Debit"
            elif head_lower in credit_heads:
                expected = "Credit"
            else:
                expected = None
            actual = "Debit" if closing > 0 else "Credit"
            if expected and expected != actual:
                signals["sign_anomalies"].append({
                    "account": account, "gl_code": gl_code, "gl_name": gl_name,
                    "closing_balance": round(closing, 2),
                    "report_head": head,
                    "normal_balance": expected,
                    "actual_balance": actual,
                    "reason": (
                        f"{head} account carries a {actual.lower()} closing balance; "
                        f"{expected.lower()} is the normal side."
                    ),
                })

    return signals

@pipeline_tool("build_relationship_analytics", domain="risk")
def build_relationship_analytics(
    canonical_tb_file: str,
    fsli_summary_file: str,
    variance_analysis_file: str = None,
    output_dir: str = None,
    **kwargs,
) -> dict:
    """Build a deterministic relationship graph (GL -> FSLI -> business process, peers,
    expected financial relationships) and screen the TB for the four relationship-domain
    signals the risk engine scores on: orphan (unmapped) nodes, non-nil suspense/clearing
    accounts, inter-company balances, and abnormal-sign accounts.

    variance_analysis_file is optional -- Single-TB mode doesn't run variance analysis
    (requires a PY/CY comparison basis), so this degrades gracefully to an empty variance
    dataset when the file isn't supplied/doesn't exist, same as when JSON parsing fails.

    This tool deliberately does NOT read risk_indicators.json/sensitive_accounts.json:
    it runs BEFORE both of those tools in the deterministic chain (see routes.py
    _run_core_analytics_chain), so those files never exist at call time. The
    risk/sensitive/variance correlation those reads were meant to perform already
    happens in build_exception_consolidator, which loads all three source files itself
    at a point in the chain where they do exist.
    """
    errors = []
    warnings = []
    artifacts = []

    out_dir = resolve_output_dir(output_dir)
    os.makedirs(out_dir, exist_ok=True)

    paths = {
        "Canonical TB": resolve_artifact_path(canonical_tb_file, out_dir, "canonical_tb.parquet"),
        "FSLI Summary": resolve_artifact_path(fsli_summary_file, out_dir, "fsli_summary.parquet"),
    }
    _hints = {
        "Canonical TB": "supply canonical_tb_file from load_tb_from_db/ingest_tb_to_live",
        "FSLI Summary": "run build_fsli_summary first",
    }

    for name, p in paths.items():
        if not p.exists():
            errors.append({"type": "FileNotFoundError", "file": str(p)})
            return {"execution_status": "FAILED", "errors": errors, "message": f"{name} not found: {p} ({_hints[name]})."}

    tb_df = pl.read_parquet(paths["Canonical TB"])
    fsli_df = pl.read_parquet(paths["FSLI Summary"])

    variance_path = resolve_artifact_path(variance_analysis_file, out_dir, "variance_analysis.json")
    var_data = safe_load_json(variance_path) if variance_path.exists() else {}
    if variance_analysis_file and not variance_path.exists():
        warnings.append(f"Variance analysis file not found: {variance_path}")

    nodes = []
    edges = []

    default_bp_mapping = {
        k: v for k, v in load_pack("relationships")["business_process_mapping"].items()
        if not k.startswith("_")
    }
    bp_mapping_path = _TOOLS_ROOT / "business_process_mapping.json"
    if bp_mapping_path.exists():
        bp_mapping = safe_load_json(bp_mapping_path) or dict(default_bp_mapping)
    else:
        bp_mapping = dict(default_bp_mapping)

    # 1. Build Nodes and Base Hierarchy
    gl_nodes = {}
    for idx, row in enumerate(tb_df.iter_rows(named=True)):
        gl_id = f"GL_{row.get('gl_code', idx)}"
        gl_name = str(row.get("gl_name", "Unknown"))
        fsli = row_fsli(row)
        bp = _get_bp(fsli, bp_mapping)

        node = {
            "id": gl_id,
            "type": "GL",
            "name": gl_name,
            "fsli": fsli,
            "business_process": bp,
        }
        gl_nodes[gl_id] = node
        nodes.append(node)

        edges.append({"source": gl_id, "target": f"FSLI_{fsli}", "type": "BELONGS_TO"})
        edges.append({"source": gl_id, "target": f"BP_{bp}", "type": "PART_OF_PROCESS"})

    # FSLI Nodes -- defensive column resolution, fsli_summary.parquet may be shaped differently
    # by whichever sibling tool produces it.
    fsli_col = first_present_column(fsli_df, ["sub_head_1", "fsli", "line_item"])
    head_col = first_present_column(fsli_df, ["main_head", "fs_head", "head"])
    if fsli_col is None:
        warnings.append(f"FSLI Summary has no recognizable FSLI column (columns: {list(fsli_df.columns)}); skipping FSLI-node enrichment.")
    else:
        for row in fsli_df.iter_rows(named=True):
            fsli = str(row.get(fsli_col, "Unmapped"))
            head = str(row.get(head_col, "Unmapped")) if head_col else "Unmapped"
            fsli_id = f"FSLI_{fsli}"
            bp = _get_bp(fsli, bp_mapping)

            if not any(n["id"] == fsli_id for n in nodes):
                nodes.append({"id": fsli_id, "type": "FSLI", "name": fsli})
                nodes.append({"id": f"HEAD_{head}", "type": "FS Head", "name": head})
                nodes.append({"id": f"BP_{bp}", "type": "Business Process", "name": bp})

                edges.append({"source": fsli_id, "target": f"HEAD_{head}", "type": "BELONGS_TO"})
                edges.append({"source": fsli_id, "target": f"BP_{bp}", "type": "PART_OF_PROCESS"})

    unique_nodes = list({n["id"]: n for n in nodes}.values())
    nodes = unique_nodes

    unique_edges = []
    seen_edges = set()
    for e in edges:
        e_tuple = (e["source"], e["target"], e["type"])
        if e_tuple not in seen_edges:
            seen_edges.add(e_tuple)
            unique_edges.append(e)
    edges = unique_edges

    # 2. Account Relationships (Peer & Offset)
    fsli_to_gls = {}
    for gl_id, gl_data in gl_nodes.items():
        fsli_to_gls.setdefault(gl_data["fsli"], []).append(gl_id)

    for fsli, gls in fsli_to_gls.items():
        if len(gls) > 1:
            for i in range(len(gls)):
                for j in range(i + 1, len(gls)):
                    edges.append({"source": gls[i], "target": gls[j], "type": "PEER_OF"})

    financial_relationships = []
    for src, tgt in _legacy_expected_pairs():
        src_id = f"FSLI_{src}"
        tgt_id = f"FSLI_{tgt}"
        if any(n["id"] == src_id for n in nodes) and any(n["id"] == tgt_id for n in nodes):
            edges.append({"source": src_id, "target": tgt_id, "type": "RELATED_TO"})
            financial_relationships.append({"source": src, "target": tgt})

    # 3. Integrate variance analytics into the graph, when a variance run exists.
    # Only variance is read here: it is the one analytics source that can legitimately
    # precede this tool (an agent may call build_variance_analysis first, and COMPARISON
    # runs produce one). risk_indicators.json/sensitive_accounts.json are NOT read -- see
    # this function's docstring for why, and where that correlation actually happens.
    risk_relationships = []
    validation_relationships = []

    if "material_variances" in var_data:
        for v in var_data["material_variances"]:
            v["theme"] = "Material Variance"
        _link_analytics(var_data["material_variances"], "TRIGGERS", "Risk Theme", nodes, edges, risk_relationships)

    relationship_graph = {
        "nodes": nodes,
        "edges": edges,
        "metadata": {
            "total_nodes": len(nodes),
            "total_edges": len(edges),
        },
    }

    # The key names below are the contract build_risk_indicators reads (see its
    # section 3, "Apply Relationship Risks"). Do not rename without updating that
    # consumer -- tests/tools/test_artifact_contracts.py asserts they stay in sync.
    signals = _screen_relationship_signals(paths["Canonical TB"])

    relationship_analytics = {
        "financial_relationships": financial_relationships,
        "hierarchy_relationships": {"nodes": len(nodes), "edges": len(edges)},
        "hierarchy_validation": {
            "orphan_nodes": signals["orphan_nodes"],
            "orphan_count": len(signals["orphan_nodes"]),
        },
        "suspense_analysis": {
            "accounts": signals["suspense_accounts"],
            "total_balance": round(sum(a["closing_balance"] for a in signals["suspense_accounts"]), 2),
        },
        "intercompany_relationships": signals["intercompany"],
        "sign_relationships": signals["sign_anomalies"],
        "business_process_mapping": bp_mapping,
        "account_relationships": {"peers_identified": len([e for e in edges if e["type"] == "PEER_OF"])},
        "validation_relationships": validation_relationships,
        "risk_relationships": risk_relationships,
        "summary": {
            "total_relationships_integrated": len(edges),
            "orphan_nodes": len(signals["orphan_nodes"]),
            "suspense_accounts": len(signals["suspense_accounts"]),
            "intercompany_accounts": len(signals["intercompany"]),
            "sign_anomalies": len(signals["sign_anomalies"]),
        },
    }

    graph_path = out_dir / "relationship_graph.json"
    analytics_path = out_dir / "relationship_analytics.json"

    write_json_atomic(relationship_graph, graph_path, indent=4)
    write_json_atomic(relationship_analytics, analytics_path, indent=4)

    artifacts.append(str(graph_path.resolve()))
    artifacts.append(str(analytics_path.resolve()))

    return {
        "execution_status": "SUCCESS",
        "pipeline_status": "SUCCESS",
        "message": (
            f"Relationship Integration Engine complete. Graph built with {len(nodes)} nodes "
            f"and {len(edges)} edges. Screened: {len(signals['orphan_nodes'])} orphan, "
            f"{len(signals['suspense_accounts'])} suspense, "
            f"{len(signals['intercompany'])} inter-company, "
            f"{len(signals['sign_anomalies'])} abnormal-sign account(s)."
        ),
        "artifacts": artifacts,
        "errors": errors,
        "warnings": warnings,
    }


_EXPECTED_BANDS = {
    "REL-01": (0.02, 0.60, "Receivables typically fall between roughly 7 days and 7 months of revenue."),
    "REL-02": (0.02, 0.28, "GST output on fully taxable revenue approaches the applicable rate; exempt and zero-rated supplies pull it down."),
    "REL-03": (0.02, 0.60, "Payables typically represent a few weeks to a few months of procurement."),
    "REL-05": (0.01, 0.35, "An annual depreciation charge is usually a modest fraction of gross block."),
    "REL-07": (0.01, 0.24, "Interest on borrowings normally falls within commercial lending rates."),
    "REL-08": (0.02, 0.40, "Statutory payroll deductions and provisions are a meaningful but minority share of payroll cost."),
    "REL-09": (0.02, 1.50, "Inventory usually represents a fraction of annual procurement, though project entities carry more."),
}

def _resolve__build_relationship_expectations(fsli_df, tb_df, keywords) -> dict:
    node = find_fsli_component(fsli_df, keywords)
    if node and node["balance"] > 0:
        return {"balance": node["balance"], "basis": "fsli", "detail": node["node_name"]}
    gl = gl_total_by_keywords(tb_df, keywords)
    if gl["balance"] > 0:
        names = [a["gl_name"] for a in gl["accounts"][:2] if a["gl_name"]]
        detail = ", ".join(names) if names else f"{gl['account_count']} ledger match(es)"
        if gl["account_count"] > 2:
            detail += f" and {gl['account_count'] - len(names)} other account(s)"
        return {"balance": gl["balance"], "basis": "gl_name", "detail": detail}
    return {"balance": 0.0, "basis": None, "detail": None}

@pipeline_tool("build_relationship_expectations", domain="risk")
def build_relationship_expectations(
    canonical_tb_file: str,
    fsli_summary_file: str = None,
    materiality_file: str = None,
    output_dir: str = None,
    **kwargs,
) -> dict:
    """Compute each expected trial-balance relationship (receivables/revenue,
    GST output/revenue, depreciation/PPE, finance cost/borrowings, statutory
    dues/payroll and others), state the expectation, quantify the gap and assign
    severity. Ratios with a missing or zero denominator are reported as information
    requests, never approximated. Writes relationship_expectations.json."""
    out_dir = resolve_output_dir(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tb_path = Path(canonical_tb_file)
    if not tb_path.exists():
        raise PipelineFileError(str(tb_path), "Canonical TB (supply canonical_tb_file)")

    tb_df = load_canonical_tb(tb_path)
    if tb_df is None or tb_df.is_empty():
        return {
            "execution_status": "SUCCESS", "pipeline_status": "WARNING", "can_continue": True,
            "message": "Canonical TB is empty -- no relationship expectations computed.",
            "artifacts": [], "errors": [],
        }

    warnings = []
    fsli_path = Path(fsli_summary_file) if fsli_summary_file else out_dir / "fsli_summary.parquet"
    fsli_df = safe_load_parquet(fsli_path) if fsli_path.exists() else None
    if fsli_df is None or fsli_df.is_empty():
        warnings.append(
            "fsli_summary.parquet not available -- components matched on ledger names only. "
            "Run build_fsli_summary for hierarchy-based matching."
        )

    mat_path = Path(materiality_file) if materiality_file else out_dir / "materiality.json"
    mat = safe_load_json(mat_path) if mat_path.exists() else {}
    perf = float((mat.get("thresholds") or {}).get("performance") or 0.0)
    data_sufficiency = "medium" if (perf and fsli_df is not None) else "low"

    pack = load_pack("relationships")
    results, findings = [], []

    for rel in pack["relationships"]:
        band = _EXPECTED_BANDS.get(rel["id"])
        if not band:
            continue  # presence-only pair -- build_counterpart_screen handles it
        low, high, band_rationale = band

        # Wave 2 Fix 2a: a revenue-shaped source relationship defers to the single shared
        # compute_total_revenue helper instead of this function's own keyword resolution --
        # one of at least three independent revenue engines that used to disagree, which is
        # exactly why a live EPIL run showed revenue stated three different ways.
        if "revenue from operations" in rel["source_keywords"]:
            _rev = compute_total_revenue(out_dir, tb_df)
            source = {"balance": _rev["balance"], "basis": _rev["basis"], "detail": "Total Revenue (shared)"}
        else:
            source = _resolve__build_relationship_expectations(fsli_df, tb_df, rel["source_keywords"])
        target = _resolve__build_relationship_expectations(fsli_df, tb_df, rel["target_keywords"])

        # sec 6.2: never compute a ratio on an absent or zero denominator.
        if source["balance"] <= 0:
            results.append({
                "id": rel["id"], "name": rel["name"], "status": "not_computed",
                "reason": f"Denominator absent -- no head matched {rel['source_keywords']}.",
                "missing_component": "source",
            })
            continue
        if target["balance"] <= 0:
            results.append({
                "id": rel["id"], "name": rel["name"], "status": "not_computed",
                "reason": "Numerator absent -- reported by build_counterpart_screen as a missing counterpart.",
                "missing_component": "target",
            })
            continue

        ratio = target["balance"] / source["balance"]
        within = low <= ratio <= high
        result = {
            "id": rel["id"], "name": rel["name"],
            "status": "within_expectation" if within else "outside_expectation",
            "observed_ratio": round(ratio, 6),
            "expected_band": {"low": low, "high": high},
            "numerator": {"head": target["detail"], "balance": round(target["balance"], 2), "basis": target["basis"]},
            "denominator": {"head": source["detail"], "balance": round(source["balance"], 2), "basis": source["basis"]},
            "formula": f"{target['detail']} / {source['detail']}",
        }
        results.append(result)
        if within:
            continue

        direction = "below" if ratio < low else "above"
        bound = low if ratio < low else high
        # Severity per sec 7.1: high when material or regularity-sensitive, else the
        # pack's own default, floored at medium.
        material = perf and source["balance"] >= perf
        severity = rel.get("default_severity", "medium") if material else "medium"
        if rel.get("regularity_flag"):
            severity = "high"

        findings.append(make_record(
            source_screen="build_relationship_expectations",
            fsli=source["detail"],
            amount=round(source["balance"], 2),
            observation=(
                f"{rel['name']}: {target['detail']} is {target['balance']:,.2f} against "
                f"{source['detail']} of {source['balance']:,.2f} -- a ratio of {ratio:.2%}."
            ),
            expectation=f"{rel['expectation']} {band_rationale} Plausible band: {low:.0%} to {high:.0%}.",
            gap=(
                f"Observed {ratio:.2%} is {direction} the plausible band bound of {bound:.0%}, "
                f"a difference of {abs(ratio - bound):.2%} in ratio terms."
            ),
            assertions=rel.get("assertions", []),
            risk_basis=["relationship"] + (["value"] if material else [])
                       + (["context"] if rel.get("regularity_flag") else []),
            risk_rating=severity,
            regularity_flag=bool(rel.get("regularity_flag")),
            data_sufficiency=data_sufficiency,
            proposed_response=(
                "Test whether one of the listed explanations accounts for the deviation "
                "before treating it as an analytical exception, then obtain the records listed."
            ),
            evidence_requested=rel.get("resolving_evidence", []),
            valid_reasons=rel.get("valid_reasons", []),
            extra={
                "relationship_id": rel["id"],
                "observed_ratio": round(ratio, 6),
                "expected_band": {"low": low, "high": high},
                "formula": result["formula"],
                "numerator_balance": round(target["balance"], 2),
                "denominator_balance": round(source["balance"], 2),
                "risk_if_inconsistent": rel.get("risk_if_inconsistent"),
            },
        ))

    computed = [r for r in results if r["status"] != "not_computed"]
    payload = {
        "methodology": (
            "Each relationship's observed ratio is computed from mapped trial-balance "
            "components and compared against a deliberately wide plausibility band. A "
            "trial balance carries no credit terms, ageing or industry context, so the "
            "bands flag the implausible rather than the merely unusual. A ratio with an "
            "absent or zero denominator is reported as not_computed with the missing "
            "component named -- never approximated (sec 6.2)."
        ),
        "knowledge_pack_version": pack["_meta"]["version"],
        "generated_at": datetime.datetime.now().isoformat(),
        "data_sufficiency": data_sufficiency,
        "severity_rubric": pack["severity_rubric"],
        "summary": {
            "relationships_with_bands": len(_EXPECTED_BANDS),
            "computed": len(computed),
            "not_computed": len(results) - len(computed),
            "within_expectation": sum(1 for r in computed if r["status"] == "within_expectation"),
            "outside_expectation": len(findings),
        },
        "results": results,
        "finding_records": findings,
        "safe_limitation": SAFE_WORDING_DISCLAIMER,
    }

    out_path = out_dir / "relationship_expectations.json"
    with atomic_write(out_path) as tmp:
        Path(tmp).write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    return {
        "execution_status": "SUCCESS",
        "pipeline_status": "SUCCESS",
        "can_continue": True,
        "message": (
            f"Relationship expectations: {len(computed)} computed, {len(findings)} outside the "
            f"plausible band, {len(results) - len(computed)} not computable from this TB."
        ),
        "artifacts": [str(out_path.resolve())],
        "errors": [],
        "warnings": warnings,
    }


def _mapping_confidence(tb_df) -> str:
    """High/Medium/Low from the share of GL accounts carrying a MAPPED status.

    Replaces a lookup of `downstream_metadata.hierarchy_confidence` on
    relationship_analytics.json -- a key no tool has ever written, so this metric
    read "Unknown" on every run since the port, and downstream_metadata.
    planning_confidence (derived from it) was pinned to "Medium" regardless of how
    well-mapped the TB actually was.

    Thresholds come from the knowledge pack shared with
    build_data_sufficiency_grade, so the two tools cannot report different
    confidence for the same trial balance.
    """
    if tb_df is None or tb_df.is_empty() or "mapped_status" not in tb_df.columns:
        return "Unknown"
    high_pct, medium_pct = mapping_confidence_thresholds()
    mapped_pct = float((tb_df["mapped_status"] == MAPPED_STATUS_MAPPED).sum()) / tb_df.height * 100
    if mapped_pct >= high_pct:
        return "High"
    if mapped_pct >= medium_pct:
        return "Medium"
    return "Low"

@pipeline_tool("build_risk_indicators", domain="risk")
def build_risk_indicators(canonical_tb_file: str, materiality_file: str = None, output_dir: str = None) -> dict:
    """Assign a composite risk score to each GL account, sourced from materiality, variance,
    and relationship analytics, and bin accounts into Critical/High/Medium/Low risk levels."""
    errors = []
    warnings = []
    artifacts = []

    out_dir = resolve_output_dir(output_dir)
    os.makedirs(out_dir, exist_ok=True)

    tb_path = Path(canonical_tb_file)
    mat_path = Path(materiality_file) if materiality_file else out_dir / "materiality.json"
    # Optional -- Single-TB mode doesn't run variance analysis (requires a PY/CY comparison
    # basis); degrade gracefully to no variance data when absent.
    var_path = out_dir / "variance_analysis.json"
    rel_path = out_dir / "relationship_analysis.json"
    if not rel_path.exists():
        rel_path = out_dir / "relationship_analytics.json"

    for p, n in [
        (tb_path, "Canonical TB (supply canonical_tb_file from load_tb_from_db/ingest_tb_to_live)"),
        (mat_path, "Materiality (run build_materiality first)"),
        (rel_path, "Relationship analytics (run build_relationship_analytics first)"),
    ]:
        if not p.exists():
            errors.append({"type": "FileNotFoundError", "file": str(p)})
            return {"execution_status": "FAILED", "errors": errors, "message": f"{n} not found."}

    with open(mat_path, "r") as f:
        mat_data = json.load(f)
    var_data = {}
    if var_path.exists():
        with open(var_path, "r") as f:
            var_data = json.load(f)
    with open(rel_path, "r") as f:
        rel_data = json.load(f)

    tb_df = pl.read_parquet(tb_path)
    tb_df = tb_df.with_columns(
        [
            (
                pl.col(c).cast(pl.Float64, strict=False).fill_null(0.0).alias(c)
                if c in tb_df.columns
                else pl.lit(0.0).alias(c)
            )
            for c in CANONICAL_TB_NUMERIC_COLUMNS
        ]
    )

    rows = list(tb_df.iter_rows(named=True))
    fsli_list = [row_fsli(r) for r in rows]
    if "sub_head_2" in tb_df.columns:
        sub_fsli_list = [(v if v is not None else "") for v in tb_df["sub_head_2"].to_list()]
    else:
        sub_fsli_list = ["" for _ in rows]

    # Initialize Risk tracking
    acct_risks = {}
    for row, fsli, sub_fsli in zip(rows, fsli_list, sub_fsli_list):
        acct_str = f"{row.get('gl_code', '')} - {row.get('gl_name', '')}"
        acct_risks[acct_str] = {
            "account": acct_str,
            "fsli": str(fsli),
            "sub_fsli": str(sub_fsli),
            "balance": round(row["closing_balance"], 2),
            "score": 0,
            "triggered_rules": [],
        }

    rule_weights = risk_rule_weights()

    def add_risk(acct_str, rule_name, source_tool):
        if acct_str in acct_risks:
            w = rule_weights.get(rule_name, 10)
            acct_risks[acct_str]["score"] += w
            acct_risks[acct_str]["triggered_rules"].append({
                "rule": rule_name,
                "source": source_tool,
                "weight": w,
            })

    def _apply_materiality_risks():
        # 1. Apply Materiality Risks
        mat_gls = mat_data.get("material_gl", [])
        if not mat_gls:
            om = mat_data.get("selected_materiality", {}).get("overall_materiality", 0)
            pm = mat_data.get("thresholds", {}).get("performance", 0)
            for row in rows:
                acct_str = f"{row.get('gl_code', '')} - {row.get('gl_name', '')}"
                cb = abs(row["closing_balance"])
                if cb > om and om > 0:
                    add_risk(acct_str, "ABOVE_OM", "Materiality")
                elif cb > pm and pm > 0:
                    add_risk(acct_str, "ABOVE_PM", "Materiality")
        else:
            for gl in mat_gls:
                add_risk(gl["account"], "ABOVE_OM", "Materiality")

    _apply_materiality_risks()

    def _apply_variance_risks():
        # 2. Apply Variance Risks
        for v in var_data.get("material_variances", []):
            if v.get("priority") == "Critical":
                add_risk(v["account"], "CRITICAL_VARIANCE", "Variance")
            elif v.get("priority") == "High":
                add_risk(v["account"], "HIGH_VARIANCE", "Variance")

        for acct in var_data.get("new_accounts", []):
            add_risk(acct["account"], "NEW_ACCOUNT", "Variance")

        for acct in var_data.get("sign_reversals", []):
            add_risk(acct["account"], "SIGN_REVERSAL", "Variance")

    _apply_variance_risks()

    def _apply_relationship_risks():
        # 3. Apply Relationship Risks
        for o in rel_data.get("hierarchy_validation", {}).get("orphan_nodes", []):
            add_risk(o["account"], "ORPHAN_NODE", "Relationship")

        for s in rel_data.get("suspense_analysis", {}).get("accounts", []):
            add_risk(s["account"], "SUSPENSE_CLEARING", "Relationship")

        for ic in rel_data.get("intercompany_relationships", []):
            add_risk(ic["account"], "INTERCOMPANY", "Relationship")

        for sr in rel_data.get("sign_relationships", []):
            add_risk(sr["account"], "SIGN_ANOMALY", "Relationship")

    _apply_relationship_risks()

    PROVISION_MOVEMENT_PCT_THRESHOLD = 100.0

    def _apply_provision_movement_risks():
        # Client-review defect registry: a provision/estimate account that jumps by a large
        # percentage (e.g. a doubtful-debt provision up 7.8x) is invisible to
        # build_variance_analysis's Extreme classification, because that check AND-gates the
        # percentage test behind abs_movement >= performance_materiality (see variance.py) --
        # a small-balance provision growing several-fold never clears that absolute floor. It
        # is equally invisible to build_estimation_exposure, which is deliberately ranked by
        # closing-balance SIZE only and explicitly indifferent to movement (see
        # estimation_keywords.json's amend_policy). This screen is the narrow, keyword-scoped
        # complement to both: reuses the same estimation-risk keyword vocabulary so it can
        # never disagree with build_estimation_exposure about what counts as an estimate/
        # provision-type account, and flags a movement pattern (large %% swing, or exactly
        # zero movement on a non-nil balance -- a stale/unchanged provision) without being
        # gated on absolute materiality, since these accounts are audit-relevant by NATURE
        # regardless of size (same rationale as the sensitive-account tags).
        for row in rows:
            gl_name = row.get("gl_name")
            if not is_estimation_risk_account(gl_name):
                continue
            acct_str = f"{row.get('gl_code', '')} - {row.get('gl_name', '')}"
            ob = float(row.get("opening_balance") or 0.0)
            cb = float(row.get("closing_balance") or 0.0)
            mov = cb - ob
            if cb == 0.0 and ob == 0.0:
                continue
            if ob != 0.0:
                pct = (mov / abs(ob)) * 100
                if abs(pct) > PROVISION_MOVEMENT_PCT_THRESHOLD:
                    add_risk(acct_str, "PROVISION_MOVEMENT_ANOMALY", "EstimationRisk")
                    continue
            if mov == 0.0 and cb != 0.0:
                add_risk(acct_str, "PROVISION_MOVEMENT_ANOMALY", "EstimationRisk")

    _apply_provision_movement_risks()

    # Read by sections 5, 6 and 7 below (planning focus, fraud indicators, output).
    critical_risks = []
    high_risks = []
    medium_risks = []
    low_risks = []

    def _aggregate_and_assign_risk_levels():
        # 4. Score Aggregation & Level Assignment
        for k, v in acct_risks.items():
            s = v["score"]
            if s > 100:
                s = 100
            v["score"] = s
            if s > 75:
                v["risk_level"] = "Critical"
                critical_risks.append(v)
            elif s >= 50:
                v["risk_level"] = "High"
                high_risks.append(v)
            elif s >= 25:
                v["risk_level"] = "Medium"
                medium_risks.append(v)
            else:
                v["risk_level"] = "Low"
                low_risks.append(v)

        critical_risks.sort(key=lambda x: x["score"], reverse=True)
        high_risks.sort(key=lambda x: x["score"], reverse=True)
        medium_risks.sort(key=lambda x: x["score"], reverse=True)

    _aggregate_and_assign_risk_levels()

    # 5. Planning Focus
    planning_focus = {
        "substantive_testing": [x["account"] for x in critical_risks + high_risks][:20],
        "analytical_review": list(set([x["fsli"] for x in medium_risks if x["fsli"] != "Unmapped"]))[:20],
        "management_inquiry": [x["account"] for x in critical_risks if any(r["rule"] in ["SIGN_REVERSAL", "SIGN_ANOMALY"] for r in x["triggered_rules"])],
    }

    # 6. Fraud Indicators (Rule based combinations)
    fraud_indicators = []
    for x in critical_risks:
        rules = [r["rule"] for r in x["triggered_rules"]]
        if "SIGN_REVERSAL" in rules and "SUSPENSE_CLEARING" in rules:
            fraud_indicators.append({"account": x["account"], "indicator": "Suspense Sign Reversal"})
        if "ORPHAN_NODE" in rules and "NEW_ACCOUNT" in rules:
            fraud_indicators.append({"account": x["account"], "indicator": "Unmapped New Account"})

    # 7. Coverage & Metadata
    total_accounts = canonical_row_count(tb_df)
    # `total_relationships_integrated` is the key build_relationship_analytics actually
    # writes; the previous `total_relationships` lookup matched nothing and reported 0 on
    # every run. `total_relationships` is still accepted as a fallback so an older
    # relationship_analytics.json on disk doesn't silently read as zero.
    rel_summary = rel_data.get("summary", {})
    relationships_consumed = rel_summary.get(
        "total_relationships_integrated", rel_summary.get("total_relationships", 0)
    )
    # Mapping confidence was read from `downstream_metadata.hierarchy_confidence`, which no
    # tool writes -- so it was always "Unknown", which in turn pinned planning_confidence to
    # "Medium" on every run regardless of how well-mapped the TB actually was. Derive it
    # from the canonical TB itself, using the same thresholds build_data_sufficiency_grade
    # applies (both now read them from the knowledge pack, so they cannot drift apart).
    coverage = {
        "accounts_evaluated": total_accounts,
        "fsli_evaluated": len(set(fsli_list)) if total_accounts else 0,
        "relationships_consumed": relationships_consumed,
        "mapping_confidence": _mapping_confidence(tb_df),
        "overall_coverage": 100.0 if total_accounts > 0 else 0.0,
    }

    downstream_metadata = {
        "tool_version": "2.0",
        "scoring_version": "1.0",
        "overall_risk_score": round(sum([x["score"] for x in acct_risks.values()]) / total_accounts, 2) if total_accounts > 0 else 0,
        "critical_risk_count": len(critical_risks),
        "high_risk_count": len(high_risks),
        "planning_confidence": "High" if coverage["mapping_confidence"] == "High" else "Medium",
        "generated_timestamp": datetime.datetime.now().isoformat(),
    }

    # Export Parquet
    pq_df = pl.DataFrame(list(acct_risks.values()))
    parquet_path = out_dir / "risk_indicators.parquet"
    if not pq_df.is_empty():
        write_parquet_atomic(pq_df.drop("triggered_rules"), parquet_path)
    else:
        write_parquet_atomic(pq_df, parquet_path)
    artifacts.append(str(parquet_path.resolve()))

    # Export JSON
    output_data = {
        "methodology": "Deterministic composite scoring based on atomic rule provenance.",
        "summary": {
            "total_accounts": total_accounts,
            "total_risks": len(critical_risks) + len(high_risks) + len(medium_risks),
            "critical": len(critical_risks),
            "high": len(high_risks),
            "medium": len(medium_risks),
            "low": len(low_risks),
        },
        "risk_categories": rule_weights,
        "critical_risks": critical_risks[:20],
        "fraud_indicators": fraud_indicators,
        "planning_focus": planning_focus,
        "coverage": coverage,
        "downstream_metadata": downstream_metadata,
        "limitations": [],
    }

    json_path = out_dir / "risk_indicators.json"
    write_json_atomic(output_data, json_path, indent=4)
    artifacts.append(str(json_path.resolve()))

    return {
        "execution_status": "SUCCESS",
        "pipeline_status": "SUCCESS",
        "message": f"Risk Intelligence Engine computed scores for {total_accounts} accounts. {len(critical_risks)} Critical risks identified.",
        "artifacts": artifacts,
        "errors": errors,
        "warnings": warnings,
    }


def _load_active_sensitive_rules(out_dir, tb_path, warnings):
    """Loads the base sensitive-rules pack and, for the related_party rule only, extends its
    keyword list with any engagement-supplied register of named counterparties. Mutates
    `warnings` in place on a parse failure. Returns (active_rules, base_rules) -- base_rules
    is also needed later by the caller to build cat_blocks."""
    register_path = out_dir / "related_party_register.json"
    if not register_path.exists():
        register_path = tb_path.parent / "related_party_register.json"
    related_party_register_names = []
    if register_path.exists():
        try:
            with open(register_path, "r", encoding="utf-8") as f:
                related_party_register_names = [
                    str(n).strip().lower() for n in (json.load(f).get("names", []) or []) if str(n).strip()
                ]
        except Exception:
            warnings.append(f"Failed to parse related_party_register.json at {register_path}.")
    base_rules = sensitive_rules()
    active_rules = [
        (tag, weight, kws + tuple(related_party_register_names) if tag == "related_party" else kws)
        for tag, weight, kws in base_rules
    ]
    return active_rules, base_rules


def _classify_sensitive_accounts(tb_df, active_sensitive_rules, perf_mat, ov_mat, risk_lookup):
    """Keyword-matches every TB row against the sensitive-rules pack, and for each match
    combines the category's base weight with a materiality-exceedance bonus and any
    inherited risk score into a 0-100 sensitivity score and priority band."""
    sensitive_accounts = []

    for row in tb_df.iter_rows(named=True):
        acct_str = f"{row.get('gl_code', '')} - {row.get('gl_name', '')}"
        text = f" {str(row.get('gl_name', '')).lower()} {str(row.get('gl_code', '')).lower()} "
        cb = row["closing_balance"]

        matched_categories = []
        for tag, weight, kws in active_sensitive_rules:
            if any(k in text for k in kws):
                matched_categories.append((tag, weight))

        if matched_categories:
            matched_categories.sort(key=lambda x: x[1], reverse=True)
            primary_cat, base_weight = matched_categories[0]

            record = {
                "gl_code": str(row.get("gl_code", "")),
                "gl_name": str(row.get("gl_name", "")),
                "account": acct_str,
                "fsli": row_fsli(row),
                "sub_fsli": str(row.get("sub_head_2", "")),
                "category": primary_cat,
                "balance": round(cb, 2),
                "sensitivity_score": 0,
                "priority": "Low",
                "risk_score": 0,
                "risk_level": "Unknown",
                "materiality_ratio": round(abs(cb) / perf_mat, 2) if perf_mat > 0 else 0,
                "triggered_rules": [],
            }

            record["sensitivity_score"] += base_weight
            record["triggered_rules"].append({"rule": primary_cat.upper(), "source": "Classification", "weight": base_weight})

            if abs(cb) > ov_mat and ov_mat > 0:
                record["sensitivity_score"] += 20
                record["triggered_rules"].append({"rule": "ABOVE_OM", "source": "Materiality", "weight": 20})
            elif abs(cb) > perf_mat and perf_mat > 0:
                record["sensitivity_score"] += 15
                record["triggered_rules"].append({"rule": "ABOVE_PM", "source": "Materiality", "weight": 15})

            if acct_str in risk_lookup:
                rs, rl = risk_lookup[acct_str]
                record["risk_score"] = rs
                record["risk_level"] = rl
                risk_additive = int((rs / 100.0) * 35)
                record["sensitivity_score"] += risk_additive
                record["triggered_rules"].append({"rule": "INHERITED_RISK", "source": "Risk Engine", "weight": risk_additive})

            s = record["sensitivity_score"]
            if s > 100:
                s = 100
            record["sensitivity_score"] = s
            if s > 75:
                record["priority"] = "Critical"
            elif s >= 50:
                record["priority"] = "High"
            elif s >= 25:
                record["priority"] = "Medium"
            else:
                record["priority"] = "Low"

            sensitive_accounts.append(record)

    return sensitive_accounts


def _summarize_sensitive_accounts_by_fsli(sensitive_accounts):
    """Groups sensitive accounts by FSLI, tracking each group's account count and its
    highest-scoring member's priority band."""
    fsli_dict = {}
    for x in sensitive_accounts:
        fsli = x["fsli"]
        if fsli not in fsli_dict:
            fsli_dict[fsli] = {"fsli": fsli, "sensitive_accounts": 0, "highest_priority": "Low", "_max_score": 0}
        fsli_dict[fsli]["sensitive_accounts"] += 1
        if x["sensitivity_score"] > fsli_dict[fsli]["_max_score"]:
            fsli_dict[fsli]["_max_score"] = x["sensitivity_score"]
            fsli_dict[fsli]["highest_priority"] = x["priority"]

    fsli_summary = []
    for k, v in fsli_dict.items():
        v.pop("_max_score")
        fsli_summary.append(v)
    fsli_summary.sort(key=lambda x: x["sensitive_accounts"], reverse=True)
    return fsli_summary


@pipeline_tool("build_sensitive_detector", domain="risk")
def build_sensitive_detector(canonical_tb_file: str, output_dir: str = None) -> dict:
    """Classify sensitive accounts (grant/subsidy, related-party, statutory dues, etc.) by
    keyword match, then combine base category weight with inherited risk score and
    materiality ratio into a sensitivity score."""
    errors = []
    warnings = []
    artifacts = []

    out_dir = resolve_output_dir(output_dir)
    os.makedirs(out_dir, exist_ok=True)

    tb_path = Path(canonical_tb_file)
    mat_path = out_dir / "materiality.json"
    risk_pq_path = out_dir / "risk_indicators.parquet"

    for p, n in [
        (tb_path, "Canonical TB (supply canonical_tb_file from load_tb_from_db/ingest_tb_to_live)"),
        (mat_path, "Materiality JSON (run build_materiality first)"),
        (risk_pq_path, "Risk Indicators Parquet (run build_risk_indicators first)"),
    ]:
        if not p.exists():
            errors.append({"type": "FileNotFoundError", "file": str(p)})
            return {"execution_status": "FAILED", "errors": errors, "message": f"{n} not found."}

    with open(mat_path, "r") as f:
        mat_data = json.load(f)
    tb_df = mapped_only(pl.read_parquet(tb_path))
    risk_df = pl.read_parquet(risk_pq_path)

    # TB-R10: optional engagement-supplied register of named counterparties (parent/JV/
    # subsidiary names, intercompany short-codes) -- no such name is derivable from this
    # pipeline's own inputs (COMPANY_DETAILS carries only the subject entity's own name/CIN/
    # FY, not its parent's), so this is a file drop-in, not an auto-derived list. Absent by
    # default; when present, extends the related_party keyword match with real counterparty
    # names instead of only the generic English/structural patterns above.
    _active_sensitive_rules, _base_rules = _load_active_sensitive_rules(out_dir, tb_path, warnings)

    if "closing_balance" not in tb_df.columns:
        tb_df = tb_df.with_columns(pl.lit(0.0).alias("closing_balance"))
    else:
        tb_df = tb_df.with_columns(pl.col("closing_balance").cast(pl.Float64, strict=False).fill_null(0.0))

    perf_mat = float(mat_data.get("thresholds", {}).get("performance", 0.0))
    ov_mat = float(mat_data.get("selected_materiality", {}).get("overall_materiality", 0.0))
    if perf_mat == 0:
        perf_mat = ov_mat * 0.75

    # Create Risk Lookup dict mapping "gl_code - gl_name" to (risk_score, risk_level)
    risk_lookup = {}
    for row in risk_df.iter_rows(named=True):
        risk_lookup[row["account"]] = (row["score"], row["risk_level"])

    sensitive_accounts = _classify_sensitive_accounts(tb_df, _active_sensitive_rules, perf_mat, ov_mat, risk_lookup)

    total_scanned = canonical_row_count(tb_df)
    critical_count = len([x for x in sensitive_accounts if x["priority"] == "Critical"])
    high_count = len([x for x in sensitive_accounts if x["priority"] == "High"])

    category_summary = {}
    for x in sensitive_accounts:
        cat = x["category"]
        if cat not in category_summary:
            category_summary[cat] = {"count": 0, "total_balance": 0.0}
        category_summary[cat]["count"] += 1
        category_summary[cat]["total_balance"] += x["balance"]
    for k in category_summary:
        category_summary[k]["total_balance"] = round(category_summary[k]["total_balance"], 2)

    fsli_summary = _summarize_sensitive_accounts_by_fsli(sensitive_accounts)

    exposure_summary = {
        "total_sensitive_balance": round(sum(abs(x["balance"]) for x in sensitive_accounts), 2),
        "material_sensitive_balance": round(sum(abs(x["balance"]) for x in sensitive_accounts if x["materiality_ratio"] > 1.0), 2),
        "high_risk_sensitive_balance": round(sum(abs(x["balance"]) for x in sensitive_accounts if x["risk_level"] in ["Critical", "High"]), 2),
        "critical_accounts": critical_count,
    }

    planning_focus = {
        "related_party_testing": [x["account"] for x in sensitive_accounts if x["category"] == "related_party" and x["priority"] in ["Critical", "High"]][:10],
        "grant_testing": [x["account"] for x in sensitive_accounts if x["category"] == "grant_subsidy" and x["priority"] in ["Critical", "High"]][:10],
        "msme_testing": [x["account"] for x in sensitive_accounts if x["category"] == "msme" and x["priority"] in ["Critical", "High"]][:10],
        "csr_testing": [x["account"] for x in sensitive_accounts if x["category"] == "csr" and x["priority"] in ["Critical", "High"]][:10],
        "statutory_testing": [x["account"] for x in sensitive_accounts if x["category"] == "statutory_dues" and x["priority"] in ["Critical", "High"]][:10],
        "advance_testing": [x["account"] for x in sensitive_accounts if x["category"] == "advance" and x["priority"] in ["Critical", "High"]][:10],
    }

    coverage = {
        "accounts_scanned": total_scanned,
        "accounts_detected": len(sensitive_accounts),
        "fsli_coverage": len(fsli_summary),
        "balance_coverage": round((exposure_summary["total_sensitive_balance"] / tb_df["closing_balance"].abs().sum() * 100), 2) if not tb_df.is_empty() and tb_df["closing_balance"].abs().sum() else 0,
        "overall_coverage": 100.0 if total_scanned > 0 else 0.0,
    }

    metadata = {
        "tool_version": "2.0",
        "classification_version": "1.0",
        "generated_at": datetime.datetime.now().isoformat(),
    }

    downstream_metadata = {
        "critical_sensitive_accounts": critical_count,
        "high_sensitive_accounts": high_count,
        "overall_sensitivity_score": round(sum(x["sensitivity_score"] for x in sensitive_accounts) / len(sensitive_accounts), 2) if sensitive_accounts else 0,
        "planning_confidence": "High",
    }

    sensitive_accounts.sort(key=lambda x: x["sensitivity_score"], reverse=True)

    cat_blocks = {tag: [] for tag, _, _ in _base_rules}
    for x in sensitive_accounts:
        cat = x["category"]
        if cat in cat_blocks:
            cat_blocks[cat].append(x)

    pq_df = pl.DataFrame(sensitive_accounts)
    parquet_path = out_dir / "sensitive_accounts.parquet"
    if not pq_df.is_empty():
        write_parquet_atomic(pq_df.drop("triggered_rules"), parquet_path)
    else:
        write_parquet_atomic(pq_df, parquet_path)
    artifacts.append(str(parquet_path.resolve()))

    output_data = {
        "methodology": "Deterministic rule-based sensitivity engine integrating Risk and Materiality scores.",
        "summary": {
            "accounts_scanned": total_scanned,
            "sensitive_accounts": len(sensitive_accounts),
            "critical": critical_count,
            "high": high_count,
            "medium": len([x for x in sensitive_accounts if x["priority"] == "Medium"]),
            "low": len([x for x in sensitive_accounts if x["priority"] == "Low"]),
        },
        "category_summary": category_summary,
        "account_sensitivity": sensitive_accounts[:20],
        "fsli_summary": fsli_summary[:10],
        "exposure_summary": exposure_summary,
        "planning_focus": planning_focus,
        "coverage": coverage,
        "downstream_metadata": downstream_metadata,
        "metadata": metadata,
        "limitations": [],
        # Wave 2 Fix 2b: this "statutory_dues" category tag is one of five independent
        # statutory-dues populations across this codebase -- see _STATUTORY_DUES_POPULATION_BASIS.
        "statutory_dues_population_basis": (
            "sensitive-pack keyword screen (backend/knowledge/sensitive); see also "
            "build_statutory_screen, build_caro_indicators, build_going_concern_screen for "
            "independently-computed statutory-dues populations"
        ),
    }
    for tag in cat_blocks:
        output_data[tag] = cat_blocks[tag][:5]

    json_path = out_dir / "sensitive_accounts.json"
    write_json_atomic(output_data, json_path, indent=4)
    artifacts.append(str(json_path.resolve()))

    return {
        "execution_status": "SUCCESS",
        "pipeline_status": "SUCCESS",
        "message": f"Sensitivity Engine classified {len(sensitive_accounts)} accounts. {critical_count} Critical priorities found.",
        "artifacts": artifacts,
        "errors": errors,
        "warnings": warnings,
    }


# Wave 2 Fix 2b: statutory dues are independently aggregated in FIVE places across this
# codebase (this screen's own STAT-TDS/STAT-GST-OUT/STAT-ITC/STAT-PF-ESI specs,
# build_sensitive_detector's "sensitive" knowledge-pack keywords, build_caro_indicators'
# "compliance" knowledge-pack keywords, and build_going_concern_screen's inline keyword
# tuple) -- each serves a genuinely different audit question, so Wave 2 does not unify
# them, but every result must name the other four so a reviewer never mistakes one
# population for the only "statutory dues" figure.
_STATUTORY_DUES_POPULATION_BASIS = (
    "keyword-band screen (STAT-TDS/STAT-GST-OUT/STAT-ITC/STAT-PF-ESI); see also "
    "build_caro_indicators, build_sensitive_detector, build_going_concern_screen for "
    "independently-computed statutory-dues populations"
)

_SCREENS = [
    {
        "id": "STAT-GST-OUT",
        "label": "GST output vs taxable revenue",
        "base": ("revenue from operations", "sales", "revenue"),
        "base_exclude": ("gst", "tax"),
        "dues": ("gst output", "output gst", "output tax", "igst payable", "cgst payable", "sgst payable", "gst payable"),
        "band": (0.02, 0.28),
        "expectation": "Book revenue should reconcile to GST returns after exempt, zero-rated and non-GST adjustments.",
        "caveats": [
            "Exempt or nil-rated supplies carry no output tax",
            "Zero-rated exports and SEZ supplies",
            "Revenue outside GST scope, including most grants and subsidies",
            "GST discharged on advances in an earlier period",
            "Credit notes reducing output liability",
            "Output tax already paid and therefore not carried as a period-end liability",
        ],
        "evidence": [
            "GSTR-1 and GSTR-3B for every period in the year",
            "Revenue-to-GST reconciliation",
            "Taxability matrix showing exempt, zero-rated and non-GST revenue",
            "Credit-note register",
        ],
        "assertions": ["Completeness", "Accuracy", "Classification"],
    },
    {
        "id": "STAT-ITC",
        "label": "Input tax credit vs purchases",
        "base": ("purchase", "cost of materials", "cost of goods"),
        "base_exclude": ("gst", "tax"),
        "dues": ("gst input", "input gst", "input tax credit", "itc", "gst receivable", "gst recoverable"),
        "band": (0.02, 0.28),
        "expectation": "Input tax credit should broadly relate to eligible taxable purchases.",
        "caveats": [
            "Purchases from composition dealers or unregistered suppliers carry no credit",
            "Credit blocked under Section 17(5)",
            "Proportionate reversal where exempt outward supplies exist",
            "Credit fully utilised against output liability, leaving no period-end balance",
        ],
        "evidence": [
            "GSTR-2B to books input reconciliation",
            "Ineligible-credit working under Section 17(5)",
            "Rule 42/43 reversal working",
            "Purchase register with taxability flags",
        ],
        "assertions": ["Existence", "Accuracy", "Cut-off"],
    },
    {
        "id": "STAT-TDS",
        "label": "TDS payable vs deductible expense base",
        "base": ("contractor", "professional", "commission", "rent", "salary", "wages", "interest paid"),
        "base_exclude": ("tds", "tax"),
        "dues": ("tds payable", "tax deducted at source", "tds on"),
        "band": (0.005, 0.15),
        "expectation": "Expenses subject to withholding should produce a TDS liability at the period end.",
        "caveats": [
            "TDS deposited before the period end leaves no closing liability",
            "Payments below the per-section threshold",
            "Payees holding lower or nil-deduction certificates",
            "Expense categories outside the withholding provisions",
        ],
        "evidence": [
            "TDS returns (Form 26Q / 24Q) for all four quarters",
            "TDS challans with deposit dates",
            "Section-wise deduction working against the expense ledger",
            "Lower/nil-deduction certificates held",
        ],
        "assertions": ["Completeness", "Accuracy"],
    },
    {
        "id": "STAT-PF-ESI",
        "label": "PF and ESI vs payroll cost",
        "base": ("salary", "salaries", "wages", "employee benefit", "staff cost"),
        "base_exclude": ("pf", "esi", "provident", "gratuity"),
        "dues": ("provident fund", "pf payable", "epf", "esi", "employee state insurance"),
        "band": (0.005, 0.30),
        "expectation": "Payroll should carry corresponding PF and ESI liabilities where the workforce is covered.",
        "caveats": [
            "Employees above the PF/ESI wage ceiling",
            "Workforce engaged through outsourced contractors, whose dues sit with the contractor",
            "Staff on government deputation with dues borne by the parent department",
            "Dues remitted before the period end",
        ],
        "evidence": [
            "Payroll register with headcount and coverage status",
            "PF and ESI challans and monthly returns (ECR)",
            "Contractor agreements and their compliance certificates",
            "Employee classification showing covered versus exempt",
        ],
        "assertions": ["Completeness", "Accuracy", "Valuation"],
    },
]

def _screen_statutory_levies(tb_df, perf, out_dir=None):
    """Compares each _SCREENS spec's dues-to-base ratio against its plausible band, and
    builds a finding record for every levy that's absent or outside expectation. Returns
    (results, findings) -- results always has one entry per spec (including the
    not-applicable/immaterial short-circuits), findings only for the ones worth raising."""
    results, findings = [], []

    for spec in _SCREENS:
        # Wave 2 Fix 2a: STAT-GST-OUT's "taxable revenue" base used to be a third,
        # independent revenue definition (client-flagged: revenue stated 3 different
        # ways) -- defer to the single shared compute_total_revenue helper instead.
        if spec["id"] == "STAT-GST-OUT" and out_dir is not None:
            _rev = compute_total_revenue(out_dir, tb_df)
            base = {"balance": _rev["balance"], "account_count": 0, "accounts": []}
        else:
            base = gl_total_by_keywords(tb_df, spec["base"], exclude=spec.get("base_exclude", ()))
        dues = gl_total_by_keywords(tb_df, spec["dues"])
        low, high = spec["band"]

        if base["balance"] <= 0:
            results.append({"id": spec["id"], "label": spec["label"], "status": "not_applicable",
                            "population_basis": _STATUTORY_DUES_POPULATION_BASIS,
                            "reason": "Base head absent -- this levy's relationship does not arise from this TB."})
            continue
        if perf and base["balance"] < perf:
            results.append({"id": spec["id"], "label": spec["label"], "status": "immaterial_base",
                            "population_basis": _STATUTORY_DUES_POPULATION_BASIS,
                            "base_balance": round(base["balance"], 2)})
            continue

        ratio = dues["balance"] / base["balance"] if base["balance"] else 0.0
        absent = dues["balance"] <= 0
        within = (not absent) and low <= ratio <= high
        status = "absent" if absent else ("within_expectation" if within else "outside_expectation")

        results.append({
            "id": spec["id"], "label": spec["label"], "status": status,
            "population_basis": _STATUTORY_DUES_POPULATION_BASIS,
            "base_balance": round(base["balance"], 2),
            "base_accounts": base["accounts"][:5],
            "dues_balance": round(dues["balance"], 2),
            "dues_accounts": dues["accounts"][:5],
            "observed_ratio": round(ratio, 6),
            "expected_band": {"low": low, "high": high},
        })

        if within:
            continue

        if absent:
            observation = (
                f"{spec['label']}: base of {base['balance']:,.2f} is present but no corresponding "
                f"statutory liability appears in the trial balance."
            )
            gap = "No statutory dues head located for this levy."
        else:
            direction = "below" if ratio < low else "above"
            bound = low if ratio < low else high
            observation = (
                f"{spec['label']}: dues of {dues['balance']:,.2f} against a base of "
                f"{base['balance']:,.2f} -- a ratio of {ratio:.2%}."
            )
            gap = f"Observed {ratio:.2%} is {direction} the plausible band bound of {bound:.0%}."

        findings.append(make_record(
            source_screen="build_statutory_screen",
            fsli=spec["label"],
            amount=round(base["balance"], 2),
            observation=observation,
            expectation=spec["expectation"],
            gap=gap,
            assertions=spec["assertions"],
            # Statutory dues are material by CONTEXT under sec 2.2 -- compliance with law
            # is decisive regardless of the amount carried at the period end.
            risk_basis=["relationship", "context"] + (["value"] if perf and base["balance"] >= perf else []),
            risk_rating="high",
            regularity_flag=True,
            data_sufficiency="low",
            proposed_response=(
                "Obtain the returns and challans listed and reconcile them to the books. The "
                "trial balance cannot establish taxability, coverage or filing status, so no "
                "compliance view can be formed without them."
            ),
            evidence_requested=spec["evidence"],
            valid_reasons=spec["caveats"],
            extra={
                "screen_id": spec["id"],
                "base_balance": round(base["balance"], 2),
                "dues_balance": round(dues["balance"], 2),
                "observed_ratio": round(ratio, 6),
                "expected_band": {"low": low, "high": high},
            },
        ))

    return results, findings


@pipeline_tool("build_statutory_screen", domain="risk")
def build_statutory_screen(
    canonical_tb_file: str,
    materiality_file: str = None,
    output_dir: str = None,
    **kwargs,
) -> dict:
    """Reconcile statutory dues against the bases they arise from -- GST output vs
    taxable revenue, input credit vs purchases, TDS vs deductible expenses, PF/ESI vs
    payroll -- and flag debit balances on statutory heads as classification questions.
    Every result carries its exemption caveats and the exact returns and challans that
    would resolve it; nothing here concludes compliance. Writes statutory_screen.json."""
    out_dir = resolve_output_dir(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tb_path = Path(canonical_tb_file)
    if not tb_path.exists():
        raise PipelineFileError(str(tb_path), "Canonical TB (supply canonical_tb_file)")

    tb_df = load_canonical_tb(tb_path)
    if tb_df is None or tb_df.is_empty():
        return {
            "execution_status": "SUCCESS", "pipeline_status": "WARNING", "can_continue": True,
            "message": "Canonical TB is empty -- no statutory screening performed.",
            "artifacts": [], "errors": [],
        }

    warnings = []
    mat_path = Path(materiality_file) if materiality_file else out_dir / "materiality.json"
    mat = safe_load_json(mat_path) if mat_path.exists() else {}
    perf = float((mat.get("thresholds") or {}).get("performance") or 0.0)
    trivial = float((mat.get("thresholds") or {}).get("clearly_trivial") or 0.0)
    if not perf:
        warnings.append("materiality.json not available -- every statutory gap is reported regardless of size.")

    results, findings = _screen_statutory_levies(tb_df, perf, out_dir=out_dir)

    # ── debit balances on statutory heads (sec 5.3, sec 9.16) ─────────────────
    statutory_kws = load_pack("sensitive")
    stat_keywords = next(
        (tuple(c["keywords"]) for c in statutory_kws["categories"] if c["tag"] == "statutory_dues"), ()
    )
    debit_dues = []
    for row in tb_df.iter_rows(named=True):
        closing = float(row.get("closing_balance") or 0.0)
        if closing <= 0:
            continue
        haystack = f" {str(row.get('gl_name') or '').lower()} {str(row.get('gl_code') or '').lower()} "
        if not any(k in haystack for k in stat_keywords):
            continue
        # A recoverable named as such is the expected debit case, not an anomaly.
        if any(t in haystack for t in ("receivable", "recoverable", "input", "refund", "advance tax")):
            continue
        if trivial and closing < trivial:
            continue
        debit_dues.append({
            "account": f"{row.get('gl_code')} - {row.get('gl_name')}",
            "gl_code": str(row.get("gl_code") or ""),
            "closing_balance": round(closing, 2),
        })

    for d in debit_dues:
        findings.append(make_record(
            source_screen="build_statutory_screen",
            account=d["account"],
            fsli="Statutory dues",
            amount=d["closing_balance"],
            source_row_id=d["gl_code"],
            normal_balance_expectation="Credit",
            observation=f"Statutory head '{d['account']}' carries a debit balance of {d['closing_balance']:,.2f}.",
            expectation="Statutory dues heads ordinarily carry a credit balance until remitted.",
            gap="Debit balance on a liability head -- typically a recoverable parked in a payables head, or an excess remittance.",
            assertions=["Classification", "Existence"],
            risk_basis=["nature", "context"],
            risk_rating="medium",
            regularity_flag=True,
            data_sufficiency="low",
            proposed_response="Establish whether the balance is a genuine recoverable and whether it should be reclassified.",
            evidence_requested=[
                "Return-wise reconciliation for this levy",
                "Challans evidencing payment and their deposit dates",
                "Refund claim or adjustment order, if the balance is a recoverable",
            ],
            valid_reasons=[
                "Excess remittance pending adjustment against a later period",
                "Refund claimed and awaiting receipt",
                "Advance tax or input credit misclassified into a payables head",
            ],
            extra={"screen_id": "STAT-DEBIT-BALANCE"},
        ))

    payload = {
        "methodology": (
            "Each statutory levy is compared against the base it arises from, using absolute "
            "ledger totals matched by keyword. Bands are wide because a trial balance cannot "
            "see taxability, coverage or remittance timing -- they flag the implausible, not "
            "the merely unusual. Every result carries its exemption caveats, and no result is "
            "a compliance conclusion: sec 10.4 treats a statutory keyword as a trigger to "
            "request the working, nothing more."
        ),
        "generated_at": datetime.datetime.now().isoformat(),
        "sensitive_pack_version": statutory_kws["_meta"]["version"],
        "data_sufficiency": "low",
        "summary": {
            "levies_screened": len(_SCREENS),
            "gaps_found": len([f for f in findings if f.get("screen_id") != "STAT-DEBIT-BALANCE"]),
            "debit_balance_heads": len(debit_dues),
            "total_findings": len(findings),
        },
        "results": results,
        "debit_balance_statutory_heads": debit_dues,
        "finding_records": findings,
        "safe_limitation": SAFE_WORDING_DISCLAIMER,
    }

    out_path = out_dir / "statutory_screen.json"
    with atomic_write(out_path) as tmp:
        Path(tmp).write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    return {
        "execution_status": "SUCCESS",
        "pipeline_status": "SUCCESS",
        "can_continue": True,
        "message": (
            f"Statutory screen: {len(_SCREENS)} levy relationship(s) evaluated, "
            f"{len(findings)} finding(s) including {len(debit_dues)} debit-balance statutory head(s). "
            "No compliance conclusion is drawn from the trial balance."
        ),
        "artifacts": [str(out_path.resolve())],
        "errors": [],
        "warnings": warnings,
    }


# ---- moved from fsli.py: tagged domain="risk", physically misplaced pre-split ----
_MUS_RELIABILITY_FACTORS = {
    # Standard zero-expected-misstatement reliability factors (Poisson-based), the same
    # table AICPA/ISA audit sampling guidance uses. Confidence not in this table falls back
    # to 95% and says so -- never interpolated, since an invented factor would misstate the
    # sample's actual statistical basis.
    0.90: 2.3,
    0.95: 3.0,
    0.99: 4.6,
}

def _mus_sample(rows, population_value, performance_materiality, confidence_level, seed):
    """Monetary Unit (PPS) Sampling: every rupee of the population has an equal chance of
    falling in the sample, so larger balances are proportionally more likely to be hit --
    any single item whose own balance exceeds the sampling interval is selected with
    certainty (a "high-value item"), never left to chance. `rows` must be pre-sorted into a
    stable order (by gl_code) so the cumulative-value walk is reproducible run to run."""
    reliability_factor = _MUS_RELIABILITY_FACTORS.get(confidence_level)
    factor_note = None
    if reliability_factor is None:
        reliability_factor = _MUS_RELIABILITY_FACTORS[0.95]
        factor_note = (
            f"Confidence level {confidence_level:.0%} is not in the standard reliability-factor "
            f"table ({', '.join(f'{c:.0%}' for c in _MUS_RELIABILITY_FACTORS)}) -- defaulted to "
            "95% (factor 3.0) rather than interpolating an unvalidated figure."
        )
    sampling_interval = performance_materiality / reliability_factor
    rng = random.Random(seed)  # nosec B311 -- statistical (MUS) sample selection, not security-sensitive
    random_start = rng.uniform(0, sampling_interval)

    cum = 0.0
    selected = []
    next_point = random_start
    for r in rows:
        bal = abs(r["closing_balance"])
        lo, hi = cum, cum + bal
        hits = 0
        while next_point < hi:
            hits += 1
            next_point += sampling_interval
        if hits:
            selected.append({
                "gl_code": r["gl_code"], "gl_name": r["gl_name"],
                "closing_balance": round(r["closing_balance"], 2),
                "cumulative_value_at_selection": round(lo, 2),
                "hits": hits,
                "selected_with_certainty": bal >= sampling_interval,
            })
        cum = hi

    return {
        "confidence_level": confidence_level,
        "reliability_factor": reliability_factor,
        "reliability_factor_note": factor_note,
        "performance_materiality": round(performance_materiality, 2),
        "sampling_interval": round(sampling_interval, 2),
        "random_start": round(random_start, 2),
        "seed": seed,
        "population_value": round(population_value, 2),
        "population_count": len(rows),
        "sample_size": len(selected),
        "certainty_items": sum(1 for s in selected if s["selected_with_certainty"]),
        "sampled_items": selected,
    }

def _stratified_sample(rows, overall_materiality, sample_pct, seed):
    """Splits the population into the SAME Critical/High/Medium/Low/Below-Threshold bands
    the Materiality sheet's Priority scale already uses (see the Legend sheet) -- one
    fewer vocabulary for a reviewer to learn -- then draws a seeded, reproducible random
    sample from each band independent of individual item size, so smaller-balance strata
    are not systematically under-represented the way MUS alone would leave them.

    PRIORITY_*_RATIO constants are defined later in this module (they belong next to
    build_materiality's own get_priority) -- looked up here, inside the function body, so
    they only need to exist by the time this runs, not by the time this function is defined."""
    strata_bands = [
        ("Critical", PRIORITY_CRITICAL_RATIO, None),
        ("High", PRIORITY_HIGH_RATIO, PRIORITY_CRITICAL_RATIO),
        ("Medium", PRIORITY_MEDIUM_RATIO, PRIORITY_HIGH_RATIO),
        ("Low", PRIORITY_LOW_RATIO, PRIORITY_MEDIUM_RATIO),
        ("Below Threshold", 0.0, PRIORITY_LOW_RATIO),
    ]
    rng = random.Random(seed)  # nosec B311 -- statistical (stratified) sample selection, not security-sensitive
    strata = {name: [] for name, _, _ in strata_bands}
    for r in rows:
        ratio = abs(r["closing_balance"]) / overall_materiality if overall_materiality else 0.0
        for name, floor, ceiling in strata_bands:
            if ratio >= floor and (ceiling is None or ratio < ceiling):
                strata[name].append(r)
                break

    results = {}
    for name, members in strata.items():
        n = max(1, math.ceil(len(members) * sample_pct)) if members else 0
        n = min(n, len(members))
        chosen = rng.sample(members, n) if n else []
        results[name] = {
            "population_count": len(members),
            "sample_size": n,
            "sampled_items": [
                {"gl_code": c["gl_code"], "gl_name": c["gl_name"], "closing_balance": round(c["closing_balance"], 2)}
                for c in sorted(chosen, key=lambda x: abs(x["closing_balance"]), reverse=True)
            ],
        }
    return {
        "sample_pct_per_stratum": sample_pct,
        "seed": seed,
        "strata": results,
    }

@pipeline_tool("build_sample_selection", domain="risk")
def build_sample_selection(
    canonical_tb_file: str,
    materiality_file: str = None,
    output_dir: str = None,
    method: str = "both",
    confidence_level: float = 0.95,
    stratified_sample_pct: float = 0.10,
    random_seed: int = 42,
    population_head: str = None,
    **kwargs,
) -> dict:
    """SA 530 audit sampling over the mapped GL population -- Monetary Unit (PPS) Sampling,
    Stratified Random Sampling, or both (`method`: "mus" | "stratified" | "both").

    Every other screen in this pipeline tests the FULL population; this is the one tool that
    deliberately does not, so every parameter that shaped the sample (method, confidence
    level / reliability factor, sampling interval, stratum bands, random seed) is recorded
    on the output -- a sample that cannot be reproduced from its own documentation is not
    audit evidence. `random_seed` makes re-running this tool with the same inputs select the
    exact same items, which is itself part of that documentation.

    population_head optionally scopes the population to one FS Head (e.g. "Assets") --
    unset samples the entire mapped, non-zero-balance population. Writes
    sample_selection.json."""
    out_dir = resolve_output_dir(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tb_path = Path(canonical_tb_file)
    if not tb_path.exists():
        raise PipelineFileError(str(tb_path), "Canonical TB (supply canonical_tb_file)")

    df = load_canonical_tb(tb_path)
    if df is None or df.is_empty():
        return {
            "execution_status": "SUCCESS", "pipeline_status": "WARNING", "can_continue": True,
            "message": "Canonical TB is empty -- no sample selected.",
            "artifacts": [], "errors": [],
        }

    materiality_file = resolve_artifact_path(materiality_file, out_dir, "materiality.json")
    mat = safe_load_json(materiality_file) or {}
    thresholds = mat.get("thresholds", {})
    overall_materiality = float(thresholds.get("overall") or 0.0)
    performance_materiality = float(thresholds.get("performance") or 0.0)
    if not overall_materiality or not performance_materiality:
        return {
            "execution_status": "FAILED",
            "errors": [{"type": "MissingDataError", "message": "materiality.json has no overall/performance threshold -- run build_materiality first."}],
            "message": "Cannot sample without materiality thresholds.",
        }

    pop = df.filter((pl.col("mapped_status") == MAPPED_STATUS_MAPPED) & (pl.col("closing_balance") != 0))
    if population_head and "report_head" in pop.columns:
        pop = pop.filter(pl.col("report_head") == population_head)
    pop = pop.sort("gl_code")
    rows = pop.select(["gl_code", "gl_name", "closing_balance"]).to_dicts()
    population_value = sum(abs(r["closing_balance"]) for r in rows)

    if not rows:
        return {
            "execution_status": "SUCCESS", "pipeline_status": "WARNING", "can_continue": True,
            "message": f"No mapped, non-zero accounts in scope{f' for report_head={population_head!r}' if population_head else ''} -- no sample selected.",
            "artifacts": [], "errors": [],
        }

    payload = {
        "methodology": (
            "SA 530 (Audit Sampling). Every other screen in this pipeline tests the full "
            "population; this tool deliberately samples instead, and records every parameter "
            "of the sample so it is reproducible from this documentation alone. Population: "
            "mapped, non-zero-balance GL accounts" + (f" under FS Head '{population_head}'" if population_head else " (whole TB)") + "."
        ),
        "population_head": population_head,
        "generated_at": datetime.datetime.now().isoformat(),
        "methods_run": [],
        "safe_limitation": SAFE_WORDING_DISCLAIMER,
    }

    if method in ("mus", "both"):
        payload["mus"] = _mus_sample(rows, population_value, performance_materiality, confidence_level, random_seed)
        payload["methods_run"].append("mus")
    if method in ("stratified", "both"):
        payload["stratified"] = _stratified_sample(rows, overall_materiality, stratified_sample_pct, random_seed)
        payload["methods_run"].append("stratified")

    # remark #13 fix: MUS and stratified are two independent SA 530 techniques with their
    # own valid sample sizes -- narrative text elsewhere kept citing "the sample size"
    # without saying which method, or vs. the sheet's combined physical row count. Add one
    # explicit, clearly-labeled combined figure so downstream text has a single canonical
    # number to cite instead of inventing its own.
    mus_size = payload.get("mus", {}).get("sample_size", 0)
    strat_total = sum(s["sample_size"] for s in payload["stratified"]["strata"].values()) if "stratified" in payload else 0
    if "mus" in payload or "stratified" in payload:
        payload["total_sampled_items_mus_plus_stratified"] = mus_size + strat_total

    out_path = out_dir / "sample_selection.json"
    with atomic_write(out_path) as tmp:
        Path(tmp).write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    sizes = []
    if "mus" in payload:
        sizes.append(f"MUS {mus_size} item(s)")
    if "stratified" in payload:
        sizes.append(f"Stratified {strat_total} item(s) across {len(payload['stratified']['strata'])} band(s)")
    if "mus" in payload and "stratified" in payload:
        sizes.append(f"combined total {mus_size + strat_total} item(s)")

    return {
        "execution_status": "SUCCESS",
        "pipeline_status": "SUCCESS",
        "can_continue": True,
        "message": f"Sample selection over {len(rows)} account(s), population value {population_value:,.0f}: " + "; ".join(sizes) + ".",
        "artifacts": [str(out_path.resolve())],
        "errors": [],
    }


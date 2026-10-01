import datetime
import hashlib
import json
import platform
import re
from pathlib import Path
from typing import Optional

import polars as pl

from modes.trial_balance.pipeline.config import settings
from modes.trial_balance.pipeline.tools._shared import *  # noqa: F401,F403
from modes.trial_balance.pipeline.tools.canonical_schema import *  # noqa: F401,F403
from modes.trial_balance.pipeline.tools.pipeline_tool import *  # noqa: F401,F403

_GENERIC_EVIDENCE = {"Ledgers", "Contracts/Agreements", "Reconciliations", "Mapping Review"}

def match_area(text: str, areas: dict, account_type: str = None) -> tuple:
    """Match an account's text against the catalogue's account areas.

    Longest keyword wins rather than first-match: "trade receivable" must beat a bare
    "receivable" when both are present, otherwise a specific area loses to a generic
    one purely because of dict ordering.

    Client-review defect registry: a flat keyword match with no structural check let a
    Liability-typed account (e.g. "Retention Money Payable") match an Asset-only area
    ("loans_advances_deposits" via the bare "retention money" keyword) purely because
    the FSLI/gl_name text happened to contain a matching phrase, even though the
    account's own already-correct FSLI classification (account_type, populated
    upstream from the canonical TB) made that area structurally impossible. Two
    independent, opt-in mechanisms fix this without touching any area that doesn't
    declare them (so every area's existing behaviour is unchanged unless it opts in):

    - An area may declare `"exclude": [...]` keywords -- if any is present, that area
      is never a candidate for this account, regardless of how well `"match"` scores
      (e.g. "revenue" excludes "loss on sale" so a disposal loss doesn't inherit
      revenue-recognition assertions just because Schedule III nets it under "Other
      Income").
    - An area may declare `"account_types": [...]` -- when the caller passes the
      account's actual type, the best keyword match from a type-COMPATIBLE area wins
      over a longer match from an incompatible one. If no type-compatible candidate
      exists, this falls back to the plain longest-match (unchanged prior behaviour)
      rather than forcing a match to unclassified -- sec 1.2 forbids inventing a
      classification, but an imperfect area is still better than no evidence when
      nothing better is available.
    """
    haystack = f" {(text or '').lower()} "
    candidates = []
    for name, spec in areas.items():
        if name == "unclassified":
            continue
        if any(kw in haystack for kw in spec.get("exclude", [])):
            continue
        best_kw_len = 0
        for kw in spec.get("match", []):
            if kw in haystack and len(kw) > best_kw_len:
                best_kw_len = len(kw)
        if best_kw_len:
            candidates.append((name, spec, best_kw_len))

    if not candidates:
        return "unclassified", areas["unclassified"]

    candidates.sort(key=lambda c: c[2], reverse=True)

    if account_type:
        for name, spec, _kw_len in candidates:
            allowed_types = spec.get("account_types")
            if not allowed_types or account_type in allowed_types:
                return name, spec

    best_name, best_spec, _ = candidates[0]
    return best_name, best_spec

@pipeline_tool("build_assertion_evidence_map", domain="reasoning")
def build_assertion_evidence_map(
    consolidated_exceptions_file: str = None,
    sensitive_accounts_file: str = None,
    output_dir: str = None,
    **kwargs,
) -> dict:
    """Map every consolidated exception to the assertions its ACCOUNT AREA exposes and
    the specific records that resolve them, from the Appendix B/C knowledge pack --
    replacing generic evidence such as "Ledgers" or "Contracts/Agreements" with named
    records (party-wise ageing, GSTR-3B, sanction order, utilisation certificate).
    Sensitive tags add their own assertions and evidence on top. Writes
    assertion_evidence_map.json."""
    out_dir = resolve_output_dir(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    exc_path = (Path(consolidated_exceptions_file) if consolidated_exceptions_file
                else out_dir / "consolidated_exceptions.json")
    if not exc_path.exists():
        raise PipelineFileError(str(exc_path), "Consolidated exceptions (run build_exception_consolidator first)")
    exc_data = safe_load_json(exc_path)

    pack = load_pack("assertions")
    areas = pack["areas"]
    tag_overrides = {k: v for k, v in pack["sensitive_tag_overrides"].items() if not k.startswith("_")}

    warnings = []
    sens_path = (Path(sensitive_accounts_file) if sensitive_accounts_file
                 else out_dir / "sensitive_accounts.json")
    sens = safe_load_json(sens_path) if sens_path.exists() else {}
    # gl_code -> sensitive tag, so an exception can pick up its tag's extra evidence.
    tag_by_gl = {}
    for a in (sens.get("account_sensitivity") or []):
        if a.get("gl_code"):
            tag_by_gl[str(a["gl_code"])] = a.get("category")
    if not sens_path.exists():
        warnings.append(
            "sensitive_accounts.json not available -- sensitive-tag evidence (RPT register, "
            "sanction orders, MSME declarations) cannot be added. Run build_sensitive_detector first."
        )

    exceptions = exc_data.get("exceptions") or []
    if not exceptions:
        warnings.append("consolidated_exceptions.json carries no exceptions to map.")

    mapped, generic_replaced, area_counts = [], 0, {}

    for exc in exceptions:
        business = exc.get("business_context") or {}
        hierarchy = exc.get("hierarchy_context") or {}
        gl_code = str(business.get("gl_code") or "")
        gl_name = str(business.get("gl_name") or exc.get("entity", {}).get("entity_name") or "")

        text = " ".join(str(v) for v in (
            gl_name, hierarchy.get("line_item"), hierarchy.get("fs_head"),
            hierarchy.get("group"), business.get("fsli"),
        ) if v)

        area_name, area = match_area(text, areas, account_type=business.get("account_type"))
        area_counts[area_name] = area_counts.get(area_name, 0) + 1

        assertions = list(area.get("assertions", []))
        evidence = list(area.get("evidence", []))
        regularity = bool(area.get("regularity_flag"))
        applied_tag = tag_by_gl.get(gl_code)

        if applied_tag and applied_tag in tag_overrides:
            override = tag_overrides[applied_tag]
            for a in override.get("assertions", []):
                if a not in assertions:
                    assertions.append(a)
            for e in override.get("evidence", []):
                if e not in evidence:
                    evidence.append(e)
            regularity = regularity or bool(override.get("regularity_flag"))

        previous = (exc.get("audit_planning") or {}).get("required_evidence") or []
        if set(previous) & _GENERIC_EVIDENCE:
            generic_replaced += 1

        mapped.append({
            "exception_id": (exc.get("metadata") or {}).get("exception_id"),
            "account": f"{gl_code} - {gl_name}".strip(" -"),
            "gl_code": gl_code,
            "gl_name": gl_name,
            "fsli": hierarchy.get("line_item") or business.get("fsli"),
            "account_area": area_name,
            "sensitive_tag": applied_tag,
            "assertions": assertions,
            "evidence_requested": evidence,
            "primary_objective": area.get("primary_objective"),
            "regularity_flag": regularity,
            "severity": (exc.get("scoring") or {}).get("severity"),
            "closing_balance": (exc.get("financial_context") or {}).get("closing_balance"),
            "previous_generic_evidence": previous,
        })

    unclassified = area_counts.get("unclassified", 0)
    if unclassified:
        warnings.append(
            f"{unclassified} exception(s) could not be matched to an account area and carry "
            "information-request evidence only. sec 1.2 forbids forcing a mapping -- request "
            "the chart of accounts rather than assuming an area."
        )

    payload = {
        "methodology": (
            "Each exception's account text is matched against the Appendix B/C account-area "
            "catalogue (longest keyword wins, so a specific area beats a generic one), and "
            "the area's assertions and named evidence are applied. A sensitive tag adds its "
            "own assertions and evidence on top. Unmatched accounts receive information-"
            "request evidence rather than a forced mapping."
        ),
        "knowledge_pack_version": pack["_meta"]["version"],
        "generated_at": datetime.datetime.now().isoformat(),
        "summary": {
            "exceptions_mapped": len(mapped),
            "generic_evidence_replaced": generic_replaced,
            "unclassified": unclassified,
            "by_account_area": dict(sorted(area_counts.items(), key=lambda kv: -kv[1])),
            "regularity_flagged": sum(1 for m in mapped if m["regularity_flag"]),
        },
        "assertion_evidence_map": mapped,
        "safe_limitation": SAFE_WORDING_DISCLAIMER,
    }

    out_path = out_dir / "assertion_evidence_map.json"
    with atomic_write(out_path) as tmp:
        Path(tmp).write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    return {
        "execution_status": "SUCCESS",
        "pipeline_status": "SUCCESS",
        "can_continue": True,
        "message": (
            f"Assertion/evidence map: {len(mapped)} exception(s) mapped to account areas, "
            f"{generic_replaced} had generic source-engine evidence replaced with named records, "
            f"{unclassified} unclassified."
        ),
        "artifacts": [str(out_path.resolve())],
        "errors": [],
        "warnings": warnings,
    }


_SEV_RANK = {"Critical": 3, "High": 2, "Medium": 1, "Low": 0, "Information Request": -1}

_AUDIT_REASONING_PROMPT_TEMPLATE = """You are an audit reasoning assistant. You are given a single \
cluster of correlated trial-balance exceptions, tagged as evidence reference {{evidence_tag}}.

Cluster payload (JSON):
{{cluster_payload}}

Write ONE audit finding about this cluster. You MUST cite the evidence tag {{evidence_tag}} in \
your `evidence_refs` field — do not invent other tags, and do not omit it.

Respond with ONLY a JSON object of this exact shape, no markdown fences, no commentary:
{
  "statement": "<one or two sentence finding, grounded only in the evidence payload above>",
  "severity": "<Critical|High|Medium|Low>",
  "audit_risk": "<why this matters for the audit>",
  "recommended_procedures": ["<procedure 1>", "<procedure 2>"],
  "evidence_refs": ["{{evidence_tag}}"]
}
"""

def _deterministic_finding(evidence_tag: str, cluster: dict) -> dict:
    """Template fallback finding used when no llm_client is supplied or every LLM attempt fails."""
    fsli = cluster.get("fsli", "Unmapped")
    bp = cluster.get("business_process", "General")
    member_count = cluster.get("executive_facts", {}).get("member_count", 0)
    total_balance = cluster.get("evidence_summary", {}).get("total_balance_at_risk", 0.0)
    rules = cluster.get("evidence_summary", {}).get("triggered_rules", [])
    return {
        # TB-R03: the condition/rule-name catalogue must never be interpolated into the
        # narrative sentence -- it reads as garbled prose ("...triggered by: ABOVE_OM,
        # SIGN_REVERSAL, ..."). Keep it in its own structured field (below) and have
        # renderers show it as a separate list, never mashed into `statement`.
        "statement": (
            f"Cluster {cluster.get('cluster_id')} ({bp} / {fsli}) contains {member_count} correlated "
            f"exception(s) totalling {total_balance:,.2f} in balance at risk."
        ),
        "severity": cluster.get("severity", "Medium"),
        "audit_risk": (
            "Correlated exceptions within the same business process/FSLI cluster increase the risk "
            "of a systemic (rather than isolated) misstatement and warrant corroboration."
        ),
        "recommended_procedures": [
            f"Review {ev}" for ev in cluster.get("evidence_summary", {}).get("required_evidence", [])
        ] or ["Obtain and review supporting ledgers/reconciliations for the flagged accounts."],
        "evidence_refs": [evidence_tag],
        "conditions_evaluated": rules,
    }

def _call_llm_for_cluster(llm_client, evidence_tag: str, cluster: dict, stats: dict) -> dict:
    """Attempt LLM narrative generation for one cluster: up to 3 tries on a malformed-JSON
    response, but gives up immediately on a connection failure. Returns None on total failure.

    The client already retries internally with its own backoff on a connection failure (see
    its own retry logging) -- retrying the SAME failure again here just re-multiplies an
    already-multiplied wait (up to 9 real attempts per cluster before this fix, which is what
    made a comparison run with several clusters look hung for minutes when the LLM endpoint
    was down). A malformed-JSON response is a different failure mode -- the model answered,
    just not usably -- and IS worth asking again."""
    prompt = _AUDIT_REASONING_PROMPT_TEMPLATE.replace("{{evidence_tag}}", evidence_tag)
    prompt = prompt.replace("{{cluster_payload}}", json.dumps(cluster, indent=2, default=str))
    messages = [{"role": "user", "content": prompt}]

    for _ in range(3):
        try:
            stats["llm_calls"] += 1
            response = llm_client.generate(messages, max_tokens=1024)
        except Exception:
            break
        try:
            content = response.get("content", "{}") if isinstance(response, dict) else getattr(response, "content", "{}")
            cleaned = content.strip()
            if cleaned.startswith("```json"):
                cleaned = cleaned[7:]
            if cleaned.startswith("```"):
                cleaned = cleaned[3:]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]
            parsed = json.loads(cleaned.strip())
            return parsed
        except Exception:
            continue  # malformed JSON from the model -- worth asking again

    stats["failures"] += 1
    return None

def _validate_grounding(finding: dict, evidence_tag: str) -> bool:
    """A finding is only kept if it cites the evidence tag it was built from -- per this repo's
    evidence-grounding convention, ungrounded findings are dropped, never surfaced."""
    refs = finding.get("evidence_refs") or []
    return evidence_tag in refs

@pipeline_tool("build_audit_reasoning", domain="reasoning")
def build_audit_reasoning(
    consolidated_exceptions_file: str,
    validation_report_file: str,
    # TB-QA-Issue-C: the previous default of 5 silently dropped Medium+ clusters from
    # every downstream section that reads `observations` (build_docx_report.py's Focus
    # Areas / Focus Sequence) while Section 5's Exception Summary rendered the full,
    # uncapped cluster list from consolidated_exceptions.json directly -- the mismatch is
    # exactly what a real run showed (13 clusters in Section 5, only 10 in Focus Areas,
    # no disclosure). Raised well above anything seen in testing (24 clusters ran fine);
    # build_docx_report.py/build_report_markdown.py also independently disclose when a
    # cap is ever hit, as a second safeguard.
    max_observations: int = 25,
    output_dir: str = None,
    llm_client=None,
    data_sufficiency_file: str = None,
    **kwargs,
) -> dict:
    """Select the highest-priority exception clusters, tag each with an evidence reference,
    and generate one evidence-grounded finding per cluster (LLM narrative if llm_client is
    given, deterministic template otherwise). Gated strictly on validate_tb_pipeline's output."""
    errors = []
    artifacts = []

    out_dir = resolve_output_dir(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Data-sufficiency limitations statement -- must appear before findings
    # per the audit-language spec's response ordering (input-quality note
    # first). Optional: absent when build_data_sufficiency_grade wasn't run.
    data_sufficiency_note = None
    if data_sufficiency_file:
        ds_path = Path(data_sufficiency_file)
        if ds_path.exists():
            with open(ds_path, "r") as f:
                ds_data = json.load(f)
            data_sufficiency_note = (
                f"Data sufficiency: {ds_data.get('grade', 'Unknown')}. {ds_data.get('rationale', '')}"
            ).strip()

    # 1. Strict Deterministic Gating
    val_path = Path(validation_report_file) if validation_report_file else out_dir / "validation_report.json"
    if not val_path.exists():
        errors.append({"type": "FileNotFoundError", "file": str(val_path)})
        return {"execution_status": "FAILED", "errors": errors, "message": "validation_report.json not found (run validate_tb_pipeline first)."}

    with open(val_path, "r") as f:
        validation_report = json.load(f)

    report_status = validation_report.get("report_readiness", {}).get("status")
    reasoning_status = validation_report.get("reasoning_readiness", {}).get("status")

    if report_status != "READY" or reasoning_status != "LLM Ready":
        errors.append({"type": "ValidationError", "message": f"Pipeline not ready. Report: {report_status}, Reasoning: {reasoning_status}"})
        return {"execution_status": "FAILED", "errors": errors, "message": "Pipeline validation failed. Cannot proceed with reasoning."}

    exc_path = Path(consolidated_exceptions_file) if consolidated_exceptions_file else out_dir / "consolidated_exceptions.json"
    if not exc_path.exists():
        errors.append({"type": "FileNotFoundError", "file": str(exc_path)})
        return {"execution_status": "FAILED", "errors": errors, "message": "Consolidated exceptions file not found (run build_exception_consolidator first)."}

    with open(exc_path, "r") as f:
        exc_data = json.load(f)

    clusters = exc_data.get("clusters", [])
    exceptions = exc_data.get("exceptions", [])

    if not clusters:
        return {
            "execution_status": "SUCCESS",
            "pipeline_status": "SUCCESS",
            "message": "No clusters found in payload, skipping reasoning.",
            "artifacts": [],
            "errors": errors,
        }

    # Sort clusters: Critical -> High -> Medium, then by score
    clusters.sort(key=lambda x: (_SEV_RANK.get(x.get("cluster_severity", "Low"), -1), x.get("cluster_score", 0.0)), reverse=True)

    selected_clusters = []
    for c in clusters:
        if len(selected_clusters) >= max_observations:
            break
        if _SEV_RANK.get(c.get("cluster_severity", "Low"), -1) > 0:  # Only Medium and above
            selected_clusters.append(c)

    if not selected_clusters:
        return {
            "execution_status": "SUCCESS",
            "pipeline_status": "SUCCESS",
            "message": "No actionable clusters (Medium or higher) found.",
            "artifacts": [],
            "errors": errors,
        }

    # Build summarized payload + evidence tags
    payload_clusters = []
    evidence_tags = {}
    for i, c in enumerate(selected_clusters):
        evidence_tag = f"E{i + 1}"
        member_excs = [e for e in exceptions if e.get("metadata", {}).get("exception_id") in c.get("member_exceptions", [])]

        audit_assertions = set()
        required_evidence = set()
        for e in member_excs:
            pl = e.get("audit_planning", {})
            audit_assertions.update(pl.get("audit_assertions", []))
            required_evidence.update(pl.get("required_evidence", []))

        evidence_summary = {
            "total_balance_at_risk": c.get("total_balance", 0.0),
            "triggered_rules": list(c.get("risk_themes", [])),
            "exception_count": c.get("member_count", 0),
            "assertions_at_risk": list(audit_assertions),
            "required_evidence": list(required_evidence),
        }

        evidence_tags[evidence_tag] = c.get("member_exceptions", [])

        payload_clusters.append({
            "evidence_tag": evidence_tag,
            "cluster_id": c.get("cluster_id"),
            "cluster_name": c.get("cluster_name"),
            "business_process": c.get("business_process"),
            "fsli": c.get("fsli"),
            "severity": c.get("cluster_severity"),
            "executive_facts": {
                "member_count": c.get("member_count"),
                "critical_count": c.get("critical_count"),
                "high_count": c.get("high_count"),
                "cluster_score": c.get("cluster_score"),
            },
            "evidence_summary": evidence_summary,
            "exception_references": c.get("member_exceptions", []),
        })

    # Generate one evidence-grounded finding per cluster: LLM narrative if available, else a
    # deterministic template finding. LLM failures never crash the pipeline -- they just fall
    # back to the deterministic finding for that cluster.
    findings = []
    stats = {"llm_calls": 0, "failures": 0}
    for cluster in payload_clusters:
        evidence_tag = cluster["evidence_tag"]
        finding = None
        if llm_client is not None:
            llm_finding = _call_llm_for_cluster(llm_client, evidence_tag, cluster, stats)
            if llm_finding is not None and _validate_grounding(llm_finding, evidence_tag):
                finding = llm_finding
            elif llm_finding is not None:
                # LLM returned something but didn't cite the evidence tag -- ungrounded, drop it
                # and fall back to the deterministic finding rather than surface an unsupported claim.
                stats["failures"] += 1

        if finding is None:
            finding = _deterministic_finding(evidence_tag, cluster)

        # Deterministic safe-wording pass -- applies regardless of whether the
        # finding came from the LLM or the template fallback, so a drifting
        # narrative prompt can never ship a banned assurance-language phrase.
        # append_disclaimer=False: the disclaimer is attached once per finding
        # as a structured safe_limitation field (matches OUT-01 schema), not
        # mashed into the free-text statement/audit_risk fields.
        finding["statement"] = apply_safe_wording(finding.get("statement", ""), append_disclaimer=False)
        if finding.get("audit_risk"):
            finding["audit_risk"] = apply_safe_wording(finding["audit_risk"], append_disclaimer=False)
        finding["safe_limitation"] = SAFE_WORDING_DISCLAIMER

        finding["cluster_id"] = cluster["cluster_id"]
        finding["evidence_uids"] = evidence_tags.get(evidence_tag, [])
        findings.append(finding)

    # audit_observations: report-consumable view of `findings`, shaped as
    # build_docx_report.py/build_report_markdown.py already expect
    # ({"observation": {priority, title, detailed_observation, ...}}) --
    # additive alongside `findings`/`clusters`, not a replacement.
    clusters_by_id = {c["cluster_id"]: c for c in payload_clusters}
    audit_observations = []
    for finding in findings:
        c = clusters_by_id.get(finding.get("cluster_id"), {})
        evidence_summary = c.get("evidence_summary", {})
        audit_observations.append({
            "observation": {
                "priority": finding.get("severity", "Medium"),
                "title": c.get("cluster_name") or f"Cluster {finding.get('cluster_id')}",
                "cluster_id": finding.get("cluster_id"),
                "detailed_observation": finding.get("statement", ""),
                "executive_summary": finding.get("statement", ""),
                "affected_assertions": evidence_summary.get("assertions_at_risk", []),
                "supporting_evidence": ", ".join(evidence_summary.get("required_evidence", [])) or finding.get("audit_risk", ""),
                "conclusion": finding.get("audit_risk", ""),
                "safe_limitation": finding.get("safe_limitation"),
                "evidence_refs": finding.get("evidence_refs", []),
                "conditions_evaluated": finding.get("conditions_evaluated") or evidence_summary.get("triggered_rules", []),
            }
        })

    payload_file = out_dir / "audit_reasoning.json"

    payload_data = {
        "payload_metadata": {
            "generated_at": datetime.datetime.now().isoformat(),
            "pipeline_version": "2.0",
            "cluster_count": len(clusters),
            "selected_clusters": len(selected_clusters),
            "selection_reason": "Severity + Score (Medium or higher)",
            "llm_used": llm_client is not None,
            "llm_stats": stats,
        },
        "data_sufficiency_note": data_sufficiency_note,
        "evidence_tags": evidence_tags,
        "clusters": payload_clusters,
        "findings": findings,
        "audit_observations": audit_observations,
    }

    write_json_atomic(payload_data, payload_file, indent=4)

    artifacts.append(str(payload_file))

    return {
        "execution_status": "SUCCESS",
        "pipeline_status": "SUCCESS",
        "message": f"Generated audit_reasoning.json with {len(findings)} evidence-grounded finding(s) for {len(selected_clusters)} clusters.",
        "artifacts": artifacts,
        "errors": errors,
        "payload_path": str(payload_file),
        "findings": findings,
    }


def _collect_screen_records(out_dir: Path) -> tuple:
    """Read every screen artifact in out_dir that carries a `finding_records` array.

    Screens write their own artifact and include a finding_records block in it;
    this tool consolidates rather than recomputes, so it never needs to know which
    screens ran. A screen that did not run simply contributes nothing.
    """
    records, sources = [], []
    for path in sorted(out_dir.glob("*.json")):
        if path.name in ("finding_records.json", "audit_reasoning.json"):
            continue
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError, UnicodeDecodeError):
            continue
        if isinstance(data, dict) and isinstance(data.get("finding_records"), list):
            found = [r for r in data["finding_records"] if isinstance(r, dict)]
            if found:
                records.extend(found)
                sources.append({"artifact": path.name, "record_count": len(found)})
    return records, sources

@pipeline_tool("build_finding_records", domain="reasoning")
def build_finding_records(output_dir: str = None, data_sufficiency_file: str = None, **kwargs) -> dict:
    """Consolidate every screen's findings into the canonical audit-finding record
    schema: observation, expectation, gap, assertion, risk_basis, regularity flag,
    data sufficiency, proposed response, evidence requested and safe limitation.

    Reads any artifact in output_dir carrying a `finding_records` array, so no
    screen needs registering here. Writes finding_records.json, risk-ranked."""
    out_dir = resolve_output_dir(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    records, sources = _collect_screen_records(out_dir)

    # sec 15.1: the data-sufficiency note precedes findings, and every record
    # inherits the run's grade unless its own screen set a more specific one.
    run_sufficiency = None
    ds_path = Path(data_sufficiency_file) if data_sufficiency_file else out_dir / "data_sufficiency.json"
    if ds_path.exists():
        try:
            with open(ds_path, encoding="utf-8") as f:
                ds = json.load(f)
            run_sufficiency = ds.get("grade")
        except (json.JSONDecodeError, OSError):
            run_sufficiency = None

    warnings = []
    for i, record in enumerate(records):
        record.setdefault("safe_limitation", SAFE_WORDING_DISCLAIMER)
        if run_sufficiency and not record.get("data_sufficiency"):
            record["data_sufficiency"] = run_sufficiency
        warnings.extend(validate_record(record, i))

    records.sort(
        key=lambda r: (
            RATING_RANK.get(str(r.get("risk_rating", "low")).lower(), 0),
            1 if r.get("regularity_flag") else 0,
            abs(float(r.get("amount") or 0.0)),
        ),
        reverse=True,
    )

    by_rating = {rating: 0 for rating in VALID_RATINGS}
    for r in records:
        by_rating[str(r.get("risk_rating", "low")).lower()] = by_rating.get(
            str(r.get("risk_rating", "low")).lower(), 0) + 1

    payload = {
        "schema": "TB finding record (spec sec 14)",
        "generated_at": datetime.datetime.now().isoformat(),
        "knowledge_pack_versions": {
            "assertions": load_pack("assertions")["_meta"]["version"],
            "language": load_pack("language")["_meta"]["version"],
        },
        "data_sufficiency": run_sufficiency,
        "summary": {
            "total_records": len(records),
            "by_risk_rating": by_rating,
            "regularity_flagged": sum(1 for r in records if r.get("regularity_flag")),
            "contributing_screens": sources,
        },
        "schema_warnings": warnings,
        "finding_records": records,
        "safe_limitation": SAFE_WORDING_DISCLAIMER,
    }

    out_path = out_dir / "finding_records.json"
    with atomic_write(out_path) as tmp:
        Path(tmp).write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    if not records:
        message = (
            "No screen findings available to consolidate -- no artifact in this run "
            "carries a finding_records array. Run the Phase-2 screens first."
        )
    else:
        message = (
            f"Consolidated {len(records)} finding record(s) from {len(sources)} screen(s): "
            f"{by_rating['high']} high, {by_rating['medium']} medium, {by_rating['low']} low, "
            f"{by_rating['information_request']} information request."
        )

    return {
        "execution_status": "SUCCESS",
        "pipeline_status": "WARNING" if warnings else "SUCCESS",
        "can_continue": True,
        "message": message,
        "artifacts": [str(out_path.resolve())],
        "errors": [],
        "warnings": warnings,
    }


_AREA_ROUTING = [
    (("ageing", "confirmation", "subsequent receipt", "customer", "debtor", "ecl",
      "sales register", "credit policy", "dispute"), "Receivables"),
    (("vendor", "payable", "msme", "gr/ir", "purchase cut-off"), "Payables & Procurement"),
    (("gst", "gstr", "tds", "provident fund", "esi", "professional tax", "challan",
      "return", "assessment", "demand order", "taxability", "credit-note", "credit note",
      "tax computation", "disputed dues"), "Statutory & Tax"),
    (("loan agreement", "sanction letter", "repayment", "covenant", "charge registration", "interest computation", "borrowing"), "Treasury & Borrowings"),
    (("fixed asset register", "title deed", "depreciation", "capitalisation", "impairment", "valuer", "physical verification of assets"), "Fixed Assets & CWIP"),
    (("stock", "inventory", "nrv", "physical verification"), "Inventory"),
    (("payroll", "actuarial", "employment contract", "headcount"), "Payroll"),
    (("sanction order", "utilisation certificate", "grant", "fund-utilisation", "unspent"), "Grants & Public Funds"),
    (("related-party", "related party", "mbp-1", "rpt"), "Related Parties"),
    (("board", "shareholder", "roc", "allotment", "minutes", "approval"), "Governance"),
    (("bank reconciliation", "bank confirmation", "cash verification", "bank trail"), "Cash & Bank"),
    (("chart of accounts", "mapping", "ledger extract"), "Data & Mapping"),
    (("journal", "audit trail", "edit log", "user access", "period lock"), "IT & Audit Trail"),
    # Remark #16 fix: EPC/construction-domain evidence themes previously fell through
    # to "General" -- these were confirmed populated on live EPIL data in an earlier
    # investigation of this same evidence-routing gap.
    (("claim", "mobilisation", "mobilization", "retention", "withheld", "recoverable",
      "bank guarantee"), "Contract Exposures & Retention"),
    (("provision", "write-off", "write off", "written off"), "Provisions & Write-Offs"),
    (("foreign", "overseas", "segment", "branch", "fcy", "exchange rate"), "Foreign Operations & Segments"),
]

_INFORMATION_REQUEST_MARKERS = ("chart of accounts", "mapping", "confirm scale", "confirm currency")

def _route(evidence: str) -> str:
    low = evidence.lower()
    for keywords, area in _AREA_ROUTING:
        if any(k in low for k in keywords):
            return area
    return "General"

def _normalise(evidence: str) -> str:
    """Collapse near-duplicate phrasings so the same record is not requested twice
    because two screens described it slightly differently."""
    return re.sub(r"[^a-z0-9 ]", "", (evidence or "").lower()).strip()

def _build_management_query(rec: dict) -> Optional[dict]:
    """The query text/candidate-explanations/corroboration triad build_request_lists derives
    from one finding's valid_reasons -- pulled out to a standalone, pure function (input:
    one finding_records row; output: None or the query fields) so the Excel Findings & Queries
    Register sheet can render the EXACT same text inline, per finding, rather than re-deriving
    or value-matching it back from management_query_list.json's independently-sorted list.
    Single source of truth for query wording (same principle as resolve_severity for severity:
    never compute the same derived text in two places)."""
    reasons = rec.get("valid_reasons") or []
    if not reasons:
        return None
    label = rec.get("account") or rec.get("fsli") or "(run-level)"
    return {
        "query": (
            f"For {label}: which of the following explains the position, and what "
            f"record supports that explanation? "
            + "; ".join(f"({i + 1}) {r}" for i, r in enumerate(reasons))
        ),
        "candidate_explanations": reasons,
        "corroborating_evidence_required": rec.get("evidence_requested", []),
    }

@pipeline_tool("build_request_lists", domain="reasoning")
def build_request_lists(
    finding_records_file: str = None,
    assertion_evidence_map_file: str = None,
    output_dir: str = None,
    **kwargs,
) -> dict:
    """Consolidate every finding's evidence into one de-duplicated, risk-ordered
    request list grouped by audit area, and produce a separate management-query list
    marked as explanations requiring corroboration rather than evidence. Writes
    evidence_request_list.json and management_query_list.json."""
    out_dir = resolve_output_dir(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    warnings = []
    fr_path = Path(finding_records_file) if finding_records_file else out_dir / "finding_records.json"
    fr = safe_load_json(fr_path) if fr_path.exists() else {}
    records = fr.get("finding_records") or []
    if not records:
        warnings.append(
            "finding_records.json carries no records -- run build_finding_records after the "
            "Phase-2 screens. Falling back to the assertion/evidence map alone."
        )

    aem_path = (Path(assertion_evidence_map_file) if assertion_evidence_map_file
                else out_dir / "assertion_evidence_map.json")
    aem = safe_load_json(aem_path) if aem_path.exists() else {}
    mapped = aem.get("assertion_evidence_map") or []
    if not mapped and not records:
        warnings.append("assertion_evidence_map.json also unavailable -- request lists will be empty.")

    # ── consolidated evidence requests ────────────────────────────────────────
    requests = {}

    def add(evidence: str, *, rating: str, regularity: bool, account: str, source: str, assertions):
        if not evidence or not str(evidence).strip():
            return
        key = _normalise(evidence)
        if not key:
            return
        entry = requests.setdefault(key, {
            "record": str(evidence).strip(),
            "audit_area": _route(evidence),
            "highest_risk_rating": "low",
            "regularity_relevant": False,
            "requested_for": [],
            "assertions_resolved": [],
            "is_information_request": any(m in str(evidence).lower() for m in _INFORMATION_REQUEST_MARKERS),
        })
        if RATING_RANK.get(rating, 0) > RATING_RANK.get(entry["highest_risk_rating"], 0):
            entry["highest_risk_rating"] = rating
        entry["regularity_relevant"] = entry["regularity_relevant"] or bool(regularity)
        if account and account not in entry["requested_for"]:
            entry["requested_for"].append(account)
        for a in (assertions or []):
            if a not in entry["assertions_resolved"]:
                entry["assertions_resolved"].append(a)
        if source not in entry.setdefault("source_screens", []):
            entry["source_screens"].append(source)

    for rec in records:
        label = rec.get("account") or rec.get("fsli") or "(run-level)"
        for ev in rec.get("evidence_requested") or []:
            add(ev,
                rating=str(rec.get("risk_rating", "low")).lower(),
                regularity=rec.get("regularity_flag", False),
                account=label,
                source=rec.get("source_screen", "unknown"),
                assertions=rec.get("assertion"))

    for m in mapped:
        for ev in m.get("evidence_requested") or []:
            add(ev,
                rating=str(m.get("severity", "low")).lower() if str(m.get("severity", "")).lower() in RATING_RANK else "medium",
                regularity=m.get("regularity_flag", False),
                account=m.get("account") or "(unmapped)",
                source="build_assertion_evidence_map",
                assertions=m.get("assertions"))

    consolidated = sorted(
        requests.values(),
        key=lambda e: (
            RATING_RANK.get(e["highest_risk_rating"], 0),
            1 if e["regularity_relevant"] else 0,
            len(e["requested_for"]),
        ),
        reverse=True,
    )
    evidence_requests = [e for e in consolidated if not e["is_information_request"]]
    information_requests = [e for e in consolidated if e["is_information_request"]]

    by_area = {}
    for e in evidence_requests:
        by_area.setdefault(e["audit_area"], []).append(e["record"])

    # ── management queries ────────────────────────────────────────────────────
    # Built from each finding's valid_reasons: those are precisely the hypotheses only
    # management can confirm. Kept in a SEPARATE artifact so an explanation can never
    # be mistaken for evidence (sec 13.1).
    queries = []
    for rec in records:
        q = _build_management_query(rec)
        if q is None:
            continue
        queries.append({
            "subject": rec.get("account") or rec.get("fsli") or "(run-level)",
            "risk_rating": rec.get("risk_rating"),
            "regularity_flag": rec.get("regularity_flag", False),
            "observation": rec.get("observation"),
            **q,
            "source_screen": rec.get("source_screen"),
        })
    queries.sort(key=lambda q: (RATING_RANK.get(str(q.get("risk_rating", "low")).lower(), 0),
                               1 if q["regularity_flag"] else 0), reverse=True)

    disclaimer = (
        "Management explanation is NOT audit evidence. Each response below is a hypothesis "
        "that guides testing and must be corroborated by the records listed against it "
        "before any finding is closed (spec sec 1.2, sec 13.1)."
    )

    ev_payload = {
        "methodology": (
            "Every finding's and mapped exception's evidence is normalised, de-duplicated, "
            "annotated with the accounts and assertions it serves, ordered by the highest "
            "risk it supports, and grouped by audit area. Data-quality requests (chart of "
            "accounts, scale confirmation) are separated from substantive evidence, since "
            "sec 6.2 treats those as information requests rather than audit findings."
        ),
        "generated_at": datetime.datetime.now().isoformat(),
        "summary": {
            "evidence_requests": len(evidence_requests),
            "information_requests": len(information_requests),
            "audit_areas": len(by_area),
            "findings_covered": len(records),
            "deduplication_saved": sum(len(e["requested_for"]) for e in consolidated) - len(consolidated),
        },
        "evidence_request_list": evidence_requests,
        "information_request_list": information_requests,
        "by_audit_area": by_area,
        "safe_limitation": SAFE_WORDING_DISCLAIMER,
    }

    mq_payload = {
        "important": disclaimer,
        "generated_at": datetime.datetime.now().isoformat(),
        "summary": {"queries": len(queries)},
        "management_query_list": queries,
        "safe_limitation": SAFE_WORDING_DISCLAIMER,
    }

    ev_path = out_dir / "evidence_request_list.json"
    mq_path = out_dir / "management_query_list.json"
    with atomic_write(ev_path) as tmp:
        Path(tmp).write_text(json.dumps(ev_payload, indent=2, default=str), encoding="utf-8")
    with atomic_write(mq_path) as tmp:
        Path(tmp).write_text(json.dumps(mq_payload, indent=2, default=str), encoding="utf-8")

    return {
        "execution_status": "SUCCESS",
        "pipeline_status": "SUCCESS",
        "can_continue": True,
        "message": (
            f"Request lists: {len(evidence_requests)} de-duplicated evidence request(s) across "
            f"{len(by_area)} audit area(s), {len(information_requests)} information request(s), "
            f"{len(queries)} management quer(ies) held separately from evidence."
        ),
        "artifacts": [str(ev_path.resolve()), str(mq_path.resolve())],
        "errors": [],
        "warnings": warnings,
    }


_SELF = {"run_log.json"}

def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()

@pipeline_tool("build_run_log", domain="reasoning")
def build_run_log(
    output_dir: str = None,
    source_file_path: str = None,
    tb_doc_id: str = None,
    session_id: str = None,
    **kwargs,
) -> dict:
    """Record this run's provenance: source file name and SHA-256, every artifact
    produced with its own hash and size, model and prompt versions, knowledge-pack and
    taxonomy versions, the tools that actually ran, and a PII scan result. Runs last so
    it records what executed rather than what was planned. Writes run_log.json.

    This is what makes a finding defensible in supervisory review -- without it,
    "which rule set produced this?" has no answer."""
    out_dir = resolve_output_dir(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    warnings = []

    # ── source file ───────────────────────────────────────────────────────────
    source = {"supplied": bool(source_file_path)}
    if source_file_path:
        sp = Path(source_file_path)
        if sp.exists():
            source.update({
                "file_name": sp.name,
                "sha256": _sha256(sp),
                "size_bytes": sp.stat().st_size,
                "modified_at": datetime.datetime.fromtimestamp(sp.stat().st_mtime).isoformat(),
            })
        else:
            source["error"] = f"Supplied source_file_path does not exist: {sp}"
            warnings.append(source["error"])
    else:
        warnings.append(
            "No source_file_path supplied -- the run log cannot record the input file's hash, "
            "which sec 16.2 requires. DB-sourced runs should pass tb_doc_id instead."
        )

    # ── artifacts produced ────────────────────────────────────────────────────
    artifacts = []
    for path in sorted(out_dir.iterdir()):
        if not path.is_file() or path.name in _SELF or path.name.startswith("."):
            continue
        try:
            artifacts.append({
                "file_name": path.name,
                "sha256": _sha256(path),
                "size_bytes": path.stat().st_size,
                "written_at": datetime.datetime.fromtimestamp(path.stat().st_mtime).isoformat(),
            })
        except OSError as e:  # noqa: PERF203 -- a locked/partial file must not fail the log
            warnings.append(f"Could not hash {path.name}: {e}")

    # ── which tools actually ran, inferred from what they wrote ───────────────
    artifact_names = {a["file_name"] for a in artifacts}
    tool_evidence = {
        "validate_layer1_tb": "layer1_results.json",
        "build_engagement_context": "engagement_context.json",
        "build_normalisation_note": "normalisation_note.json",
        "build_fsli_summary": "fsli_summary.parquet",
        "build_financial_snapshot": "financial_snapshot_statistics.json",
        "build_financial_ratios": "financial_ratios.json",
        "build_audit_ratio_pack": "audit_ratio_pack.json",
        "build_materiality": "materiality.json",
        "build_materiality_lens": "materiality_lens.json",
        "build_relationship_analytics": "relationship_analytics.json",
        "build_counterpart_screen": "counterpart_screen.json",
        "build_relationship_expectations": "relationship_expectations.json",
        "build_abnormal_sign_screen": "abnormal_sign_screen.json",
        "build_risk_indicators": "risk_indicators.json",
        "build_sensitive_detector": "sensitive_accounts.json",
        "build_anomaly_scanner": "anomaly_findings.json",
        "build_statutory_screen": "statutory_screen.json",
        "build_public_sector_lens": "public_sector_lens.json",
        "build_going_concern_screen": "going_concern_screen.json",
        "build_caro_indicators": "caro_indicators.json",
        "build_override_indicators": "override_indicators.json",
        "build_exception_consolidator": "consolidated_exceptions.json",
        "build_assertion_evidence_map": "assertion_evidence_map.json",
        "build_data_sufficiency_grade": "data_sufficiency.json",
        "build_finding_records": "finding_records.json",
        "build_request_lists": "evidence_request_list.json",
        "build_audit_reasoning": "audit_reasoning.json",
    }
    tools_run = sorted(t for t, a in tool_evidence.items() if a in artifact_names)
    tools_absent = sorted(t for t, a in tool_evidence.items() if a not in artifact_names)

    # ── PII scan over rendered narrative ──────────────────────────────────────
    # Reports masks at the render boundary (see _masking.py). Recording what was
    # PRESENT lets a reviewer distinguish "no identifiers found" from "masking never
    # ran" -- two very different assurances.
    pii = {}
    for name in ("finding_records.json", "evidence_request_list.json", "management_query_list.json"):
        p = out_dir / name
        if p.exists():
            found = find_identifiers(p.read_text(encoding="utf-8", errors="replace"))
            if found:
                pii[name] = found
    if pii:
        warnings.append(
            "Identifier-shaped strings detected in narrative artifacts -- confirm the report "
            f"writers applied _masking before distribution: {pii}"
        )

    ds = safe_load_json(out_dir / "data_sufficiency.json") if (out_dir / "data_sufficiency.json").exists() else {}

    log = {
        "purpose": (
            "Run provenance per spec sec 16.2. Records the input, every artifact with its hash, "
            "and the exact rule-set versions that produced them, so a finding can be traced and "
            "defended in supervisory review, and so sec 16.2's 're-run affected analyses when "
            "mappings change' is actionable."
        ),
        "run": {
            "run_timestamp": datetime.datetime.now().isoformat(),
            "session_id": session_id,
            "tb_doc_id": tb_doc_id,
            "output_dir": str(out_dir.resolve()),
            "python_version": platform.python_version(),
            "platform": platform.platform(),
        },
        "source_file": source,
        "versions": {
            "prompt_version": PROMPT_VERSION,
            "llm_model": settings.LLM_MODEL,
            "llm_base_url": settings.LLM_BASE_URL,
            "knowledge_packs": pack_versions(),
        },
        "data_sufficiency": {"grade": ds.get("grade"), "rationale": ds.get("rationale")} if ds else None,
        "tools_run": tools_run,
        "tools_not_run": tools_absent,
        "artifacts": artifacts,
        "pii_scan": pii or {"result": "no identifier-shaped strings found in narrative artifacts"},
        "retention_note": (
            "sec 16.2 also requires preserving the original files and the transformations made "
            "during normalisation, restricting access to the engagement team, and marking "
            "AI-generated observations in the working papers. Those are process controls outside "
            "this pipeline's enforcement -- this log evidences the versioning half only."
        ),
    }

    out_path = out_dir / "run_log.json"
    with atomic_write(out_path) as tmp:
        Path(tmp).write_text(json.dumps(log, indent=2, default=str), encoding="utf-8")

    return {
        "execution_status": "SUCCESS",
        "pipeline_status": "WARNING" if warnings else "SUCCESS",
        "can_continue": True,
        "message": (
            f"Run log: {len(artifacts)} artifact(s) hashed, {len(tools_run)} tool(s) evidenced as "
            f"run, prompt v{PROMPT_VERSION}, {len(log['versions']['knowledge_packs'])} rule-set "
            "version(s) recorded."
        ),
        "artifacts": [str(out_path.resolve())],
        "errors": [],
        "warnings": warnings,
    }


@pipeline_tool("validate_tb_pipeline", domain="reasoning")
def validate_tb_pipeline(output_dir: str = None, **kwargs) -> dict:
    """Check pipeline artifact completeness, financial (assets vs liabilities+equity)
    integrity, and reasoning readiness before the audit-reasoning stage is allowed to run."""
    errors = []
    warnings = []
    artifacts = []

    out_dir = resolve_output_dir(output_dir)
    if not out_dir.exists():
        errors.append({"type": "FileNotFoundError", "file": str(out_dir)})
        return {"execution_status": "FAILED", "errors": errors, "message": "Output directory not found."}

    def load_json(name):
        p = out_dir / name
        if p.exists():
            with open(p, "r") as f:
                return json.load(f)
        return None

    def load_pq(name):
        p = out_dir / name
        if p.exists():
            return pl.read_parquet(p)
        return pl.DataFrame()

    # 1. Pipeline Integrity
    # variance_analysis.json is intentionally excluded -- variance analysis requires a PY/CY
    # comparison basis and is disabled for Single-TB runs, so its absence is not a completeness
    # gap in that mode.
    # Filename -> the tool that produces it, so a FAILED integrity check tells the agent
    # exactly which tool to run next instead of just naming a missing file.
    # Every canonical_tb.parquet now comes from either load_tb_from_db (MAIN) or
    # ingest_tb_to_live (LIVE -- both live-template and uploaded documents go through
    # this single entry point) -- neither produces a separate processing_manifest.json/
    # grouping_ground_truth.parquet, so there is no longer an upload-vs-DB-sourced
    # distinction to make here.
    _PRODUCING_TOOL = {
        "canonical_tb.parquet": "load_tb_from_db or ingest_tb_to_live",
        "layer1_results.json": "validate_layer1_tb",
        "financial_snapshot.json": "build_financial_snapshot",
        "materiality.json": "build_materiality",
        "relationship_analytics.json": "build_relationship_analytics",
        "risk_indicators.json": "build_risk_indicators",
        "sensitive_accounts.json": "build_sensitive_detector",
        "consolidated_exceptions.json": "build_exception_consolidator",
    }

    expected_files = list(_PRODUCING_TOOL.keys())

    missing_files = [f for f in expected_files if not (out_dir / f).exists()]

    integrity_status = "PASS" if not missing_files else "FAILED"

    if missing_files:
        missing_with_hints = [f"{f} (run {_PRODUCING_TOOL[f]})" for f in missing_files]
        errors.append({"type": "PipelineIntegrityError", "message": f"Missing artifacts: {missing_with_hints}"})

    stats_data = load_json("financial_snapshot_statistics.json") or {}
    exc_pq = load_pq("consolidated_exceptions.parquet")

    # Financial Balance Validation -- sourced from financial_snapshot_statistics.json's
    # flat total_assets/total_liabilities/total_equity/total_revenue/total_expenses keys
    # (the actual shape build_financial_snapshot.py writes; financial_snapshot.json itself
    # is a node-tree with no "metric"/"balance" pairs, so it can't be read as a metrics list).
    #
    # Nets the current period's P&L into the equity side before comparing against assets,
    # rather than comparing Assets to Liabilities+Equity alone. Most real trial balances are
    # interim/unclosed -- retained earnings hasn't yet absorbed the current period's profit
    # or loss, so Equity is understated by exactly that amount until period-end closing
    # entries run. This isn't a closed-books assumption: for any internally-consistent TB,
    # Assets + Liabilities + Equity + Revenue + Expenses sums to exactly zero (it's the same
    # signed identity the whole TB's closing_balance column already sums to zero on) whether
    # or not the books have been closed, so comparing against that full sum is correct either
    # way -- it just also happens to make an unclosed TB's normal, non-error imbalance net out.
    fin_status = "UNKNOWN"
    fin_diff = 0.0
    if stats_data:
        total_assets = stats_data.get("total_assets", 0.0)
        # TB-R17: financial_snapshot_statistics.json's total_equity is now written as a
        # positive, real-world value (build_financial_snapshot negates the raw,
        # credit-normal closing-balance sign -- see fsli.py). total_liabilities/
        # total_revenue are still raw-signed (negative), so this identity -- which relies
        # on the raw signed sum netting to zero, same as TB-012's Assets+Liabilities+
        # Equity+Revenue+Expenses check -- must subtract the now-positive total_equity
        # back to its original negative contribution rather than add it.
        effective_liab_eq = (
            stats_data.get("total_liabilities", 0.0)
            - stats_data.get("total_equity", 0.0)
            + stats_data.get("total_revenue", 0.0)
            + stats_data.get("total_expenses", 0.0)
        )
        fin_diff = abs(total_assets + effective_liab_eq)
        fin_status = "PASS" if fin_diff < 1.0 else "FAILED"
        if fin_status == "FAILED":
            errors.append({"type": "FinancialIntegrityError", "message": f"Assets vs Liab&Eq(+period P&L) diff: {fin_diff}"})

    exceptions_checked = len(exc_pq) if not exc_pq.is_empty() else 0

    data_score = 100 if integrity_status == "PASS" else 50
    hierarchy_score = 100
    mapping_score = 100
    financial_score = 100 if fin_status == "PASS" else 40
    analytics_score = 100
    sensitivity_score = 100
    exception_score = 100 if exceptions_checked > 0 else 0
    reasoning_readiness_score = 100 if exceptions_checked > 0 else 0
    reporting_readiness_score = 100 if len(errors) == 0 else 0

    gov_score = (data_score + financial_score + exception_score + reasoning_readiness_score) / 4.0

    report_ready = (len(errors) == 0)
    reasoning_ready = (reasoning_readiness_score > 90 and report_ready)

    validation_results = []
    validation_results.append({
        "validation_type": "Pipeline Integrity",
        "severity": "Critical",
        "status": integrity_status,
        "message": f"Missing files: {missing_with_hints}" if missing_files else "All artifacts present",
    })
    validation_results.append({
        "validation_type": "Financial Integrity",
        "severity": "Critical",
        "status": fin_status,
        "message": f"Balance diff: {fin_diff}",
    })
    validation_results.append({
        "validation_type": "Reasoning Readiness",
        "severity": "High",
        "status": "PASS" if reasoning_ready else "FAILED",
        "message": f"Exceptions ready for LLM: {exceptions_checked}",
    })

    val_pq_df = pl.DataFrame(validation_results)
    val_pq_path = out_dir / "validation_results.parquet"
    write_parquet_atomic(val_pq_df, val_pq_path)
    artifacts.append(str(val_pq_path.resolve()))

    report_json = {
        "methodology": "Deterministic Pipeline Governance Validation",
        "pipeline_integrity": {
            "overall_status": integrity_status,
            "missing_outputs": missing_files,
            "invalid_versions": [],
            "dependency_failures": [],
        },
        "financial_validation": {
            "status": fin_status,
            "difference": fin_diff,
        },
        "quality_scores": {
            "data": data_score,
            "hierarchy": hierarchy_score,
            "mapping": mapping_score,
            "financial": financial_score,
            "analytics": analytics_score,
            "sensitivity": sensitivity_score,
            "exception": exception_score,
            "reasoning_readiness": reasoning_readiness_score,
            "reporting_readiness": reporting_readiness_score,
        },
        "reasoning_readiness": {
            "status": "LLM Ready" if reasoning_ready else "Not Ready",
            "confidence": reasoning_readiness_score,
        },
        "report_readiness": {
            "status": "READY" if report_ready else "BLOCKED",
            "blockers": missing_files,
            "warnings": [],
            "confidence": reporting_readiness_score,
        },
        "pipeline_health": {
            "governance_score": gov_score,
            "overall_health": "Healthy" if report_ready else "Unhealthy",
        },
        "downstream_metadata": {
            "validation_score": gov_score,
            "reasoning_ready": reasoning_ready,
            "report_ready": report_ready,
        },
        "errors": errors,
        "warnings": warnings,
    }

    rep_path = out_dir / "validation_report.json"
    write_json_atomic(report_json, rep_path, indent=4)
    artifacts.append(str(rep_path.resolve()))

    return {
        "execution_status": "SUCCESS" if report_ready else "FAILED",
        "pipeline_status": "SUCCESS" if report_ready else "FAILED",
        "message": "Pipeline validation passed." if report_ready else "Pipeline validation failed.",
        "artifacts": artifacts,
        "errors": errors,
        "warnings": warnings,
    }





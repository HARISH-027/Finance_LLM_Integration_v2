import datetime
import logging
from dataclasses import replace
from pathlib import Path
from typing import Optional

import polars as pl

from modes.trial_balance.pipeline.db import (
    db_cursor,
    fetch_live_document,
    fetch_live_lines,
    fetch_main_document,
    fetch_main_document_by_entity_fy,
    fetch_main_lines,
    list_main_documents,
    upsert_live_document_and_lines,
)
from modes.trial_balance.pipeline.db import list_priority_companies as _db_list_priority_companies
from modes.trial_balance.pipeline.tools._shared import *  # noqa: F401,F403
from modes.trial_balance.pipeline.tools.canonical_schema import *  # noqa: F401,F403
from modes.trial_balance.pipeline.tools.classify_document import (
    DataQualityConfirmationRequiredError,
    QualityGateFailedError,
    classify_tb_document,
)
from modes.trial_balance.pipeline.tools.document_metadata import FrameworkResolution
from modes.trial_balance.pipeline.tools.erp_adapter_integration import enrich_parsed_input_with_erp_adapter
from modes.trial_balance.pipeline.tools.erp_persistence import upsert_tb_erp_classification, upsert_tb_scope
from modes.trial_balance.pipeline.tools.input_dispatch import UnsupportedInputError, parse_tb_input
from modes.trial_balance.pipeline.tools.input_grouping_shape import GroupingShape
from modes.trial_balance.pipeline.tools.input_intake_checks import assess_ingestion_intake
from modes.trial_balance.pipeline.tools.input_legacy_formats import PasswordProtectedError, UnsupportedWorkbookFormatError
from modes.trial_balance.pipeline.tools.input_scenario_a import TemplateStructureError
from modes.trial_balance.pipeline.tools.input_scenario_b import DuplicateTbNoGroupingError, InputClassificationError
from modes.trial_balance.pipeline.tools.input_scenario_c import CombinedWorkbookStructureError
from modes.trial_balance.pipeline.tools.input_scenario_d import AmbiguousGroupingSourceError, AmbiguousTbYearError
from modes.trial_balance.pipeline.tools.pipeline_tool import *  # noqa: F401,F403
from modes.trial_balance.pipeline.tools.tb_erp_contracts import ERPType

logger = logging.getLogger(__name__)


@pipeline_tool("delete_db_document", domain="db_bridge")
def delete_db_document(tb_doc_id: str) -> dict:
    """Delete a Trial Balance document and its GL lines from LIVE staging by tb_doc_id.
    Only operates on live_document_table/live_tb_table; MAIN (document_table/tb_table)
    is never deleted from here."""
    with db_cursor(dict_rows=False) as cur:
        cur.execute("DELETE FROM live_tb_table WHERE tb_doc_id = %s", (tb_doc_id,))
        cur.execute("DELETE FROM live_document_table WHERE tb_doc_id = %s", (tb_doc_id,))
        deleted = cur.rowcount

    return {
        "execution_status": "SUCCESS",
        "pipeline_status": "SUCCESS",
        "message": f"Deleted live document {tb_doc_id} (document row removed: {bool(deleted)}).",
        "artifacts": [],
        "tb_doc_id": tb_doc_id,
    }


def _normalized_rows_to_canonical_dicts(tb_doc_id: str, rows: list, custom_fields: dict = None) -> list:
    """Projects the engine's NormalizedRow output onto the existing 16-column
    canonical shape (CANONICAL_TB_ALL_COLUMNS). custom_field_1/2/3 are
    caller-supplied passthrough (no source TB/grouping data maps to them) --
    same reserved/not-yet-defined passthrough contract build_canonical_tb's
    old company_details parameter offered."""
    custom_fields = custom_fields or {}
    return [
        {
            "tb_doc_id": tb_doc_id, "gl_code": r.gl_code, "gl_name": r.gl_name,
            "opening_balance": r.opening, "debit": r.debit, "credit": r.credit, "closing_balance": r.closing,
            "bs_pl": r.bs_pl, "sub_head_2": r.sub_head_2, "sub_head_1": r.sub_head_1, "main_head": r.main_head,
            "account_type": r.account_type, "mapped_status": r.mapped_status,
            "custom_field_1": custom_fields.get("custom_field_1"),
            "custom_field_2": custom_fields.get("custom_field_2"),
            "custom_field_3": custom_fields.get("custom_field_3"),
        }
        for r in rows
    ]


def _ingest_one_parsed_document(
    parsed, standard, llm_client, accept_data_quality_risk, out_dir, custom_fields=None,
    intake_warnings=None, metadata_overrides=None, persist_to_live=True, user_id=None,
    source_files=None, erp_type=None,
) -> dict:
    """Classifies and writes ONE ParsedInput (one fiscal year) to LIVE
    staging. Shared by the single-document and multi-year (Scenario D)
    paths below so there is exactly one place that does the classify ->
    write -> parquet sequence.

    `metadata_overrides` ({"company_name"/"cin"/"financial_year": value}) applies
    per-field, user-supplied corrections on top of the parser's own detected
    metadata -- each key only overrides when its value is non-empty, so a blank
    field leaves that piece of metadata exactly as the parser produced it. Applied
    BEFORE build_tb_doc_id() below, so the generated doc id reflects the correction
    too, not just the display fields in doc_row.

    `persist_to_live=False` runs the exact same classify/quality-gate/confirmation
    chain and still writes canonical_tb.parquet, but skips both the duplicate-
    document check and the LIVE write entirely -- used by the query-analysis
    staging path (session-only, nothing reaches live_document_table/live_tb_table).

    `source_files`/`erp_type`: TB Multi-ERP Adaptability wiring (Tally/SAP only --
    Oracle not built). `source_files` is the original input file list, only passed
    by ingest_tb_to_live when exactly one ParsedInput was produced (Scenario D's
    per-year file association isn't recoverable here, so multi-year documents are
    never enriched). Best-effort: enrich_parsed_input_with_erp_adapter never raises
    and returns None for anything that isn't Tally/SAP-shaped, leaving this function's
    behavior identical to before this wiring existed."""
    metadata = parsed.metadata
    if metadata_overrides:
        metadata = replace(metadata, **{k: v for k, v in metadata_overrides.items() if v})

    erp_enrichment = None
    if source_files is not None:
        erp_enrichment = enrich_parsed_input_with_erp_adapter(source_files, parsed, explicit_erp_type=erp_type)
        if erp_enrichment is not None:
            parsed = replace(parsed, grouping_hints=erp_enrichment.enriched_grouping_hints)

    tb_doc_id = build_tb_doc_id(
        company_name=metadata.company_name,
        fy_start=metadata.fy_period_start,
        fy_end=metadata.fy_period_end,
        financial_year=metadata.financial_year,
        fallback_bytes=(metadata.tb_doc_name or "unknown").encode("utf-8"),
    )

    try:
        result = classify_tb_document(
            parsed.tb_rows, parsed.grouping_hints, parsed.grouping_shape or GroupingShape.HIERARCHICAL_TRAIL,
            standard, llm_client, accept_data_quality_risk=accept_data_quality_risk,
        )
    except QualityGateFailedError as exc:
        return {
            "execution_status": "FAILED", "pipeline_status": "FAILED", "can_continue": False,
            "message": str(exc), "artifacts": [], "financial_year": metadata.financial_year,
        }
    except DataQualityConfirmationRequiredError as exc:
        return {
            "execution_status": "SUCCESS", "pipeline_status": "CONFIRMATION_REQUIRED", "can_continue": False,
            "message": exc.requirement.reason, "artifacts": [], "financial_year": metadata.financial_year,
            "confirmation": {
                "tier": exc.tier, "reason": exc.requirement.reason,
                "incomplete_count": exc.requirement.incomplete_count, "total_count": exc.requirement.total_count,
            },
        }

    canonical_dicts = _normalized_rows_to_canonical_dicts(tb_doc_id, result.rows, custom_fields)
    out_df = pl.DataFrame(canonical_dicts) if canonical_dicts else pl.DataFrame(schema={c: pl.Utf8 for c in CANONICAL_TB_ALL_COLUMNS})
    out_df = out_df.select(CANONICAL_TB_ALL_COLUMNS)

    doc_row = {
        "entity_id": metadata.entity_id,
        "entity_name": metadata.entity_name,
        "cin": metadata.cin,
        "company_name": metadata.company_name,
        "fy_period_start": metadata.fy_period_start,
        "fy_period_end": metadata.fy_period_end,
        "tb_doc_id": tb_doc_id,
        "tb_doc_name": metadata.tb_doc_name,
        "statement_type": metadata.statement_type,
        "financial_year": metadata.financial_year,
        "has_grouping": metadata.has_grouping,
        "grouping_doc_id": tb_doc_id,
        "grouping_doc_name": metadata.grouping_doc_name,
        "document_version": 1,
        "modification_dump": None,
        "user_id": user_id,
    }

    replaced_existing_document = None
    if persist_to_live:
        # DUPLICATE_UPLOAD (informational only -- upsert_live_document is delete-
        # then-insert by design, see its own docstring; this note does not block).
        existing_doc = fetch_live_document(tb_doc_id)
        replaced_existing_document = (
            {"tb_doc_id": tb_doc_id, "previous_tb_doc_name": existing_doc.get("tb_doc_name")}
            if existing_doc else None
        )
        _, rows_written = upsert_live_document_and_lines(doc_row, out_df.to_dicts())
        if erp_enrichment is not None:
            try:
                upsert_tb_scope(tb_doc_id, erp_enrichment.scope)
                upsert_tb_erp_classification(tb_doc_id, erp_enrichment.classification_by_gl_code)
            except Exception:
                logger.warning("ERP scope/classification persistence failed for %s; LIVE document still written", tb_doc_id, exc_info=True)
    else:
        rows_written = out_df.height

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "canonical_tb.parquet"
    write_parquet_atomic(out_df, out_path)

    message = (
        f"Ingested {rows_written} GL lines into LIVE staging as {tb_doc_id} "
        f"(quality tier {result.quality_tier.tier}, {result.metrics.mapped_pct}% mapped)."
        if persist_to_live else
        f"Classified {rows_written} GL lines as {tb_doc_id} (quality tier {result.quality_tier.tier}, "
        f"{result.metrics.mapped_pct}% mapped) -- staged in-session only, not written to LIVE."
    )

    return {
        "execution_status": "SUCCESS",
        "pipeline_status": "WARNING" if result.quality_gate_report.warnings else "SUCCESS",
        "message": message,
        "artifacts": [str(out_path)],
        "tb_doc_id": tb_doc_id,
        "financial_year": metadata.financial_year,
        "intake_warnings": (intake_warnings or []) + [
            {"code": "QUALITY_GATE", "severity": "WARNING", "message": w} for w in result.quality_gate_report.warnings
        ],
        "replaced_existing_document": replaced_existing_document,
        "persisted_to_live": persist_to_live,
        "erp_type": erp_enrichment.erp_type.value if erp_enrichment else None,
        "erp_findings": erp_enrichment.findings if erp_enrichment else [],
        "db_refs": (
            {"tb_doc_id": tb_doc_id, "table": "live_tb_table", "ingested_at": datetime.datetime.now(datetime.timezone.utc).isoformat()}
            if persist_to_live else None
        ),
        "quality_tier": result.quality_tier.tier,
        "run_metrics": {
            "total_rows": result.metrics.total_rows,
            "mapped_count": result.metrics.mapped_count,
            "unmapped_count": result.metrics.unmapped_count,
            "unmatched_count": result.metrics.unmatched_count,
            "mapped_pct": result.metrics.mapped_pct,
            "staged_narrowing_triggered_count": result.metrics.staged_narrowing_triggered_count,
            "repair_attempted_count": result.metrics.repair_attempted_count,
            "repair_succeeded_count": result.metrics.repair_succeeded_count,
        },
        "sanity_flags": {
            "keyword_conflict_count": len(result.sanity_flags.keyword_conflicts),
            "inconsistent_name_group_count": len(result.sanity_flags.inconsistent_name_groups),
        },
    }


@pipeline_tool("ingest_tb_to_live", domain="db_bridge")
def ingest_tb_to_live(
    tb_grouping_template_path: str,
    grouping_file_path: str = None,
    output_dir: str = None,
    standard: str = None,
    accept_data_quality_risk: bool = False,
    custom_fields: dict = None,
    company_name: str = None,
    cin: str = None,
    financial_year: str = None,
    persist_to_live: bool = True,
    user_id: str = None,
    erp_type: str = None,
) -> dict:
    """Ingest one or two client-submitted TB workbook(s) into LIVE staging
    (live_document_table/live_tb_table) and produce canonical_tb.parquet
    artifact(s). This is the single ingestion entry point for BOTH live/
    interactive TB submissions AND uploaded documents (the upload UI's
    /upload + /upload-mapped route this straight through) — both write to
    the same LIVE tables via the same native parsing/classification engine;
    they differ only in where the input files came from. Never promotes to
    MAIN — that happens in a separate process.

    Input format is auto-detected content-first, never by filename or a
    fixed template contract: pass just `tb_grouping_template_path` for a
    normalized TB_GROUPING_TEMPLATE.xlsx (Scenario A) or a single combined
    workbook with TB rows and grouping data in the same sheet (Scenario
    C); pass both `tb_grouping_template_path` and `grouping_file_path` for
    two separate TB + grouping files (Scenario B), in either order --
    which file is which is decided from content, not argument order. If
    either input packs more than one fiscal year across its sheets
    (Scenario D), every detected year is ingested as its own LIVE
    document; the response's "documents" list carries one entry per year.

    `standard` (AS/IND_AS) is auto-resolved when not given: an explicit
    COMPANY_STANDARDS value in a template's COMPANY_DETAILS sheet wins,
    else a vocabulary heuristic over the TB/grouping content. If neither
    resolves (or the file names more than one standard ambiguously), this
    returns FAILED rather than silently defaulting — pass `standard`
    explicitly to force it.

    `custom_fields` ({"custom_field_1"/"custom_field_2"/"custom_field_3": value})
    is reserved/not-yet-defined passthrough, written verbatim onto every
    output row when supplied; no business logic assigned here.

    `company_name`/`cin`/`financial_year` are optional user-supplied corrections on
    top of whatever the parser auto-detected from the workbook's own COMPANY_DETAILS
    sheet -- each applies independently (a blank one leaves that field exactly as
    parsed), and only for the ordinary one-document case (ignored, not failed, when
    the input resolves to more than one fiscal year -- Scenario D).

    `persist_to_live=False` runs the identical parse/classify/quality-gate chain
    and still writes canonical_tb.parquet, but never writes to LIVE (no dup-check,
    no live_document_table/live_tb_table row) -- for staging a TB for a downstream
    feature that only needs the canonical Parquet, not a durable LIVE document.

    Classification runs through the full native engine
    (backend/tools/classify_document.py and friends): quality gate ->
    data-quality tier -> confirmation gate -> taxonomy resolution (EXACT/
    ALIAS/CANDIDATE_AUTO/DETERMINISTIC_RULE) -> LLM classification
    (single-shot + staged narrowing for ambiguous vocabulary) -> the
    Validation Gate, which is the only code path that can mark a row
    MAPPED. A document needing a human data-quality confirmation returns
    pipeline_status "CONFIRMATION_REQUIRED" rather than writing anything
    to LIVE; pass accept_data_quality_risk=True to proceed anyway once a
    human has reviewed the reason.

    `user_id` (the authenticated caller's id, threaded in by router.py from
    Depends(require_user)) is recorded on the LIVE document row it creates --
    that's what GET/DELETE /documents/{doc_id} check ownership against. None
    is accepted (not required) since this tool is also callable deterministically
    without an HTTP request in front of it; router.py always supplies it.

    `erp_type` ("tally"/"sap" -- Oracle not yet built) opts into the TB Multi-ERP
    Adaptability upgrade's ERP-specific classification/scope extraction and
    scope-aware quality-gate findings (see erp_adapter_integration.py). Automatic
    detection also runs when this is omitted, but resolves to UNKNOWN for almost
    any real upload (it relies on the file's own path containing "SAP"/"Tally" as
    a segment, true for this project's own test fixtures, essentially never true
    for a real upload's temp path) -- so in practice this is currently opt-in only.
    Internal/pipeline-layer only for now: not yet exposed through router.py's
    /upload-mapped request model. Ignored (falls back to auto-detection) for an
    unrecognized value, and for any input that resolves to more than one fiscal
    year (Scenario D) -- there's no way to recover which source file backs which
    year's document at this layer, so multi-year documents are never enriched.
    Never blocks or fails ingestion: on any adapter/detection error, ingestion
    proceeds exactly as it would without this parameter."""
    input_files = [Path(tb_grouping_template_path)]
    if grouping_file_path:
        input_files.append(Path(grouping_file_path))
    missing = [str(p) for p in input_files if not p.exists()]
    if missing:
        raise PipelineFileError(", ".join(missing), "TB/Grouping input file")

    try:
        parsed_inputs = parse_tb_input(input_files, framework=standard)
    except (
        TemplateStructureError, CombinedWorkbookStructureError, InputClassificationError,
        DuplicateTbNoGroupingError, AmbiguousGroupingSourceError, AmbiguousTbYearError, UnsupportedInputError,
        # Ingestion error catalog Section 1 (file/upload level) -- previously
        # fell through to the pipeline_tool decorator's generic redacted-
        # exception handling, losing the friendly message these already carry.
        PasswordProtectedError, UnsupportedWorkbookFormatError,
    ) as exc:
        return {
            "execution_status": "FAILED", "pipeline_status": "FAILED", "can_continue": False,
            "message": f"{type(exc).__name__}: {exc}", "artifacts": [],
        }

    unresolved = [
        p for p in parsed_inputs
        if p.metadata.framework is None or p.metadata.framework_resolution in (
            FrameworkResolution.AMBIGUOUS, FrameworkResolution.UNRESOLVED,
        )
    ]
    if unresolved:
        return {
            "execution_status": "FAILED", "pipeline_status": "FAILED", "can_continue": False,
            "message": (
                "Could not resolve the accounting standard (AS vs IND_AS) for "
                f"{[p.metadata.financial_year or p.metadata.tb_doc_name for p in unresolved]} -- "
                "pass standard=\"AS\" or standard=\"IND_AS\" explicitly."
            ),
            "artifacts": [],
        }

    from modes.trial_balance.pipeline.agent import get_agent

    llm_client = get_agent().llm_client
    base_out_dir = resolve_output_dir(output_dir)
    multi_year = len(parsed_inputs) > 1

    erp_type_enum = None
    if erp_type:
        try:
            erp_type_enum = ERPType(erp_type.strip().lower())
        except ValueError:
            erp_type_enum = None  # unrecognized value -- falls back to auto-detection, never raises

    # A single explicit company_name/cin/financial_year override doesn't make sense
    # forced onto several distinct fiscal-year documents detected from one file
    # (Scenario D) -- only applied for the ordinary one-document case. Accounting
    # standard has no such restriction (passed to parse_tb_input above, uniformly).
    metadata_overrides = (
        {"company_name": company_name, "cin": cin, "financial_year": financial_year}
        if not multi_year else None
    )

    # Ingestion error catalog Sections 3/4 (sheet/header/formula-error
    # WARNINGs) -- one best-effort re-scan per input file, shared across
    # every fiscal-year document parsed from it (Scenario D can produce
    # several documents from the same file).
    intake_warnings = [w for f in input_files for w in assess_ingestion_intake(str(f))]

    documents = [
        _ingest_one_parsed_document(
            parsed, parsed.metadata.framework, llm_client, accept_data_quality_risk,
            # Multi-year: one subdirectory per detected fiscal year, so
            # each document's canonical_tb.parquet doesn't overwrite the
            # next -- single-document case keeps the original flat path
            # unchanged for backward compatibility.
            (base_out_dir / parsed.metadata.financial_year) if multi_year else base_out_dir,
            custom_fields=custom_fields,
            intake_warnings=intake_warnings,
            metadata_overrides=metadata_overrides,
            persist_to_live=persist_to_live,
            user_id=user_id,
            source_files=(input_files if not multi_year else None),
            erp_type=erp_type_enum,
        )
        for parsed in parsed_inputs
    ]

    if not multi_year:
        return documents[0]

    all_succeeded = all(d["execution_status"] == "SUCCESS" and d["pipeline_status"] not in ("FAILED", "CONFIRMATION_REQUIRED") for d in documents)
    return {
        "execution_status": "SUCCESS",
        "pipeline_status": "SUCCESS" if all_succeeded else "WARNING",
        "message": f"Detected {len(documents)} fiscal year(s) across the supplied input; ingested each as its own LIVE document.",
        "artifacts": [a for d in documents for a in d.get("artifacts", [])],
        "documents": documents,
    }


@pipeline_tool("list_db_documents", domain="db_bridge")
def list_db_documents(entity_id: Optional[str] = None, financial_year: Optional[str] = None) -> dict:
    """List Trial Balance documents already ingested into the MAIN database, optionally filtered
    by entity_id and/or financial_year. Use this when the user wants to analyze an existing
    document instead of uploading a new file."""
    docs = list_main_documents(entity_id=entity_id, financial_year=financial_year)
    return {
        "execution_status": "SUCCESS",
        "pipeline_status": "SUCCESS",
        "message": f"Found {len(docs)} document(s) in MAIN.",
        "artifacts": [],
        "documents": docs,
    }


@pipeline_tool("list_priority_companies", domain="db_bridge")
def list_priority_companies_tool() -> dict:
    """List every company in the priority_companies suggestion table (company_name/
    cin/financial_years) -- backs the upload picker's optional Company Details
    fields' autocomplete. Not MAIN/LIVE data; a small standalone reference table
    seeded from the client's Priority Companies List and grown from user-supplied
    overrides (see modes.trial_balance.pipeline.db.learn_priority_company)."""
    companies = _db_list_priority_companies()
    return {
        "execution_status": "SUCCESS",
        "pipeline_status": "SUCCESS",
        "message": f"Found {len(companies)} compan{'y' if len(companies) == 1 else 'ies'}.",
        "artifacts": [],
        "companies": companies,
    }


def _write_canonical_from_document_and_lines(
    doc: dict, lines: list, output_filename: str, output_dir: str, source_label: str,
) -> dict:
    """Shared by load_tb_from_db and load_tb_from_live: both resolve a document
    + its GL lines from wherever they live (MAIN vs LIVE), then need the exact
    same canonical_tb.parquet + tb_metadata.json write -- one implementation
    so the artifact shape/contract can't drift between the two sources."""
    resolved_doc_id = doc["tb_doc_id"]
    df = pl.DataFrame(lines)
    missing_cols = [c for c in CANONICAL_TB_ALL_COLUMNS if c not in df.columns]
    if missing_cols:
        df = df.with_columns([pl.lit(None).alias(c) for c in missing_cols])
    df = df.select(CANONICAL_TB_ALL_COLUMNS)

    out_dir = resolve_output_dir(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / output_filename
    write_parquet_atomic(df, out_path)

    artifacts = [str(out_path)]

    # tb_metadata.json is what every report builder (build_excel_report/build_docx_report/
    # build_report_markdown) reads for the entity/FY shown in report titles and headers --
    # no tool in this codebase writes it, so DB-sourced runs previously always fell back to
    # "Unknown Source". company_name/entity_name/financial_year here are the real document
    # fields, which the report builders prefer directly over their filename-regex fallback
    # (see _resolve_entity_and_fy in each report builder).
    tb_metadata_path = out_dir / "tb_metadata.json"
    write_json_atomic(
        {
            "source_file": doc.get("tb_doc_name") or resolved_doc_id,
            "company_name": doc.get("company_name"),
            "entity_name": doc.get("entity_name"),
            "financial_year": doc.get("financial_year"),
            "cin": doc.get("cin"),
        },
        tb_metadata_path,
        indent=4,
        default=str,
    )
    artifacts.append(str(tb_metadata_path))

    return {
        "execution_status": "SUCCESS",
        "pipeline_status": "SUCCESS",
        "message": f"Loaded {len(df)} GL lines for {resolved_doc_id} from {source_label}.",
        "artifacts": artifacts,
        "document": doc,
        "tb_doc_id": resolved_doc_id,
    }


@pipeline_tool("load_tb_from_db", domain="db_bridge")
def load_tb_from_db(
    tb_doc_id: str = None,
    entity_id: str = None,
    financial_year: str = None,
    output_filename: str = "canonical_tb.parquet",
    output_dir: str = None,
) -> dict:
    """Load a Trial Balance from the MAIN database (document_table/tb_table) into a canonical
    Parquet artifact, identified either by tb_doc_id directly, or by entity_id + financial_year
    (used for prior-year comparison lookups). Read-only against MAIN; never touches staging/live.
    Use load_tb_from_live for a document that only exists in LIVE staging (e.g. a document
    freshly ingested via ingest_tb_to_live this session, not yet promoted to MAIN)."""
    if not tb_doc_id and not (entity_id and financial_year):
        raise PipelineDBError("Must provide either tb_doc_id, or both entity_id and financial_year.")

    doc = fetch_main_document(tb_doc_id) if tb_doc_id else fetch_main_document_by_entity_fy(entity_id, financial_year)
    if not doc:
        return {
            "execution_status": "FAILED",
            "pipeline_status": "FAILED",
            "can_continue": False,
            "message": f"No MAIN document found for tb_doc_id={tb_doc_id!r} entity_id={entity_id!r} fy={financial_year!r}",
            "artifacts": [],
        }

    resolved_doc_id = doc["tb_doc_id"]
    lines = fetch_main_lines(resolved_doc_id)
    if not lines:
        return {
            "execution_status": "FAILED",
            "pipeline_status": "FAILED",
            "can_continue": False,
            "message": f"MAIN document {resolved_doc_id} has no GL lines in tb_table.",
            "artifacts": [],
        }

    return _write_canonical_from_document_and_lines(doc, lines, output_filename, output_dir, "MAIN")


@pipeline_tool("load_tb_from_live", domain="db_bridge")
def load_tb_from_live(
    tb_doc_id: str,
    output_filename: str = "canonical_tb.parquet",
    output_dir: str = None,
) -> dict:
    """Load a Trial Balance from LIVE staging (live_document_table/live_tb_table) into a
    canonical Parquet artifact, identified by tb_doc_id. Read-only against LIVE; never
    touches MAIN. This is the LIVE counterpart to load_tb_from_db -- use it for a document
    ingest_tb_to_live just wrote this session that hasn't been promoted to MAIN (promotion
    is a separate, out-of-scope process), which load_tb_from_db can never resolve since it
    only ever reads MAIN."""
    doc = fetch_live_document(tb_doc_id)
    if not doc:
        return {
            "execution_status": "FAILED",
            "pipeline_status": "FAILED",
            "can_continue": False,
            "message": f"No LIVE document found for tb_doc_id={tb_doc_id!r}",
            "artifacts": [],
        }

    lines = fetch_live_lines(tb_doc_id)
    if not lines:
        return {
            "execution_status": "FAILED",
            "pipeline_status": "FAILED",
            "can_continue": False,
            "message": f"LIVE document {tb_doc_id} has no GL lines in live_tb_table.",
            "artifacts": [],
        }

    return _write_canonical_from_document_and_lines(doc, lines, output_filename, output_dir, "LIVE")







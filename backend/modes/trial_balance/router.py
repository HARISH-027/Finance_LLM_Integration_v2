"""TB-v2 mode router — implements the mode-gateway contract the existing
finance_llm_integrated_v1 frontend expects (see frontend/src/api/client.js).
Every route is thin: parse input, call one tool directly (deterministic
lookups) or hand off to the agent (multi-step reasoning), return JSON.

Routes below call tools (ingest_tb_to_live, load_tb_from_db, build_*, etc.) via
`call_tool()`, which raises ToolNotAvailableError with a clear 424 response if a
module isn't registered — this file does not hardcode tool signatures beyond what
PROMPT.md documents.

Note: frontend/src/api/client.js also calls `/pdfs/health`, `POST /pdfs`, and
`GET/DELETE /pdfs` -- a separate PDF-evidence/embedding sub-feature with its
own database and embedding endpoint, not part of the Trial Balance pipeline
this router implements. Deliberately not built here; flagged so it isn't
rediscovered as a missing-endpoint gap later.
"""

import json
import logging
import re
import uuid
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from app.auth.deps import CurrentUser, require_user
from modes.trial_balance.pipeline.agent import ToolNotAvailableError, call_tool, get_agent, llm_reachable
from modes.trial_balance.pipeline.agent_memory import invoke_scoped
from modes.trial_balance.pipeline.config import settings
from modes.trial_balance.pipeline.db import add_findings, append_turns, create_session, delete_conversation, fetch_live_document, fetch_main_document, find_latest_session, get_messages, history_for_agent, learn_priority_company, list_conversations, new_conversation_id, update_session_status
from modes.trial_balance.pipeline.tools import PipelineDBError, PipelineFileError, resolve_output_dir, verify_packs
from modes.trial_balance.pipeline.valkey_client import preview_store_get, preview_store_set, preview_store_update

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/trial-balance", tags=["trial-balance"])

# Standalone TB-v2 calls this from its own main.py, which this gateway doesn't
# use -- module load here is the equivalent startup-time hook, verifying every
# versioned knowledge pack (taxonomy/keyword/weight JSON) is well-formed before
# the first request rather than failing deep inside a tool call.
verify_packs()

_UPLOAD_ROOT = settings.OUTPUT_DIR / "uploads"
_UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)

# Token -> upload metadata, backing /preview and /upload-mapped. Backed by Valkey
# (backend/valkey_client.py's preview_store_*, 30-min TTL) rather than an
# in-process dict -- the prior in-process store didn't survive a restart and
# broke under any multi-worker deployment.


def _tool_error_response(exc: Exception):
    if isinstance(exc, ToolNotAvailableError):
        return JSONResponse(status_code=424, content={"detail": str(exc)})
    if isinstance(exc, (PipelineFileError, PipelineDBError)):
        # Both carry deliberately-constructed, caller-safe messages (a labelled missing
        # artifact, a summarised DB failure) -- these are meant to be actionable to the
        # caller, so they pass through unchanged.
        return JSONResponse(status_code=400, content={"detail": str(exc)})
    # Anything else is an unclassified internal failure. str(exc) on those routinely
    # carries infrastructure detail (DB DSNs with host:port, LLM endpoint URLs, absolute
    # server paths), so it goes to the log and the client gets a correlation id instead.
    error_id = uuid.uuid4().hex[:12]
    logger.exception("Unhandled tool error (error_id=%s)", error_id)
    return JSONResponse(
        status_code=500,
        content={
            "detail": "Internal server error. Quote the error_id when reporting this.",
            "error_id": error_id,
        },
    )


def _verify_document_access(tb_doc_id: Optional[str], user: CurrentUser) -> None:
    """MAIN documents are shared across every authenticated user (see list_documents'
    docstring); LIVE documents are private to whoever created them. A tb_doc_id can
    exist in BOTH tables at once -- every document promoted to MAIN before this
    isolation feature existed still has its original LIVE row too, now an orphan
    with no recorded owner (confirmed live: 7 of 40 MAIN documents in this
    environment). MAIN is checked FIRST and always wins when present: a document
    visible in the firm's shared, already-promoted corpus must stay usable
    regardless of what its old LIVE staging row looks like -- checking LIVE first
    would 404 every one of those as "owned by nobody," blocking a real document
    every authenticated user can already see via GET /documents.

    Only falls through to the LIVE-ownership check for a tb_doc_id that is NOT in
    MAIN at all -- i.e. someone's own in-progress, not-yet-promoted upload. Raises
    404 (never 403) if that LIVE document belongs to someone else, or has no
    recorded owner (an orphan, invisible to everyone rather than reassigned to
    whoever asks first). A tb_doc_id in neither table passes through unchanged --
    a natural "not found" from whatever tool call runs next.

    NOT used by delete_document: deletion only ever touches LIVE (never MAIN,
    see its own docstring), so it needs its own strict, LIVE-only ownership
    check -- the MAIN shortcut here must never let a MAIN document's existence
    authorize deleting someone else's (or an orphaned) LIVE row."""
    if not tb_doc_id:
        return
    if fetch_main_document(tb_doc_id):
        return
    live_doc = fetch_live_document(tb_doc_id)
    if live_doc and live_doc.get("user_id") != user.user_id:
        raise HTTPException(status_code=404, detail="Document not found.")


# ── Request models ───────────────────────────────────────────────────────────

class UploadMappedRequest(BaseModel):
    token: str
    # No longer used server-side: ingest_tb_to_live's native parser auto-detects TB
    # columns (backend/tools/input_header_detect.py), so there is no manual mapping
    # step any more. Kept optional, accepted-but-ignored, so an existing client that
    # still sends it doesn't break.
    column_mapping: Optional[dict] = None
    # Token returned by /audit/upload-grouping for the matching grouping workbook.
    # Genuinely optional: ingest_tb_to_live can parse a single self-contained
    # workbook (a normalized TB_GROUPING_TEMPLATE.xlsx, or any combined TB+grouping
    # sheet) with no second file at all -- the same way a Live template submission
    # can. Only a two-separate-files upload (TB + a distinct grouping/chart-of-
    # accounts workbook) needs this.
    grouping_token: Optional[str] = None
    # Explicit opt-in to proceed past a CASE_3/high-incomplete-CASE_2
    # CONFIRMATION_REQUIRED result (see ingest_tb_to_live's own docstring).
    # Defaults False so a first attempt always stops for review; the picker
    # re-submits with this set True only after the user clicks "Proceed
    # anyway" on the surfaced confirmation reason.
    accept_data_quality_risk: bool = False
    # Optional user-supplied corrections from the picker's Company Details
    # fields (Upload new tab only) -- each applies independently on top of
    # ingest_tb_to_live's own auto-parse/auto-detect; a blank field changes
    # nothing. See ingest_tb_to_live's docstring for the multi-year caveat.
    company_name: Optional[str] = None
    cin: Optional[str] = None
    financial_year: Optional[str] = None
    standard: Optional[str] = None
    # Optional ERP source of the uploaded TB ("tally"/"sap"/"oracle"); blank
    # keeps ingest_tb_to_live's auto-detection. Opts the document into that
    # ERP's adapter (e.g. Tally: Dr/Cr re-signing, Group/Ledger split,
    # main_head-only classification).
    erp_type: Optional[str] = None
    # "SFS" (standalone) or "CFS" (consolidated); blank keeps whatever the file
    # itself declares (STATEMENT_TYPE) or leaves it unset.
    statement_type: Optional[str] = None
    # False for the query-analysis staging flow (TbRunPicker's 'query' mode) --
    # runs the identical classify/quality-gate chain and still writes
    # canonical_tb.parquet, but never writes to LIVE. True (default) is the
    # ordinary upload-and-audit path, unchanged.
    persist_to_live: bool = True


class AskRequest(BaseModel):
    doc_id: Optional[str] = None
    question: str
    # NOTE: session_id used to be declared here and doubled as the Valkey
    # conversation-memory key -- removed rather than left as a dead field
    # (same "declared but never read" cleanup this file already did once for
    # upload_doc_ids) now that conversation_id below is the single id driving
    # both the Valkey cache key and the durable Postgres record. Pydantic's
    # default extra="ignore" means an old client still sending session_id
    # keeps working, it's just silently unused.
    #
    # Which durable, 90-day-retained conversation this turn belongs to (see
    # pipeline/db.py's pipeline_chat_messages). Optional and defaulted, same
    # contract as the platform's own QueryRequest.conversation_id: an existing
    # caller sending only `question` keeps working, and the server starts a
    # new conversation when it's absent. Prior turns are read from the
    # database against the AUTHENTICATED user, never trusted from the client.
    conversation_id: Optional[str] = None


# NOTE: `upload_doc_ids` was declared on this model, AuditRequest and
# AuditWorkbookRequest but never read by any handler -- accepted by the schema and
# silently discarded, which reads to a caller as a supported feature. Removed rather
# than wired up: nothing in the pipeline consumes a list of upload doc ids today.
# Pydantic's default extra="ignore" means a client still sending the field is
# unaffected; it is simply no longer advertised in the OpenAPI schema.
class AskGeneralRequest(BaseModel):
    question: str
    conversation_id: Optional[str] = None


class AuditRequest(BaseModel):
    doc_id: str
    doc_id_prior: Optional[str] = None
    entity: Optional[str] = None
    engagement_context: Optional[str] = None
    framework: Optional[str] = None
    grouping_token: Optional[str] = None
    # Same contract as AskRequest.conversation_id above -- the frontend never sends this
    # today (each audit run always starts its own fresh conversation), kept as a real
    # optional field for the same forward-compatible reason /ask has it.
    conversation_id: Optional[str] = None
    # Display-only, never read for identity/authorization/routing -- this handler has no
    # cheap way to resolve doc_id to a filename itself; the frontend already has it in
    # hand (state.documents), same trust level as UploadMappedRequest's company_name/cin.
    doc_label: Optional[str] = None


class ValidateRequest(BaseModel):
    doc_id: str
    doc_id_prior: Optional[str] = None


class AuditWorkbookRequest(BaseModel):
    doc_id: str
    doc_id_prior: Optional[str] = None
    format: Optional[str] = None  # "xlsx" | "docx", default "xlsx"


# ── Health ───────────────────────────────────────────────────────────────────

@router.get("/health")
def health():
    from modes.trial_balance.pipeline.health_checks import check_storage_health

    storage = check_storage_health()
    # Postgres is the only hard dependency among the four storage engines --
    # Valkey/MinIO are designed to degrade gracefully to a no-op when down
    # (see valkey_client.py/object_store.py's module docstrings), and DuckDB
    # is in-process with no external service to be down. A Postgres outage is
    # the only one of the four that should read as the app itself being
    # unhealthy, not just running in a degraded mode.
    postgres_ok = storage["postgres"]["status"] == "ok"
    return {
        "status": "ok" if postgres_ok else "degraded",
        "mode": "trial-balance",
        "available": True,
        "storage": storage,
    }


# ── Upload / preview / mapped-upload ────────────────────────────────────────

@router.post("/upload")
async def upload(file: UploadFile = File(...)):
    token = str(uuid.uuid4())
    dest_dir = _UPLOAD_ROOT / token
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = dest_dir / file.filename
    with open(dest_path, "wb") as f:
        f.write(await file.read())

    preview_store_set(token, {"file_path": str(dest_path), "filename": file.filename})

    try:
        result = call_tool("preview_excel_data", excel_path=str(dest_path))
    except ToolNotAvailableError as exc:
        return _tool_error_response(exc)

    if result.get("execution_status") != "SUCCESS":
        raise HTTPException(status_code=422, detail={"needs_mapping": True, "preview_token": token, **result})

    preview_store_update(token, {"preview": result})
    return {"token": token, "filename": file.filename, **result}


@router.get("/preview")
def preview(token: str):
    entry = preview_store_get(token)
    if not entry:
        raise HTTPException(status_code=404, detail="Unknown or expired preview token.")
    return entry.get("preview", entry)


@router.post("/upload-mapped")
def upload_mapped(request: UploadMappedRequest, user: CurrentUser = Depends(require_user)):
    entry = preview_store_get(request.token)
    if not entry:
        raise HTTPException(status_code=404, detail="Unknown or expired preview token.")

    # grouping_token is optional -- a self-contained workbook (a normalized
    # TB_GROUPING_TEMPLATE.xlsx, Scenario A, or any combined TB+grouping sheet,
    # Scenario C) needs no second file; ingest_tb_to_live auto-detects which shape
    # it got from content alone. Only resolve a grouping file when one was actually
    # supplied.
    grouping_file_path = None
    if request.grouping_token:
        grouping_entry = preview_store_get(request.grouping_token)
        if not grouping_entry:
            raise HTTPException(status_code=404, detail="Unknown or expired grouping token.")
        grouping_file_path = grouping_entry["file_path"]

    # resolve_output_dir() anchors a RELATIVE path to the repo root, so passing the bare
    # token wrote every upload's parquet/JSON artifacts to <repo_root>/<uuid>/ -- one
    # stray UUID directory at the project root per upload, outside the configured
    # OUTPUT_DIR entirely. The uploaded source files already live in
    # OUTPUT_DIR/uploads/<token> (see _UPLOAD_ROOT above); the artifacts derived from
    # them belong beside their source, which is also how /audit scopes its own run
    # directory (sessions/<session_id>).
    #
    # persist_to_live=False is the query-analysis staging path: nothing reaches
    # LIVE, so there's no reason to key its output off the upload token -- it gets
    # a real pipeline_sessions row (mode="CHAT_QUERY", the same mode /ask and
    # /ask-general already use for this feature area) under sessions/<session_id>,
    # the same convention /audit uses, so a future query feature can find it.
    query_session_id = None
    if request.persist_to_live:
        output_dir = str(_UPLOAD_ROOT / request.token)
    else:
        query_session_id = create_session(mode="CHAT_QUERY", source="chat_upload", user_id=user.user_id)
        output_dir = str(resolve_output_dir(f"sessions/{query_session_id}"))

    try:
        result = call_tool(
            "ingest_tb_to_live",
            tb_grouping_template_path=entry["file_path"],
            grouping_file_path=grouping_file_path,
            output_dir=output_dir,
            accept_data_quality_risk=request.accept_data_quality_risk,
            company_name=request.company_name,
            cin=request.cin,
            financial_year=request.financial_year,
            standard=request.standard,
            erp_type=request.erp_type or None,
            statement_type=request.statement_type or None,
            persist_to_live=request.persist_to_live,
            user_id=user.user_id,
        )
    except (ToolNotAvailableError, PipelineFileError, PipelineDBError) as exc:
        if query_session_id:
            update_session_status(query_session_id, "FAILED", error_message=str(exc))
        return _tool_error_response(exc)

    if query_session_id:
        # CONFIRMATION_REQUIRED writes nothing (no canonical_tb.parquet either --
        # see ingest_tb_to_live's own early-return, before the write) -- not a
        # real completion, so this attempt's session is marked FAILED same as a
        # genuine failure; "Proceed anyway" creates a fresh session on retry
        # rather than resurrecting this one, same one-session-per-attempt model
        # /audit already uses.
        succeeded = result.get("execution_status") == "SUCCESS" and result.get("pipeline_status") in ("SUCCESS", "WARNING")
        update_session_status(
            query_session_id,
            "SUCCESS" if succeeded else "FAILED",
            error_message=None if succeeded else result.get("message"),
        )
        result["session_id"] = query_session_id

    preview_store_update(request.token, {"processed": result})

    # Grow the Company Details suggestion list from this explicit user override --
    # never from ingest_tb_to_live's own auto-detected value, only from what the
    # user actually typed. Best-effort: a failure here must never fail the upload
    # itself, matching this route's existing graceful-degradation style.
    if request.company_name and result.get("execution_status") == "SUCCESS":
        try:
            learn_priority_company(request.company_name, cin=request.cin, financial_year=request.financial_year)
        except PipelineDBError:
            logger.exception("learn_priority_company failed for %r -- upload result unaffected.", request.company_name)

    def _attach_validation(doc: dict) -> dict:
        canonical_tb_file = next((a for a in doc.get("artifacts", []) if a.endswith("canonical_tb.parquet")), None)
        if doc.get("execution_status") != "SUCCESS" or not canonical_tb_file:
            return doc
        try:
            validation = call_tool(
                "validate_layer1_tb", canonical_tb_file=canonical_tb_file, output_dir=str(Path(canonical_tb_file).parent),
            )
        except (ToolNotAvailableError, PipelineFileError, PipelineDBError) as exc:
            validation = _tool_error_response(exc)
        doc["layer1_validation"] = validation
        return doc

    # Scenario D (multi-year input) ingests every detected fiscal year as its own LIVE
    # document -- each carries its own canonical_tb.parquet, so each needs its own
    # Layer 1 validation rather than the single top-level one the old single-document-
    # only upload path always returned.
    if "documents" in result:
        result["documents"] = [_attach_validation(doc) for doc in result["documents"]]
    else:
        result = _attach_validation(result)

    preview_store_update(request.token, {"layer1_validation": result.get("layer1_validation")})
    return result


# ── Priority companies (suggestion list, not MAIN/LIVE) ─────────────────────

@router.get("/companies/priority")
def list_priority_companies():
    """Backs the upload picker's optional Company Details fields' autocomplete.
    No query params -- small enough (order-of-hundreds rows) to fetch in full and
    filter client-side, same as /documents already does for TbRunPicker's own
    company/period pickers."""
    try:
        result = call_tool("list_priority_companies")
    except ToolNotAvailableError as exc:
        return _tool_error_response(exc)
    return result


# ── Documents (MAIN, read-only) ─────────────────────────────────────────────

@router.get("/documents")
def list_documents(
    entity_id: Optional[str] = None, financial_year: Optional[str] = None,
    user: CurrentUser = Depends(require_user),
):
    """Lists MAIN documents -- deliberately shared across every authenticated user,
    not scoped by caller. MAIN is populated by a separate promotion process this
    codebase doesn't own and never records an uploader, so it's treated as the
    firm's shared, already-finalized reference corpus rather than per-user data
    (unlike LIVE documents below, which TB's own code creates and does scope)."""
    try:
        result = call_tool("list_db_documents", entity_id=entity_id, financial_year=financial_year)
    except ToolNotAvailableError as exc:
        return _tool_error_response(exc)
    return result


@router.get("/documents/{doc_id}")
def get_document(doc_id: str, user: CurrentUser = Depends(require_user)):
    """Reads from MAIN (load_tb_from_db) -- shared, see list_documents' docstring."""
    try:
        result = call_tool("load_tb_from_db", tb_doc_id=doc_id)
    except ToolNotAvailableError as exc:
        return _tool_error_response(exc)
    if result.get("execution_status") != "SUCCESS":
        raise HTTPException(status_code=404, detail=result.get("message", "Document not found."))

    canonical_tb_file = (result.get("artifacts") or [None])[0]
    if canonical_tb_file:
        try:
            validation = call_tool(
                "validate_layer1_tb",
                canonical_tb_file=canonical_tb_file,
                output_dir=str(Path(canonical_tb_file).parent),
            )
        except ToolNotAvailableError as exc:
            validation = _tool_error_response(exc)
        result["layer1_validation"] = validation

    return result


@router.delete("/documents/{doc_id}")
def delete_document(doc_id: str, user: CurrentUser = Depends(require_user)):
    """delete_db_document only ever touches LIVE tables (never MAIN, never deletes
    it -- see pipeline/db.py's module docstring), so this is exactly the kind of
    TB-own-data write full isolation applies to. Deliberately does NOT call
    _verify_document_access -- that helper lets a MAIN document's existence pass
    a caller through regardless of LIVE ownership, which is correct for read/
    analyze operations but would be a real bug here: it must never let someone
    delete another user's (or an orphaned) LIVE row just because a MAIN row with
    the same tb_doc_id happens to exist. This check is LIVE-only, strict, no
    MAIN shortcut."""
    owner = fetch_live_document(doc_id)
    if not owner or owner.get("user_id") != user.user_id:
        raise HTTPException(status_code=404, detail="Document not found.")
    try:
        result = call_tool("delete_db_document", tb_doc_id=doc_id)
    except ToolNotAvailableError as exc:
        return _tool_error_response(exc)
    return result


# ── Chat ─────────────────────────────────────────────────────────────────────

_LLM_UNAVAILABLE_ANSWER = (
    "The assistant is temporarily unavailable. Your trial balance data and any "
    "completed audit reports are unaffected -- please retry shortly."
)


def _user_scoped_memory_key(user_id: str, raw_session_id: str) -> str:
    """agentchat:{session_id} in valkey_client.py was keyed only by this raw,
    client-supplied-or-echoed string -- unverified against any authenticated
    identity, so a caller who learned/guessed another session's id could
    resume or overwrite that session's conversation memory. Prefixing with the
    CALLER's own authenticated user_id (never client-supplied) makes a
    collision land in a different, inaccessible Valkey key instead."""
    return f"{user_id}:{raw_session_id}"


def _resolve_conversation(user: CurrentUser, conversation_id: Optional[str]) -> tuple:
    """(conversation_id, durable_history). If the caller supplied a
    conversation_id, it must be theirs -- history_for_agent() returns []
    both for "doesn't exist" and "belongs to someone else" (pipeline_chat_
    messages' user_id-scoped WHERE clause, never checked after the fact), so
    an empty result 404s without confirming which. A fresh conversation_id is
    minted when none was supplied (turn 1)."""
    if conversation_id:
        history = history_for_agent(user.user_id, conversation_id)
        if not history:
            raise HTTPException(status_code=404, detail="No such conversation.")
        return conversation_id, history
    return new_conversation_id(), []


def _persist_turn(
    user: CurrentUser, conversation_id: str, tb_doc_id: Optional[str], question: str, answer: str,
    payload: Optional[dict] = None,
) -> None:
    """Best-effort, after a successful answer only -- a failed call leaves no
    orphan question (matches append_turns' own contract). Logged, never
    raised: the answer already succeeded, only the durable record of it
    would be lost. `payload` is the full structured response (currently only
    an /audit run's envelope) stored alongside the assistant turn so a
    reopened conversation can render it richly -- /ask and /ask-general don't
    pass one, same as before."""
    try:
        append_turns(user.user_id, conversation_id, tb_doc_id, question, answer, payload=payload)
    except Exception as e:
        logger.warning("[chat history] failed to persist turn for conversation %s: %s", conversation_id, e)


@router.post("/ask")
def ask(request: AskRequest, user: CurrentUser = Depends(require_user)):
    """A question about a specific uploaded/analyzed Trial Balance (doc_id)."""
    _verify_document_access(request.doc_id, user)
    conversation_id, durable_history = _resolve_conversation(user, request.conversation_id)
    # Same DB-write guard as /audit: a Postgres hiccup here must degrade gracefully
    # rather than escape as an unhandled 500.
    try:
        session_id = create_session(mode="CHAT_QUERY", source="ask", tb_doc_id=request.doc_id, user_id=user.user_id)
    except Exception as e:
        return _tool_error_response(e)

    # The agent swallows connection failures and returns the raw error text as its
    # answer, which put an internal endpoint URL in front of the end user. Probe once
    # up front (same rationale as /audit) and answer plainly instead.
    if not llm_reachable():
        logger.warning("LLM endpoint unreachable (%s) -- /ask answered as unavailable.", settings.LLM_BASE_URL)
        update_session_status(session_id, "FAILED", error_message="LLM endpoint unreachable")
        return {"answer": _LLM_UNAVAILABLE_ANSWER, "session_id": session_id, "conversation_id": conversation_id, "available": False}

    try:
        prompt = f"Trial Balance doc_id: {request.doc_id}\nQuestion: {request.question}"
        # conversation_id (stable across every turn once minted -- see
        # _resolve_conversation) is now the ONE identifier a client tracks for
        # continuity, driving both the Valkey fast-path cache key AND the
        # durable Postgres store below. request.session_id (legacy, pre-dates
        # conversation_id) is no longer used for memory keying -- keeping it
        # as the key would mean a client had to track two separate ids to get
        # full continuity, one for the 2-hour Valkey cache and a different one
        # for the 90-day durable record. This is still isolated per key --
        # never the agent's old shared, cross-tenant default memory (see
        # agent_memory.py) -- and never guessable/reusable across two
        # different users. durable_history seeds the agent's REAL working
        # memory (not just a display list) whenever Valkey's 2-hour cache for
        # this conversation has already expired but pipeline_chat_messages
        # still has it.
        memory_key = _user_scoped_memory_key(user.user_id, conversation_id)
        response = invoke_scoped(get_agent(), prompt, session_id=memory_key, durable_history=durable_history)
        update_session_status(session_id, "SUCCESS")
        _persist_turn(user, conversation_id, request.doc_id, request.question, response)
        return {"answer": response, "session_id": session_id, "conversation_id": conversation_id, "available": True}
    except Exception as e:
        update_session_status(session_id, "FAILED", error_message=str(e))
        return _tool_error_response(e)


@router.post("/ask-general")
def ask_general(request: AskGeneralRequest, user: CurrentUser = Depends(require_user)):
    """A question with no Trial Balance attached — answered from reference corpora."""
    conversation_id, durable_history = _resolve_conversation(user, request.conversation_id)
    try:
        session_id = create_session(mode="CHAT_QUERY", source="ask-general", user_id=user.user_id)
    except Exception as e:
        return _tool_error_response(e)

    if not llm_reachable():
        logger.warning("LLM endpoint unreachable (%s) -- /ask-general answered as unavailable.", settings.LLM_BASE_URL)
        update_session_status(session_id, "FAILED", error_message="LLM endpoint unreachable")
        return {
            "answer": _LLM_UNAVAILABLE_ANSWER, "session_id": session_id, "conversation_id": conversation_id,
            "computed": None, "guardrail": None, "available": False,
        }

    try:
        # See /ask's own comment: conversation_id (not request.session_id) is
        # what drives both the Valkey fast-path key and durable_history's
        # Postgres fallback now.
        memory_key = _user_scoped_memory_key(user.user_id, conversation_id)
        response = invoke_scoped(get_agent(), request.question, session_id=memory_key, durable_history=durable_history)
        update_session_status(session_id, "SUCCESS")
        _persist_turn(user, conversation_id, None, request.question, response)
        return {
            "answer": response, "session_id": session_id, "conversation_id": conversation_id,
            "computed": None, "guardrail": None, "available": True,
        }
    except Exception as e:
        update_session_status(session_id, "FAILED", error_message=str(e))
        return _tool_error_response(e)


# ── Chat history (durable, 90-day-retained conversations) ───────────────────
#
# Mirrors financial_statement/router.py's three conversation routes exactly:
# user_id-scoped inside the SQL (see pipeline/db.py), never checked afterward,
# a miss is always 404 (never 403 -- don't confirm another user's conversation
# exists). Unlike FS's platform-DB-backed artha_fs_messages, these read/write
# TB's own Postgres (pipeline_chat_messages) so retention can reuse
# soft_delete_expired_sessions' already-proven 90-day pattern.

@router.get("/conversations")
def list_conversations_route(user: CurrentUser = Depends(require_user)):
    return {"conversations": list_conversations(user.user_id)}


@router.get("/conversations/{conversation_id}")
def get_conversation(conversation_id: str, user: CurrentUser = Depends(require_user)):
    messages = get_messages(user.user_id, conversation_id)
    if not messages:
        raise HTTPException(status_code=404, detail="No such conversation.")
    return {"conversation_id": conversation_id, "messages": messages}


@router.delete("/conversations/{conversation_id}", status_code=204)
def delete_conversation_route(conversation_id: str, user: CurrentUser = Depends(require_user)):
    removed = delete_conversation(user.user_id, conversation_id)
    if not removed:
        raise HTTPException(status_code=404, detail="No such conversation.")
    return None


# ── Audit (single-TB or comparison) ─────────────────────────────────────────

_LAYER1_SEVERITY_MAP = {"Blocking": "high", "Warning": "medium", "Info": "low"}


def _run_layer1_precheck(
    session_id: str, tb_doc_id: str, output_dir: str, run_full_analytics: bool = True
) -> Optional[dict]:
    """Deterministically load a DB-sourced doc (MAIN, or LIVE for a document
    ingest_tb_to_live wrote this session and hasn't been promoted to MAIN yet)
    and run validate_layer1_tb against it, recording any HALTED/WARNING rule
    outcomes as pipeline_findings. Returns the validation tool response, or
    None if the doc isn't resolvable from either (e.g. a stale/unknown upload
    token) -- in that case the agent's own dynamic flow handles it later.

    Tries MAIN first (the common case -- most audits run against already-
    promoted documents), only falling back to LIVE when MAIN doesn't resolve,
    so a doc_id that happens to exist in both never silently prefers the
    staging copy over the promoted one.

    run_full_analytics=False skips the full single-TB analytics chain below (used for
    the COMPARISON PY leg, which only ever needs canonical_tb.parquet -- running
    materiality/exceptions/reasoning against a period nothing downstream consumes would
    be pure waste)."""
    try:
        loaded = call_tool("load_tb_from_db", tb_doc_id=tb_doc_id, output_dir=output_dir)
    except ToolNotAvailableError:
        return None
    if loaded.get("execution_status") != "SUCCESS":
        try:
            loaded = call_tool("load_tb_from_live", tb_doc_id=tb_doc_id, output_dir=output_dir)
        except ToolNotAvailableError:
            return None
        if loaded.get("execution_status") != "SUCCESS":
            return None

    canonical_tb_file = loaded["artifacts"][0]

    # TB-R18: entity_profile.json must exist deterministically, before the agent's own
    # analytics phase runs -- build_materiality.py's basis selector and build_sensitive_
    # detector.py's related-party register both read it. Live testing (Phase 1) showed
    # the agent's tool-calling loop is not reliable for pipeline-critical steps the
    # response contract depends on, so this is not left to agent discretion.
    try:
        call_tool("build_entity_profile", canonical_tb_file=canonical_tb_file, output_dir=output_dir)
    except ToolNotAvailableError:
        logger.warning("build_entity_profile not available -- downstream tools fall back to their own defaults.")
    except Exception:
        logger.exception("build_entity_profile failed for output_dir=%s", output_dir)

    # TB-R12: estimation-exposure is a size-ranked screen independent of movement/dormancy,
    # needs only canonical_tb.parquet -- deterministic for the same reliability reason as
    # build_entity_profile above.
    try:
        call_tool("build_estimation_exposure", canonical_tb_file=canonical_tb_file, output_dir=output_dir)
    except ToolNotAvailableError:
        logger.warning("build_estimation_exposure not available -- report skips that section.")
    except Exception:
        logger.exception("build_estimation_exposure failed for output_dir=%s", output_dir)

    # TB-R11: gated on the entity_profile.json flag just written above, so this must run
    # after it, not concurrently.
    try:
        call_tool("build_fx_exposure", canonical_tb_file=canonical_tb_file, output_dir=output_dir)
    except ToolNotAvailableError:
        logger.warning("build_fx_exposure not available -- report skips that section.")
    except Exception:
        logger.exception("build_fx_exposure failed for output_dir=%s", output_dir)

    # TB-R19: needs only canonical_tb.parquet -- deterministic for the same reliability
    # reason as the other analytics screens above.
    try:
        call_tool("build_mapping_quality", canonical_tb_file=canonical_tb_file, output_dir=output_dir)
    except ToolNotAvailableError:
        logger.warning("build_mapping_quality not available -- report skips that section.")
    except Exception:
        logger.exception("build_mapping_quality failed for output_dir=%s", output_dir)

    try:
        validation = call_tool("validate_layer1_tb", canonical_tb_file=canonical_tb_file, output_dir=output_dir)
    except ToolNotAvailableError as exc:
        return _tool_error_response(exc)

    if validation.get("execution_status") == "SUCCESS" and validation.get("artifacts"):
        try:
            with open(validation["artifacts"][0]) as f:
                rule_results = json.load(f)
            findings = [
                {
                    "category": "layer1_validation",
                    "severity": _LAYER1_SEVERITY_MAP.get(r.get("severity"), "medium"),
                    "statement": f"[{r['rule']}] {r['rule_name']}: {r['message']}",
                    "evidence_uids": [r["rule"]],
                }
                for r in rule_results
                if r.get("status") in ("HALTED", "WARNING")
            ]
            if findings:
                add_findings(session_id, findings)
        except Exception:
            logger.exception("Failed to record layer1_validation findings for session %s", session_id)

    if run_full_analytics:
        _run_core_analytics_chain(canonical_tb_file, output_dir)
    else:
        # Chat-query coverage (Q25, common-size % change vs PY): the PY leg otherwise never
        # gets a snapshot_drilldown.parquet at all, so chat_query_fsli_table's common_size
        # metric has nothing to diff CY against for this comparison run. build_fsli_summary
        # is a prerequisite for build_financial_snapshot (see _run_core_analytics_chain's own
        # note) -- both are cheap, deterministic, and don't touch materiality/exceptions/
        # reasoning, which stay skipped for PY as before.
        try:
            call_tool("build_fsli_summary", canonical_tb_file=canonical_tb_file, output_dir=output_dir)
            call_tool("build_financial_snapshot", canonical_tb_file=canonical_tb_file, output_dir=output_dir)
        except ToolNotAvailableError:
            logger.warning("build_fsli_summary/build_financial_snapshot not available for PY leg -- common-size PY comparison unavailable.")
        except Exception:
            logger.exception("PY-leg financial snapshot chain failed for output_dir=%s", output_dir)

        # Wave 2 Fix 4a: materiality was previously CY-only, so every PY-side movement
        # (continuity breaks included) was graded against a threshold derived from a year
        # it doesn't belong to. build_materiality's own inputs (financial_snapshot_
        # statistics.json, fsli_summary.parquet, canonical_tb.parquet, snapshot_drilldown.
        # parquet) are already produced by the two calls just above -- build_entity_profile
        # also already ran earlier in this same function -- so this has no unmet
        # dependency, it was simply never invoked for PY before.
        try:
            call_tool("build_materiality", canonical_tb_file=canonical_tb_file, output_dir=output_dir)
        except ToolNotAvailableError:
            logger.warning("build_materiality not available for PY leg -- PY-side movements have no PY materiality threshold.")
        except Exception:
            logger.exception("PY-leg materiality failed for output_dir=%s", output_dir)

        # Remark #30 fix: validate_layer2_tb (TB-012/013/016/017/018/020 -- 6 rules,
        # including TB-012's Assets=Liabilities+Equity identity check, the single most
        # consequential integrity rule) previously only ran inside
        # _run_core_analytics_chain, which run_full_analytics=False (this branch) never
        # invokes -- so PY only ever got layer1's ~20+ rules, never layer2's 6. It only
        # needs canonical_tb_file + the layer1_results.json validate_layer1_tb already
        # wrote above (no dependency on materiality/exceptions/reasoning), so it can run
        # here directly, decoupled from the full analytics chain.
        layer1_results_file = Path(output_dir) / "layer1_results.json"
        if layer1_results_file.exists():
            try:
                call_tool(
                    "validate_layer2_tb",
                    canonical_tb_file=canonical_tb_file,
                    layer1_results_file=str(layer1_results_file),
                    output_dir=output_dir,
                )
            except ToolNotAvailableError:
                logger.warning("validate_layer2_tb not available for PY leg -- PY register omits layer2 rules.")
            except Exception:
                logger.exception("PY-leg validate_layer2_tb failed for output_dir=%s", output_dir)

        # Remark #21 fix: the comparative report's Sections 6/8/10 and ratio-trend
        # section were CY-only because these 4 tools never ran for PY, even though
        # every prerequisite they need (fsli_summary.parquet, financial_snapshot_
        # statistics.json) was already produced by the calls above in this same
        # branch. Same call pattern as _run_core_analytics_chain's CY-leg calls
        # (routes.py ~567-600); build_risk_indicators needs relationship_analytics.
        # json to exist first, same ordering constraint as the CY leg.
        try:
            call_tool("build_financial_ratios", canonical_tb_file=canonical_tb_file, output_dir=output_dir)
            call_tool("build_audit_ratio_pack", canonical_tb_file=canonical_tb_file, output_dir=output_dir)
            call_tool(
                "build_relationship_analytics",
                canonical_tb_file=canonical_tb_file,
                fsli_summary_file=str(Path(output_dir) / "fsli_summary.parquet"),
                output_dir=output_dir,
            )
            call_tool("build_risk_indicators", canonical_tb_file=canonical_tb_file, output_dir=output_dir)
        except ToolNotAvailableError:
            logger.warning("Ratio/relationship/risk tools not available for PY leg -- comparative Sections 6/8/10 and ratio-trend stay CY-only.")
        except Exception:
            logger.exception("PY-leg ratio/relationship/risk chain failed for output_dir=%s", output_dir)

    return validation


def _run_core_analytics_chain(canonical_tb_file: str, output_dir: str) -> None:
    """Deterministically runs the full single-TB analytics chain (financial snapshot ->
    FSLI summary -> materiality -> relationship analytics -> risk indicators -> sensitive
    detector -> anomaly scanner -> exception consolidator -> pipeline validation -> data
    sufficiency -> audit reasoning).

    Root cause this fixes: live testing (re-running the same EPIL/APCPL documents multiple
    times) showed the agent's tool-calling loop skips a DIFFERENT step of this chain on
    different runs of the SAME document -- materiality.json missing on one run, consolidated_
    exceptions.json/audit_reasoning.json missing on another, both fully correct on a third.
    Not a data or taxonomy bug (same document, same data, different outcome), and not
    fixable by better agent prompting -- the same "don't trust agent discretion for
    response-contract-critical output" fix already applied to report generation and the
    Phase-2 screens above, extended to cover this chain too.

    Each step degrades independently on failure (log and continue, matching every other
    tool call in this file) -- one missing prerequisite means that section reads "not
    available" in the report, not that the whole /audit response fails."""
    out_dir = Path(output_dir)

    def run(tool_name: str, **kwargs):
        try:
            return call_tool(tool_name, **kwargs)
        except ToolNotAvailableError:
            logger.warning("%s not available -- downstream steps/report sections degrade accordingly.", tool_name)
        except Exception:
            logger.exception("%s failed for output_dir=%s", tool_name, output_dir)
        return None

    # ── TB-012/013/016/017/018/020 patch runs FIRST (TB-R19/R21 fix) ─────────
    # Previously this block ran after build_exception_consolidator/build_assertion_
    # evidence_map/validate_tb_pipeline (further down this chain), so all three
    # consumed layer1_results.json while TB-012 etc. were still SKIPPED stubs -- a
    # genuinely HALTED TB-012 (the TB doesn't foot) never reached exception
    # consolidation or pipeline validation at all. validate_layer2_tb only needs
    # canonical_tb_file + the layer1_results.json validate_layer1_tb already wrote
    # before this chain started, so it has no dependency on anything below and can
    # safely move to the front -- every downstream consumer of layer1_results.json
    # now sees the real, patched status.
    layer1_results_file = out_dir / "layer1_results.json"
    layer2_result = None
    if layer1_results_file.exists():
        layer2_result = run(
            "validate_layer2_tb",
            canonical_tb_file=canonical_tb_file,
            layer1_results_file=str(layer1_results_file),
            output_dir=output_dir,
        )
        if layer2_result and layer2_result.get("can_continue") is False:
            logger.error(
                "validate_layer2_tb HALTED for output_dir=%s: %s -- analytics below will run "
                "(per this chain's graceful-degradation contract) but the report must surface "
                "this HALT; see _finalize_audit_result.",
                output_dir, layer2_result.get("halted_rules"),
            )
        run(
            "build_data_sufficiency_grade",
            layer1_results_file=str(layer1_results_file),
            canonical_tb_file=canonical_tb_file,
            output_dir=output_dir,
        )

    # build_financial_snapshot actually requires fsli_summary.parquet to already exist
    # (confirmed by direct reproduction -- its own PipelineFileError names
    # "FSLI Summary (run build_fsli_summary first)"), the reverse of what the file names
    # suggest. fsli_summary must run first.
    # ── context and normalisation (spec sec 2.1, sec 4.2) ────────────────────
    # Both run first for a stated reason. build_engagement_context writes the
    # applicability gates that build_caro_indicators and build_public_sector_lens
    # read, so it must precede them. build_normalisation_note records what was done
    # to the numbers before analysis, and sec 4.2 requires that note to precede any
    # audit finding.
    # Client-review defect registry: currency/source_system were always passed as
    # None here, so build_engagement_context always fell back to "unknown" regardless
    # of what the SAP/ERP adapter captured at ingestion (db_bridge.py's doc_row is the
    # source of truth; tb_metadata.json is its already-established downstream copy,
    # same file _tb_filename_suffix above reads defensively). Best-effort: an absent
    # or unreadable tb_metadata.json leaves this call exactly as it was before.
    tb_erp_system = tb_currency = None
    tb_metadata_path = out_dir / "tb_metadata.json"
    if tb_metadata_path.exists():
        try:
            with open(tb_metadata_path, "r") as f:
                _tb_meta = json.load(f)
            tb_erp_system = _tb_meta.get("erp_system")
            tb_currency = _tb_meta.get("currency")
        except Exception:
            logger.warning("Could not read tb_metadata.json for erp_system/currency at output_dir=%s", output_dir)
    run(
        "build_engagement_context", canonical_tb_file=canonical_tb_file, output_dir=output_dir,
        currency=tb_currency, source_system=tb_erp_system,
    )
    run(
        "build_normalisation_note",
        canonical_tb_file=canonical_tb_file,
        layer1_results_file=str(out_dir / "layer1_results.json"),
        output_dir=output_dir,
    )

    run("build_fsli_summary", canonical_tb_file=canonical_tb_file, output_dir=output_dir)
    run("build_financial_snapshot", canonical_tb_file=canonical_tb_file, output_dir=output_dir)
    # chat_get_financial_ratios (Q41/Q83) reads financial_ratios.json -- depends on
    # snapshot_drilldown.parquet/financial_snapshot_statistics.json just written above.
    run("build_financial_ratios", canonical_tb_file=canonical_tb_file, output_dir=output_dir)
    # Audit-analytical ratios (sec 8.3) -- debtor/creditor/inventory intensity,
    # depreciation proxy, finance-cost ratio. Complements build_financial_ratios'
    # liquidity/leverage set rather than replacing it; reads the same snapshot.
    run("build_audit_ratio_pack", canonical_tb_file=canonical_tb_file, output_dir=output_dir)
    run("build_materiality", canonical_tb_file=canonical_tb_file, output_dir=output_dir)
    # SA 530 sampling with default parameters (95% confidence, whole mapped population,
    # both MUS and stratified). Deliberately the one screen in this chain that samples
    # rather than tests the full population -- every parameter that shaped the sample is
    # recorded on its own output so it can be re-run with different parameters (a narrower
    # population_head, a different confidence level) without touching this default.
    run("build_sample_selection", canonical_tb_file=canonical_tb_file, output_dir=output_dir)
    # build_risk_indicators' own required-file check demands relationship_analytics.json
    # already exist, so relationship_analytics must run before it, not after.
    run(
        "build_relationship_analytics",
        canonical_tb_file=canonical_tb_file,
        fsli_summary_file=str(out_dir / "fsli_summary.parquet"),
        output_dir=output_dir,
    )
    # ── relationship screens (sec 6, sec 7, sec 5.3) ─────────────────────────
    # All three read canonical_tb + fsli_summary + materiality, so they sit after
    # build_materiality and before the risk engines that consume their findings.
    # build_counterpart_screen answers "is the counterpart there at all?";
    # build_relationship_expectations answers "given both are there, is the
    # relationship plausible?" -- they are complementary, not alternatives.
    for _screen in (
        "build_counterpart_screen",
        "build_relationship_expectations",
        "build_abnormal_sign_screen",
    ):
        run(_screen, canonical_tb_file=canonical_tb_file, output_dir=output_dir)

    run("build_risk_indicators", canonical_tb_file=canonical_tb_file, output_dir=output_dir)
    run("build_sensitive_detector", canonical_tb_file=canonical_tb_file, output_dir=output_dir)

    # Materiality by nature/context (sec 2.2, sec 8.1) reads sensitive_accounts.json,
    # so it must follow build_sensitive_detector -- NOT build_materiality, which is
    # where it would naturally seem to belong. Without the sensitive population it
    # can elevate nothing, which is the whole point of the tool.
    run(
        "build_materiality_lens",
        layer1_results_file=str(out_dir / "layer1_results.json"),
        output_dir=output_dir,
    )

    run(
        "build_anomaly_scanner",
        canonical_tb_file=canonical_tb_file,
        output_dir=output_dir,
        llm_client=get_agent().llm_client,
    )

    # Wave 2 Fix 3: -R/-P/-M clearing pairs (e.g. GL 20950021/20950022) reported gross
    # instead of netted -- must run BEFORE build_going_concern_screen so netted_balances.
    # parquet exists for it to read; going_concern_screen degrades gracefully to its prior
    # gross behavior if this is ever skipped, so ordering here is a should, not a hard
    # dependency, but running it after would defeat the point entirely.
    run("build_netting_screen", canonical_tb_file=canonical_tb_file, output_dir=output_dir)

    # Wave 3 Fix 1: must run BEFORE build_deposit_margin_money_screen so
    # contract_exposure.json's bank_guarantee category exists for that screen's
    # guarantee-book cross-reference; degrades gracefully (cross-reference unavailable,
    # not a failure) if this is ever skipped.
    run("build_contract_exposure_lens", canonical_tb_file=canonical_tb_file, output_dir=output_dir)
    run("build_foreign_operations_lens", canonical_tb_file=canonical_tb_file, output_dir=output_dir)

    # ── India-specific and public-sector screens (sec 9.16, sec 10, sec 12) ───
    # build_override_indicators reads anomaly_findings.json to reframe the
    # round-sum/Benford signals rather than recomputing them, so it follows the
    # anomaly scanner. The CARO and public-sector screens read the applicability
    # gates written by build_engagement_context at the top of this chain.
    for _screen in (
        "build_statutory_screen",
        "build_public_sector_lens",
        "build_going_concern_screen",
        "build_caro_indicators",
        "build_override_indicators",
        "build_deposit_margin_money_screen",
        "build_provisions_writeoff_screen",
        "build_msme_interest_screen",
        # Remarks #25/#26/#27 (Wave 8): single-TB-callable -- #25 (unbilled revenue vs
        # receivables) degrades to a balance-only report without a prior-period TB
        # (wired with one in the comparison chain below); #26/#27 need only this period.
        "build_unbilled_revenue_screen",
        "build_dta_recoverability_screen",
        "build_wip_contract_asset_screen",
    ):
        run(_screen, canonical_tb_file=canonical_tb_file, output_dir=output_dir)

    run("build_exception_consolidator", canonical_tb_file=canonical_tb_file, output_dir=output_dir)

    # Re-maps the exceptions just consolidated from source-engine-derived assertions
    # ("Ledgers", "Contracts/Agreements") to account-area assertions and named
    # records (Appendix B/C). Must follow build_exception_consolidator.
    run("build_assertion_evidence_map", output_dir=output_dir)
    run("validate_tb_pipeline", output_dir=output_dir)
    # (validate_layer2_tb / build_data_sufficiency_grade already ran at the top of this
    # chain -- see the TB-R19/R21 fix above -- so build_exception_consolidator and
    # validate_tb_pipeline above now see the real, patched layer1_results.json.)
    # ── consolidated output (sec 13, sec 14, sec 15.1) ───────────────────────
    # build_finding_records collects every screen's findings into the specification's
    # record shape; build_request_lists then turns them into the de-duplicated
    # evidence list and the separate management-query list sec 15.1 requires.
    run(
        "build_finding_records",
        data_sufficiency_file=str(out_dir / "data_sufficiency.json"),
        output_dir=output_dir,
    )
    run("build_request_lists", output_dir=output_dir)

    run(
        "build_audit_reasoning",
        consolidated_exceptions_file=str(out_dir / "consolidated_exceptions.json"),
        validation_report_file=str(out_dir / "validation_report.json"),
        data_sufficiency_file=str(out_dir / "data_sufficiency.json"),
        output_dir=output_dir,
        llm_client=get_agent().llm_client,
    )

    # Runs LAST so it records what actually executed rather than what was planned --
    # it infers the tool list from the artifacts present on disk (sec 16.2).
    run("build_run_log", output_dir=output_dir)


def _run_comparison_analytics_chain(report_dir: Path, cy_dir: Path, py_dir: Path) -> None:
    """Deterministically runs the PY-vs-CY comparison analytics tools (precheck,
    structural delta, variance, sign-check, reasoning) that build_comparison_report
    reads from -- live testing showed the agent's tool-calling loop reliably runs
    the CY single-period analytics chain but was skipping this whole comparison
    chain entirely (agent-driven ordering per system_prompt.py has no fixed
    sequence requirement), leaving build_comparison_report with an empty
    PY-comparison shell. Mirrors the same "don't trust agent discretion for
    outputs the response contract depends on" fix already applied to report
    generation. Skips a step whose output file already exists (agent may have
    already run it), same idempotent-and-harmless pattern as the report calls."""
    py_canonical_tb = py_dir / "canonical_tb.parquet"
    cy_canonical_tb = cy_dir / "canonical_tb.parquet"
    if not py_canonical_tb.exists() or not cy_canonical_tb.exists():
        logger.warning("Skipping comparison analytics chain: PY or CY canonical_tb.parquet missing.")
        return

    report_dir.mkdir(parents=True, exist_ok=True)
    common_kwargs = {
        "py_canonical_tb": str(py_canonical_tb),
        "cy_canonical_tb": str(cy_canonical_tb),
        "output_dir": str(report_dir),
    }

    # TB-QA-followup: these four always run, unconditionally -- they used to be skipped
    # when their output file already existed, on the assumption that an existing file
    # means a prior, correct run. Live testing showed the agent's own tool-calling loop can
    # call run_comparison_variance before CY's materiality.json is ready, writing a
    # NO_THRESHOLD-poisoned comparison_variance.parquet that this "skip if exists" guard
    # then permanently locked in, since it never got recomputed once materiality.json
    # became available. Matches the same "calling these even when the agent already built
    # them is harmless, they overwrite deterministically" pattern already used for report
    # generation in _finalize_audit_result.
    try:
        precheck_result = call_tool("run_comparison_prechecks", **common_kwargs)
        if precheck_result and precheck_result.get("can_continue") is False:
            logger.error(
                "run_comparison_prechecks HALTED for report_dir=%s: %s -- comparison analytics "
                "below will still run (graceful-degradation contract) but the report must "
                "surface this HALT.",
                report_dir, precheck_result.get("halted_checks"),
            )
        call_tool("run_comparison_structural", **common_kwargs)
        materiality_file = cy_dir / "materiality.json"
        call_tool(
            "run_comparison_variance",
            materiality_file=str(materiality_file) if materiality_file.exists() else None,
            **common_kwargs,
        )
        call_tool("run_comparison_sign_check", **common_kwargs)
        if not (report_dir / "comparison_reasoning.json").exists():
            call_tool(
                "build_comparison_reasoning",
                precheck_file=str(report_dir / "precheck_results.json"),
                structural_file=str(report_dir / "structural_delta.json"),
                variance_file=str(report_dir / "comparison_variance.parquet"),
                sign_check_file=str(report_dir / "sign_convention_flags.json"),
                output_dir=str(report_dir),
                llm_client=get_agent().llm_client,
            )
        # Remark #25 (Wave 8): re-run with the PY leg's canonical TB now available, so
        # the movement-correlation check (unbilled revenue vs receivables) actually runs
        # instead of the single-period, balance-only fallback the CY-only chain wrote
        # earlier. Overwrites cy_dir's unbilled_revenue_screen.json in place -- same
        # deterministic-overwrite pattern the four calls above already use.
        call_tool(
            "build_unbilled_revenue_screen",
            canonical_tb_file=str(cy_canonical_tb),
            prior_canonical_tb_file=str(py_canonical_tb),
            output_dir=str(cy_dir),
        )
    except ToolNotAvailableError:
        logger.warning("Comparison analytics tool(s) not available -- skipping.")
    except Exception:
        logger.exception("Comparison analytics chain failed for report_dir=%s", report_dir)


def _finalize_audit_result(mode: str, run_dir: Path, entity_hint: Optional[str] = None) -> dict:
    """Deterministically generate all 3 report formats and assemble the structured
    `result` object the frontend actually reads -- the agent's raw closing text
    (get_agent().invoke()'s return value) is never returned to the client on its own.
    build_excel_report/build_docx_report are the full-fidelity downloadable
    deliverables (served by /audit/workbook); build_report_markdown is the in-chat
    display outcome. All three are complementary, not alternatives, so all three
    always run here rather than depending on the agent's tool-calling discretion --
    that's what was leaving TB_Audit.xlsx/TB_Audit_Report.docx uncalled before."""
    is_comparison = mode == "COMPARISON"
    report_dir = run_dir / "comparison" if is_comparison else run_dir
    cy_dir = run_dir / "cy" if is_comparison else run_dir
    py_dir = run_dir / "py"
    canonical_tb_file = str(cy_dir / "canonical_tb.parquet")

    result: dict = {}
    if entity_hint:
        result["entity"] = entity_hint

    if not Path(canonical_tb_file).exists():
        logger.warning("Skipping report generation: canonical_tb.parquet not found at %s", canonical_tb_file)
        return result

    # TB-R19: surface a genuinely HALTED rule (post validate_layer2_tb patch, most
    # commonly TB-012 -- the TB doesn't foot) on the API response itself, not just in
    # logs -- "block" must mean the caller can tell the report is standing on a broken
    # invariant, not that the pipeline silently reports as if everything passed.
    layer1_results_path = cy_dir / "layer1_results.json"
    if layer1_results_path.exists():
        try:
            with open(layer1_results_path, encoding="utf-8") as f:
                layer1_results = json.load(f)
            halted = [r for r in layer1_results if r.get("status") == "HALTED"]
            if halted:
                result["tb_integrity_halted"] = True
                result["tb_integrity_halted_rules"] = [
                    {"rule": r.get("rule"), "message": r.get("message")} for r in halted
                ]
        except Exception:
            logger.exception("Failed to read layer1_results.json for run_dir=%s", run_dir)

    if is_comparison:
        _run_comparison_analytics_chain(report_dir, cy_dir, py_dir)

    try:
        if is_comparison:
            call_tool("build_comparison_report", output_dir=str(report_dir))
            markdown_response = call_tool("build_comparison_report_markdown", output_dir=str(report_dir))
        else:
            call_tool("build_excel_report", canonical_tb_file=canonical_tb_file, output_dir=str(report_dir))
            call_tool("build_docx_report", canonical_tb_file=canonical_tb_file, output_dir=str(report_dir))
            markdown_response = call_tool(
                "build_report_markdown", canonical_tb_file=canonical_tb_file, output_dir=str(report_dir)
            )
        result["report"] = (markdown_response.get("data") or {}).get("markdown")
    except ToolNotAvailableError:
        logger.warning("Report tool(s) not available -- skipping deterministic report generation.")
    except Exception:
        logger.exception("Deterministic report generation failed for run_dir=%s", run_dir)

    materiality_path = cy_dir / "materiality.json"
    if materiality_path.exists():
        try:
            with open(materiality_path) as f:
                mat = json.load(f)
            result["materiality"] = {
                "overall_materiality": mat.get("selected_materiality", {}).get("overall_materiality"),
                "benchmark_used": mat.get("summary", {}).get("benchmark_used"),
                "benchmark_analysis": mat.get("benchmark_analysis", []),
            }
        except Exception:
            logger.exception("Failed to read materiality.json for run_dir=%s", run_dir)

    if is_comparison:
        # build_comparison_reasoning writes comparison_reasoning.json into the same
        # output_dir build_comparison_report/_markdown read it from (report_dir, i.e.
        # run_dir/comparison) -- shape: {"summary": {"executive_summary": ...},
        # "observations": [{"observation": {priority, title, executive_summary,
        # detailed_observation, affected_assertions, recommended_procedures, ...}}]}.
        # No cluster_id (unlike SINGLE_TB's audit_reasoning.json), so reference_id
        # falls back to an F01/F02/... index, matching build_comparison_report.py's
        # own DOCX heading convention.
        comparison_reasoning_path = report_dir / "comparison_reasoning.json"
        if comparison_reasoning_path.exists():
            try:
                with open(comparison_reasoning_path) as f:
                    reasoning = json.load(f)
                result["context_note"] = reasoning.get("summary", {}).get("executive_summary")
                findings = []
                summary: dict = {}
                for i, obs_wrap in enumerate(reasoning.get("observations", []), start=1):
                    obs = obs_wrap.get("observation", obs_wrap) if isinstance(obs_wrap, dict) else {}
                    severity = str(obs.get("priority", "")).lower()
                    summary[severity] = summary.get(severity, 0) + 1
                    findings.append({
                        "risk_rating": severity,
                        "reference_id": f"F{i:02d}",
                        "observation": obs.get("detailed_observation") or obs.get("executive_summary"),
                        "evidence_requested": obs.get("recommended_procedures", []),
                    })
                result["findings"] = findings
                result["findings_summary"] = summary
            except Exception:
                logger.exception("Failed to read comparison_reasoning.json for run_dir=%s", run_dir)
    else:
        audit_reasoning_path = run_dir / "audit_reasoning.json"
        if audit_reasoning_path.exists():
            try:
                with open(audit_reasoning_path) as f:
                    reasoning = json.load(f)
                result["context_note"] = reasoning.get("data_sufficiency_note")
                findings = []
                summary: dict = {}
                for finding in reasoning.get("findings", []):
                    severity = str(finding.get("severity", "")).lower()
                    summary[severity] = summary.get(severity, 0) + 1
                    findings.append({
                        "risk_rating": severity,
                        "reference_id": finding.get("cluster_id"),
                        "observation": finding.get("statement"),
                        "evidence_requested": finding.get("recommended_procedures", []),
                    })
                result["findings"] = findings
                result["findings_summary"] = summary
            except Exception:
                logger.exception("Failed to read audit_reasoning.json for run_dir=%s", run_dir)

    return result


@router.post("/audit")
def audit(request: AuditRequest, user: CurrentUser = Depends(require_user)):
    """Routes to the SINGLE_TB chain when doc_id_prior is absent, or the
    COMPARISON chain when both doc_id and doc_id_prior are given."""
    _verify_document_access(request.doc_id, user)
    _verify_document_access(request.doc_id_prior, user)
    # See AuditRequest.conversation_id's own comment: the frontend never sends this today,
    # so this always takes _resolve_conversation's "mint a new one" branch in practice --
    # each audit run gets its own fresh conversation. durable_history is discarded; an
    # audit run doesn't feed the agent's chat memory the way /ask does.
    conversation_id, _ = _resolve_conversation(user, request.conversation_id)
    mode = "COMPARISON" if request.doc_id_prior else "SINGLE_TB"
    # create_session() is a live DB write and the Postgres host has been observed to go
    # briefly unreachable. Outside the try below it produced a bare unhandled 500, unlike
    # every other failure mode in this handler; inside its own guard it degrades the same
    # graceful way. session_id is needed by the main try's own except clause, so this
    # cannot simply be folded into that block.
    try:
        session_id = create_session(
            mode=mode,
            source="audit",
            tb_doc_id=request.doc_id,
            tb_doc_id_prior=request.doc_id_prior,
            entity_id=request.entity,
            user_id=user.user_id,
        )
    except Exception as e:
        return _tool_error_response(e)

    run_dir = resolve_output_dir(f"sessions/{session_id}")
    try:
        cy_validation = _run_layer1_precheck(session_id, request.doc_id, f"{run_dir}/cy" if mode == "COMPARISON" else str(run_dir))
        py_validation = None

        if mode == "COMPARISON":
            py_validation = _run_layer1_precheck(session_id, request.doc_id_prior, f"{run_dir}/py", run_full_analytics=False)
            prompt = (
                f"Run a COMPARISON audit. Current-year doc_id: {request.doc_id}. "
                f"Prior-year doc_id: {request.doc_id_prior}. "
                f"Use output_dir={run_dir}/cy for the CY single-TB chain, "
                f"output_dir={run_dir}/py for the PY load, and "
                f"output_dir={run_dir}/comparison for every comparison tool call. "
                f"Layer 1 validation has already run for both periods (see session findings) "
                f"-- do not re-run validate_layer1_tb unless a downstream tool explicitly needs its artifacts."
            )
        else:
            prompt = (
                f"Run a SINGLE_TB audit on doc_id: {request.doc_id}. "
                f"Use output_dir={run_dir} for every tool call in this run. "
                f"Layer 1 validation has already run (see session findings) "
                f"-- do not re-run validate_layer1_tb unless a downstream tool explicitly needs its artifacts."
            )
        if request.engagement_context:
            prompt += f"\nEngagement context: {request.engagement_context}"
        if request.framework:
            prompt += f"\nFramework: {request.framework}"
        if request.grouping_token:
            entry = preview_store_get(request.grouping_token)
            if entry:
                prompt += f"\nGrouping file: {entry['file_path']}"

        # A COMPARISON run leans on the agent for far more of its own tool orchestration
        # than SINGLE_TB does (most of SINGLE_TB's real work already ran deterministically
        # above, in _run_layer1_precheck/_run_core_analytics_chain) -- when the LLM
        # endpoint is down, that difference is what made a comparison look hung for many
        # minutes: the agent's own retry loop (already layered on top of the LLM client's
        # own internal retries) burns through iterations against a connection that will
        # never succeed. One cheap TCP probe here (a few ms when reachable) avoids ever
        # starting that doomed loop; skipping the agent call outright, not aborting the
        # whole request, since _finalize_audit_result already reports "not available" for
        # any section whose backing artifact never got written, same as any other
        # degraded-but-not-fatal gap in this pipeline.
        if llm_reachable():
            # Isolated per audit session_id so this run's orchestration prompt/response
            # never shares the agent's memory with a concurrent /ask conversation or
            # another audit run (see agent_memory.py).
            invoke_scoped(get_agent(), prompt, session_id=session_id)
        else:
            logger.warning(
                "LLM endpoint unreachable (%s) -- skipping agent orchestration for this "
                "%s run; proceeding with whatever the deterministic chain already produced.",
                settings.LLM_BASE_URL, mode,
            )
        result = _finalize_audit_result(mode, Path(run_dir), entity_hint=request.entity)
        update_session_status(session_id, "SUCCESS")
        response_envelope = {
            "session_id": session_id,
            "mode": mode,
            "result": result,
            "layer1_validation": {"cy": cy_validation, "py": py_validation},
        }

        # Records this run as a conversation so it appears in the sidebar the same way a
        # chat turn does (see AuditCard.jsx: it renders from exactly {result, doc,
        # priorDoc, pdfIds} -- this payload reproduces that same shape verbatim, so a
        # reopened conversation renders identically to the live result). pdfIds is always
        # [] -- upload_doc_ids isn't accepted by this endpoint (removed earlier), so no
        # live audit run tracks it server-side today either.
        label = "Two TB Comparative Analysis" if mode == "COMPARISON" else "Single TB Analysis"
        doc_display = request.doc_label or request.doc_id
        question = f"{label} — {doc_display}" + (f" vs {request.doc_id_prior}" if request.doc_id_prior else "")
        _persist_turn(
            user, conversation_id, request.doc_id, question,
            answer=f"{label} completed for {doc_display}.",
            payload={
                "result": response_envelope,
                "doc": {"doc_id": request.doc_id, "filename": doc_display},
                "priorDoc": {"doc_id": request.doc_id_prior} if request.doc_id_prior else None,
                "pdfIds": [],
            },
        )

        return {**response_envelope, "conversation_id": conversation_id}
    except Exception as e:
        update_session_status(session_id, "FAILED", error_message=str(e))
        return _tool_error_response(e)


@router.post("/audit/upload-grouping")
async def audit_upload_grouping(
    file: UploadFile = File(...),
    doc_id: Optional[str] = Form(None),
    doc_id_2: Optional[str] = Form(None),
):
    token = str(uuid.uuid4())
    dest_dir = _UPLOAD_ROOT / token
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = dest_dir / file.filename
    with open(dest_path, "wb") as f:
        f.write(await file.read())

    preview_store_set(token, {
        "file_path": str(dest_path),
        "filename": file.filename,
        "doc_id": doc_id,
        "doc_id_2": doc_id_2,
        "is_grouping": True,
    })

    try:
        result = call_tool("preview_excel_data", excel_path=str(dest_path), is_grouping=True)
    except ToolNotAvailableError as exc:
        return _tool_error_response(exc)

    if result.get("execution_status") != "SUCCESS":
        raise HTTPException(status_code=422, detail={"needs_mapping": True, "preview_token": token, **result})

    preview_store_update(token, {"preview": result})
    return {"token": token, "filename": file.filename, **result}


# ── Validation ───────────────────────────────────────────────────────────────

@router.post("/validate")
def validate(request: ValidateRequest, user: CurrentUser = Depends(require_user)):
    _verify_document_access(request.doc_id, user)
    try:
        result = call_tool("load_tb_from_db", tb_doc_id=request.doc_id)
        if result.get("execution_status") != "SUCCESS":
            # Same MAIN-then-LIVE fallback as _run_layer1_precheck -- a
            # freshly ingest_tb_to_live'd document (not yet promoted to MAIN)
            # would otherwise never validate right after upload.
            result = call_tool("load_tb_from_live", tb_doc_id=request.doc_id)
            if result.get("execution_status") != "SUCCESS":
                return result
        canonical_tb_file = result["artifacts"][0]
        output_dir = str(Path(canonical_tb_file).parent)
        validation = call_tool("validate_layer1_tb", canonical_tb_file=canonical_tb_file, output_dir=output_dir)
        return validation
    except ToolNotAvailableError as exc:
        return _tool_error_response(exc)


# ── Report download ──────────────────────────────────────────────────────────

# (mode, format) -> filename, matching what build_excel_report/build_docx_report/
# build_comparison_report actually write (see backend/tools/reports/build_excel_report.py,
# build_docx_report.py and backend/tools/comparison/build_comparison_report.py).
_WORKBOOK_FILENAMES = {
    ("SINGLE_TB", "xlsx"): "TB_Audit.xlsx",
    ("SINGLE_TB", "docx"): "TB_Audit_Report.docx",
    ("COMPARISON", "xlsx"): "TB_Comparison_Audit.xlsx",
    ("COMPARISON", "docx"): "TB_Comparison_Report.docx",
}
_WORKBOOK_MEDIA_TYPES = {
    "xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
}


def _tb_filename_suffix(run_dir: Path, mode: str) -> str:
    """Filesystem-safe suffix for downloadable report filenames, derived from the
    input TB's own identity -- tb_metadata.json's source_file (the uploaded
    filename) if present, else entity_name/company_name, else nothing. Read from
    the CY period's dir for COMPARISON (that's where load_tb_from_db writes it),
    the run's own dir for SINGLE_TB."""
    meta_dir = run_dir / "cy" if mode == "COMPARISON" else run_dir
    meta_path = meta_dir / "tb_metadata.json"
    if not meta_path.exists():
        return ""
    try:
        with open(meta_path, "r") as f:
            meta = json.load(f)
    except Exception:
        return ""
    raw = Path(meta.get("source_file", "")).stem or meta.get("company_name") or meta.get("entity_name") or ""
    safe = re.sub(r"[^A-Za-z0-9]+", "_", raw).strip("_")
    return f"_{safe}" if safe else ""


def _fetch_report_from_object_store(session_id: str, file_path: Path) -> bool:
    """MinIO is the primary, replica-independent copy of a generated report --
    local disk (under TB_OUTPUT_DIR) is a fast-path cache, not the source of
    truth, since a report generated by one backend replica during /audit isn't
    guaranteed to be visible to a different replica serving this download later
    (or after local scratch was reclaimed). Looks up this exact file's
    storage_uri (recorded at upload time by pipeline_tool.py's
    _upload_and_record_artifact_files) and rehydrates it onto local disk at the
    SAME path a fresh /audit run would have written it to, so this and every
    later request for the same report hit the local fast-path again. Returns
    False (never raises) if MinIO is disabled, the upload never happened, or
    the fetch itself fails -- the caller treats that as "report not found",
    same as before this existed."""
    from modes.trial_balance.pipeline.db import list_artifact_files_for_session
    from modes.trial_balance.pipeline.object_store import download_artifact

    try:
        files = list_artifact_files_for_session(session_id)
    except PipelineDBError:
        logger.warning("[audit_workbook] could not list artifact files for session %s", session_id)
        return False

    target = str(file_path)
    match = next((f for f in files if f.get("artifact_path") == target and f.get("storage_uri")), None)
    if not match:
        return False

    file_path.parent.mkdir(parents=True, exist_ok=True)
    return download_artifact(match["storage_uri"], target)


@router.post("/audit/workbook")
def audit_workbook(request: AuditWorkbookRequest, user: CurrentUser = Depends(require_user)):
    """Binary download of the generated report for a completed /audit run.
    The frontend requests this by doc_id (not session_id), so this looks up
    the most recent successful session for that doc_id/doc_id_prior pair and
    recomputes its run_dir the exact same way /audit itself does -- no extra
    pipeline_sessions column needed.

    Scoped to sessions THE CALLER created (find_latest_session's user_id filter)
    -- a report is the output of one user's own audit run, private to them, even
    when the underlying doc_id is a shared MAIN document another user could also
    run their own audit against."""
    mode = "COMPARISON" if request.doc_id_prior else "SINGLE_TB"
    session = find_latest_session(request.doc_id, request.doc_id_prior, user_id=user.user_id)
    if not session:
        raise HTTPException(
            status_code=404,
            detail=f"No completed audit run found for doc_id={request.doc_id!r} "
            f"doc_id_prior={request.doc_id_prior!r}.",
        )

    run_dir = resolve_output_dir(f"sessions/{session['session_id']}")
    report_dir = Path(run_dir) / "comparison" if mode == "COMPARISON" else Path(run_dir)

    fmt = (request.format or "xlsx").lower()
    base_filename = _WORKBOOK_FILENAMES.get((mode, fmt))
    if not base_filename:
        raise HTTPException(status_code=400, detail=f"No report available for mode={mode!r} format={fmt!r}.")

    file_path = report_dir / base_filename
    if not file_path.exists() and not _fetch_report_from_object_store(session["session_id"], file_path):
        raise HTTPException(
            status_code=404,
            detail=f"Report not found for session {session['session_id']} (expected {file_path}).",
        )

    stem, ext = base_filename.rsplit(".", 1)
    download_filename = f"{stem}{_tb_filename_suffix(Path(run_dir), mode)}.{ext}"

    return FileResponse(path=str(file_path), filename=download_filename, media_type=_WORKBOOK_MEDIA_TYPES[fmt])

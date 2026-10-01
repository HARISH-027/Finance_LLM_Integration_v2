"""Master database module (flattened, single-file — mirrors backend/tools.py).

Layout: connection pool + cursor context manager -> MAIN/LIVE repository (over
document_table/tb_table and live_document_table/live_tb_table) -> pipeline_sessions /
pipeline_findings / pipeline_artifacts session store. Formerly
backend/db/{connection,repository,session_store}.py.
"""

import json
import uuid
from contextlib import contextmanager
from typing import Optional

import psycopg2
import psycopg2.errorcodes
import psycopg2.extras
import psycopg2.pool

from modes.trial_balance.pipeline.config import settings

# Friendly messages for the ingestion error catalog's Section 7 "DB
# constraint safety net" (TB-v2-git/Trial_Balance_ingestion_error.md) --
# these mirror app-level checks (quality_gate.py's new WARN entries) and
# should never normally fire; they exist so a gap in app-level validation
# still surfaces a readable message instead of a raw psycopg2 traceback.
_CONSTRAINT_MESSAGES = {
    psycopg2.errorcodes.UNIQUE_VIOLATION: (
        "A record with this identifier already exists. Please check for a duplicate submission."
    ),
    psycopg2.errorcodes.NOT_NULL_VIOLATION: (
        "A required field (e.g. GL Code or GL Name) was missing when writing to the database. "
        "Please check the source file for blank required columns."
    ),
    psycopg2.errorcodes.FOREIGN_KEY_VIOLATION: (
        "Internal error: line items could not be linked to their document. Please contact support."
    ),
    psycopg2.errorcodes.CHECK_VIOLATION: (
        "Internal error: a value did not meet the required database constraint. Please contact support."
    ),
    psycopg2.errorcodes.STRING_DATA_RIGHT_TRUNCATION: (
        "A value in this file is too long for its column (e.g. an account name or classification "
        "label over 255 characters). Please shorten it and re-upload."
    ),
}


def _translate_db_error(e: Exception) -> str:
    pgcode = getattr(e, "pgcode", None)
    return _CONSTRAINT_MESSAGES.get(pgcode, f"Database operation failed: {e}")

# modes.trial_balance.pipeline.tools imports db_cursor/repository functions from THIS module at its own
# module level, so importing modes.trial_balance.pipeline.tools back here at module level would circularly
# deadlock whichever of the two modules is imported first. PipelineDBError /
# CANONICAL_TB_ALL_COLUMNS / DOCUMENT_OPTIONAL_COLUMNS are only ever needed inside a
# function body below, never at module load time, so each is imported locally at the
# point of use instead -- that fully breaks the cycle in both directions.

# =============================================================================
# CONNECTION (formerly backend/db/connection.py)
# =============================================================================

_pool = None


def _get_pool():
    global _pool
    if _pool is None:
        from modes.trial_balance.pipeline.tools import PipelineDBError

        try:
            # ThreadedConnectionPool, not SimpleConnectionPool: nearly every route in
            # router.py is a plain `def`, which FastAPI runs in a threadpool executor, so
            # concurrent requests from different authenticated users already call
            # getconn()/putconn() from multiple OS threads simultaneously.
            # SimpleConnectionPool's own docstring says it "can't be shared across
            # different threads" -- this was silently unsafe before multi-user traffic
            # was a real scenario.
            _pool = psycopg2.pool.ThreadedConnectionPool(
                settings.DB_POOL_MIN,
                settings.DB_POOL_MAX,
                host=settings.DB_HOST,
                port=settings.DB_PORT,
                dbname=settings.DB_NAME,
                user=settings.DB_USER,
                password=settings.DB_PASSWORD,
            )
        except psycopg2.OperationalError as e:
            raise PipelineDBError(f"Could not connect to trial_balance_db: {e}")
    return _pool


@contextmanager
def db_cursor(dict_rows: bool = True):
    """Checkout a pooled connection + cursor. Commits on success, rolls back on error."""
    from modes.trial_balance.pipeline.tools import PipelineDBError

    pool = _get_pool()
    conn = pool.getconn()
    try:
        cursor_factory = psycopg2.extras.RealDictCursor if dict_rows else None
        with conn.cursor(cursor_factory=cursor_factory) as cur:
            yield cur
        conn.commit()
    except Exception as e:
        conn.rollback()
        if isinstance(e, PipelineDBError):
            raise
        raise PipelineDBError(_translate_db_error(e))
    finally:
        pool.putconn(conn)


# =============================================================================
# REPOSITORY (formerly backend/db/repository.py)
#
# Over document_table/tb_table (MAIN, read-only from this codebase) and
# live_document_table/live_tb_table (LIVE, write-capable for interactive
# ingestion). Operations are split by write scope on purpose — nothing in this
# section can write to MAIN, and nothing reads from LIVE for comparison/report
# purposes, so "which table a UI action touches" is structural, not a runtime flag.
# =============================================================================

# ingest_to_db.py (TB_ingestion) writes mapped_status lowercase ("mapped"/
# "unmapped") straight from the finished TB_GROUPING workbook's own "Mapped
# Status" column text. Every TB-v2 tool compares against the uppercase
# MAPPED_STATUS_* constants (backend/tools.py), so a MAIN row's raw value
# would silently never match -- every DB-sourced document would look 100%
# unmapped downstream. Normalized once here, the single MAIN-read choke
# point, rather than patched in every caller.
_MAPPED_STATUS_UPPER = {"mapped": "MAPPED", "unmapped": "UNMAPPED", "unmatched": "UNMATCHED"}


def _normalize_mapped_status(rows: list) -> list:
    for row in rows:
        status = row.get("mapped_status")
        if isinstance(status, str):
            row["mapped_status"] = _MAPPED_STATUS_UPPER.get(status.strip().lower(), status.strip().upper())
    return rows

# ---------------------------------------------------------------------------
# MAIN (document_table / tb_table) — READ ONLY
# ---------------------------------------------------------------------------


def list_main_documents(entity_id: Optional[str] = None, financial_year: Optional[str] = None) -> list:
    query = "SELECT * FROM document_table WHERE 1=1"
    params = []
    if entity_id:
        query += " AND entity_id = %s"
        params.append(entity_id)
    if financial_year:
        query += " AND financial_year = %s"
        params.append(financial_year)
    query += " ORDER BY fy_period_start DESC NULLS LAST, id DESC"
    with db_cursor() as cur:
        cur.execute(query, params)
        return [dict(r) for r in cur.fetchall()]


def fetch_main_document(tb_doc_id: str) -> Optional[dict]:
    with db_cursor() as cur:
        cur.execute("SELECT * FROM document_table WHERE tb_doc_id = %s LIMIT 1", (tb_doc_id,))
        row = cur.fetchone()
        return dict(row) if row else None


def fetch_main_document_by_entity_fy(entity_id: str, financial_year: str) -> Optional[dict]:
    with db_cursor() as cur:
        cur.execute(
            "SELECT * FROM document_table WHERE entity_id = %s AND financial_year = %s "
            "ORDER BY document_version DESC NULLS LAST, id DESC LIMIT 1",
            (entity_id, financial_year),
        )
        row = cur.fetchone()
        return dict(row) if row else None


# ---------------------------------------------------------------------------
# priority_companies -- company/CIN/financial-year suggestion list for the
# upload picker's optional Company Details fields (see database/schema.sql
# for the table). Not MAIN/LIVE -- its own small reference table, seeded once
# from the client's Priority Companies List and grown at request time from
# explicit user-supplied overrides only (never from the parser's own guesses).
# ---------------------------------------------------------------------------


def list_priority_companies() -> list:
    with db_cursor() as cur:
        cur.execute(
            "SELECT company_name, cin, financial_years FROM priority_companies "
            "WHERE deleted_at IS NULL ORDER BY company_name"
        )
        return [dict(r) for r in cur.fetchall()]


def upsert_priority_companies(rows: list) -> int:
    """Bulk upsert, used by backend/scripts/ingest_priority_companies.py. The
    client's list is authoritative, so a re-run overwrites cin/financial_years
    outright on conflict -- safe to re-run if the client sends an updated list."""
    if not rows:
        return 0
    with db_cursor() as cur:
        psycopg2.extras.execute_batch(
            cur,
            """
            INSERT INTO priority_companies (company_name, cin, financial_years)
            VALUES (%(company_name)s, %(cin)s, %(financial_years)s)
            ON CONFLICT (lower(company_name)) WHERE deleted_at IS NULL DO UPDATE SET
                cin = EXCLUDED.cin,
                financial_years = EXCLUDED.financial_years,
                updated_at = now()
            """,
            [
                {
                    "company_name": r["company_name"],
                    "cin": r.get("cin"),
                    "financial_years": psycopg2.extras.Json(r.get("financial_years") or []),
                }
                for r in rows
            ],
        )
        return len(rows)


def learn_priority_company(company_name: str, cin: Optional[str] = None, financial_year: Optional[str] = None) -> None:
    """Upserts one company from a live user-supplied override (see
    routes.py's /upload-mapped). Merges rather than overwrites -- a new
    financial_year is appended to whatever is already there (never replacing
    prior years), and cin only fills in if the existing row has none -- this
    never downgrades a row the bulk client-list ingestion already populated."""
    if not company_name or not company_name.strip():
        return
    with db_cursor() as cur:
        cur.execute(
            """
            INSERT INTO priority_companies (company_name, cin, financial_years)
            VALUES (%(company_name)s, %(cin)s, %(financial_years)s)
            ON CONFLICT (lower(company_name)) WHERE deleted_at IS NULL DO UPDATE SET
                cin = COALESCE(priority_companies.cin, EXCLUDED.cin),
                financial_years = COALESCE((
                    SELECT jsonb_agg(DISTINCT elem)
                    FROM jsonb_array_elements(priority_companies.financial_years || EXCLUDED.financial_years) AS elem
                ), '[]'::jsonb),
                updated_at = now()
            """,
            {
                "company_name": company_name.strip(),
                "cin": cin.strip() if cin else None,
                "financial_years": psycopg2.extras.Json([financial_year] if financial_year else []),
            },
        )


def fetch_main_lines(tb_doc_id: str) -> list:
    from modes.trial_balance.pipeline.tools import CANONICAL_TB_ALL_COLUMNS

    with db_cursor() as cur:
        # Column list is a fixed internal constant, never user input; the actual
        # value (tb_doc_id) is passed as a %s parameter below (reviewed 2026-09).
        cur.execute(
            f"SELECT {', '.join(CANONICAL_TB_ALL_COLUMNS)} FROM tb_table WHERE tb_doc_id = %s",  # nosec B608
            (tb_doc_id,),
        )
        return _normalize_mapped_status([dict(r) for r in cur.fetchall()])


# ---------------------------------------------------------------------------
# LIVE (live_document_table / live_tb_table) — WRITE (upsert) for interactive ingestion
# ---------------------------------------------------------------------------


def fetch_live_lines(tb_doc_id: str) -> list:
    """LIVE counterpart to fetch_main_lines -- same column contract
    (CANONICAL_TB_ALL_COLUMNS), same table shape (live_tb_table is
    column-for-column identical to tb_table by design), reading GL lines for
    a document that only exists in staging, not yet promoted to MAIN."""
    from modes.trial_balance.pipeline.tools import CANONICAL_TB_ALL_COLUMNS

    with db_cursor() as cur:
        # Column list is a fixed internal constant, never user input; the actual
        # value (tb_doc_id) is passed as a %s parameter below (reviewed 2026-09).
        cur.execute(
            f"SELECT {', '.join(CANONICAL_TB_ALL_COLUMNS)} FROM live_tb_table WHERE tb_doc_id = %s",  # nosec B608
            (tb_doc_id,),
        )
        return _normalize_mapped_status([dict(r) for r in cur.fetchall()])


def fetch_live_document(tb_doc_id: str) -> Optional[dict]:
    """Used to detect (informationally, never to block -- see
    upsert_live_document's own "delete-then-insert by design" docstring)
    that a LIVE row already exists for this tb_doc_id before it gets
    replaced by a new upload."""
    with db_cursor() as cur:
        cur.execute("SELECT * FROM live_document_table WHERE tb_doc_id = %s LIMIT 1", (tb_doc_id,))
        row = cur.fetchone()
        return dict(row) if row else None


def _upsert_live_document_exec(cur, doc: dict) -> None:
    cur.execute("DELETE FROM live_document_table WHERE tb_doc_id = %s", (doc["tb_doc_id"],))
    cur.execute(
        """
        INSERT INTO live_document_table
            (entity_id, entity_name, cin, company_name, fy_period_start, fy_period_end,
             tb_doc_id, tb_doc_name, statement_type, financial_year, has_grouping,
             grouping_doc_id, grouping_doc_name, document_version, modification_dump,
             erp_system, currency, custom_field_3, user_id)
        VALUES (%(entity_id)s, %(entity_name)s, %(cin)s, %(company_name)s,
                %(fy_period_start)s, %(fy_period_end)s, %(tb_doc_id)s, %(tb_doc_name)s,
                %(statement_type)s, %(financial_year)s, %(has_grouping)s,
                %(grouping_doc_id)s, %(grouping_doc_name)s, %(document_version)s,
                %(modification_dump)s, %(erp_system)s, %(currency)s, %(custom_field_3)s,
                %(user_id)s)
        """,
        doc,
    )


def _fill_document_optional_columns(doc: dict) -> dict:
    from modes.trial_balance.pipeline.tools import DOCUMENT_OPTIONAL_COLUMNS

    return {**{c: None for c in DOCUMENT_OPTIONAL_COLUMNS}, **doc}


def upsert_live_document(doc: dict) -> str:
    """Delete-then-insert by tb_doc_id (no unique constraint exists to ON CONFLICT against).
    erp_system/currency/custom_field_3 are optional document-level fields --
    passed through as optional/nullable (doc.get, not doc[...]) so callers
    never have to supply them."""
    doc = _fill_document_optional_columns(doc)
    with db_cursor(dict_rows=False) as cur:
        _upsert_live_document_exec(cur, doc)
    return doc["tb_doc_id"]


def _insert_live_lines_batch_exec(cur, tb_doc_id: str, rows: list) -> int:
    from modes.trial_balance.pipeline.tools import CANONICAL_TB_ALL_COLUMNS

    if not rows:
        return 0
    cols = CANONICAL_TB_ALL_COLUMNS
    values = [tuple(r.get(c) for c in cols) for r in rows]
    cur.execute("DELETE FROM live_tb_table WHERE tb_doc_id = %s", (tb_doc_id,))
    # Column list is a fixed internal constant, never user input; row values are
    # passed through execute_values' own parameterization (reviewed 2026-09).
    psycopg2.extras.execute_values(
        cur,
        f"INSERT INTO live_tb_table ({', '.join(cols)}) VALUES %s",  # nosec B608
        values,
    )
    return len(values)


def insert_live_lines_batch(tb_doc_id: str, rows: list) -> int:
    """Batched insert via execute_values. `rows` are dicts already shaped to
    CANONICAL_TB_ALL_COLUMNS (custom_field_1/2/3 optional/nullable)."""
    if not rows:
        return 0
    with db_cursor(dict_rows=False) as cur:
        return _insert_live_lines_batch_exec(cur, tb_doc_id, rows)


def upsert_live_document_and_lines(doc: dict, rows: list) -> tuple:
    """Atomic combination of upsert_live_document + insert_live_lines_batch:
    the document row and its GL lines are written under the SAME pooled
    connection/cursor inside ONE db_cursor transaction, so they commit or
    roll back together. Before this, ingest_tb_to_live called the two
    functions separately (two independent transactions) -- a crash between
    them could leave a document row with no lines, or stale lines sitting
    under a document row that no longer matches them. Returns
    (tb_doc_id, rows_written)."""
    doc = _fill_document_optional_columns(doc)
    tb_doc_id = doc["tb_doc_id"]
    with db_cursor(dict_rows=False) as cur:
        _upsert_live_document_exec(cur, doc)
        rows_written = _insert_live_lines_batch_exec(cur, tb_doc_id, rows)
    return tb_doc_id, rows_written


# =============================================================================
# SESSION STORE (formerly backend/db/session_store.py)
#
# pipeline_sessions / pipeline_findings CRUD. Modeled on
# trial_balance_v1/postgres_session_store.py, scoped to what the frontend's
# /documents, /documents/{id}, findings/evidence endpoints need.
# =============================================================================


def create_session(mode: str, source: str, tb_doc_id: Optional[str] = None,
                    tb_doc_id_prior: Optional[str] = None, entity_id: Optional[str] = None,
                    financial_year: Optional[str] = None, user_id: Optional[str] = None) -> str:
    """user_id is the authenticated caller's id (router.py threads it in from
    Depends(require_user)) -- get_session()/find_latest_session()/list_sessions()
    still return rows regardless of owner; router.py is where the ownership
    check against the CURRENT caller happens, once per route, not here."""
    session_id = str(uuid.uuid4())
    with db_cursor() as cur:
        cur.execute(
            """
            INSERT INTO pipeline_sessions
                (session_id, mode, tb_doc_id, tb_doc_id_prior, source, status, entity_id, financial_year, user_id)
            VALUES (%s, %s, %s, %s, %s, 'RUNNING', %s, %s, %s)
            """,
            (session_id, mode, tb_doc_id, tb_doc_id_prior, source, entity_id, financial_year, user_id),
        )
    return session_id


def update_session_status(session_id: str, status: str, error_message: Optional[str] = None) -> None:
    with db_cursor() as cur:
        cur.execute(
            "UPDATE pipeline_sessions SET status = %s, error_message = %s, updated_at = now() "
            "WHERE session_id = %s AND deleted_at IS NULL",
            (status, error_message, session_id),
        )


def get_session(session_id: str) -> Optional[dict]:
    with db_cursor() as cur:
        cur.execute(
            "SELECT * FROM pipeline_sessions WHERE session_id = %s AND deleted_at IS NULL",
            (session_id,),
        )
        row = cur.fetchone()
        return dict(row) if row else None


def find_latest_session(tb_doc_id: str, tb_doc_id_prior: Optional[str] = None,
                         status: Optional[str] = "SUCCESS", user_id: Optional[str] = None) -> Optional[dict]:
    """Most recent session for a doc_id (+ optional prior-year doc_id for
    COMPARISON runs) -- used by /audit/workbook to locate which session's
    output_dir holds the report for a given doc_id, since the frontend
    requests downloads by doc_id, not session_id.

    `user_id`, when given, filters to sessions THAT caller created. Without it,
    this would return the globally-latest session for a doc_id regardless of who
    ran it -- fine for a shared MAIN document, wrong for /audit/workbook's own
    per-run report download, where the report is the caller's own analysis
    output, not shared data, even when the underlying doc_id is shared MAIN."""
    query = "SELECT * FROM pipeline_sessions WHERE tb_doc_id = %s AND deleted_at IS NULL"
    params = [tb_doc_id]
    if tb_doc_id_prior:
        query += " AND tb_doc_id_prior = %s"
        params.append(tb_doc_id_prior)
    else:
        query += " AND tb_doc_id_prior IS NULL"
    if status:
        query += " AND status = %s"
        params.append(status)
    if user_id:
        query += " AND user_id = %s"
        params.append(user_id)
    query += " ORDER BY created_at DESC LIMIT 1"
    with db_cursor() as cur:
        cur.execute(query, params)
        row = cur.fetchone()
        return dict(row) if row else None


def list_sessions(limit: int = 50, offset: int = 0) -> list:
    with db_cursor() as cur:
        cur.execute(
            "SELECT * FROM pipeline_sessions WHERE deleted_at IS NULL "
            "ORDER BY created_at DESC LIMIT %s OFFSET %s",
            (limit, offset),
        )
        return [dict(r) for r in cur.fetchall()]


def add_findings(session_id: str, findings: list) -> int:
    """Each finding: {category, severity, statement, evidence_uids: [...], gl_code?}."""
    if not findings:
        return 0
    with db_cursor(dict_rows=False) as cur:
        for f in findings:
            cur.execute(
                """
                INSERT INTO pipeline_findings (session_id, category, severity, statement, evidence_uids, gl_code)
                VALUES (%s, %s, %s, %s, %s, %s)
                """,
                (
                    session_id,
                    f["category"],
                    f["severity"],
                    f["statement"],
                    json.dumps(f.get("evidence_uids", [])),
                    f.get("gl_code"),
                ),
            )
    return len(findings)


def record_tool_artifacts(run_id: str, tool_name: str, artifact_paths: list, pipeline_status: Optional[str] = None) -> None:
    """Best-effort index of which tool wrote what, for a given pipeline_sessions.session_id.
    Purely additive/observability -- downstream tools never read from pipeline_artifacts,
    they still get their input paths from the producing tool's own `artifacts` response
    list (see the pipeline_tool decorator in backend/tools.py)."""
    with db_cursor(dict_rows=False) as cur:
        cur.execute(
            """
            INSERT INTO pipeline_artifacts (session_id, tool_name, artifact_paths, pipeline_status)
            VALUES (%s, %s, %s, %s)
            """,
            (run_id, tool_name, json.dumps(artifact_paths or []), pipeline_status),
        )


def record_artifact_files(session_id: str, tool_name: str, files: list) -> None:
    """Per-file durable-artifact-plane metadata, one row per file in `files`.
    Each entry: {artifact_path, checksum_sha256, size_bytes, format,
    storage_backend, storage_uri, uploaded_at}. Purely additive/observability,
    same posture as record_tool_artifacts -- never read back by any tool, and
    a DB hiccup here must not fail the tool call that produced these files
    (see backend/tools/pipeline_tool.py's _register_artifacts, the only
    caller)."""
    if not files:
        return
    with db_cursor(dict_rows=False) as cur:
        for f in files:
            cur.execute(
                """
                INSERT INTO pipeline_artifact_files
                    (session_id, tool_name, artifact_path, checksum_sha256, size_bytes,
                     format, storage_backend, storage_uri, uploaded_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    session_id,
                    tool_name,
                    f["artifact_path"],
                    f["checksum_sha256"],
                    f["size_bytes"],
                    f["format"],
                    f.get("storage_backend", "local"),
                    f.get("storage_uri"),
                    f.get("uploaded_at"),
                ),
            )


def soft_delete_expired_sessions(older_than_days: int = 90) -> list:
    """Marks pipeline_sessions.deleted_at for every session older than the
    retention window (resolved decision: 90 days) that isn't already soft-
    deleted. Purely a Postgres-side flag flip -- does not touch any local
    file or MinIO object; backend/scripts/cleanup_sessions.py is the
    separate, deliberately-run step that acts on this flag to remove local
    scratch once a durable MinIO copy is confirmed for every file. Returns
    the list of session_ids just flagged, for the caller to log/report."""
    with db_cursor() as cur:
        cur.execute(
            """
            UPDATE pipeline_sessions
            SET deleted_at = now()
            WHERE deleted_at IS NULL
              AND created_at < now() - (%s || ' days')::interval
            RETURNING session_id
            """,
            (older_than_days,),
        )
        return [r["session_id"] for r in cur.fetchall()]


def list_artifact_files_for_session(session_id: str) -> list:
    """All pipeline_artifact_files rows for one session -- used by
    backend/scripts/cleanup_sessions.py to confirm every local file has a
    recorded durable (MinIO) copy before that session's local directory is
    removed."""
    with db_cursor() as cur:
        cur.execute(
            "SELECT * FROM pipeline_artifact_files WHERE session_id = %s AND deleted_at IS NULL",
            (session_id,),
        )
        return [dict(r) for r in cur.fetchall()]


def list_expired_sessions(limit: int = 1000) -> list:
    """Soft-deleted pipeline_sessions rows -- the population
    cleanup_sessions.py iterates over."""
    with db_cursor() as cur:
        cur.execute(
            "SELECT * FROM pipeline_sessions WHERE deleted_at IS NOT NULL ORDER BY deleted_at LIMIT %s",
            (limit,),
        )
        return [dict(r) for r in cur.fetchall()]


def get_findings(session_id: str, severity: Optional[str] = None, category: Optional[str] = None) -> list:
    query = "SELECT * FROM pipeline_findings WHERE session_id = %s AND deleted_at IS NULL"
    params = [session_id]
    if severity:
        query += " AND severity = %s"
        params.append(severity)
    if category:
        query += " AND category = %s"
        params.append(category)
    with db_cursor() as cur:
        cur.execute(query, params)
        return [dict(r) for r in cur.fetchall()]


# =============================================================================
# CHAT HISTORY -- pipeline_chat_messages (0002_add_chat_history.sql)
#
# Mirrors financial_statement/conversations.py's shape: one append-only table,
# "a conversation" is a GROUP BY over conversation_id, user_id is in the WHERE
# clause of every read/delete and never checked afterward (router.py turns an
# empty get_messages() into a 404, not a 403 -- don't confirm another user's
# conversation exists). Lives in TB's own Postgres, not the platform DB, so its
# retention reuses soft_delete_expired_sessions' exact pattern below instead of
# the platform's artha_fs_messages, which has no retention at all today.
# =============================================================================


def new_conversation_id() -> str:
    return str(uuid.uuid4())


# A title is the first user turn, trimmed. Mirrors financial_statement/
# conversations.py's own _title_from() exactly (same char cap, same
# word-boundary trim, same fallback) -- ConversationList.jsx (shared by both
# modes) has always rendered a `title` field; TB's list_conversations only
# ever computed title_src (the raw text) and left turning it into a display
# title as "for the caller to build" (see its own prior docstring), but no
# caller ever did, so every TB conversation showed a blank title in the
# sidebar until now.
_TITLE_CHARS = 70


def _title_from(text: str) -> str:
    text = " ".join((text or "").split())
    if len(text) <= _TITLE_CHARS:
        return text or "New conversation"
    return text[:_TITLE_CHARS].rsplit(" ", 1)[0] + "…"


def list_conversations(user_id: str, limit: int = 50) -> list:
    """One row per conversation_id, most recently active first. tb_doc_id is
    whichever the first turn recorded (NULL for an /ask-general-only thread)."""
    with db_cursor() as cur:
        cur.execute(
            """
            SELECT conversation_id,
                   min(created_at) AS started_at,
                   max(created_at) AS last_at,
                   count(*) AS n_messages,
                   (array_agg(tb_doc_id ORDER BY seq))[1] AS tb_doc_id,
                   (array_agg(content ORDER BY seq) FILTER (WHERE role = 'user'))[1] AS title_src
            FROM pipeline_chat_messages
            WHERE user_id = %s AND deleted_at IS NULL
            GROUP BY conversation_id
            ORDER BY max(created_at) DESC
            LIMIT %s
            """,
            (user_id, limit),
        )
        rows = [dict(r) for r in cur.fetchall()]
        for row in rows:
            row["title"] = _title_from(row["title_src"])
        return rows


def get_messages(user_id: str, conversation_id: str) -> list:
    """Empty for both "no such conversation" and "belongs to someone else" --
    the caller (router.py) turns that into a 404 without confirming which.
    Malformed conversation_id degrades to "not found" rather than a server
    error."""
    from modes.trial_balance.pipeline.tools import PipelineDBError

    try:
        with db_cursor() as cur:
            cur.execute(
                """
                SELECT seq, role, content, payload, tb_doc_id, created_at
                FROM pipeline_chat_messages
                WHERE user_id = %s AND conversation_id = %s AND deleted_at IS NULL
                ORDER BY seq
                """,
                (user_id, conversation_id),
            )
            return [dict(r) for r in cur.fetchall()]
    except PipelineDBError:
        return []


def history_for_agent(user_id: str, conversation_id: str, turns: int = 8) -> list:
    """Last `turns` messages as {"role", "content"} pairs, for seeding the
    agent's rehydrated Memory (pipeline/agent_memory.py) when Valkey's 2-hour
    window has already expired. Unlike Financial Statement's history_for_
    rewriter, TB has no query-rewrite step -- this feeds straight into agent
    memory, not a rewrite prompt."""
    rows = get_messages(user_id, conversation_id)
    return [{"role": r["role"], "content": r["content"]} for r in rows[-turns:]]


def append_turns(user_id: str, conversation_id: str, tb_doc_id: Optional[str],
                  question: str, answer_text: str, payload: Optional[dict] = None) -> None:
    """Records one user+assistant turn pair in a single transaction. seq is
    computed inside the same transaction as the insert; the unique index on
    (conversation_id, seq) is what makes a concurrent-turn collision
    impossible, not merely unlikely. Called only after the agent has already
    answered -- a failed call leaves no orphan question, matching
    conversations.py's own append_turns contract."""
    with db_cursor(dict_rows=False) as cur:
        cur.execute(
            "SELECT coalesce(max(seq), 0) AS n FROM pipeline_chat_messages WHERE conversation_id = %s",
            (conversation_id,),
        )
        next_seq = cur.fetchone()[0]
        cur.execute(
            """
            INSERT INTO pipeline_chat_messages
                (message_id, conversation_id, user_id, tb_doc_id, seq, role, content, payload)
            VALUES (%s, %s, %s, %s, %s, 'user', %s, NULL),
                   (%s, %s, %s, %s, %s, 'assistant', %s, %s)
            """,
            (
                str(uuid.uuid4()), conversation_id, user_id, tb_doc_id, next_seq + 1, question,
                str(uuid.uuid4()), conversation_id, user_id, tb_doc_id, next_seq + 2, answer_text,
                json.dumps(payload) if payload is not None else None,
            ),
        )


def delete_conversation(user_id: str, conversation_id: str) -> int:
    """Hard delete -- a user removing their own conversation should genuinely
    remove it, not just hide it from themselves. Soft-delete (deleted_at) is
    reserved for the retention job below, a different actor with a different
    intent. Malformed conversation_id degrades to "0 rows removed" rather than
    a server error."""
    from modes.trial_balance.pipeline.tools import PipelineDBError

    try:
        with db_cursor(dict_rows=False) as cur:
            cur.execute(
                "DELETE FROM pipeline_chat_messages WHERE user_id = %s AND conversation_id = %s",
                (user_id, conversation_id),
            )
            return cur.rowcount or 0
    except PipelineDBError:
        return 0


def soft_delete_expired_chat_messages(older_than_days: int = 90) -> list:
    """Same shape as soft_delete_expired_sessions -- marks deleted_at for every
    message older than the retention window that isn't already soft-deleted.
    Called alongside soft_delete_expired_sessions by
    backend/scripts/cleanup_sessions.py's --older-than-days flag, so TB's
    entire retention story (sessions + chat) stays on one scheduled job.
    Returns the distinct conversation_ids just flagged, for the caller to
    log/report."""
    with db_cursor() as cur:
        cur.execute(
            """
            UPDATE pipeline_chat_messages
            SET deleted_at = now()
            WHERE deleted_at IS NULL
              AND created_at < now() - (%s || ' days')::interval
            RETURNING conversation_id
            """,
            (older_than_days,),
        )
        # DISTINCT isn't valid directly inside RETURNING -- one row per message
        # updated, so dedupe conversation_ids in Python instead (order-preserving,
        # a UPDATE...RETURNING has no natural order to lose here anyway).
        return list(dict.fromkeys(r["conversation_id"] for r in cur.fetchall()))

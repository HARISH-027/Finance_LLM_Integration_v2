




import polars as pl

# Bumped on every change to the master system prompt (backend/Prompt.md);
# recorded in each run's run_log.json so a finding can be traced to the
# prompt that produced it. Formerly backend/agent/system_prompt.py.
PROMPT_VERSION = "2.0"

CANONICAL_TB_COLUMNS = [
    "tb_doc_id",
    "gl_code",
    "gl_name",
    "opening_balance",
    "debit",
    "credit",
    "closing_balance",
    "bs_pl",
    "sub_head_2",
    "sub_head_1",
    "main_head",
    "account_type",
    "mapped_status",
]

CANONICAL_TB_OPTIONAL_COLUMNS = ["custom_field_1", "custom_field_2", "custom_field_3"]

CANONICAL_TB_ALL_COLUMNS = CANONICAL_TB_COLUMNS + CANONICAL_TB_OPTIONAL_COLUMNS

DOCUMENT_OPTIONAL_COLUMNS = ["erp_system", "currency", "custom_field_3", "user_id"]

CANONICAL_TB_NUMERIC_COLUMNS = ["opening_balance", "debit", "credit", "closing_balance"]

MAPPED_STATUS_MAPPED = "MAPPED"

MAPPED_STATUS_UNMAPPED = "UNMAPPED"

MAPPED_STATUS_UNMATCHED = "UNMATCHED"

def validate_canonical_columns(columns) -> list:
    """Return the list of missing required columns, empty if the frame is valid."""
    have = set(columns)
    return [c for c in CANONICAL_TB_COLUMNS if c not in have]


def mapped_only(df: pl.DataFrame) -> pl.DataFrame:
    """Analysis-stage rows only -- excludes UNMAPPED/UNMATCHED. Validation-stage functions
    (control_totals, data-sufficiency grading, Layer 1/2 validation, Section 2/3 scope tables)
    must NOT call this -- they intentionally cover the full population, mapped and unmapped
    together, since that's the whole point of a validation/integrity check."""
    if df is None or "mapped_status" not in df.columns:
        return df
    return df.filter(pl.col("mapped_status") == MAPPED_STATUS_MAPPED)



# ============================================================================
# SHARED HELPERS (formerly backend/tools/_*.py and per-domain _shared.py)
# ============================================================================


__all__ = [
    'PROMPT_VERSION',
    'CANONICAL_TB_COLUMNS',
    'CANONICAL_TB_OPTIONAL_COLUMNS',
    'CANONICAL_TB_ALL_COLUMNS',
    'DOCUMENT_OPTIONAL_COLUMNS',
    'CANONICAL_TB_NUMERIC_COLUMNS',
    'MAPPED_STATUS_MAPPED',
    'MAPPED_STATUS_UNMAPPED',
    'MAPPED_STATUS_UNMATCHED',
    'validate_canonical_columns',
    'mapped_only',
]

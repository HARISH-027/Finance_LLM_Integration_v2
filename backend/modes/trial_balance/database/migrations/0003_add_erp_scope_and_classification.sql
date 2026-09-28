-- Trial Balance Multi-ERP Adaptability, Phase 9: persistence for ScopeMetadata
-- and ClassificationEvidence (tb_erp_contracts.py) -- data with no existing
-- home in the schema. tb_validation_result/tb_artifact are NOT added here --
-- pipeline_findings/pipeline_artifacts already cover that need (confirmed by
-- reading db.py: add_findings/record_tool_artifacts). tb_taxonomy_resolution
-- is deliberately NOT added -- tb_models.py's TaxonomyResolution dataclass
-- states it is "never persisted as a DB column"; overriding that is an open
-- decision (see the design doc's own section 35), not made here.
--
-- No FK (project convention -- see 0001's own comment). tb_doc_id is TEXT,
-- matching document_table/live_document_table's existing column type.
-- Idempotent -- safe to apply more than once.

CREATE TABLE IF NOT EXISTS tb_scope (
    tb_doc_id       TEXT PRIMARY KEY,
    entity          TEXT,
    company_code    TEXT,
    ledger          TEXT,
    fiscal_year     TEXT,
    period_from     INTEGER,
    period_to       INTEGER,
    currency        TEXT,
    account_range   TEXT,
    business_scope  TEXT,
    profit_center   TEXT,
    segment         TEXT,
    scope_type      TEXT NOT NULL DEFAULT 'unknown',
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ
);

CREATE TABLE IF NOT EXISTS tb_erp_classification (
    id                      UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    tb_doc_id               TEXT NOT NULL,
    gl_code                 TEXT NOT NULL,
    erp_type                TEXT NOT NULL,
    account_identity        JSONB NOT NULL DEFAULT '{}',
    account_classification  JSONB NOT NULL DEFAULT '{}',
    financial_statement     JSONB NOT NULL DEFAULT '{}',
    business_dimensions     JSONB NOT NULL DEFAULT '{}',
    created_at              TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_tb_erp_classification_doc_gl
    ON tb_erp_classification(tb_doc_id, gl_code);

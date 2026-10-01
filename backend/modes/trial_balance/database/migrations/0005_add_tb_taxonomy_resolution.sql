-- Trial Balance: resolves the open decision migration 0003 explicitly deferred
-- ("tb_taxonomy_resolution is deliberately NOT added -- tb_models.py's
-- TaxonomyResolution dataclass states it is 'never persisted as a DB column';
-- overriding that is an open decision (design doc section 35), not made here").
--
-- Per the design doc's own persistence model (documentation/
-- TB_Multi_ERP_Adaptability_Implementation_Upgrade_Design.md, sections 12/20/21),
-- TaxonomyResolution was always meant to be its own persisted table -- keyed the
-- same way as tb_erp_classification (tb_doc_id, gl_code) -- not a column bolted
-- onto the Canonical TB row schema. Client-review defect registry (§Area 2, "per-
-- account mapping confidence level omitted"): the confidence a row was resolved
-- with was already computed during classification (taxonomy_resolver.py /
-- resolution_metadata.py) and simply discarded before persistence -- no auditor
-- could see whether a mapped account was an exact/alias match or a shaky LLM
-- guess. This table gives that data a real home without touching
-- CANONICAL_TB_COLUMNS or any of its many consumers.
--
-- mapping_confidence is the reconciled High/Medium/Low vocabulary the client
-- spec asks for (deterministic resolver methods -> High; CANDIDATE_AUTO bucketed
-- by its own confidence_score; the LLM path's own retrieval_confidence passed
-- through where available). NULL for UNMAPPED/UNMATCHED rows -- mapped_status
-- already conveys that; this column only ever answers "how sure were we", not
-- "did we map it at all". confidence_score keeps the resolver's raw float
-- (EXACT=1.0/ALIAS=0.95/DETERMINISTIC_RULE=0.85/CANDIDATE_AUTO's real score) for
-- anyone who wants the number behind the bucket, not just the bucket.
--
-- No FK (project convention -- see 0001's own comment). tb_doc_id/gl_code are
-- TEXT, matching tb_erp_classification's existing column types.
-- Idempotent -- safe to apply more than once.

CREATE TABLE IF NOT EXISTS tb_taxonomy_resolution (
    id                   UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    tb_doc_id            TEXT NOT NULL,
    gl_code              TEXT NOT NULL,
    mapped_status        TEXT NOT NULL,
    resolution_method    TEXT,
    mapping_confidence   TEXT,
    confidence_score     DOUBLE PRECISION,
    evidence             TEXT,
    taxonomy_node_id     INTEGER,
    taxonomy_version_id  INTEGER,
    created_at           TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_tb_taxonomy_resolution_doc_gl
    ON tb_taxonomy_resolution(tb_doc_id, gl_code);

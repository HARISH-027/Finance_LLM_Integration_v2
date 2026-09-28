-- Trial Balance: rename document-level custom_field_1/custom_field_2 to
-- erp_system/currency on BOTH document_table (MAIN) and live_document_table
-- (LIVE) -- the actual field names canonical_schema.py's
-- DOCUMENT_OPTIONAL_COLUMNS and db.py's live_document_table INSERT SQL have
-- expected since the Multi-ERP Adaptability upgrade's Phase 0 (see the
-- design doc's own section 1.6.5/34/35 item 0). custom_field_3 stays
-- reserved/untouched on both tables.
--
-- This exact rename was applied ad-hoc earlier (not as a saved migration
-- file) and was lost when document_table/live_document_table were reset --
-- an external process owns document_table's data (see 0001's own comment),
-- so an unmigrated column rename does not survive that process's own
-- schema regeneration. Saving it here as a real migration this time so it's
-- reproducible.
--
-- Does NOT touch tb_table/live_tb_table's OWN custom_field_1/2/3 -- those
-- are a separate, line-level concept (CANONICAL_TB_OPTIONAL_COLUMNS) that
-- was never renamed and must stay as-is.
--
-- Idempotent -- safe to apply more than once.

-- Postgres has no "RENAME COLUMN IF EXISTS old_name" -- a plain RENAME
-- would fail on a second apply once custom_field_1/2 no longer exist.
-- Guarding each rename by information_schema is what makes this safe to
-- run more than once.
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'document_table' AND column_name = 'custom_field_1') THEN
        ALTER TABLE document_table RENAME COLUMN custom_field_1 TO erp_system;
    END IF;
    IF EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'document_table' AND column_name = 'custom_field_2') THEN
        ALTER TABLE document_table RENAME COLUMN custom_field_2 TO currency;
    END IF;
    IF EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'live_document_table' AND column_name = 'custom_field_1') THEN
        ALTER TABLE live_document_table RENAME COLUMN custom_field_1 TO erp_system;
    END IF;
    IF EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'live_document_table' AND column_name = 'custom_field_2') THEN
        ALTER TABLE live_document_table RENAME COLUMN custom_field_2 TO currency;
    END IF;
END $$;

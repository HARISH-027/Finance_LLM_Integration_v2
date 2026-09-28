"""ERP-adaptability persistence (TB Multi-ERP Adaptability upgrade, Phase 9
-- see modes/trial_balance/documentation/
TB_Multi_ERP_Adaptability_Implementation_Upgrade_Design.md, sections 20/22).

Persists ScopeMetadata (tb_scope table) and ClassificationEvidence
(tb_erp_classification table) -- the two Phase 1 contracts that had no
existing home anywhere in this schema. tb_validation_result/tb_artifact were
NOT added as new tables: pipeline_findings/pipeline_artifacts (db.py's
existing add_findings/record_tool_artifacts) already cover that need.
tb_taxonomy_resolution was deliberately not added -- tb_models.py's
TaxonomyResolution dataclass states it is "never persisted as a DB column";
overriding that is an open decision (design doc section 35), not made here.

Purely additive: not wired into input_dispatch.py/db_bridge.py's live
ingestion path. Imports db.py's existing db_cursor/add_findings rather than
modifying db.py itself.
"""

from __future__ import annotations

import json
from typing import Dict, List, Optional

from modes.trial_balance.pipeline.db import db_cursor
from modes.trial_balance.pipeline.tools.tb_erp_contracts import (
    ClassificationEvidence,
    ERPType,
    ScopeMetadata,
    ScopeType,
)


def upsert_tb_scope(tb_doc_id: str, scope: ScopeMetadata) -> None:
    """Upsert by tb_doc_id (primary key) -- re-running ingestion for the
    same document replaces its scope row, never accumulates duplicates."""
    with db_cursor(dict_rows=False) as cur:
        cur.execute(
            """
            INSERT INTO tb_scope
                (tb_doc_id, entity, company_code, ledger, fiscal_year, period_from, period_to,
                 currency, account_range, business_scope, profit_center, segment, scope_type, updated_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, now())
            ON CONFLICT (tb_doc_id) DO UPDATE SET
                entity = EXCLUDED.entity, company_code = EXCLUDED.company_code, ledger = EXCLUDED.ledger,
                fiscal_year = EXCLUDED.fiscal_year, period_from = EXCLUDED.period_from,
                period_to = EXCLUDED.period_to, currency = EXCLUDED.currency,
                account_range = EXCLUDED.account_range, business_scope = EXCLUDED.business_scope,
                profit_center = EXCLUDED.profit_center, segment = EXCLUDED.segment,
                scope_type = EXCLUDED.scope_type, updated_at = now()
            """,
            (tb_doc_id, scope.entity, scope.company_code, scope.ledger, scope.fiscal_year,
             scope.period_from, scope.period_to, scope.currency, scope.account_range,
             scope.business_scope, scope.profit_center, scope.segment, scope.scope_type.value),
        )


def fetch_tb_scope(tb_doc_id: str) -> Optional[ScopeMetadata]:
    with db_cursor() as cur:
        cur.execute("SELECT * FROM tb_scope WHERE tb_doc_id = %s", (tb_doc_id,))
        row = cur.fetchone()
        if row is None:
            return None
        return ScopeMetadata(
            entity=row["entity"], company_code=row["company_code"], ledger=row["ledger"],
            fiscal_year=row["fiscal_year"], period_from=row["period_from"], period_to=row["period_to"],
            currency=row["currency"], account_range=row["account_range"], business_scope=row["business_scope"],
            profit_center=row["profit_center"], segment=row["segment"], scope_type=ScopeType(row["scope_type"]),
        )


def upsert_tb_erp_classification(tb_doc_id: str, classification_by_gl_code: Dict[str, ClassificationEvidence]) -> int:
    """Delete-then-insert by tb_doc_id -- same pattern insert_live_lines_batch
    already uses for live_tb_table (re-ingesting a document replaces its
    classification rows, never accumulates duplicates)."""
    with db_cursor(dict_rows=False) as cur:
        cur.execute("DELETE FROM tb_erp_classification WHERE tb_doc_id = %s", (tb_doc_id,))
        for gl_code, evidence in classification_by_gl_code.items():
            cur.execute(
                """
                INSERT INTO tb_erp_classification
                    (tb_doc_id, gl_code, erp_type, account_identity, account_classification,
                     financial_statement, business_dimensions)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                """,
                (tb_doc_id, gl_code, evidence.erp_type.value, json.dumps(evidence.account_identity),
                 json.dumps(evidence.account_classification), json.dumps(evidence.financial_statement),
                 json.dumps(evidence.business_dimensions)),
            )
    return len(classification_by_gl_code)


def fetch_tb_erp_classification(tb_doc_id: str) -> Dict[str, ClassificationEvidence]:
    with db_cursor() as cur:
        cur.execute("SELECT * FROM tb_erp_classification WHERE tb_doc_id = %s", (tb_doc_id,))
        return {
            row["gl_code"]: ClassificationEvidence(
                erp_type=ERPType(row["erp_type"]), account_identity=row["account_identity"],
                account_classification=row["account_classification"],
                financial_statement=row["financial_statement"], business_dimensions=row["business_dimensions"],
            )
            for row in cur.fetchall()
        }


def findings_to_pipeline_finding_dicts(findings: list, *, category: str = "erp_adaptability") -> List[dict]:
    """Converts any of Phase 3/5/6's Finding-shaped objects (TallyAdapterFinding/
    SAPAdapterFinding/ValidationFinding -- all (severity, code, message)) into
    the dict shape db.py's EXISTING add_findings(session_id, findings) already
    expects: {category, severity, statement, evidence_uids, gl_code}.

    Known limitation, not silently worked around: none of these Finding types
    carry a structured gl_code field today (only baked into the free-text
    message for row-level findings) -- gl_code is left None here rather than
    regex-parsed out of the message text. Making Finding carry an optional
    gl_code would require touching three already-shipped Phase 3/5/6 files;
    flagged as a follow-up, not done in this phase."""
    return [
        {
            "category": category,
            "severity": f.severity.upper() if f.severity.islower() else f.severity,
            "statement": f.message,
            "evidence_uids": [f.code],
            "gl_code": None,
        }
        for f in findings
    ]

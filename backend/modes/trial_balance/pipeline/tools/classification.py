"""Step 3: single-shot batch classification for GL rows that reached a
grouping match but couldn't resolve directly.

The model SELECTS from the closed Schedule III taxonomy tree -- never
generates free text -- and every returned selection is validated against
the taxonomy before being trusted (`snap_to_taxonomy`), so an off-taxonomy
string can never make it into output. account_type is never requested from
the LLM at all; it's derived deterministically after grouping is resolved.

Ported from TB_normalization_v1's core/classify.py.
"""

from __future__ import annotations

import json
import logging

from modes.trial_balance.pipeline.tools import taxonomy_repository as repo
from modes.trial_balance.pipeline.tools._shared import Standard
from modes.trial_balance.pipeline.tools.candidate_retrieval import (
    EXPANSION_TOP_K,
    INJECTED_GENERIC_SCORE,
    all_generic_bucket_nodes,
    classify_confidence,
    filter_generic_buckets_by_bs_pl,
    rank_and_cap_candidates,
)
from modes.trial_balance.pipeline.tools.classify_examples import FEW_SHOT
from modes.trial_balance.pipeline.tools.llm_call import call_with_empty_batch_retry, read_content
from modes.trial_balance.pipeline.tools.taxonomy_repository import build_taxonomy_tree, derive_account_type, snap_to_taxonomy
from modes.trial_balance.pipeline.tools.tb_models import MatchResult
from modes.trial_balance.pipeline.tools.vocab import AMBIGUOUS_VOCAB, LIAB_ABBREV_RE

logger = logging.getLogger(__name__)

BATCH_SIZE = 15

SYSTEM_PROMPT = """You are a financial audit assistant classifying General Ledger \
(GL) accounts from a Trial Balance into a closed Schedule III grouping taxonomy \
(Companies Act 2013). You will be given the taxonomy as a TREE, not a flat list: \
a JSON array where each item is one main_head (with its bs_pl) and a nested \
"sub_heads" list; each sub_head has its own sub_head_1 name and a \
"sub_head_2_options" list that belongs ONLY to that sub_head_1.

For each GL account, descend the tree in three steps, in order:
1. Find the ONE main_head entry (and its bs_pl) that best matches the account.
2. Within THAT entry's "sub_heads" list ONLY, pick the best-matching sub_head_1.
3. Copy sub_head_2 VERBATIM from THAT sub_head_1's own "sub_head_2_options" list \
ONLY -- never pull a sub_head_2 string from a different sub_head_1's list, even \
if it reads as a closer textual match. Never invent new wording, never paraphrase, \
never merge fields from two different branches of the tree.

If nothing in the tree plausibly fits an account, use null for main_head, \
sub_head_1, and sub_head_2 for that row -- never force a weak guess. A row you're \
not confident about should come back null, not your closest guess.

Do not return an account_type field -- it is not requested.

IMPORTANT: bs_pl must be EXACTLY the two-letter code "BS" or "PL" -- never the \
main_head text itself.

Use the GL name and, if given, a "hint" (a label pulled from that company's own \
grouping workbook) as evidence -- the hint may be incomplete, wrong, or just a \
note/section label, so use your own accounting judgement about which taxonomy \
branch actually fits. Respond with ONLY a JSON array, one object per input row, \
in any order. Each object MUST include "gl_code" copied VERBATIM from the input \
row it classifies -- this is how your answer gets matched back to the right row, \
so never omit it, never alter it, and never invent a gl_code that wasn't given to \
you. Each object's keys: gl_code, bs_pl, main_head, sub_head_1, sub_head_2. Return \
exactly one object per input row -- no duplicates, no extra rows, no missing rows. \
No prose, no markdown fences."""

CANDIDATE_ID_SYSTEM_PROMPT = """You are a financial audit assistant classifying General Ledger \
(GL) accounts from a Trial Balance into ONE of a pre-selected shortlist of Schedule III taxonomy \
candidates (Companies Act 2013). You will be given a numbered candidate list -- a JSON array where \
each item has an "id", "main_head", "sub_head_1", and "sub_head_2".

For each GL account, pick the ONE candidate id that best matches -- copy the id VERBATIM from the \
list. Never invent a new id, never combine fields from two different candidates, never return a \
main_head/sub_head_1/sub_head_2 string of your own.

If nothing in the candidate list plausibly fits an account, use null for candidate_id for that row \
-- never force a weak guess, and never fall back to inventing your own taxonomy strings just \
because the list doesn't contain a good match.

Respond with ONLY a JSON array, one object per input row, in any order. Each object MUST include \
"gl_code" copied VERBATIM from the input row -- this is how your answer gets matched back to the \
right row. Each object's keys: gl_code, candidate_id. Return exactly one object per input row -- no \
duplicates, no extra rows, no missing rows. No prose, no markdown fences."""


def gl_name_conflicts_with_hint(gl_name: str, hint: str) -> bool:
    """For the ambiguous-vocabulary trigger list, the GL name's own wording
    is trusted over a grouping hint that might contradict it. Tells the
    caller WHETHER the trigger vocabulary is present -- it doesn't resolve
    the conflict itself; that's still the LLM's job, informed by the
    prompt wording in staged_narrowing.STAGE1_SYSTEM_PROMPT."""
    text = f"{gl_name} {hint}".lower()
    return any(term in text for term in AMBIGUOUS_VOCAB) or bool(LIAB_ABBREV_RE.search(text))


# Below this many candidate nodes, a batch-level narrowed tree risks
# excluding the right answer for some row in the batch -- fall back to the
# full tree rather than gamble on a too-thin shortlist.
MIN_CANDIDATES_FOR_NARROWING = 8

# Retrieval top_k for each row's OWN query, before the per-batch union.
# Per-row retrieval guarantees each row's own text determines whether ITS
# candidates make the shortlist, independent of what else is in the batch
# (a batch-combined query was measured to drown out individual rows'
# vocabulary).
_PER_ROW_TOP_K = 15


def _identity(node: dict) -> tuple:
    return (node["bs_pl"], node["main_head"], node["sub_head_1"], node["sub_head_2"])


def _row_candidates(query_text: str, standard: Standard, *, top_k: int = _PER_ROW_TOP_K) -> list:
    """Embedding-first retrieval for one row's own query text, graceful
    fallback to keyword scoring. Returns SCORED (node, score) pairs."""
    scored = repo.get_candidates_with_scores_embedding(query_text, standard, top_k=top_k)
    if scored:
        return scored
    return repo.get_candidates_with_scores(query_text, standard, top_k=top_k)


def _row_candidates_with_confidence(query_text: str, standard: Standard) -> tuple:
    scored = _row_candidates(query_text, standard, top_k=_PER_ROW_TOP_K)
    top_score = scored[0][1] if scored else None
    second_score = scored[1][1] if len(scored) > 1 else None
    return scored, classify_confidence(top_score, second_score)


def _adaptive_row_candidates(scored: list, confidence, query_text: str, standard: Standard) -> list:
    """If `confidence` doesn't reach HIGH, retrieves ONCE more at the
    confidence-indicated expanded top_k (30 or 60) and uses that instead --
    one extra retrieval call at most per row, never a second LLM
    round-trip."""
    expanded_k = EXPANSION_TOP_K[confidence]
    if expanded_k <= _PER_ROW_TOP_K:
        return scored
    return _row_candidates(query_text, standard, top_k=expanded_k)


def _majority_bs_pl(candidates: list) -> str:
    """The most common bs_pl among a row's OWN retrieved candidates --
    deliberately NOT the grouping hint's own bs_pl, which can itself be
    wrong."""
    if not candidates:
        return ""
    counts: dict = {}
    for node in candidates:
        counts[node["bs_pl"]] = counts.get(node["bs_pl"], 0) + 1
    return max(counts, key=counts.get)


def _get_candidates_or_none(
    batch: list, standard: Standard, *, use_adaptive_candidates: bool = True,
) -> tuple:
    """Retrieves candidates PER ROW, then unions the results into one
    shortlist shown to the LLM for the whole batch call. Unions in
    catch-all leaves ("Other expenses", ...) compatible with each row's OWN
    majority-retrieved bs_pl, ranks + caps the union.

    Returns (candidates_or_none, confidence_by_gl). candidates is None
    when the union is too few to trust, so callers fall back to the
    unmodified full tree.

    FULL-TREE FALLBACK DISABLED FOR NOW (explicit instruction: "comment it
    out entirely for now, map it for future look case to work when
    required the full-tree fallback, always use candidates for now"). The
    MIN_CANDIDATES_FOR_NARROWING threshold check below is commented out,
    not deleted -- always uses whatever candidates were retrieved, even
    below that threshold. To re-enable the safety net, uncomment the two
    lines in the block below. Known side effect of disabling it: in the
    (rare) case where retrieval finds literally ZERO candidates for every
    row in a batch (embedding client down AND no keyword matches AND no
    matching generic bucket), this still returns an empty list rather than
    None -- callers will show the LLM an empty shortlist for that batch,
    not the full tree."""
    scored_nodes: dict = {}
    known_main_heads: set = set()
    known_bs_pls: set = set()
    confidence_by_gl: dict = {}
    generic_nodes = all_generic_bucket_nodes(standard)

    for row in batch:
        query_text = f"{row.gl_name} {row.hint}"
        row_scored, confidence = _row_candidates_with_confidence(query_text, standard)
        confidence_by_gl[row.gl_code] = confidence.value
        if use_adaptive_candidates:
            row_scored = _adaptive_row_candidates(row_scored, confidence, query_text, standard)
        for node, score in row_scored:
            identity = _identity(node)
            if identity not in scored_nodes or score > scored_nodes[identity][1]:
                scored_nodes[identity] = (node, score)

        if row.known_main_head:
            known_main_heads.add(row.known_main_head)
        if row.known_bs_pl:
            known_bs_pls.add(row.known_bs_pl)

        majority_bs_pl = _majority_bs_pl([node for node, _ in row_scored])
        for node in filter_generic_buckets_by_bs_pl(generic_nodes, bs_pl=majority_bs_pl):
            identity = _identity(node)
            scored_nodes.setdefault(identity, (node, INJECTED_GENERIC_SCORE))

    # Disabled for now -- see docstring above. Uncomment to restore the
    # full-tree fallback for a too-thin candidate shortlist:
    # if len(scored_nodes) < MIN_CANDIDATES_FOR_NARROWING:
    #     return None, confidence_by_gl
    return (
        rank_and_cap_candidates(scored_nodes, known_main_heads=frozenset(known_main_heads), known_bs_pls=frozenset(known_bs_pls)),
        confidence_by_gl,
    )


def _tree_from_candidates(candidates: list) -> list:
    tree: dict = {}
    for node in candidates:
        entry = tree.setdefault(node["main_head"], {"bs_pl": node["bs_pl"], "main_head": node["main_head"], "sub_heads": []})
        sub = next((s for s in entry["sub_heads"] if s["sub_head_1"] == node["sub_head_1"]), None)
        if sub is None:
            sub = {"sub_head_1": node["sub_head_1"], "sub_head_2_options": []}
            entry["sub_heads"].append(sub)
        sub["sub_head_2_options"].append(node["sub_head_2"])
    return list(tree.values())


def _build_tree_messages(batch: list, standard: Standard, taxonomy_tree: list) -> list:
    input_rows = [{"gl_code": r.gl_code, "gl_name": r.gl_name, "hint": r.hint} for r in batch]
    user_content = (
        f"Taxonomy tree ({standard}) -- descend main_head -> sub_heads -> "
        f"sub_head_2_options, never cross branches:\n{json.dumps(taxonomy_tree)}\n\n"
        f"Worked examples (selection mechanic, not exhaustive):\n{json.dumps(FEW_SHOT[standard])}\n\n"
        f"Classify these {len(batch)} GL rows (respond with a JSON array of exactly "
        f"{len(batch)} objects, same order not required):\n{json.dumps(input_rows)}"
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def _build_candidate_id_messages(batch: list, candidates: list, *, use_hierarchical_candidate_context: bool = True) -> tuple:
    """The LLM selects a candidate_id from `candidates` instead of echoing
    taxonomy strings. IDs are request-scoped labels ("C1", "C2", ...)
    assigned here, meaningful only for resolving this one response back to
    a node already fetched from the authoritative taxonomy."""
    candidate_lookup = {f"C{i + 1}": node for i, node in enumerate(candidates)}
    if use_hierarchical_candidate_context:
        candidate_list = [
            {"id": cid, "bs_pl": node["bs_pl"], "main_head": node["main_head"], "sub_head_1": node["sub_head_1"], "sub_head_2": node["sub_head_2"]}
            for cid, node in candidate_lookup.items()
        ]
    else:
        candidate_list = [
            {"id": cid, "main_head": node["main_head"], "sub_head_1": node["sub_head_1"], "sub_head_2": node["sub_head_2"]}
            for cid, node in candidate_lookup.items()
        ]
    input_rows = [{"gl_code": r.gl_code, "gl_name": r.gl_name, "hint": r.hint} for r in batch]
    user_content = (
        f"Candidate list:\n{json.dumps(candidate_list)}\n\n"
        f"Classify these {len(batch)} GL rows (respond with a JSON array of exactly "
        f"{len(batch)} objects, same order not required):\n{json.dumps(input_rows)}"
    )
    messages = [
        {"role": "system", "content": CANDIDATE_ID_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
    return messages, candidate_lookup


def _call_batch_once(
    batch: list, standard: Standard, llm_client, *, use_candidates: bool = True,
    use_candidate_id_contract: bool = False, use_adaptive_candidates: bool = True,
    use_hierarchical_candidate_context: bool = True,
) -> dict:
    """A single LLM call attempt for one batch. Returns {gl_code:
    MatchResult} for whatever resolved, or {} on any failure (parse error,
    off-taxonomy answer, explicit null, or an empty/malformed response) --
    never raises."""
    candidate_lookup = None
    confidence_by_gl: dict = {}
    if use_candidates:
        candidates, confidence_by_gl = _get_candidates_or_none(batch, standard, use_adaptive_candidates=use_adaptive_candidates)
    else:
        candidates = None

    if candidates is not None and use_candidate_id_contract:
        messages, candidate_lookup = _build_candidate_id_messages(
            batch, candidates, use_hierarchical_candidate_context=use_hierarchical_candidate_context,
        )
    elif candidates is not None:
        messages = _build_tree_messages(batch, standard, _tree_from_candidates(candidates))
    else:
        messages = _build_tree_messages(batch, standard, build_taxonomy_tree(standard))

    try:
        response = llm_client.generate(messages, max_tokens=max(2000, 120 * len(batch)), temperature=0)
        content = read_content(response)
        cleaned = content.strip()
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        if cleaned.startswith("```"):
            cleaned = cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        parsed = json.loads(cleaned.strip())
    except Exception:
        return {}

    if not isinstance(parsed, list):
        return {}

    results: dict = {}
    for selection in parsed:
        if not isinstance(selection, dict) or not selection.get("gl_code"):
            continue

        if candidate_lookup is not None:
            candidate_id = selection.get("candidate_id")
            if not candidate_id or candidate_id not in candidate_lookup:
                continue  # missing, null, or out-of-set id -- unresolved, never coerced to a nearest match.
            node = candidate_lookup[candidate_id]
            match = snap_to_taxonomy(node["bs_pl"], node["main_head"], node["sub_head_1"], node["sub_head_2"], standard)
        else:
            if not selection.get("main_head") or not selection.get("sub_head_1") or not selection.get("sub_head_2"):
                continue  # explicit null -- nothing plausible for this row, leave unresolved.
            match = snap_to_taxonomy(
                selection.get("bs_pl"), selection.get("main_head"), selection.get("sub_head_1"),
                selection.get("sub_head_2"), standard,
            )

        if match is None:
            continue  # off-taxonomy answer -- rejected, never trusted verbatim.
        gl_code = str(selection["gl_code"])
        results[gl_code] = MatchResult(
            bs_pl=match["bs_pl"], main_head=match["main_head"], sub_head_1=match["sub_head_1"],
            sub_head_2=match["sub_head_2"], account_type=derive_account_type(match["bs_pl"], match["main_head"], standard),
            retrieval_confidence=confidence_by_gl.get(gl_code),
        )
    return results


def classify_batch(
    batch: list, standard: Standard, llm_client, *, use_candidates: bool = True,
    use_candidate_id_contract: bool = False, use_adaptive_candidates: bool = True,
    use_hierarchical_candidate_context: bool = True,
) -> dict:
    """Classifies up to BATCH_SIZE rows in one LLM call (with one
    retry-on-empty-batch), matched back by the LLM-echoed gl_code, never
    by array position. Returns {} immediately, no call made, if
    llm_client is None or batch is empty."""
    if not llm_client or not batch:
        return {}
    return call_with_empty_batch_retry(
        lambda: _call_batch_once(
            batch, standard, llm_client, use_candidates=use_candidates,
            use_candidate_id_contract=use_candidate_id_contract, use_adaptive_candidates=use_adaptive_candidates,
            use_hierarchical_candidate_context=use_hierarchical_candidate_context,
        )
    )


def classify_all(
    rows: list, standard: Standard, llm_client, *, use_candidates: bool = True,
    use_candidate_id_contract: bool = False, use_adaptive_candidates: bool = True,
    use_hierarchical_candidate_context: bool = True,
) -> dict:
    """Chunks `rows` into BATCH_SIZE-sized calls and merges results. This
    is the true choke point every caller funnels through (normalize()'s
    single-shot path and validation_gate.py's repair pass both call this),
    so the known-bad `use_candidates=True` + `use_adaptive_candidates=False`
    combination (a narrow, non-adaptive shortlist can omit the correct
    node) is guarded here once."""
    if use_candidates and not use_adaptive_candidates:
        logger.warning(
            "use_candidates=True with use_adaptive_candidates=False recreates a known-bad "
            "combination -- forcing use_adaptive_candidates=True."
        )
        use_adaptive_candidates = True
    results: dict = {}
    for i in range(0, len(rows), BATCH_SIZE):
        results.update(
            classify_batch(
                rows[i:i + BATCH_SIZE], standard, llm_client, use_adaptive_candidates=use_adaptive_candidates,
                use_candidates=use_candidates, use_candidate_id_contract=use_candidate_id_contract,
                use_hierarchical_candidate_context=use_hierarchical_candidate_context,
            )
        )
    return results

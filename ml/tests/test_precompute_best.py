"""Предрасчёт демо-карточек: лучшая из нескольких генераций (ml/precompute.py, --best-of)."""

from ml.agent.orchestrator import load_card
from ml.llm.client import LLMClient
from ml.precompute import best_of, card_score
from ml.tests.conftest import needs_data


@needs_data
def test_best_of_saves_the_lowest_score_card():
    card = best_of("U_1595", "n_2026_09_11_cbr_rate", 2, None, client=LLMClient(mock=True, use_cache=False))
    assert load_card(card.card_id).trace_id == card.trace_id


@needs_data
def test_card_score_prefers_fewer_issues_and_inferences():
    card = load_card(best_of("U_1595", "n_2026_09_11_cbr_rate", 1, None, client=LLMClient(mock=True, use_cache=False)).card_id)
    worse = card.model_copy(update={"meta": card.meta.model_copy(update={"verifier_issues": ["x"]})})
    assert card_score(card) < card_score(worse)

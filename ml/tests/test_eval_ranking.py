"""Бейзлайны и метрика персонализации eval: на маленьких синтетических примерах."""

import pytest

from ml.eval.run_eval import auc_score, material_auc, popularity_loo


def test_auc_score_ties_and_empty():
    assert auc_score([3, 4], [1, 2]) == 1.0
    assert auc_score([1], [2]) == 0.0
    assert auc_score([1, 3], [2]) == pytest.approx(0.5)
    assert auc_score([1], [1]) == 0.5
    assert auc_score([], [1]) is None and auc_score([1], []) is None


def test_popularity_loo_excludes_own_labels():
    gold = {"a": {"x": 2, "y": 0}, "b": {"x": 0}, "c": {"x": 1, "z": 1}}
    assert popularity_loo(gold, "a") == {"x": 0.5, "z": 1}
    assert popularity_loo(gold, "c") == {"x": 1, "y": 0}


def test_material_auc_skips_one_class_items_and_weights():
    gold = {"a": {"x": 2, "y": 1}, "b": {"x": 0, "y": 1}, "c": {"x": 1}}
    scores = {"a": {"x": 0.9, "y": 0.5}, "b": {"x": 0.1, "y": 0.4}, "c": {"x": 0.2}}
    r = material_auc(scores, gold, ["a", "b", "c"])
    # y: только положительные, пропущен; x: pos a=0.9, c=0.2, neg b=0.1 -> AUC 1
    assert r["n"] == 1 and r["median"] == 1.0 and r["weighted"] == 1.0
    scores["c"]["x"] = 0.05
    assert material_auc(scores, gold, ["a", "b", "c"])["median"] == pytest.approx(0.5)


def test_count_repaired_only_counts_interpret_calls():
    from ml.eval.run_eval import count_repaired

    ev = [{"event": "llm_call", "task": "interpret", "repaired": True},
          {"event": "llm_call", "task": "interpret", "repaired": False},
          {"event": "llm_call", "task": "claim_verify", "repaired": True},
          {"event": "verify", "task": "interpret", "repaired": True}]
    assert count_repaired(ev) == 1
    assert count_repaired(ev, "claim_verify") == 1


def test_disputed_variants_pick_grades():
    from ml.eval.run_eval import load_gold_v2, load_gold_v2_disputed

    row = next(l for l in open("ml/eval/labels.jsonl", encoding="utf-8") if '"llm_disputed"' in l)
    import json
    r = json.loads(row)
    c = {"id": r["company"]}
    assert load_gold_v2()(c, r["item"]) is None
    assert load_gold_v2_disputed("A")(c, r["item"]) == r["grade_a"]
    assert load_gold_v2_disputed("min")(c, r["item"]) == min(r["grade_a"], r["grade_b"])
    assert load_gold_v2_disputed("max")(c, r["item"]) == max(r["grade_a"], r["grade_b"])

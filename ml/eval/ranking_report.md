# Eval ранжирования ленты: прогон `20261021_123058`

Без вызовов LLM. Файл генерируется `python -m ml.eval.run_eval --ranking-only [--set v2]` и перезаписывается. Разметка: v1 ручная (`ml/eval/labels.py`), v2 ручная + согласованная двумя LLM (`ml/eval/labels.jsonl`).

Параметры: ALPHA=0.6, MIN_SCORE=0.25, FEATURE_WEIGHT=0.45, OFF_SECTOR_DAMP=0.5, ADJACENCY_WEIGHT=0.35, BROAD_WEIGHT_NO_HOOKS=0.35, смежность отраслей: flows+rules, устаревшая аналитика банка (старше 180 дн.) в ленту не попадает.

## Ранжирование ленты (без LLM)

Пул после дедупликации: 26 материалов. Дубликаты схлопнуты: 1/1.

| | Лента (ml.ranking) | Только BM25 по профилю |
|---|---|---|
| nDCG@5 | 0.79 | 0.65 |
| recall@5 | 98% | 84% |
| precision / recall отсева | 87% / 87% | — |

| Компания | nDCG@5 | В ленте / релевантных | Пропущены | Лишние |
|---|---|---|---|---|
| U_1595 (R) | 0.86 | 20 / 18 | — | h_2026_09_08_pharma, h_2026_09_10_grain_duty |
| U_3263 (T) | 0.76 | 18 / 17 | — | bank_consumption_2026_08_05 |
| U_4217 (P) | 0.77 | 18 / 14 | — | bank_gold_2026_08, bank_macro_1h26, h_2026_09_09_housing, h_2026_09_11_metal_prices |
| U_26030 (F) | 0.80 | 19 / 17 | — | bank_consumption_2026_08_05, n_2026_09_08_retail |
| U_177581 (C) | 0.76 | 17 / 15 | n_2026_09_03_petrol | bank_consumption_2026_08_05, bank_gold_2026_08, n_2026_09_10_trucking |
| U_200890 (M) | 1.00 | 9 / 14 | bank_macro_7m26, bank_rate_2026_09_11, h_2026_09_09_housing, h_2026_09_11_rate_en, n_2026_09_10_trucking, n_2026_09_11_cbr_rate | bank_bond_brief_2026_09_14 |
| U_183405 (B) | 0.68 | 6 / 11 | bank_macro_7m26, bank_rate_2026_09_11, h_2026_09_11_rate_en, n_2026_09_10_trucking, n_2026_09_11_cbr_rate | — |
| U_236447 (W) | 0.79 | 11 / 11 | n_2026_09_06_noisy, n_2026_09_10_trucking, n_2026_09_11_cbr_rate | bank_bond_brief_2026_09_14, bank_fi_strategy_3q26, bank_gold_2026_08 |
| group_134463_sb (G) | 0.66 | 19 / 20 | bank_inflation_2026_09_02, h_2026_09_11_metal_prices, n_2026_09_03_petrol | bank_consumption_2026_08_05, h_2026_09_10_grain_duty |

## Честная оценка ленты

Параметры ранжирования (ALPHA, MIN_SCORE, FEATURE_WEIGHT, OFF_SECTOR_DAMP, ADJACENCY_WEIGHT) подбираются по сетке из 180 вариантов (лучший F1 отсева, при равенстве nDCG@5) без одной группы и проверяются на ней: группа = компанию, фолдов 9. В скобках 95% интервал bootstrap по компаниям (2000 выборок).

| Метрика | Текущие параметры | Подбор на всех (in-sample) | Подбор без группы (CV) | Разрыв in-sample − CV |
|---|---|---|---|---|
| nDCG@5 | 0.79 [0.73; 0.85] | 0.76 | 0.74 [0.66; 0.83] | 0.02 |
| recall@5 | 98% [93%; 100%] | 96% | 96% [89%; 100%] | 0% |
| precision отсева | 87% [82%; 91%] | 88% | 85% [80%; 89%] | 3% |
| recall отсева | 87% [75%; 96%] | 87% | 85% [74%; 95%] | 1% |
| F1 отсева | 87% [80%; 92%] | 87% | 85% [79%; 90%] | 2% |

Параметры, выбранные на всех компаниях: {'ALPHA': 0.7, 'MIN_SCORE': 0.25, 'FEATURE_WEIGHT': 0.45, 'OFF_SECTOR_DAMP': 0.5, 'ADJACENCY_WEIGHT': 0.35}.

Разброс выбранных параметров по фолдам: ALPHA 0.6, 0.7; MIN_SCORE 0.2, 0.25, 0.3; FEATURE_WEIGHT 0.35, 0.45; OFF_SECTOR_DAMP 0.5; ADJACENCY_WEIGHT 0.25, 0.35, 0.45.

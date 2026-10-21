"""Какая задача идёт на какую модель. Окончательно подстроить по eval."""

from __future__ import annotations

import os
from dataclasses import dataclass

# Только модели из списка кейса. Имена у провайдера задаются в .env.
ALLOWED_MODELS = {
    "gemma": ("LLM_MODEL_GEMMA", "google/gemma-4-26b-a4b-it"),
    "glm": ("LLM_MODEL_GLM", "z-ai/glm-5.3-flash"),
    "qwen": ("LLM_MODEL_QWEN", "qwen/qwen3.6-35b-a3b"),
    "deepseek": ("LLM_MODEL_DEEPSEEK", "deepseek/deepseek-v4.1-flash"),
    "kimi": ("LLM_MODEL_KIMI", "moonshotai/kimi-k2.6"),
    "qwen_vl": ("LLM_MODEL_QWEN_VL", "qwen/qwen3-vl-30b-a3b-instruct"),
}


@dataclass(frozen=True)
class Route:
    family: str
    temperature: float
    max_tokens: int
    fallback: str | None = None
    # reasoning_effort провайдера. "none" отключает рассуждения: без этого qwen, deepseek и kimi
    # тратят весь max_tokens на reasoning и возвращают пустой content. None: параметр не передаём.
    reasoning: str | None = "none"


ROUTES: dict[str, Route] = {
    "parse_item": Route("gemma", 0.0, 1500, fallback="qwen", reasoning=None),
    "relevance": Route("gemma", 0.0, 600, fallback="qwen", reasoning=None),
    "plan": Route("gemma", 0.0, 500, fallback="qwen", reasoning=None),
    # Общая важность материала для ленты (ml/ingest/importance.py): один вызов на материал, а не на пару.
    "importance": Route("gemma", 0.0, 300, fallback="qwen", reasoning=None),
    # Сверка противоречий событие/банк (ml/agent/consistency.py): один короткий вызов на карточку.
    "consistency": Route("gemma", 0.0, 600, fallback="qwen", reasoning=None),
    "interpret": Route("glm", 0.2, 2500, fallback="qwen"),
    # Стадия 1 двухстадийной генерации (ml/agent/channels.py): та же модель, короткий ответ.
    "channels": Route("glm", 0.0, 900, fallback="qwen"),
    "verify": Route("gemma", 0.0, 800, fallback="qwen", reasoning=None),
    # Проверка утверждений по одному: Qwen строже Gemma (ml/agent/claim_verify.py).
    "claim_verify": Route("qwen", 0.0, 2000, fallback="gemma"),
    "followup_qa": Route("glm", 0.2, 1000, fallback="qwen"),
    # Судья eval: другая модель, чем интерпретатор (GLM), чтобы не оценивать самого себя.
    "eval_judge": Route("deepseek", 0.0, 1200),
    # Второй судья, включается run_eval --second-judge. Kimi в ~15 раз дороже DeepSeek;
    # Qwen почти не различает карточки (factual=2 в 99%), поэтому по умолчанию выключен.
    "eval_judge_2": Route("qwen", 0.0, 1200),
    # Разметка эталона релевантности (ml/eval/label_llm.py): две модели разных семейств, обе не интерпретатор.
    "label_a": Route("deepseek", 0.0, 3000),
    "label_b": Route("qwen", 0.0, 3000),
    "ocr_cleanup": Route("gemma", 0.0, 3000, reasoning=None),
    # Чтение страниц PDF vision-моделью (ml/ingest/pdf_vl.py). В пилоте таблица A2 заняла ~1900 токенов ответа,
    # лимит с запасом; зацикливание режет ml.ingest.page_check.degenerate_reason, а не лимит.
    "page_vl": Route("qwen_vl", 0.0, 4500, reasoning=None),
    "chart_vl": Route("qwen_vl", 0.0, 2500, reasoning=None),
}


def model_name(family: str) -> str:
    env, default = ALLOWED_MODELS[family]
    return os.getenv(env, default).strip() or default


def allowed_model_names() -> set[str]:
    return {model_name(f) for f in ALLOWED_MODELS}


def route(task: str) -> Route:
    if task not in ROUTES:
        raise KeyError(f"нет маршрута для задачи {task!r}")
    return ROUTES[task]

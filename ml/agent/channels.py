"""Двухстадийная генерация: сначала каналы влияния, потом текст.

Стадия 1: модель по профилю, событию и фрагментам банка возвращает каналы
(`ml.topics.IMPACT_CHANNELS`) с силой, механизмом, признаками профиля и
источниками. Код отбрасывает канал без источника события или банка и без
заполненного признака профиля. Стадия 2 (`interpret`) пишет карточку только
по оставшимся каналам. Если каналов нет, карточка короткая, а relevance.score
ограничивается сверху: натянуть связь без канала нельзя.

Включается ML_CHANNELS=1. По умолчанию выключено: на dev (прогон 20261021_164631) factual
упал на 0,16, доля карточек с неподтверждёнными утверждениями выросла на 24 п.п.
"""

from __future__ import annotations

import os
from pathlib import Path

from pydantic import BaseModel

from ml.agent.verifier import _path_value
from ml.llm.client import LLMClient
from ml.prompts import render
from ml.schemas import PROFILE_SOURCE_ID, CompanyProfile, ImpactChannel, InsightCard, ParsedItem
from ml.topics import IMPACT_CHANNELS

RULES_PATH = Path(__file__).resolve().parent.parent / "prompts" / "interpret_channels.md"
# Потолок relevance.score: без каналов связь не доказана; только слабые каналы — не «напрямую».
NO_CHANNEL_MAX = 0.25
WEAK_ONLY_MAX = 0.6
_STRENGTH = {"strong": 2, "weak": 1, "none": 0}


class ChannelsResult(BaseModel):
    channels: list[ImpactChannel] = []


def enabled() -> bool:
    return os.getenv("ML_CHANNELS", "0") == "1"


def taxonomy() -> list[str]:
    return [f"{c.key}: {c.meaning}; признаки: {', '.join(c.features)}" for c in IMPACT_CHANNELS.values()]


def _fix_path(path: str, profile: CompanyProfile) -> str | None:
    """Полный путь заполненного поля профиля; «loan» -> «bank_products.loan». None, если поле пустое или его нет."""
    def filled(p: str) -> bool:
        return _path_value(profile, p) not in (None, [], "", False)

    if filled(path):
        return path
    cands = {f"{group}.{path}" for group in ("industry", "size", "geo", "bank_products", "group", "network")}
    found = [p for p in cands if filled(p)]
    return found[0] if len(found) == 1 else None


def validate(raw: list[ImpactChannel], profile: CompanyProfile, allowed_ids: set[str]) -> tuple[list[ImpactChannel], list[str]]:
    """Проверенные каналы и список отброшенных с причиной. «none» не ошибка, а проверенное отсутствие связи."""
    best: dict[str, ImpactChannel] = {}
    dropped: list[str] = []
    for ch in raw:
        if ch.channel not in IMPACT_CHANNELS:
            dropped.append(f"{ch.channel}: нет в таксономии")
            continue
        title = IMPACT_CHANNELS[ch.channel].title
        if ch.strength == "none":
            dropped.append(f"{title}: связи нет")
            continue
        evidence = [e for e in dict.fromkeys(ch.evidence) if e in allowed_ids and e != PROFILE_SOURCE_ID]
        features = [f for f in dict.fromkeys(_fix_path(x, profile) for x in ch.profile_features) if f]
        if not evidence:
            dropped.append(f"{title}: нет источника")
            continue
        if not features:
            dropped.append(f"{title}: нет признака в профиле")
            continue
        ch = ch.model_copy(update={"evidence": evidence, "profile_features": features})
        if ch.channel not in best or _STRENGTH[ch.strength] > _STRENGTH[best[ch.channel].strength]:
            best[ch.channel] = ch
    kept = sorted(best.values(), key=lambda c: -_STRENGTH[c.strength])
    return kept, [d for d in dict.fromkeys(dropped) if d.split(":")[0] not in {IMPACT_CHANNELS[c.channel].title for c in kept}]


def extract(client: LLMClient, profile: CompanyProfile, item: ParsedItem, item_source_id: str, bank_block: list[dict],
            reasons: list[str], trace=None) -> tuple[list[ImpactChannel], list[str]]:
    allowed = [item_source_id, *(h["id"] for h in bank_block)]
    p = render("channels", taxonomy=taxonomy(), profile=profile.prompt_view(), reasons=reasons or ["явных связей не найдено"],
               item=item.model_dump(mode="json", exclude={"dedup_cluster"}), item_source_id=item_source_id,
               allowed_source_ids=allowed, bank_chunks=bank_block or "нет подходящих фрагментов")
    res, _ = client.complete_json("channels", p.system, p.user, ChannelsResult, prompt_version=p.version, trace=trace,
                                  mock_context={"profile": profile, "item_source_id": item_source_id, "bank_ids": allowed[1:]})
    kept, dropped = validate(res.channels, profile, set(allowed))
    if trace is not None:
        trace.log("channels", kept=[f"{c.channel}:{c.strength}" for c in kept], dropped=dropped)
    return kept, dropped


def prompt_vars(kept: list[ImpactChannel] | None, dropped: list[str]) -> dict[str, str | list]:
    """Переменные промпта interpret: правила и блок CHANNELS (пустые, если стадия 1 выключена)."""
    if kept is None:
        return {"channel_rules": "", "channels": ""}
    block = [{"channel": c.channel, "title": IMPACT_CHANNELS[c.channel].title, "strength": c.strength,
              "direction": c.direction, "mechanism": c.mechanism, "profile_features": c.profile_features,
              "evidence": c.evidence} for c in kept]
    text = "CHANNELS:\n" + (render_list(block) if block else "пусто") + \
        ("\nНЕ ПОДТВЕРДИЛИСЬ: " + "; ".join(dropped) if dropped else "")
    return {"channel_rules": RULES_PATH.read_text(encoding="utf-8").strip(), "channels": text}


def render_list(block: list[dict]) -> str:
    import json

    return json.dumps(block, ensure_ascii=False, indent=1)


def enforce(card: InsightCard, kept: list[ImpactChannel], dropped: list[str]) -> InsightCard:
    """Потолок relevance.score по каналам и одна строка в limitations о неподтверждённых каналах."""
    cap = NO_CHANNEL_MAX if not kept else WEAK_ONLY_MAX if all(c.strength == "weak" for c in kept) else 1.0
    update: dict = {"channels": kept}
    if card.relevance.score > cap:
        update["relevance"] = card.relevance.model_copy(update={"score": cap})
    not_found = [d.split(":")[0] for d in dropped if not d.endswith("связи нет")]
    if not_found:
        line = "каналы влияния без подтверждения в источниках или профиле: " + ", ".join(not_found)
        if line not in card.limitations:
            update["limitations"] = [*card.limitations, line]
    return card.model_copy(update=update)

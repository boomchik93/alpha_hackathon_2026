"""Роли пользователей компании: одна компания, несколько человек с разными задачами.

Роль меняет акцент `why_it_matters` и порядок `suggested_actions` в карточке, но не факты, не `bank_view` и не источники.
"""

from __future__ import annotations

from enum import Enum


class Role(str, Enum):
    owner = "owner"
    ceo = "ceo"
    cfo = "cfo"
    treasurer = "treasurer"
    chief_accountant = "chief_accountant"
    other = "other"


ROLE_TITLES = {
    Role.owner: "собственник", Role.ceo: "генеральный директор", Role.cfo: "финансовый директор", Role.treasurer: "казначей",
    Role.chief_accountant: "главный бухгалтер", Role.other: "другая роль",
}

ROLE_FOCUS = {
    Role.owner: "стоимость долга и инвестиционная программа; влияние на устойчивость и стоимость бизнеса",
    Role.ceo: "общий эффект для бизнеса: спрос, издержки, конкуренция, ключевые риски и возможности",
    Role.cfo: "стоимость долга и инвестиционная программа; планирование бюджета и ковенантов",
    Role.treasurer: "ликвидность, валютные позиции и депозиты; управление остатками и платёжным календарём",
    Role.chief_accountant: "налоги, регулирование и расчёты; изменения требований к учёту и отчётности",
    Role.other: "",
}


def card_suffix(role: Role | None) -> str:
    return f"_{role.value}" if role else ""


def role_block(role: Role | None) -> str:
    """Блок ROLE для промпта; без роли или для «other» пустой."""
    if not role or not ROLE_FOCUS[role]:
        return ""
    return (f"ROLE: карточку читает {ROLE_TITLES[role]}. Акцент: {ROLE_FOCUS[role]}. "
            "Это меняет акцент why_it_matters и порядок suggested_actions (сначала то, что важно этой роли). "
            "Факты, bank_view и источники от роли не зависят.")

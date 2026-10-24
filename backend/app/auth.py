"""Вход, сессии и проверка доступа.

- Пользователь видит только свои компании; привязка к группе даёт доступ к группе и компаниям из её `member_ids`.
- Эксперт и админ видят любую компанию (демо-переключатель) и могут выбрать роль карточки параметром `role`.
- Роль обычного пользователя берётся из его профиля: фронт её не передаёт.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated

from fastapi import Depends, Request

from backend.app.db import DB
from backend.app.errors import ApiError


@dataclass(frozen=True)
class User:
    id: int
    login: str
    display_name: str
    role: str
    access: str               # user | expert | admin
    companies: tuple[str, ...]

    @property
    def privileged(self) -> bool:
        return self.access in ("expert", "admin")


def unauthorized(message: str = "нужен вход: заголовок Authorization: Bearer <token>") -> ApiError:
    return ApiError(401, "unauthorized", message)


def forbidden(message: str) -> ApiError:
    return ApiError(403, "forbidden", message)


def _db(request: Request) -> DB:
    return request.app.state.db


def bearer_token(request: Request) -> str:
    scheme, _, token = request.headers.get("Authorization", "").partition(" ")
    if scheme.lower() != "bearer" or not token.strip():
        raise unauthorized()
    return token.strip()


def current_user(request: Request) -> User:
    db = _db(request)
    row = db.session_user(bearer_token(request))
    if row is None:
        raise unauthorized("сессия не найдена или истекла, войдите заново")
    return User(row["id"], row["login"], row["display_name"], row["role"], row["access"], tuple(db.user_companies(row["id"])))


CurrentUser = Annotated[User, Depends(current_user)]


def require_privileged(user: CurrentUser) -> User:
    if not user.privileged:
        raise forbidden("доступно только эксперту и администратору")
    return user


PrivilegedUser = Annotated[User, Depends(require_privileged)]


async def group_members(request: Request, group_id: str) -> set[str]:
    """Компании группы из профиля ML; кэшируются на время жизни процесса (состав группы в выгрузке не меняется)."""
    cache: dict[str, set[str]] = request.app.state.group_members
    if group_id not in cache:
        profile = await request.app.state.ml.get(f"/companies/{group_id}")
        cache[group_id] = set(profile.get("member_ids") or [])
    return cache[group_id]


async def can_access(request: Request, user: User, company_id: str) -> bool:
    if user.privileged or company_id in user.companies:
        return True
    for linked in user.companies:
        if linked.startswith("group") and company_id in await group_members(request, linked):
            return True
    return False


async def ensure_company(request: Request, user: User, company_id: str) -> None:
    if not await can_access(request, user, company_id):
        raise forbidden(f"нет доступа к компании {company_id}")


def effective_role(user: User, requested: str | None) -> str | None:
    """Роль, для которой пишется карточка. Роль «other» карточку не меняет (ml/roles.py), её в ML не передаём."""
    if requested is not None and requested != user.role and not user.privileged:
        raise forbidden("роль задаётся профилем пользователя")
    role = requested or user.role
    return None if role == "other" else role

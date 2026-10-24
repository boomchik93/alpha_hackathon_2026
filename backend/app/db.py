"""SQLite-хранилище бэкенда: пользователи, сессии, карточки, просмотры, обратная связь, журнал обращений.

Одно соединение на процесс под блокировкой: бэкенд запускается одним процессом, запросы к базе короткие.
Пароли хранятся как scrypt с солью, токены сессий как sha256: утечка базы не даёт ни паролей, ни действующих токенов.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import secrets
import sqlite3
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id            INTEGER PRIMARY KEY,
    login         TEXT NOT NULL UNIQUE,
    password_hash TEXT NOT NULL,
    display_name  TEXT NOT NULL,
    role          TEXT NOT NULL,              -- ml/roles.py: owner, ceo, cfo, treasurer, chief_accountant, other
    access        TEXT NOT NULL               -- user | expert | admin
);
CREATE TABLE IF NOT EXISTS user_companies (
    user_id    INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    company_id TEXT NOT NULL,                 -- компания или группа (group_...)
    PRIMARY KEY (user_id, company_id)
);
CREATE TABLE IF NOT EXISTS sessions (
    token_hash TEXT PRIMARY KEY,
    user_id    INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    created_at TEXT NOT NULL,
    expires_at TEXT NOT NULL
);
-- Карточки, которые бэкенд отдал: по card_id находим компанию для проверки доступа и трассу.
CREATE TABLE IF NOT EXISTS cards (
    card_id    TEXT PRIMARY KEY,
    company_id TEXT NOT NULL,
    item_id    TEXT NOT NULL,
    role       TEXT,
    trace_id   TEXT NOT NULL,
    card_json  TEXT NOT NULL,
    saved_at   TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS views (
    user_id    INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    company_id TEXT NOT NULL,
    item_id    TEXT NOT NULL,
    viewed_at  TEXT NOT NULL,
    PRIMARY KEY (user_id, company_id, item_id)
);
CREATE TABLE IF NOT EXISTS feedback (
    id         INTEGER PRIMARY KEY,
    ts         TEXT NOT NULL,
    user_id    INTEGER NOT NULL REFERENCES users(id),
    access     TEXT NOT NULL,                 -- user | expert | admin: оценки экспертов в сводке отдельно (К08)
    role       TEXT NOT NULL,
    card_id    TEXT NOT NULL,
    company_id TEXT NOT NULL,
    rating     TEXT NOT NULL,
    reason     TEXT,
    comment    TEXT,
    synced_to_ml INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS audit (
    id         INTEGER PRIMARY KEY,
    ts         TEXT NOT NULL,
    user_id    INTEGER,
    action     TEXT NOT NULL,                 -- login, login_failed, insight, followup, feedback, trace, source
    company_id TEXT,
    item_id    TEXT,
    card_id    TEXT,
    trace_id   TEXT,
    detail     TEXT
);
CREATE INDEX IF NOT EXISTS idx_audit_ts ON audit(ts);
"""

ACCESS_LEVELS = ("user", "expert", "admin")
# scrypt: стандартная библиотека, параметры по рекомендации OWASP для интерактивного входа.
_SCRYPT = {"n": 2**14, "r": 8, "p": 1}


def now() -> datetime:
    return datetime.now(timezone.utc)


def iso(dt: datetime) -> str:
    return dt.isoformat(timespec="seconds")


def hash_password(password: str, salt: bytes | None = None) -> str:
    salt = salt or secrets.token_bytes(16)
    digest = hashlib.scrypt(password.encode(), salt=salt, **_SCRYPT)
    return f"scrypt${salt.hex()}${digest.hex()}"


def verify_password(password: str, stored: str) -> bool:
    try:
        _, salt_hex, digest_hex = stored.split("$")
    except ValueError:
        return False
    digest = hashlib.scrypt(password.encode(), salt=bytes.fromhex(salt_hex), **_SCRYPT)
    return hmac.compare_digest(digest.hex(), digest_hex)


# Сравнение с этим хэшем при неизвестном логине: время ответа не выдаёт, есть ли такой пользователь.
_DUMMY_HASH = hash_password("dummy", salt=b"\0" * 16)


def _token_hash(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


class DB:
    def __init__(self, path: Path | str):
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._lock = threading.Lock()
        with self._lock, self._conn:
            self._conn.executescript(SCHEMA)

    def close(self) -> None:
        self._conn.close()

    def _all(self, sql: str, args: tuple = ()) -> list[sqlite3.Row]:
        with self._lock:
            return self._conn.execute(sql, args).fetchall()

    def _one(self, sql: str, args: tuple = ()) -> sqlite3.Row | None:
        with self._lock:
            return self._conn.execute(sql, args).fetchone()

    def _write(self, sql: str, args: tuple = ()) -> int:
        with self._lock, self._conn:
            return self._conn.execute(sql, args).lastrowid

    # --- пользователи -------------------------------------------------------
    def upsert_user(self, login: str, password: str, display_name: str, role: str, access: str, companies: list[str]) -> int:
        if access not in ACCESS_LEVELS:
            raise ValueError(f"неизвестный уровень доступа {access}")
        with self._lock, self._conn:
            row = self._conn.execute("SELECT id, password_hash FROM users WHERE login=?", (login,)).fetchone()
            if row is None:
                uid = self._conn.execute(
                    "INSERT INTO users (login,password_hash,display_name,role,access) VALUES (?,?,?,?,?)",
                    (login, hash_password(password), display_name, role, access)).lastrowid
            else:
                uid = row["id"]
                # Хэш пересчитывается, только если пароль в сиде сменился: scrypt медленный намеренно.
                pw = row["password_hash"] if verify_password(password, row["password_hash"]) else hash_password(password)
                self._conn.execute("UPDATE users SET password_hash=?, display_name=?, role=?, access=? WHERE id=?",
                                   (pw, display_name, role, access, uid))
            self._conn.execute("DELETE FROM user_companies WHERE user_id=?", (uid,))
            self._conn.executemany("INSERT INTO user_companies (user_id, company_id) VALUES (?,?)", [(uid, c) for c in companies])
        return uid

    def authenticate(self, login: str, password: str) -> sqlite3.Row | None:
        row = self._one("SELECT * FROM users WHERE login=?", (login,))
        ok = verify_password(password, row["password_hash"] if row else _DUMMY_HASH)
        return row if row and ok else None

    def user(self, user_id: int) -> sqlite3.Row | None:
        return self._one("SELECT * FROM users WHERE id=?", (user_id,))

    def user_companies(self, user_id: int) -> list[str]:
        return [r["company_id"] for r in self._all(
            "SELECT company_id FROM user_companies WHERE user_id=? ORDER BY rowid", (user_id,))]

    # --- сессии ---------------------------------------------------------------
    def create_session(self, user_id: int, ttl: timedelta) -> tuple[str, datetime]:
        token, expires = secrets.token_urlsafe(32), now() + ttl
        self._write("INSERT INTO sessions (token_hash,user_id,created_at,expires_at) VALUES (?,?,?,?)",
                    (_token_hash(token), user_id, iso(now()), iso(expires)))
        return token, expires

    def session_user(self, token: str) -> sqlite3.Row | None:
        return self._one("SELECT u.* FROM sessions s JOIN users u ON u.id=s.user_id WHERE s.token_hash=? AND s.expires_at>?",
                         (_token_hash(token), iso(now())))

    def delete_session(self, token: str) -> None:
        self._write("DELETE FROM sessions WHERE token_hash=?", (_token_hash(token),))

    def prune_sessions(self) -> None:
        self._write("DELETE FROM sessions WHERE expires_at<=?", (iso(now()),))

    # --- карточки и просмотры ------------------------------------------------------
    def save_card(self, card: dict, company_id: str, item_id: str) -> None:
        self._write(
            "INSERT OR REPLACE INTO cards (card_id,company_id,item_id,role,trace_id,card_json,saved_at) VALUES (?,?,?,?,?,?,?)",
            (card["card_id"], company_id, item_id, card.get("role"), card["trace_id"],
             json.dumps(card, ensure_ascii=False), iso(now())))

    def card(self, card_id: str) -> sqlite3.Row | None:
        return self._one("SELECT * FROM cards WHERE card_id=?", (card_id,))

    def mark_viewed(self, user_id: int, company_id: str, item_id: str) -> None:
        self._write("INSERT OR REPLACE INTO views (user_id,company_id,item_id,viewed_at) VALUES (?,?,?,?)",
                    (user_id, company_id, item_id, iso(now())))

    def viewed_items(self, user_id: int, company_id: str) -> set[str]:
        return {r["item_id"] for r in self._all("SELECT item_id FROM views WHERE user_id=? AND company_id=?", (user_id, company_id))}

    # --- обратная связь ----------------------------------------------------------------
    def add_feedback(self, user_id: int, access: str, role: str, card: sqlite3.Row, rating: str, reason: str | None,
                     comment: str | None) -> int:
        return self._write(
            "INSERT INTO feedback (ts,user_id,access,role,card_id,company_id,rating,reason,comment,synced_to_ml) "
            "VALUES (?,?,?,?,?,?,?,?,?,0)",
            (iso(now()), user_id, access, role, card["card_id"], card["company_id"], rating, reason, comment))

    def mark_feedback_synced(self, feedback_id: int) -> None:
        self._write("UPDATE feedback SET synced_to_ml=1 WHERE id=?", (feedback_id,))

    def feedback_rows(self) -> list[dict[str, Any]]:
        return [dict(r) for r in self._all("SELECT * FROM feedback ORDER BY id")]

    # --- журнал ---------------------------------------------------------------------------
    def audit(self, action: str, user_id: int | None = None, *, company_id: str | None = None, item_id: str | None = None,
              card_id: str | None = None, trace_id: str | None = None, detail: str | None = None) -> None:
        self._write("INSERT INTO audit (ts,user_id,action,company_id,item_id,card_id,trace_id,detail) VALUES (?,?,?,?,?,?,?,?)",
                    (iso(now()), user_id, action, company_id, item_id, card_id, trace_id, detail))

    def audit_rows(self, limit: int = 200) -> list[dict[str, Any]]:
        return [dict(r) for r in self._all("SELECT * FROM audit ORDER BY id DESC LIMIT ?", (limit,))]


def load_seed(db: DB, path: Path) -> int:
    """Тестовые учётные записи из JSON: [{login, password, display_name, role, access, companies}]. Возвращает число записей."""
    users = json.loads(path.read_text(encoding="utf-8"))
    for u in users:
        db.upsert_user(u["login"], u["password"], u["display_name"], u["role"], u["access"], u.get("companies", []))
    return len(users)

"""SQLite-хранилище статей и событий (кластеров статей об одном и том же)."""
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS articles (
    id           TEXT PRIMARY KEY,
    url          TEXT NOT NULL,
    source_id    TEXT NOT NULL,
    source_name  TEXT NOT NULL,
    source_type  TEXT NOT NULL,
    lang         TEXT NOT NULL,
    scope        TEXT NOT NULL,
    title        TEXT NOT NULL,
    summary      TEXT,
    text         TEXT,
    text_status  TEXT,            -- NULL = не пробовали, ok, failed, skipped
    published_at TEXT NOT NULL,
    fetched_at   TEXT NOT NULL,
    relevant     INTEGER NOT NULL,
    rule_tags    TEXT NOT NULL,
    tokens       TEXT NOT NULL,
    event_id     TEXT
);
CREATE INDEX IF NOT EXISTS idx_articles_event ON articles(event_id);
CREATE INDEX IF NOT EXISTS idx_articles_pub ON articles(published_at);

CREATE TABLE IF NOT EXISTS events (
    id             TEXT PRIMARY KEY,
    created_at     TEXT NOT NULL,
    updated_at     TEXT NOT NULL
);
"""


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Store:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(path)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)

    def close(self) -> None:
        self.conn.close()

    # --- статьи -------------------------------------------------------------
    def has_article(self, article_id: str) -> bool:
        row = self.conn.execute("SELECT 1 FROM articles WHERE id=?", (article_id,)).fetchone()
        return row is not None

    def insert_article(self, a: dict) -> None:
        self.conn.execute(
            """INSERT OR IGNORE INTO articles
               (id,url,source_id,source_name,source_type,lang,scope,title,summary,text,text_status,
                published_at,fetched_at,relevant,rule_tags,tokens)
               VALUES (:id,:url,:source_id,:source_name,:source_type,:lang,:scope,:title,:summary,:text,
                       :text_status,:published_at,:fetched_at,:relevant,:rule_tags,:tokens)""",
            a,
        )

    def articles_needing_text(self) -> list[sqlite3.Row]:
        return self.conn.execute(
            "SELECT id,url FROM articles WHERE relevant=1 AND text_status IS NULL"
        ).fetchall()

    def set_article_text(self, article_id: str, text: str | None) -> None:
        self.conn.execute(
            "UPDATE articles SET text=?, text_status=? WHERE id=?",
            (text, "ok" if text else "failed", article_id),
        )

    def unclustered_articles(self) -> list[sqlite3.Row]:
        return self.conn.execute(
            "SELECT * FROM articles WHERE relevant=1 AND event_id IS NULL ORDER BY published_at"
        ).fetchall()

    def clustered_members(self, since_iso: str) -> list[sqlite3.Row]:
        return self.conn.execute(
            "SELECT id,event_id,tokens,published_at FROM articles "
            "WHERE event_id IS NOT NULL AND published_at>=?",
            (since_iso,),
        ).fetchall()

    def assign_event(self, article_id: str, event_id: str) -> None:
        self.conn.execute("UPDATE articles SET event_id=? WHERE id=?", (event_id, article_id))
        self.conn.execute("UPDATE events SET updated_at=? WHERE id=?", (now_iso(), event_id))

    def create_event(self, event_id: str) -> None:
        ts = now_iso()
        self.conn.execute(
            "INSERT OR IGNORE INTO events (id,created_at,updated_at) VALUES (?,?,?)",
            (event_id, ts, ts),
        )

    # --- события ------------------------------------------------------------
    def event_articles(self, event_id: str) -> list[sqlite3.Row]:
        return self.conn.execute(
            "SELECT * FROM articles WHERE event_id=? ORDER BY published_at", (event_id,)
        ).fetchall()

    def events_for_export(self, since_iso: str) -> list[sqlite3.Row]:
        return self.conn.execute(
            """SELECT e.* , MAX(a.published_at) AS last_pub
               FROM events e JOIN articles a ON a.event_id=e.id
               GROUP BY e.id HAVING last_pub >= ? ORDER BY last_pub DESC""",
            (since_iso,),
        ).fetchall()

    def commit(self) -> None:
        self.conn.commit()

    def stats(self) -> dict:
        c = self.conn
        return {
            "articles": c.execute("SELECT COUNT(*) FROM articles").fetchone()[0],
            "relevant_articles": c.execute("SELECT COUNT(*) FROM articles WHERE relevant=1").fetchone()[0],
            "events": c.execute("SELECT COUNT(*) FROM events").fetchone()[0],
        }

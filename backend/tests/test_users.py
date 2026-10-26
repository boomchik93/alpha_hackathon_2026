"""Вход, доступ к компаниям, просмотры и «новое», фильтры ленты, сводка, обратная связь, трасса, первоисточник, журнал."""
from backend.app.sources import BANK_DOCS

from .fakes import COMPANY, GROUP, GROUP_MEMBER, ITEM, OTHER, login
from .test_api import CARD_ID, INSIGHT, assert_error

# --- вход и сессии ------------------------------------------------------------


def test_login_me_logout(client):
    resp = client.post("/api/auth/login", json={"login": "alice", "password": "pw-alice"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["user"] == {"login": "alice", "display_name": "Алиса", "role": "cfo", "access": "user", "companies": [COMPANY]}
    headers = {"Authorization": f"Bearer {body['token']}"}
    assert client.get("/api/me", headers=headers).json()["login"] == "alice"
    assert client.post("/api/auth/logout", headers=headers).json() == {"ok": True}
    assert_error(client.get("/api/me", headers=headers), 401, "unauthorized")


def test_bad_credentials_and_missing_token(client):
    assert_error(client.post("/api/auth/login", json={"login": "alice", "password": "wrong"}), 401, "unauthorized")
    assert_error(client.post("/api/auth/login", json={"login": "nobody", "password": "x"}), 401, "unauthorized")
    assert_error(client.get(f"/api/companies/{COMPANY}/feed"), 401, "unauthorized")
    assert_error(client.get("/api/me", headers={"Authorization": "Bearer forged"}), 401, "unauthorized")


def test_password_is_not_stored_in_plain_text(client):
    rows = client.app.state.db._all("SELECT password_hash FROM users")
    assert rows and all(r["password_hash"].startswith("scrypt$") and "pw-" not in r["password_hash"] for r in rows)


# --- доступ к компаниям ---------------------------------------------------------

def test_user_sees_only_own_companies(client, alice):
    assert [c["company_id"] for c in client.get("/api/companies", headers=alice).json()] == [COMPANY]
    for url in (f"/api/companies/{OTHER}", f"/api/companies/{OTHER}/feed", f"/api/companies/{OTHER}/insights/{ITEM}",
                f"/api/me/summary?company_id={OTHER}"):
        assert_error(client.get(url, headers=alice), 403, "forbidden")


def test_group_link_gives_access_to_members(client):
    grp = login(client, "grp")
    assert sorted(c["company_id"] for c in client.get("/api/companies", headers=grp).json()) == [GROUP_MEMBER, GROUP]
    assert client.get(f"/api/companies/{GROUP_MEMBER}/feed", headers=grp).status_code == 200
    assert_error(client.get(f"/api/companies/{COMPANY}", headers=grp), 403, "forbidden")


def test_expert_sees_everything_and_can_pick_role(client):
    exp = login(client, "exp")
    assert len(client.get("/api/companies", headers=exp).json()) == 4
    card = client.get(f"/api/companies/{OTHER}/insights/{ITEM}", params={"role": "treasurer"}, headers=exp).json()
    assert card["role"] == "treasurer"
    # Роль «other» карточку не меняет: в ML роль не передаётся.
    assert client.get(f"/api/companies/{OTHER}/insights/{ITEM}", headers=exp).json()["role"] is None


def test_user_cannot_override_role(client, alice):
    assert_error(client.get(INSIGHT, params={"role": "owner"}, headers=alice), 403, "forbidden")
    assert client.get(INSIGHT, params={"role": "cfo"}, headers=alice).status_code == 200


def test_other_users_card_and_job_are_hidden(client, fake_ml, alice):
    assert client.get(INSIGHT, headers=alice).status_code == 200
    grp = login(client, "grp")
    assert_error(client.post(f"/api/cards/{CARD_ID}/followup", json={"question": "?"}, headers=grp), 403, "forbidden")
    assert_error(client.post(f"/api/cards/{CARD_ID}/feedback", json={"rating": "up"}, headers=grp), 403, "forbidden")
    assert_error(client.get(f"/api/cards/{CARD_ID}/trace", headers=grp), 403, "forbidden")

    fake_ml.delay_s = 1.0
    job = client.get(f"/api/companies/{COMPANY}/insights/n_fx", headers=alice).json()
    assert_error(client.get(job["poll_url"], headers=grp), 404, "not_found")
    assert client.get(job["poll_url"], headers=login(client, "exp")).status_code == 200


# --- лента, «новое», сводка --------------------------------------------------------

def test_feed_marks_new_until_opened(client, alice):
    url = f"/api/companies/{COMPANY}/feed"
    feed = client.get(url, headers=alice).json()
    assert all(e["is_new"] for e in feed["entries"]) and feed["new_count"] == 3  # только основная лента
    client.get(INSIGHT, headers=alice)
    feed = client.get(url, headers=alice).json()
    assert {e["item_id"]: e["is_new"] for e in feed["entries"]}[ITEM] is False and feed["new_count"] == 2
    # Просмотры у каждого пользователя свои; несуществующий материал в историю не попадает.
    assert client.get(url, headers=login(client, "bob")).json()["new_count"] == 3
    client.get(f"/api/companies/{COMPANY}/insights/x_missing", headers=alice)
    assert "x_missing" not in client.app.state.db.viewed_items(1, COMPANY)


def test_feed_filters(client, alice):
    url = f"/api/companies/{COMPANY}/feed"
    ids = lambda **p: [e["item_id"] for e in client.get(url, params=p, headers=alice).json()["entries"]]  # noqa: E731
    assert ids(tier="maybe") == ["n_maybe"]
    assert "n_maybe" not in ids(tier="main")
    assert ids(source_kind="bank_analytics") == ["bank_rate_2026_09_11"]
    assert ids(topic="fx") == ["n_fx"]
    assert ids(date_from="2026-10-21") == [ITEM, "n_maybe"]
    assert ids(date_to="2026-10-15") == ["bank_rate_2026_09_11"]
    assert "bank_rate_2026_09_11" not in ids(hide_stale="true")
    feed = client.get(url, params={"tier": "maybe"}, headers=alice).json()
    assert feed["maybe_count"] == 1 and feed["fetched_at"]


def test_summary_widget(client, alice):
    summary = client.get("/api/me/summary", headers=alice).json()
    assert summary["company_id"] == COMPANY and summary["new_count"] == 3
    assert summary["top"] == {"item_id": ITEM, "title": f"Материал {ITEM}", "score": 0.9, "published_at": "2026-10-23", "is_new": True}
    client.get(INSIGHT, headers=alice)
    summary = client.get("/api/me/summary", headers=alice).json()
    assert summary["new_count"] == 2 and summary["top"]["item_id"] == "bank_rate_2026_09_11"
    assert_error(client.get("/api/me/summary", headers=login(client, "exp")), 422, "validation_error")


# --- обратная связь ----------------------------------------------------------------------

def test_feedback_saved_and_summarized_by_access(client, fake_ml, alice):
    client.get(INSIGHT, headers=alice)
    resp = client.post(f"/api/cards/{CARD_ID}/feedback", json={"rating": "down", "reason": "obvious"}, headers=alice)
    assert resp.json() == {"ok": True, "card_id": CARD_ID, "synced_to_ml": True}
    assert fake_ml.feedback_calls[-1][1].role == "cfo"  # роль из профиля дошла до ML

    exp = login(client, "exp")
    client.post(f"/api/cards/{CARD_ID}/feedback", json={"rating": "up"}, headers=exp)
    summary = client.get("/api/feedback/summary", headers=exp).json()
    assert summary["user"]["down"] == 1 and summary["user"]["reasons"] == {"obvious": 1}
    assert summary["user"]["by_role"]["cfo"]["total"] == 1
    assert summary["expert"]["total"] == 1 and summary["expert"]["down"] == 0
    assert "ml" in summary
    # Вовлечённость из журнала: одно открытие карточки и две оценки после него.
    assert summary["usage"]["actions"]["insight"]["count"] == 1
    assert summary["usage"]["actions"]["feedback"] == {"count": 2, "users": 2}
    assert summary["usage"]["actions_per_card"] == 2.0
    assert summary["items_by_day"] is None or len(summary["items_by_day"]) == 14
    assert_error(client.get("/api/feedback/summary", headers=alice), 403, "forbidden")
    assert_error(client.post(f"/api/cards/{CARD_ID}/feedback", json={"rating": "meh"}, headers=alice), 422, "validation_error")


def test_feedback_kept_when_ml_fails(client, fake_ml, alice, monkeypatch):
    from ml import feedback as ml_feedback

    client.get(INSIGHT, headers=alice)

    def broken(card_id, fb):
        raise RuntimeError("диск переполнен")

    monkeypatch.setattr(ml_feedback, "record", broken)
    resp = client.post(f"/api/cards/{CARD_ID}/feedback", json={"rating": "up"}, headers=alice)
    assert resp.status_code == 200 and resp.json()["synced_to_ml"] is False
    assert client.app.state.db.feedback_rows()[-1]["synced_to_ml"] == 0


# --- трасса, первоисточник, журнал -----------------------------------------------------------

def test_trace_user_view_hides_internals(client, alice):
    client.get(INSIGHT, headers=alice)
    trace = client.get(f"/api/cards/{CARD_ID}/trace", headers=alice).json()
    # Шаг рассуждения без инструмента приходит с tool=null: фронт показывает только пояснение.
    assert trace["plan"] == [{"tool": "compute_exposure", "why": "найти признаки профиля"},
                             {"tool": None, "why": "интерпретация, затем самопроверка"}]
    assert trace["tool_calls"] == [{"tool": "compute_exposure", "ms": 12}]
    assert trace["checks"] == [{"attempt": 0, "issues": 1}, {"attempt": 1, "issues": 0}] and trace["retries"] == 1
    assert trace["replans"] == [{"trigger": "bank_miss", "why": "переформулировать запрос"}]
    assert trace["company_features_used"] == ["bank_products.loan"] and trace["sources"][0]["id"] == "src_n_rate"
    assert "СЕКРЕТНЫЙ" not in str(trace) and "raw" not in trace and "cost_rub" not in trace

    full = client.get(f"/api/cards/{CARD_ID}/trace", headers=login(client, "exp")).json()
    assert full["cost_rub"] == 0.42 and full["raw"]["events"][3]["event"] == "llm_call"


def test_source_pdf_whitelist(client, alice, tmp_path):
    pdf_dir = tmp_path / "analytics"
    pdf_dir.mkdir()
    (pdf_dir / "ALFA-7M26.pdf").write_bytes(b"%PDF-1.4 test")
    (pdf_dir / "secret.pdf").write_bytes(b"%PDF-1.4 secret")
    resp = client.get("/api/sources/bank_macro_7m26/pdf", headers=alice)
    assert resp.status_code == 200 and resp.headers["content-type"] == "application/pdf" and resp.content.startswith(b"%PDF")
    assert_error(client.get("/api/sources/secret/pdf", headers=alice), 404, "not_found")
    assert_error(client.get("/api/sources/bank_rate_2026_09_11/pdf", headers=alice), 404, "not_found")  # нет файла
    assert_error(client.get("/api/sources/bank_macro_7m26/pdf"), 401, "unauthorized")


def test_bank_docs_match_ml():
    from ml.ingest.analytics import DOCS

    assert BANK_DOCS == {meta.doc_id: stem for stem, meta in DOCS.items()}


def test_audit_log(client, alice):
    client.post("/api/auth/login", json={"login": "alice", "password": "bad"})
    client.get(INSIGHT, headers=alice)
    client.post(f"/api/cards/{CARD_ID}/followup", json={"question": "Что с кредитами?"}, headers=alice)
    adm = login(client, "adm")
    actions = {r["action"]: r for r in client.get("/api/admin/audit", headers=adm).json()}
    assert {"login", "login_failed", "insight", "followup"} <= set(actions)
    assert actions["followup"]["detail"] == "Что с кредитами?" and actions["followup"]["trace_id"] == "tr_q"
    assert actions["insight"]["company_id"] == COMPANY and actions["insight"]["item_id"] == ITEM
    assert_error(client.get("/api/admin/audit", headers=login(client, "exp")), 403, "forbidden")

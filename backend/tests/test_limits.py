"""Лимиты частоты: новые карточки и вопросы на пользователя, неудачные входы на логин и адрес."""
import pytest

from backend.app.errors import ApiError
from backend.app.ratelimit import RateLimiter

from .fakes import COMPANY, login, make_client, ml_transport
from .test_api import CARD_ID, INSIGHT, assert_error


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


def test_sliding_window():
    clock = Clock()
    rl = RateLimiter(clock)
    for _ in range(2):
        rl.hit(("k",), 2, 60, "стоп")
    with pytest.raises(ApiError) as e:
        rl.hit(("k",), 2, 60, "стоп")
    assert e.value.code == "rate_limited" and e.value.retry_after == 60
    clock.t += 30
    with pytest.raises(ApiError) as e:
        rl.check(("k",), 2, 60, "стоп")
    assert e.value.retry_after == 30
    clock.t += 30  # первые события вышли из окна
    rl.hit(("k",), 2, 60, "стоп")
    rl.hit(("other",), 2, 60, "стоп")  # ключи независимы


def test_new_cards_limited_per_user_but_joining_is_free(fake_ml, tmp_path):
    fake_ml.delay_s = 1.0
    with make_client(ml_transport(), tmp_path, limit_insights_per_10m=1, insight_sync_wait_s=0) as client:
        alice, bob = login(client, "alice"), login(client, "bob")
        assert client.get(INSIGHT, headers=alice).status_code == 202
        assert client.get(INSIGHT, headers=alice).status_code == 202  # та же карточка ещё генерируется
        body = assert_error(client.get(f"/api/companies/{COMPANY}/insights/n_fx", headers=alice), 429, "rate_limited")
        assert body["retry_after"] > 0
        assert client.get(f"/api/companies/{COMPANY}/insights/n_fx", headers=bob).status_code == 202  # лимит у каждого свой


def test_followups_limited(fake_ml, tmp_path):
    with make_client(ml_transport(), tmp_path, limit_followups_per_10m=1) as client:
        alice = login(client, "alice")
        assert client.get(INSIGHT, headers=alice).status_code == 200
        assert client.post(f"/api/cards/{CARD_ID}/followup", json={"question": "Что с кредитами?"}, headers=alice).status_code == 200
        assert_error(client.post(f"/api/cards/{CARD_ID}/followup", json={"question": "А с лизингом?"}, headers=alice),
                     429, "rate_limited")
        # Вопрос по несуществующей карточке лимит не тратит и не проходит проверку раньше 404.
        assert_error(client.post("/api/cards/ins_bad/followup", json={"question": "?"}, headers=alice), 404, "not_found")


def test_login_failures_limited_per_login(fake_ml, tmp_path):
    with make_client(ml_transport(), tmp_path, limit_login_failures_per_login=2) as client:
        bad = {"login": "alice", "password": "wrong"}
        for _ in range(2):
            assert_error(client.post("/api/auth/login", json=bad), 401, "unauthorized")
        # Даже верный пароль не проверяется, пока окно не истекло.
        resp = client.post("/api/auth/login", json={"login": "alice", "password": "pw-alice"})
        body = assert_error(resp, 429, "rate_limited")
        assert resp.headers["Retry-After"] == str(body["retry_after"])
        assert client.post("/api/auth/login", json={"login": "bob", "password": "pw-bob"}).status_code == 200


def test_successful_login_resets_counter(fake_ml, tmp_path):
    with make_client(ml_transport(), tmp_path, limit_login_failures_per_login=2) as client:
        client.post("/api/auth/login", json={"login": "alice", "password": "wrong"})
        login(client, "alice")
        client.post("/api/auth/login", json={"login": "alice", "password": "wrong"})
        assert client.post("/api/auth/login", json={"login": "alice", "password": "pw-alice"}).status_code == 200


def test_login_failures_limited_per_ip(fake_ml, tmp_path):
    with make_client(ml_transport(), tmp_path, limit_login_failures_per_ip=3) as client:
        for user in ("alice", "bob", "nobody"):
            assert_error(client.post("/api/auth/login", json={"login": user, "password": "wrong"}), 401, "unauthorized")
        assert_error(client.post("/api/auth/login", json={"login": "grp", "password": "pw-grp"}), 429, "rate_limited")

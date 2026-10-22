"""Реальные новости: текст статьи, сбор с делением по дате, сигналы материала и причины по каналам."""

from datetime import date

from ml.eval import collect_real as cr
from ml.ingest.importance import SOFT_SOURCE_MAX, item_signals, with_importance
from ml.ingest.news_fetch import RawItem, article_text
from ml.llm.client import LLMClient
from ml.schemas import BankProducts, CompanyProfile, DataQuality, Geo, GroupInfo, Industry, Network, ParsedItem, Size
from ml.topics import channel_reasons

PAGE = """<html><head><script>var x = '<p>скрипт не текст статьи, длинная строка для проверки фильтра</p>';</script></head>
<body><header><p>Меню сайта: экономика, политика, общество и всё остальное подряд</p></header>
<p>Банк России сохранил ключевую ставку на уровне 16% годовых, сообщил регулятор по итогам заседания совета директоров.</p>
<p>Подписывайтесь на наш канал в Telegram, чтобы первыми узнавать о главных новостях дня и недели</p>
<p>Аналитики ожидают, что регулятор вернётся к снижению ставки не раньше следующего года &mdash; по мере замедления инфляции.</p>
<p>Коротко.</p></body></html>"""


def test_article_text_keeps_paragraphs_and_drops_junk():
    t = article_text(PAGE)
    assert t.startswith("Банк России сохранил ключевую ставку") and "Аналитики ожидают" in t and "—" in t
    assert "скрипт" not in t and "Меню сайта" not in t and "Подписывайтесь" not in t and "Коротко" not in t


def _rss(items: list[tuple[str, str, str, str]]) -> bytes:
    rows = "".join(f"<item><title>{t}</title><link>{l}</link><pubDate>{d}</pubDate><description>{x}</description></item>"
                   for t, l, d, x in items)
    return f"<rss><channel><title>Лента</title>{rows}</channel></rss>".encode()


def _long(topic: str, d: int, n: int, k: int) -> str:
    """Уникальный текст: иначе дедупликация по словам склеит все записи."""
    return f"{topic}: " + " ".join(f"q{d % 10}{n}{k}{j:02d}" for j in range(60))


def test_collect_window_split_quota_and_ids():
    days = {16: "Fri, 16 Oct 2026 10:00:00 +0300", 18: "Sun, 18 Oct 2026 10:00:00 +0300",
            21: "Wed, 21 Oct 2026 10:00:00 +0300", 22: "Thu, 22 Oct 2026 10:00:00 +0300", 1: "Thu, 01 Oct 2026 10:00:00 +0300"}
    topics = ["ключевая ставка", "инфляция цены", "бензин топливо", "курс рубля доллар", "зарплаты рынок труда"]
    items = [(f"Тема {t} {d} {k}", f"https://x/{d}/{k}/{n}", days[d], _long(t, d, n, k)) for d in days for n, t in enumerate(topics) for k in range(2)]
    items.append(("Короткая", "https://x/short", days[16], "мало"))
    out, stats = cr.collect({"u": (_rss(items), ())}, date(2026, 10, 22), 7, 10, 3, seed=1, fetched_at="t",
                            full_text=lambda it, n: it)
    assert len(out) == 10 and stats["holdout"] == round(10 * cr.HOLDOUT_SHARE) and stats["dev"] == 6
    assert all(i.id.startswith("r_") and "2026-10-01" != i.published_at for i in out)
    assert not any(i.title == "Короткая" for i in out)
    assert all((stats["split"][i.id] == "holdout") == (i.published_at >= "2026-10-20") for i in out)
    again, _ = cr.collect({"u": (_rss(items), ())}, date(2026, 10, 22), 7, 10, 3, seed=1, fetched_at="t", full_text=lambda it, n: it)
    assert [i.id for i in again] == [i.id for i in out]


def test_parse_feed_filters_categories():
    body = (b"<rss><channel><title>K</title>"
            b"<item><title>a</title><link>https://k/1</link><pubDate>Wed, 21 Oct 2026 10:00:00 +0300</pubDate><category>\xd0\x9c\xd0\xb8\xd1\x80</category><description>x</description></item>"
            b"<item><title>b</title><link>https://k/2</link><pubDate>Wed, 21 Oct 2026 10:00:00 +0300</pubDate><category>\xd0\xad\xd0\xba\xd0\xbe\xd0\xbd\xd0\xbe\xd0\xbc\xd0\xb8\xd0\xba\xd0\xb0</category><description>y</description></item>"
            b"</channel></rss>")
    assert [i.url for i in cr.parse_feed(body, "u", ("Экономика",))] == ["https://k/2"]
    assert len(cr.parse_feed(body, "u")) == 2


def _item(event_type: str = "monetary_policy", topics=("key_rate",)) -> ParsedItem:
    return ParsedItem.model_validate({"item_id": "n_x", "source": {"kind": "news", "title": "t", "published_at": "2026-10-20"},
                                      "title": "Банк России сохранил ставку", "event_type": event_type, "topics": list(topics)})


def test_signals_from_mock_and_soft_source_cap(monkeypatch):
    client = LLMClient(mock=True, use_cache=False)
    it = with_importance(_item(), "текст", client)
    assert it.importance >= 0.9 and it.scope == "economy" and it.channels == ["cost_of_money"]
    from ml.llm import mock

    monkeypatch.setitem(mock.RESPONDERS, "importance", lambda ctx: {"importance": 0.9, "scope": "economy", "source_type": "rumor",
                                                                     "channels": ["cost_of_money", "нет_такого", "fx"], "reason": "x"})
    res = item_signals(_item(), "текст", client)
    assert res.importance == SOFT_SOURCE_MAX and res.channels == ["cost_of_money", "fx"]


def _profile(**bp) -> CompanyProfile:
    return CompanyProfile(company_id="U_1", eligible_big=True, industry=Industry(okved_code="47.71"), size=Size(), geo=Geo(),
                          bank_products=BankProducts(**bp), group=GroupInfo(), network=Network(), data_quality=DataQuality())


def test_channel_reasons_use_profile_hooks_only():
    r = channel_reasons(_profile(loan=True, rko=True), ["cost_of_money", "liquidity"], "economy")
    assert [p for _, _, p in r] == ["bank_products.loan"]  # РКО связи не доказывает
    assert channel_reasons(_profile(), ["cost_of_money"], "economy")[0][2] == "eligible_big"
    assert channel_reasons(_profile(), ["cost_of_money"], "sector") == []


def test_pool_accumulates_only_new_and_freezes_from_start(tmp_path):
    days = {20: "Tue, 20 Oct 2026 10:00:00 +0300", 23: "Fri, 23 Oct 2026 10:00:00 +0300", 25: "Sun, 25 Oct 2026 10:00:00 +0300",
            26: "Mon, 26 Oct 2026 10:00:00 +0300"}
    topics = ["ключевая ставка", "инфляция цены", "бензин топливо", "курс рубля доллар"]
    items = [(f"Тема {t} {d} {k}", f"https://p/{d}/{k}/{n}", days[d], _long(t, d, n, k)) for d in days for n, t in enumerate(topics) for k in range(2)]
    pool = tmp_path / "pool.jsonl"
    first = cr.accumulate({"u": (_rss(items[:20]), ())}, "t1", pool, full_text=lambda it, n: it)
    again = cr.accumulate({"u": (_rss(items), ())}, "t2", pool, full_text=lambda it, n: it)
    assert first["new"] == 20 and again["new"] == len(items) - 20 and again["pool"] == len(items)
    loaded = cr.load_pool(pool)
    assert len({i.id for i in loaded}) == len(loaded) and {i.fetched_at for i in loaded[:20]} == {"t1"}
    out, stats = cr.collect_from_pool(loaded, date(2026, 10, 23), date(2026, 10, 26), 10, 2, seed=1, fetched_at="t3")
    assert len(out) == 10 and all(i.published_at >= "2026-10-23" for i in out)
    assert all((stats["split"][i.id] == "holdout") == (i.published_at >= "2026-10-25") for i in out)
    assert {i.fetched_at for i in out} <= {"t1", "t2"}  # дата сбора записи сохраняется


def test_set_paths_keep_real_names():
    assert cr.set_paths()[0].name == "real_news.jsonl" and cr.set_paths("real2")[1].name == "real2_news.lock"
    assert cr.labels_path("real2").name == "labels_real2.jsonl"


def test_freeze_without_holdout_part_uses_full_quota(tmp_path):
    days = {23: "Fri, 23 Oct 2026 10:00:00 +0300", 24: "Sat, 24 Oct 2026 10:00:00 +0300"}
    topics = ["ключевая ставка", "инфляция цены", "бензин топливо", "курс рубля доллар"]
    items = [(f"Тема {t} {d} {k}", f"https://f/{d}/{k}/{n}", days[d], _long(t, d, n, k)) for d in days for n, t in enumerate(topics) for k in range(2)]
    pool = tmp_path / "pool.jsonl"
    cr.accumulate({"u": (_rss(items), ())}, "t", pool, full_text=lambda it, n: it)
    out, stats = cr.collect_from_pool(cr.load_pool(pool), date(2026, 10, 23), date(2026, 10, 24), 10, 0, seed=1, fetched_at="t")
    assert len(out) == 10 and stats["holdout"] == 0

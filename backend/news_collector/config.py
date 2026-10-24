"""Настройки сборщика новостей и список источников."""
from pathlib import Path
from urllib.parse import quote_plus

BASE_DIR = Path(__file__).resolve().parent
# var/ в .gitignore: база и выгрузки не попадают в репозиторий
VAR_DIR = BASE_DIR.parents[1] / "var" / "news_collector"
DB_PATH = VAR_DIR / "news.db"
OUTPUT_PATH = VAR_DIR / "news.json"            # события с тегами по правилам
RAW_ITEMS_PATH = VAR_DIR / "raw_items.jsonl"   # формат ml.ingest.news_fetch.RawItem
TAXONOMY_PATH = VAR_DIR / "taxonomy.json"

HTTP_HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; alpha-hack-news-bot/0.1)"}
HTTP_TIMEOUT = 15

MAX_AGE_DAYS = 3          # старше не берём
CLUSTER_WINDOW_DAYS = 3   # окно поиска дубликатов среди событий
FULLTEXT_CHARS = 6000     # сколько символов полного текста хранить


def _gnews(query: str, lang: str) -> str:
    q = quote_plus(f"{query} when:2d")
    if lang == "ru":
        return f"https://news.google.com/rss/search?q={q}&hl=ru&gl=RU&ceid=RU:ru"
    return f"https://news.google.com/rss/search?q={q}&hl=en-US&gl=US&ceid=US:en"


def _src(id, name, url, lang, scope, type="media", always=False, fulltext=True, core=False):
    return {
        "id": id, "name": name, "url": url, "lang": lang, "scope": scope,
        "type": type,               # primary | media | aggregator
        "always_relevant": always,  # пропускать предфильтр (официальные источники)
        "fulltext": fulltext,       # пробовать скачать полный текст по ссылке
        # core: издания, чьи ленты уже используются в проекте (ml/eval/collect_real.py).
        # Остальные организаторами не подтверждены и включаются только флагом --sources all.
        "core": core,
    }


SOURCES = [
    # --- Россия: официальные источники ---
    _src("cbr_press", "Банк России: пресс-релизы", "https://www.cbr.ru/rss/RssPress", "ru", "russia", "primary", True, core=True),
    _src("cbr_events", "Банк России: события", "https://www.cbr.ru/rss/eventrss", "ru", "russia", "primary", True, core=True),
    # --- Россия: СМИ ---
    _src("lenta_econ", "Lenta.ru: экономика", "https://lenta.ru/rss/news/economics", "ru", "russia", core=True),
    _src("tass", "ТАСС", "https://tass.ru/rss/v2.xml?sections=ekonomika", "ru", "russia", core=True),
    _src("interfax", "Интерфакс", "https://www.interfax.ru/rss.asp?r=business", "ru", "russia", core=True),
    _src("kommersant_econ", "Коммерсантъ: экономика", "https://www.kommersant.ru/RSS/section-economics.xml", "ru", "russia", core=True),
    _src("kommersant_business", "Коммерсантъ: бизнес", "https://www.kommersant.ru/RSS/section-business.xml", "ru", "russia", core=True),
    _src("vedomosti_econ", "Ведомости: экономика", "https://www.vedomosti.ru/rss/rubric/economics", "ru", "russia", core=True),
    _src("vedomosti_business", "Ведомости: бизнес", "https://www.vedomosti.ru/rss/rubric/business", "ru", "russia", core=True),
    _src("rbc", "РБК", "https://rssexport.rbc.ru/rbcnews/news/30/full.rss", "ru", "russia"),
    _src("prime", "ПРАЙМ", "https://1prime.ru/export/rss2/index.xml", "ru", "russia"),
    _src("finmarket", "Финмаркет", "https://www.finmarket.ru/rss/mainnews.asp", "ru", "russia"),
    _src("rg", "Российская газета", "https://rg.ru/xml/index.xml", "ru", "russia"),
    # --- Международные: официальные ---
    _src("fed", "ФРС США: пресс-релизы", "https://www.federalreserve.gov/feeds/press_all.xml", "en", "international", "primary", True),
    _src("ecb", "ЕЦБ: пресс-релизы", "https://www.ecb.europa.eu/rss/press.html", "en", "international", "primary", True),
    _src("eia", "US EIA", "https://www.eia.gov/rss/todayinenergy.xml", "en", "international", "primary"),
    # --- Международные: СМИ ---
    _src("bbc_business", "BBC Business", "http://feeds.bbci.co.uk/news/business/rss.xml", "en", "international"),
    _src("guardian_business", "The Guardian: Business", "https://www.theguardian.com/business/rss", "en", "international"),
    _src("cnbc", "CNBC", "https://www.cnbc.com/id/10001147/device/rss/rss.html", "en", "international"),
    _src("nyt_business", "NYT: Business", "https://rss.nytimes.com/services/xml/rss/nyt/Business.xml", "en", "international"),
    _src("wsj_markets", "WSJ: Markets", "https://feeds.a.dj.com/rss/RSSMarketsMain.xml", "en", "international"),
    _src("scmp_econ", "SCMP: China Economy", "https://www.scmp.com/rss/92/feed", "en", "international"),
    _src("nikkei", "Nikkei Asia", "https://asia.nikkei.com/rss/feed/nar", "en", "international"),
    _src("dw_business", "DW Business", "https://rss.dw.com/rdf/rss-en-bus", "en", "international"),
    _src("oilprice", "OilPrice", "https://oilprice.com/rss/main", "en", "international"),
    # --- Тематические запросы Google News (текст по ссылке недоступен, берём заголовок и аннотацию) ---
    *[
        _src(f"gnews_ru_{i}", "Google News (RU)", _gnews(q, "ru"), "ru", "russia", "aggregator", fulltext=False)
        for i, q in enumerate([
            "ключевая ставка", "курс рубля", "санкции против России", "налоги для бизнеса",
            "господдержка МСП", "лизинг", "импорт экспорт таможня", "кредитование бизнеса",
            "логистика грузоперевозки", "нефть газ экспорт", "параллельный импорт", "банкротство компаний",
        ])
    ],
    *[
        _src(f"gnews_en_{i}", "Google News (EN)", _gnews(q, "en"), "en", "international", "aggregator", fulltext=False)
        for i, q in enumerate([
            "russia sanctions", "russia economy", "oil price OPEC", "ruble", "russia china trade",
            "russia export ban", "EU sanctions package russia", "global supply chain shipping",
            "central bank interest rates", "russia india oil", "russia turkey trade",
        ])
    ],
]


def select_sources(mode: str = "core") -> list[dict]:
    """core: только ленты, уже используемые в проекте; all: все, включая не подтверждённые организаторами."""
    return [s for s in SOURCES if mode == "all" or s["core"]]

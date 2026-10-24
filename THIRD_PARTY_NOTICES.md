# Сторонние компоненты и данные

Версии зависимостей из `requirements.lock` (Python 3.13). Лицензии указаны по метаданным установленных пакетов на 2026-10-21; перед публикацией сверьте их с исходниками пакетов.

## Python-зависимости (`requirements.lock`)

| Пакет | Версия | Лицензия |
|---|---|---|
| annotated-doc | 0.0.5 | MIT |
| annotated-types | 0.8.0 | MIT |
| anyio | 4.15.1 | MIT |
| certifi | 2026.10.19 | MPL-2.0 |
| click | 8.5.0 | BSD-3-Clause |
| fastapi | 0.156.2 | MIT |
| h11 | 0.16.0 | MIT |
| httpcore | 1.0.9 | BSD-3-Clause |
| httpx | 0.28.1 | BSD-3-Clause |
| idna | 3.20 | BSD-3-Clause |
| iniconfig | 2.3.1 | MIT |
| numpy | 2.5.4 | BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0 |
| opentelemetry-api | 1.45.1 | Apache-2.0 |
| packaging | 26.3 | Apache-2.0 OR BSD-2-Clause |
| pandas | 3.0.6 | BSD-3-Clause |
| pillow | 13.0.0 | MIT-CMU (HPND) |
| pluggy | 1.7.0 | MIT |
| pyarrow | 26.0.0 | Apache-2.0 |
| pydantic | 2.14.1 | MIT |
| pydantic_core | 2.50.1 | MIT |
| Pygments | 2.21.0 | BSD-2-Clause |
| pytest | 9.1.1 | MIT |
| python-dateutil | 2.9.0.post0 | Apache-2.0 / BSD-3-Clause (двойная) |
| python-dotenv | 1.2.4 | BSD-3-Clause |
| rank-bm25 | 0.2.2 | Apache-2.0 |
| six | 1.17.0 | MIT |
| starlette | 1.9.0 | BSD-3-Clause |
| typing-inspection | 0.4.4 | MIT |
| typing_extensions | 4.16.0 | PSF-2.0 |
| uvicorn | 0.54.0 | BSD-3-Clause |

## Необязательные зависимости (`requirements-lemma.txt`, `ML_TOKENIZER=lemma`)

В `requirements.lock` не входят; в образ Docker попадают только с `--build-arg WITH_LEMMA=1`.

| Пакет | Версия | Лицензия |
|---|---|---|
| pymorphy3 | 2.0.6 | MIT |
| pymorphy3-dicts-ru | 2.4.417150.4580142 | MIT |
| DAWG2-Python | 0.9.0 | MIT |

## Системные программы (чтение PDF)

| Программа | Назначение | Лицензия |
|---|---|---|
| tesseract-ocr (языки rus, eng) | OCR сканов аналитики: запасной вариант и сверка с чтением моделью | Apache-2.0 |
| poppler-utils (`pdftoppm`) | рендер страниц PDF в изображения | GPL-2.0 / GPL-3.0 |

Программы вызываются как отдельные процессы и не входят в код проекта.

## Языковые модели (API организаторов, OpenAI-совместимый)

Используются только модели из списка кейса, через API организаторов; внешние AI-сервисы не подключены. Лицензии и условия модели определяет провайдер.

| Модель | Роль |
|---|---|
| google/gemma-4-26b-a4b-it | разбор новостей, важность материала, сверка противоречий |
| z-ai/glm-5.3-flash | карточки, ответы на уточняющие вопросы |
| qwen/qwen3.6-35b-a3b | резерв, проверка утверждений (выкл.), второй судья eval, разметка |
| deepseek/deepseek-v4.1-flash | судья eval, разметка |
| qwen/qwen3-vl-30b-a3b-instruct | чтение страниц PDF |
| moonshotai/kimi-k2.6 | разрешён, в пайплайне не используется |

Эмбеддинги e5 (`intfloat/multilingual-e5-base`, MIT) подключаются только как локальный сервер (`EMBEDDER_URL`) и в данной версии не включены; данные в интернет не отправляются. Если e5 будет включён, добавить сюда версию весов и источник загрузки.

## Данные

- Структурированные данные (CSV) и PDF аналитики Альфа-Банка: материалы кейса от организаторов, обезличены, в репозиторий не входят (`data/` в `.gitignore`).
- Синтетические новости (`ml/fixtures/news.jsonl`, `ml/eval/holdout_news.jsonl`, `kind: synthetic_test`): написаны командой для демо и eval, реальных источников не имеют.
- Синтетические фикстуры тестов (`ml/tests/fixtures/rss_sample.xml`, `ml/eval/contradictions.jsonl`): написаны командой.
- RSS-источники: ML-слой по умолчанию их не использует (`NEWS_SOURCE=fixtures`). Список лент для `holdout2` определяется отдельно и должен быть подтверждён организаторами; после подтверждения указать здесь названия и условия использования. Ленты сборщика новостей описаны в разделе ниже.

## Сборщик новостей (`backend/news_collector/`)

Зависимости сборщика (`backend/news_collector/requirements.txt`, в `requirements.lock` не входят; версии на 2026-10-24):

| Пакет | Версия | Лицензия |
|---|---|---|
| feedparser | 6.0.14 | BSD-2-Clause |
| requests | 2.34.2 | Apache-2.0 |
| trafilatura | 2.3.1 | Apache-2.0 |

Сборщик не вызывает модели: теги расставляются правилами по ключевым словам, разбор новостей остаётся за ML-слоем (Gemma через `ml/llm`, с учётом бюджета). Тексты забираются только из открытых лент и страниц без авторизации; User-Agent `alpha-hack-news-bot`.

Источники делятся на две группы (флаг `--sources`, в коде `core` в `backend/news_collector/config.py`). Организаторами кейса не подтверждён ни один из них; условия использования каждого издания нужно проверить перед публичным показом.

| Группа | Ленты | Режим по умолчанию |
|---|---|---|
| core: те же издания, что в `ml/eval/collect_real.py` | Банк России (пресс-релизы, события), Lenta.ru (экономика), ТАСС, Интерфакс, Коммерсантъ (экономика, бизнес), Ведомости (экономика, бизнес) | включены (`--sources core`) |
| не подтверждены, только `--sources all` | РБК, ПРАЙМ, Финмаркет, «Российская газета»; ФРС США, ЕЦБ, US EIA; BBC, The Guardian, CNBC, NYT, WSJ, SCMP, Nikkei Asia, DW, OilPrice; тематические запросы Google News (RU и EN, берутся заголовок и аннотация, ссылки ведут через редирект Google) | выключены |

"""Сборщик новостей: python backend/news_collector/main.py {run,collect,export,status}"""
import argparse
import logging
import time

import config
from collector import collect
from dedup import cluster_new_articles
from export import export
from store import Store

log = logging.getLogger("main")


def cycle(store: Store, args) -> None:
    """Один полный проход: сбор -> события -> выгрузка."""
    sources = config.select_sources(args.sources)
    log.info("сбор: %s", collect(store, sources, prefilter=not args.no_prefilter))
    log.info("события: %s", cluster_new_articles(store))
    log.info("выгрузка: %s", export(store, days=args.days))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=["run", "collect", "export", "status"])
    p.add_argument("--sources", choices=["core", "all"], default="core",
                   help="core: ленты, уже используемые в проекте; all: все, включая не подтверждённые организаторами")
    p.add_argument("--once", action="store_true", help="run: один проход вместо бесконечного цикла")
    p.add_argument("--interval", type=int, default=600, help="run: пауза между проходами, сек")
    p.add_argument("--days", type=int, default=2, help="export: за сколько дней отдавать новости")
    p.add_argument("--no-prefilter", action="store_true", help="не отсеивать нерелевантное по ключевым словам")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s", datefmt="%H:%M:%S")
    logging.getLogger("trafilatura").setLevel(logging.CRITICAL)  # жалуется на каждую нечитаемую страницу

    store = Store(config.DB_PATH)
    try:
        if args.command == "collect":
            log.info("сбор: %s", collect(store, config.select_sources(args.sources), prefilter=not args.no_prefilter))
            log.info("события: %s", cluster_new_articles(store))
        elif args.command == "export":
            log.info("выгрузка: %s", export(store, days=args.days))
        elif args.command == "status":
            print(store.stats())
        else:
            while True:
                try:
                    cycle(store, args)
                except Exception:  # noqa: BLE001 - живой режим не должен падать от одного сбоя
                    log.exception("проход завершился ошибкой")
                if args.once:
                    break
                time.sleep(args.interval)
    finally:
        store.close()


if __name__ == "__main__":
    main()

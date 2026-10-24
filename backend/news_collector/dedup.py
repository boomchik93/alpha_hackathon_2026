"""Группировка статей об одном событии (дубли и пересказы разных изданий)."""
import hashlib
import json
import re
from datetime import datetime, timedelta, timezone

import config
from store import Store

_STOP = set("""
и в во на с со по для из за от до о об не что как это к у а но или да так же его ее их мы вы они он она при про
над под после перед между через также который которые которая году года год сообщил сообщила сообщили заявил
заявила рассказал рассказала стало будет были был была были может могут the a an of to in on for and or is are
was were be by with at from as that this it its has have will after over new says say said about into amid
""".split())
_WORD = re.compile(r"[a-zа-я0-9]+")


def tokenize(*texts: str) -> set[str]:
    """Основы слов (первые 6 букв) и числа: хватает, чтобы узнавать одну новость в разных заголовках."""
    text = " ".join(t for t in texts if t).lower().replace("ё", "е")
    out = set()
    for w in _WORD.findall(text):
        if w in _STOP:
            continue
        if w.isdigit():
            out.add(w)
        elif len(w) >= 3:
            out.add(w[:6])
    return out


def _same_event(a: set, b: set) -> bool:
    common = len(a & b)
    if common < 3:
        return False
    jaccard = common / len(a | b)
    return jaccard >= 0.45 or (common >= 4 and common / min(len(a), len(b)) >= 0.7)


def cluster_new_articles(store: Store) -> dict:
    """Приписывает каждой новой статье событие: существующее или новое."""
    since = (datetime.now(timezone.utc) - timedelta(days=config.CLUSTER_WINDOW_DAYS)).isoformat(timespec="seconds")
    members = [(m["event_id"], set(json.loads(m["tokens"]))) for m in store.clustered_members(since)]
    stats = {"joined": 0, "created": 0}

    for art in store.unclustered_articles():
        toks = set(json.loads(art["tokens"]))
        match = next((eid for eid, mt in members if _same_event(toks, mt)), None)
        if match:
            store.assign_event(art["id"], match)
            stats["joined"] += 1
        else:
            match = hashlib.sha1(art["id"].encode()).hexdigest()[:12]
            store.create_event(match)
            store.assign_event(art["id"], match)
            stats["created"] += 1
        members.append((match, toks))
    store.commit()
    return stats

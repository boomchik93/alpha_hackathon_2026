import os
import tempfile

# До импорта ml: карточки и трассы тестов не должны перезаписывать рабочие в var/.
os.environ["ML_RUNS_DIR"] = tempfile.mkdtemp(prefix="ml_runs_")

import pytest

from ml.config import settings
from ml.llm import client as client_mod


@pytest.fixture(autouse=True)
def mock_llm(monkeypatch):
    """Тесты никогда не ходят в настоящий API и не тратят бюджет."""
    monkeypatch.setattr(client_mod, "_default", client_mod.LLMClient(mock=True, use_cache=False))


@pytest.fixture(autouse=True)
def heuristic_feed():
    """Тесты ленты проверяют эвристику и не зависят от обученной модели пары (ml/ranking/pair_model.json);
    путь с моделью проверяется явно (test_pair_model.py)."""
    from ml.ranking import ranker

    ranker.set_model(None)
    yield
    ranker.clear_model()


def _built() -> bool:
    return settings.paths.profiles.exists() and settings.paths.chunks.exists() and (settings.paths.var / "items.jsonl").exists()


needs_data = pytest.mark.skipif(not _built(), reason="нет var/: запустите make build")
needs_csv = pytest.mark.skipif(not (settings.paths.structured / "df_Uid_data_hack.csv").exists(), reason="нет data/structured")

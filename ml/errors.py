"""Общие исключения ML-слоя."""


class NotFound(KeyError):
    """Компания, материал или карточка не найдены. Сервер отвечает 404; прочие KeyError (маршрут, промпт) 404 не дают."""

    def __str__(self) -> str:
        return str(self.args[0]) if self.args else ""

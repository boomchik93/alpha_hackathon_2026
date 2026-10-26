# Как работать с репозиторием

## Ветки

- `main` — сдаваемая версия; версия на дедлайн отмечена тегом (tag и commit hash в корневом README).
- `ml/dev` — основная ветка разработки: в неё влиты ветки бэкенда (`backend/feat-*`) и фронтенда (`frontend/dev`).
- Новая работа — в короткой ветке от `ml/dev` с префиксом направления (`backend/feat-auth`, `frontend/fix-feed`), затем pull request в `ml/dev`. В `main` попадает только проверенное через pull request.

## Порядок работы

```bash
git clone git@github.com:boomchik93/alpha_hackathon_2026.git
git checkout -b backend/feat-x origin/ml/dev   # префикс по своему направлению
# ... работа ...
make test && make backend-test                   # перед pull request
git add <файлы>                                  # не `git add .`: следите, чтобы не попали .env и данные
git commit -m "backend: ручка /companies"
git push -u origin backend/feat-x
```

Перед pull request подтяните свежую ветку: `git fetch && git merge origin/ml/dev`.

## Коммиты

- Сообщение по-русски или по-английски, одной строкой: что изменилось и зачем. Префикс направления приветствуется: `ml:`, `backend:`, `frontend:`, `product:`.
- Один коммит — одна законченная мысль.

## Что нельзя коммитить

- `.env`, ключи, токены, пароли.
- Архивы и сырые данные кейса (`data/`, `*.zip`), артефакты сборки (`var/`, `var_live/`, `dist/`), виртуальные окружения, `node_modules/`. Демо-данные для жюри выкладываются архивом в релиз (`make demo-bundle`), не в git.
- Чужие персональные данные.

## Pull request

Описание: что сделано, как проверить, что затронуто в чужих папках. Если меняется контракт между слоями (поля ответа API), отметьте владельцев соседних направлений.

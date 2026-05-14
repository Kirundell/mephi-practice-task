# СберАвтоподписка - предсказание целевого действия

Задача: создать модель, которая предсказывает вероятность того, что пользователь совершит целевое действие (оставит заявку, закажет звонок и пр.) на сайте.

## Структура

```
.
├── notebook.ipynb        # EDA + ML + интерпретация (аналитический отчёт)
├── train.py              # обучение модели -> model.pkl
├── test_api.py           # тест API на реальных данных
├── api/
│   ├── main.py           # FastAPI: POST /predict, GET /health
│   ├── schemas.py        # Pydantic-схемы
│   └── preprocessing.py  # clean_sessions + build_features (общий код)
├── target_actions.py     # список целевых event_action
├── model.pkl             # обученная модель
├── metrics.json          # метрики на test
├── requirements.txt
└── README.md
```

`api/preprocessing.py` - источник для трансформации данных. Используется и в notebook, и в `train.py`, и в API: гарантирует, что обучение и serve работают на идентичной матрице фичей.

## Установка

```powershell
git clone https://github.com/Kirundell/mephi-practice-task.git
cd mephi-practice-task
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Данные

Датасет (`ga_sessions.pkl`, `ga_hits.csv`) **в репозитории не лежит** из-за большого объема. Для запуска `notebook.ipynb`, `train.py` или `test_api.py` положите файлы в **родительскую** директорию репозитория:

```
parent/
├── ga_sessions.pkl
├── ga_hits.csv
└── mephi-practice-task/   ← этот репозиторий
    ├── train.py
    └── ...
```

API (`uvicorn api.main:app`) **данные не требует** - он работает на готовом `model.pkl`.

## Запуск API

```powershell
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

Документация Swagger: http://localhost:8000/docs

### Эндпоинты

- `GET /health` - статус сервиса.
- `POST /predict` - предсказание для одного визита.

### Пример запроса

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "9999999999.1234567890",
    "visit_date": "2022-03-15",
    "visit_time": "14:23:01",
    "visit_number": 2,
    "utm_source": "ZpYIoDJMcFzVoPFsHGJL",
    "utm_medium": "cpc",
    "device_category": "mobile",
    "device_os": "Android",
    "device_browser": "Chrome",
    "device_screen_resolution": "412x915",
    "geo_country": "Russia",
    "geo_city": "Moscow"
  }'
```

Ответ:

```json
{
  "session_id": "9999999999.1234567890",
  "prediction": 0,
  "probability": 0.0234,
  "threshold": 0.5
}
```


### Пример запуска
<img width="1094" height="484" alt="image" src="https://github.com/user-attachments/assets/7ef3febe-f1b0-4ef8-968d-09300f2ffe3d" />

<img width="915" height="143" alt="image" src="https://github.com/user-attachments/assets/05b7c7ea-3419-4f2d-8c28-12f223492fe6" />


### Тест API на реальных данных

Требует наличия `ga_sessions.pkl` в родительской папке. Запускать в отдельном терминале, пока uvicorn работает:

```powershell
python test_api.py --n 100
```

Скрипт берёт N случайных визитов, прогоняет через `/predict` и печатает latency (p50/p95/p99) + распределение предсказаний.

## Переобучение модели (опционально)

`model.pkl` уже лежит в репозитории. Для обучения с нуля нужно:

```powershell
python train.py                    # с Optuna (30 trials по умолчанию)
python train.py --quick            # без Optuna, быстрый прогон
python train.py --n-trials 50      # больше trials
```

После завершения перезаписываются `model.pkl` и `metrics.json`.

## Метрики

Финальные значения в `metrics.json`.

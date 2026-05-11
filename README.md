# СберАвтоподписка - предсказание целевого действия

Задача: создать модель, которая предсказывает вероятность того, что пользователь совершит целевое действие (оставит заявку, закажет звонок и пр.) на сайте.

## Структура

```
solution/
├── notebook.ipynb        # EDA + ML + интерпретация (аналитический отчёт)
├── train.py              # обучение модели -> model.pkl
├── api/
│   ├── main.py           # FastAPI: POST /predict, GET /health
│   ├── schemas.py        # Pydantic-схемы
│   └── preprocessing.py  # clean_sessions + build_features (общий код)
├── target_actions.py     # список целевых event_action
├── model.pkl             # артефакт после train.py
├── metrics.json          # метрики на test
├── requirements.txt
└── README.md
```

`api/preprocessing.py` - источник для трансформации данных. Используется и в notebook, и в `train.py`, и в API: гарантирует, что обучение и serve работают на идентичной матрице фичей.

## Установка

```powershell
cd "2 sem/Practice task/solution"
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Обучение

```powershell
python train.py                    # с Optuna (30 trials по умолчанию)
python train.py --quick            # без Optuna
python train.py --n-trials 50      # больше trials
```

После завершения создаются `model.pkl` и `metrics.json`.

## API

```powershell
uvicorn api.main:app --host 0.0.0.0 --port 8000
```


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

## Метрики

Финальные значения см. в `metrics.json` после обучения.

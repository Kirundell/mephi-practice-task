"""Чистка и фиче-инжиниринг для ga_sessions.

Логика трансформации сырых данных в признаки для модели, используется в трёх местах:
  1. EDA notebook - для распределений, корреляций, CR в разрезах фичей.
  2. train.py - для обучения модели.
  3. api/main.py - для трансформации входящего JSON.

"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Колонки в ga_sessions, которые приходят на вход
RAW_COLUMNS = [
    "session_id",
    "client_id",
    "visit_date",
    "visit_time",
    "visit_number",
    "utm_source",
    "utm_medium",
    "utm_campaign",
    "utm_adcontent",
    "utm_keyword",
    "device_category",
    "device_os",
    "device_brand",
    "device_model",
    "device_screen_resolution",
    "device_browser",
    "geo_country",
    "geo_city",
]

# Категориальные колонки, которые идут в модель
CATEGORICAL_FEATURES = [
    "utm_source",
    "utm_medium",
    "utm_campaign",
    "utm_adcontent",
    "device_category",
    "device_os",
    "device_brand",
    "device_browser",
    "geo_country",
    "geo_city",
]

# Числовые/бинарные фичи, построенные из сырых данных
NUMERIC_FEATURES = [
    "visit_number_clipped",
    "visit_hour",
    "visit_dayofweek",
    "visit_month",
    "is_weekend",
    "is_organic",
    "is_social",
    "is_moscow_region",
    "is_russia",
    "is_mobile",
    "screen_area",
]

ORGANIC_MEDIUMS = {"organic", "referral", "(none)"}

# Зашифрованные ID соцсетей
SOCIAL_SOURCES = {
    "QxAxdyPLuQMEcrdZWdWb",
    "MvfHsxITijuriZxsqZqt",
    "ISrKoXQCxqqYvAZICvjs",
    "IZEXUFLARCUMynmHNBGo",
    "PlbkrSYoHuZBWfYjYnfw",
    "gVRrcxiDQubJiljoTbGm",
}

MOSCOW_CITIES = {
    "Moscow",
    "Zelenograd",
    "Krasnogorsk",
    "Khimki",
    "Mytishchi",
    "Lyubertsy",
    "Balashikha",
    "Odintsovo",
    "Podolsk",
    "Reutov",
}


def clean_sessions(
    df: pd.DataFrame, *, nan_drop_threshold: float = 0.9, verbose: bool = False
) -> pd.DataFrame:
    """Дедупликация, типизация, обработка пропусков для ga_sessions:
      - доля NaN > nan_drop_threshold -> колонка удаляется.
      - категориальные NaN -> 'unknown'.
      - числовые NaN -> медиана.

    verbose=True печатает промежуточные shape и список удалённых колонок
    (полезно в EDA notebook, в train.py и API оставляем False).
    """
    df = df.copy()

    df = df.drop_duplicates(subset="session_id", keep="first")
    if verbose:
        print(f"after dedup: shape={df.shape}")

    df["visit_date"] = pd.to_datetime(df["visit_date"], errors="coerce")
    df["visit_number"] = (
        pd.to_numeric(df["visit_number"], errors="coerce").fillna(1).astype(int)
    )

    nan_share = df.isna().mean()
    cols_to_drop = nan_share[nan_share > nan_drop_threshold].index.tolist()
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)
        if verbose:
            print(f"dropped (NaN > {nan_drop_threshold:.0%}): {cols_to_drop}")

    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].fillna("unknown")
    for col in df.select_dtypes(include=["number"]).columns:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())

    if verbose:
        print(
            f"clean_sessions done: shape={df.shape}, NaN total={int(df.isna().sum().sum())}"
        )
    return df


def fill_nans(df: pd.DataFrame) -> pd.DataFrame:
    """Лёгкая версия clean_sessions для serve-time в API.

    Без дедупликации и удаления колонок (бессмысленно на одной строке):
    только заполняем NaN на 'unknown' для строк и 0 для чисел, чтобы
    схема колонок совпала с обучающим набором.
    """
    df = df.copy()
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].fillna("unknown")
    for col in df.select_dtypes(include=["number"]).columns:
        df[col] = df[col].fillna(0)
    return df


def _parse_hour(value) -> int:
    """visit_time в формате 'HH:MM:SS' -> час.
    Для NaN/мусора возвращаем 12 как дефолт."""
    if pd.isna(value):
        return 12
    s = str(value)
    try:
        return int(s.split(":")[0])
    except (ValueError, IndexError):
        return 12


def _parse_screen_area(value) -> float:
    """device_screen_resolution в формате 'WIDTHxHEIGHT' -> площадь экрана.
    Разделителем может быть латинская 'x' или кириллическая 'х'."""
    if pd.isna(value):
        return 0.0
    s = str(value).lower().replace(" ", "")
    for sep in ("x", "х"):
        if sep in s:
            parts = s.split(sep)
            try:
                return float(parts[0]) * float(parts[1])
            except (ValueError, IndexError):
                return 0.0
    return 0.0


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Построить матрицу фичей из очищенного ga_sessions.

    Возвращает DataFrame с CATEGORICAL_FEATURES + NUMERIC_FEATURES
    """
    df = df.copy()

    df["visit_hour"] = df["visit_time"].map(_parse_hour)

    vd = pd.to_datetime(df["visit_date"], errors="coerce")
    df["visit_dayofweek"] = vd.dt.dayofweek.fillna(0).astype(int)
    df["visit_month"] = vd.dt.month.fillna(1).astype(int)
    df["is_weekend"] = (df["visit_dayofweek"] >= 5).astype(int)

    df["visit_number_clipped"] = df["visit_number"].clip(1, 20).astype(int)

    df["is_organic"] = df["utm_medium"].isin(ORGANIC_MEDIUMS).astype(int)
    df["is_social"] = df["utm_source"].isin(SOCIAL_SOURCES).astype(int)
    df["is_moscow_region"] = df["geo_city"].isin(MOSCOW_CITIES).astype(int)
    df["is_russia"] = (df["geo_country"] == "Russia").astype(int)
    df["is_mobile"] = (df["device_category"] == "mobile").astype(int)

    df["screen_area"] = (
        df["device_screen_resolution"].map(_parse_screen_area).astype(float)
    )

    cat_cols = [c for c in CATEGORICAL_FEATURES if c in df.columns]
    num_cols = [c for c in NUMERIC_FEATURES if c in df.columns]

    for c in cat_cols:
        df[c] = df[c].astype(str)

    return df[cat_cols + num_cols]

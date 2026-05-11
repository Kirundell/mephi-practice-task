"""Обучение модели предсказания целевого действия.

Запуск:
    python train.py
    python train.py --quick           # без Optuna, быстрый прогон
    python train.py --n-trials 30     # Optuna с N trials

На выходе:
    model.pkl - Pipeline (preprocessing + XGBClassifier) + threshold
    metrics.json - итоговые метрики на test
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from api.preprocessing import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    build_features,
    clean_sessions,
)
from target_actions import TARGET_ACTIONS

DATA_DIR = ROOT.parent
SESSIONS_PATH = DATA_DIR / "ga_sessions.pkl"
HITS_PATH = DATA_DIR / "ga_hits.pkl"

MODEL_PATH = ROOT / "model.pkl"
METRICS_PATH = ROOT / "metrics.json"

RANDOM_STATE = 42


def load_sessions() -> pd.DataFrame:
    df = pd.read_pickle(SESSIONS_PATH)
    print(f"sessions: {df.shape}")
    return df


def load_target_session_ids() -> set[str]:
    """Собрать set session_id с целевым event_action.

    ga_hits на 15.7M строк не помещается в память целиком, поэтому
    читаем CSV чанками только по двум нужным колонкам.
    """
    print("\nloading target session ids")
    csv_path = HITS_PATH.with_suffix(".csv")
    target_set: set[str] = set()
    rows_read = 0
    target_actions_set = set(TARGET_ACTIONS)
    for chunk in pd.read_csv(
        csv_path,
        usecols=["session_id", "event_action"],
        chunksize=2_000_000,
        low_memory=False,
    ):
        mask = chunk["event_action"].isin(target_actions_set)
        target_set.update(chunk.loc[mask, "session_id"].unique())
        rows_read += len(chunk)
        print(f"read {rows_read:,} rows")
    print(f"target sessions: {len(target_set)}")
    return target_set


def build_target(sessions: pd.DataFrame, target_session_ids: set[str]) -> pd.Series:
    y = sessions["session_id"].isin(target_session_ids).astype(int)
    print(f"target distribution: positives={int(y.sum())}, CR={y.mean():.4%}")
    return y


def make_xgb_pipeline(cat_cols, num_cols, params):
    from xgboost import XGBClassifier

    pre = ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(
                    handle_unknown="ignore", min_frequency=20, sparse_output=True
                ),
                cat_cols,
            ),
            ("num", "passthrough", num_cols),
        ],
        remainder="drop",
        sparse_threshold=1.0,
    )
    clf = XGBClassifier(
        objective="binary:logistic",
        eval_metric="auc",
        tree_method="hist",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        **params,
    )
    return Pipeline([("pre", pre), ("clf", clf)])


def make_lr_pipeline(cat_cols, num_cols):
    pre = ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(
                    handle_unknown="ignore", min_frequency=50, sparse_output=True
                ),
                cat_cols,
            ),
            ("num", "passthrough", num_cols),
        ],
        remainder="drop",
        sparse_threshold=1.0,
    )
    clf = LogisticRegression(
        max_iter=200, C=1.0, solver="liblinear", random_state=RANDOM_STATE
    )
    return Pipeline([("pre", pre), ("clf", clf)])


def find_best_threshold(y_true, y_proba) -> tuple[float, float]:
    """Подобрать порог бинаризации по максимальному F1 на тесте.

    XGBoost с scale_pos_weight на сильно несбалансированных данных раздувает
    proba класса 1, поэтому дефолтный порог 0.5 даёт много ложных
    срабатываний. Перебираем все пороги через precision_recall_curve,
    берём тот, где F1 = 2 * P * R / (P + R) максимален. Этот порог
    сохраняется в model.pkl и используется в API при ответе на /predict.
    """
    p, r, t = precision_recall_curve(y_true, y_proba)
    f1 = 2 * p * r / np.clip(p + r, 1e-9, None)
    best = int(np.argmax(f1[:-1])) if len(f1) > 1 else 0
    return float(t[best]) if len(t) > 0 else 0.5, float(f1[best])


def run_optuna(X_train, y_train, cat_cols, num_cols, n_trials):
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    pos = int(y_train.sum())
    neg = len(y_train) - pos
    scale_pos_weight = neg / max(pos, 1)

    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)

    def objective(trial):
        params = {
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 100, 600),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10, log=True),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
            "scale_pos_weight": scale_pos_weight,
        }

        scores = []
        for tr_idx, va_idx in skf.split(X_train, y_train):
            pipe = make_xgb_pipeline(cat_cols, num_cols, params)
            pipe.fit(X_train.iloc[tr_idx], y_train.iloc[tr_idx])
            proba = pipe.predict_proba(X_train.iloc[va_idx])[:, 1]
            scores.append(roc_auc_score(y_train.iloc[va_idx], proba))
        return float(np.mean(scores))

    study = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE)
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    print(f"best CV ROC-AUC: {study.best_value:.4f}")
    print(f"best params: {study.best_params}")
    best_params = dict(study.best_params)
    best_params["scale_pos_weight"] = scale_pos_weight
    return best_params


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--quick", action="store_true", help="без Optuna"
    )
    parser.add_argument(
        "--n-trials", type=int, default=30, help="число Optuna trials"
    )
    args = parser.parse_args()

    t0 = time.time()
    sessions = load_sessions()
    target_ids = load_target_session_ids()

    print("\ncleaning sessions")
    sessions_clean = clean_sessions(sessions, verbose=True)

    print("\nbuilding target")
    y = build_target(sessions_clean, target_ids)

    print("\nbuilding features")
    X = build_features(sessions_clean)
    print(f"feature matrix: {X.shape}")
    cat_cols = [c for c in CATEGORICAL_FEATURES if c in X.columns]
    num_cols = [c for c in NUMERIC_FEATURES if c in X.columns]
    print(f"categorical: {cat_cols}")
    print(f"numeric: {num_cols}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    print(f"\nsplit: train={len(X_train)}, test={len(X_test)}")

    print("\nbaseline: LogisticRegression")
    lr = make_lr_pipeline(cat_cols, num_cols)
    lr.fit(X_train, y_train)
    lr_proba = lr.predict_proba(X_test)[:, 1]
    print(f"ROC-AUC = {roc_auc_score(y_test, lr_proba):.4f}")
    print(f"PR-AUC = {average_precision_score(y_test, lr_proba):.4f}")

    print("\nmain: XGBoost")
    pos = int(y_train.sum())
    neg = len(y_train) - pos
    if args.quick:
        params = {
            "max_depth": 6,
            "learning_rate": 0.1,
            "n_estimators": 300,
            "min_child_weight": 5,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "reg_alpha": 0.01,
            "reg_lambda": 1.0,
            "gamma": 0.0,
            "scale_pos_weight": neg / max(pos, 1),
        }
        print(f"quick mode, params={params}")
    else:
        print(f"running Optuna with {args.n_trials} trials")
        params = run_optuna(X_train, y_train, cat_cols, num_cols, args.n_trials)

    pipe = make_xgb_pipeline(cat_cols, num_cols, params)
    pipe.fit(X_train, y_train)
    proba = pipe.predict_proba(X_test)[:, 1]

    roc = roc_auc_score(y_test, proba)
    pr = average_precision_score(y_test, proba)
    threshold, best_f1 = find_best_threshold(y_test, proba)
    pred = (proba >= threshold).astype(int)

    print(f"\nROC-AUC = {roc:.4f}")
    print(f"PR-AUC = {pr:.4f}")
    print(f"threshold = {threshold:.4f}")
    print(f"F1@thr = {best_f1:.4f}")
    print(f"F1 sklearn = {f1_score(y_test, pred):.4f}")

    artifact = {
        "pipeline": pipe,
        "threshold": threshold,
        "feature_columns": cat_cols + num_cols,
        "categorical_columns": cat_cols,
        "numeric_columns": num_cols,
    }
    joblib.dump(artifact, MODEL_PATH)
    print(f"saved {MODEL_PATH}")

    metrics = {
        "roc_auc_test": roc,
        "pr_auc_test": pr,
        "f1_at_threshold": best_f1,
        "threshold": threshold,
        "baseline_lr_roc_auc": float(roc_auc_score(y_test, lr_proba)),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "positive_rate_train": float(y_train.mean()),
        "feature_columns": cat_cols + num_cols,
        "params": {
            k: float(v) if isinstance(v, np.floating) else v for k, v in params.items()
        },
        "training_time_sec": round(time.time() - t0, 1),
    }
    METRICS_PATH.write_text(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"saved {METRICS_PATH}")


if __name__ == "__main__":
    main()

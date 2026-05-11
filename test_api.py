"""Smoke-тест API на реальных данных.

Берёт N случайных визитов из ga_sessions.pkl, отправляет каждый в
/predict, печатает сводку по latency и распределению предсказаний.

Запуск (в отдельном терминале, пока uvicorn работает):
    python test_api.py
    python test_api.py --n 500 --url http://localhost:8000
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).parent
SESSIONS_PATH = ROOT.parent / "ga_sessions.pkl"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=100, help="число визитов для теста")
    parser.add_argument("--url", default="http://localhost:8000", help="базовый URL API")
    args = parser.parse_args()

    print(f"loading {SESSIONS_PATH.name}...")
    sessions = pd.read_pickle(SESSIONS_PATH)
    sample = sessions.sample(args.n, random_state=42).reset_index(drop=True)
    print(f"sampled {len(sample)} visits")

    print(f"\nhealthcheck: {args.url}/health")
    r = requests.get(f"{args.url}/health", timeout=5)
    print(f"  {r.status_code} {r.json()}")

    print(f"\nsending {args.n} requests to {args.url}/predict...")
    latencies = []
    predictions = []
    probabilities = []
    errors = 0

    for i, row in sample.iterrows():
        payload = row.where(pd.notna(row), None).to_dict()
        # visit_date в pandas Timestamp -> строка
        if hasattr(payload.get("visit_date"), "strftime"):
            payload["visit_date"] = payload["visit_date"].strftime("%Y-%m-%d")
        payload["visit_number"] = int(payload["visit_number"])

        t0 = time.perf_counter()
        try:
            r = requests.post(f"{args.url}/predict", json=payload, timeout=10)
            r.raise_for_status()
            data = r.json()
            latencies.append((time.perf_counter() - t0) * 1000)
            predictions.append(data["prediction"])
            probabilities.append(data["probability"])
        except Exception as e:
            errors += 1
            print(f"  request {i} failed: {e}")

    if not latencies:
        print("\nno successful responses :(")
        return

    lat = pd.Series(latencies)
    pred = pd.Series(predictions)
    proba = pd.Series(probabilities)

    print(f"\n=== latency (ms) ===")
    print(f"  count = {len(lat)} (errors: {errors})")
    print(f"  mean  = {lat.mean():.1f}")
    print(f"  p50   = {lat.quantile(0.5):.1f}")
    print(f"  p95   = {lat.quantile(0.95):.1f}")
    print(f"  p99   = {lat.quantile(0.99):.1f}")
    print(f"  max   = {lat.max():.1f}")

    print(f"\n=== predictions ===")
    print(f"  predicted positive: {int(pred.sum())} / {len(pred)} ({pred.mean():.1%})")
    print(f"  probability: min={proba.min():.4f}  median={proba.median():.4f}  max={proba.max():.4f}")

    print(f"\n=== sample responses (первые 5) ===")
    for i in range(min(5, len(predictions))):
        print(f"  {i}: prediction={predictions[i]}  probability={probabilities[i]:.4f}  "
              f"latency={latencies[i]:.0f}ms")


if __name__ == "__main__":
    main()

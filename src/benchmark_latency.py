"""Latency benchmarks for the prediction pipelines.

outputs/reports/latency_benchmarks.csv
outputs/reports/latency_benchmarks.md
"""
from __future__ import annotations

import time
from statistics import mean, stdev
from typing import Callable

import pandas as pd

from src.config import REPORTS_DIR
from src.speech_features import iter_wav_files

WARM_RUNS_FAST = 20
WARM_RUNS_SLOW = 10  


def _bench(predictor: Callable, payloads, warm_runs: int):
    cold_t0 = time.perf_counter()
    predictor(payloads[0])
    cold_ms = (time.perf_counter() - cold_t0) * 1000.0

    times_ms = []
    for i in range(warm_runs):
        payload = payloads[i % len(payloads)]
        t = time.perf_counter()
        predictor(payload)
        times_ms.append((time.perf_counter() - t) * 1000.0)
    return cold_ms, mean(times_ms), (stdev(times_ms) if len(times_ms) > 1 else 0.0)


def main() -> None:
    text_payloads = [
        "I am so happy today",
        "This is the worst day ever",
        "What just happened, I cannot believe it",
        "I am scared and shaking",
        "Just another normal day",
    ]
    rows = []

    print("Benchmarking text models...")
    from src.predict_text import predict_emotion as classical
    for model_key, label in (("logreg", "logreg (text)"),
                              ("linearsvc", "linearsvc (text)")):
        try:
            fn = lambda payload, _k=model_key: classical(payload, model_name=_k)
            cold, warm_mean, warm_std = _bench(fn, text_payloads, WARM_RUNS_FAST)
            rows.append({
                "model": label,
                "cold_start_ms": round(cold, 1),
                "warm_mean_ms": round(warm_mean, 2),
                "warm_std_ms": round(warm_std, 2),
                "n_warm": WARM_RUNS_FAST,
            })
        except Exception as e:
            print(f"  {label} skipped: {type(e).__name__}: {e}")

    try:
        from src.predict_text_distilbert import predict_emotion as bert
        cold, warm_mean, warm_std = _bench(bert, text_payloads, WARM_RUNS_SLOW)
        rows.append({
            "model": "distilbert (text)",
            "cold_start_ms": round(cold, 1),
            "warm_mean_ms": round(warm_mean, 2),
            "warm_std_ms": round(warm_std, 2),
            "n_warm": WARM_RUNS_SLOW,
        })
    except Exception as e:
        print(f"  distilbert skipped: {type(e).__name__}: {e}")

    print("Benchmarking speech models...")
    from src.predict_speech import predict_emotion as speech_pred
    files = list(iter_wav_files())[:5]
    if files:
        for model_key, label in (("random_forest", "random_forest (speech)"),
                                  ("svm_rbf", "svm_rbf (speech)")):
            try:
                fn = lambda payload, _k=model_key: speech_pred(payload, model_name=_k)
                cold, warm_mean, warm_std = _bench(fn, files, WARM_RUNS_FAST)
                rows.append({
                    "model": label,
                    "cold_start_ms": round(cold, 1),
                    "warm_mean_ms": round(warm_mean, 2),
                    "warm_std_ms": round(warm_std, 2),
                    "n_warm": WARM_RUNS_FAST,
                })
            except Exception as e:
                print(f"  {label} skipped: {type(e).__name__}: {e}")

    df = pd.DataFrame(rows)
    print()
    print(df.to_string(index=False))

    out_csv = REPORTS_DIR / "latency_benchmarks.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nWrote {out_csv}")

    md = ["# Latency benchmarks", "",
          "Cold-start measures the first call (includes lazy model load).",
          "Warm latency is averaged over N=20 subsequent calls (10 for DistilBERT).",
          "Hardware: CPU only.", "",
          "| Model | Cold-start (ms) | Warm mean (ms) | Warm std-dev (ms) | N warm |",
          "|---|---:|---:|---:|---:|"]
    for r in rows:
        md.append(
            f"| {r['model']} | {r['cold_start_ms']} | "
            f"{r['warm_mean_ms']} | {r['warm_std_ms']} | {r['n_warm']} |"
        )

    out_md = REPORTS_DIR / "latency_benchmarks.md"
    out_md.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"Wrote {out_md}")


if __name__ == "__main__":
    main()

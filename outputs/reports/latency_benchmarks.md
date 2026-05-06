# Latency benchmarks

Cold-start measures the first call (includes lazy model load).
Warm latency is averaged over N=20 subsequent calls (10 for DistilBERT).
Hardware: CPU only.

| Model | Cold-start (ms) | Warm mean (ms) | Warm std-dev (ms) | N warm |
|---|---:|---:|---:|---:|
| logreg (text) | 1959.2 | 3.75 | 1.33 | 20 |
| distilbert (text) | 20176.4 | 39.27 | 5.92 | 10 |
| svm_rbf (speech) | 4742.5 | 26.05 | 8.66 | 20 |

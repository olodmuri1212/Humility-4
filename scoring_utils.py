# scoring_utils.py
from __future__ import annotations
from typing import Iterable, List
import math

def _as_list(values: Iterable[float]) -> List[float]:
    xs = [float(v) for v in values if v is not None]
    return xs

def median(values: Iterable[float]) -> float:
    xs = sorted(_as_list(values))
    n = len(xs)
    if n == 0: return 0.0
    mid = n // 2
    return (xs[mid] if n % 2 else 0.5 * (xs[mid-1] + xs[mid]))

def trimmed_mean(values: Iterable[float], trim: float = 0.1) -> float:
    xs = sorted(_as_list(values))
    n = len(xs)
    if n == 0: return 0.0
    k = int(math.floor(n * trim))
    xs = xs[k:n-k] if n - 2*k > 0 else xs
    return sum(xs) / len(xs)

def winsorized_mean(values: Iterable[float], winsor: float = 0.1) -> float:
    xs = sorted(_as_list(values))
    n = len(xs)
    if n == 0: return 0.0
    k = int(math.floor(n * winsor))
    low = xs[k] if n > 0 else 0.0
    high = xs[n - k - 1] if n > 0 else 0.0
    ws = [min(max(x, low), high) for x in xs]
    return sum(ws) / len(ws)

def recency_weights(n: int, decay: float = 0.8) -> List[float]:
    # i=0..n-1 → older..newer ; newer gets higher weight
    ws = [decay ** (n-1-i) for i in range(n)]
    s = sum(ws) or 1.0
    return [w / s for w in ws]

def trimmed_recency_weighted_mean(values: Iterable[float], trim: float = 0.15, decay: float = 0.8) -> float:
    xs = _as_list(values)
    n = len(xs)
    if n == 0: return 0.0
    # pair (value, index) to keep recency
    pairs = list(enumerate(xs))  # (i, x_i) with i increasing in time
    # sort by value for trimming
    by_val = sorted(pairs, key=lambda p: p[1])
    k = int(math.floor(n * trim))
    kept = by_val[k:n-k] if n - 2*k > 0 else by_val
    # recency weights computed by original index
    ws = recency_weights(n, decay)
    num = sum(ws[i] * x for i, x in kept)
    den = sum(ws[i] for i, _ in kept) or 1.0
    return num / den

def ema(values: Iterable[float], beta: float = 0.7) -> float:
    xs = _as_list(values)
    if not xs: return 0.0
    y = xs[0]
    for x in xs[1:]:
        y = beta * y + (1 - beta) * x
    return y

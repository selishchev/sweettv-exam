"""MAP@K as defined on the course slide: AP@K = sum_r P@r * rel(r) / min(K, |R_u|)."""
import numpy as np


def apk(actual, predicted, k=5):
    if not actual:
        return 0.0
    hits, score, seen = 0, 0.0, set()
    for i, p in enumerate(predicted[:k]):
        if p in actual and p not in seen:
            hits += 1
            score += hits / (i + 1.0)
        seen.add(p)
    return score / min(len(actual), k)


def mapk(truth, preds, users, k=5):
    return float(np.mean([apk(truth.get(u, set()), preds.get(u, []), k) for u in users]))


def slide_example():
    """Course slide example: relevant = {A, B, C}; X = A,B,D,C,E -> 0.917; Y = D,A,E,B,C -> 0.533."""
    R = {'A', 'B', 'C'}
    return apk(R, list('ABDCE')), apk(R, list('DAEBC'))

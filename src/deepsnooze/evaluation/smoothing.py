import numpy as np
from scipy.stats import mode


def smooth_predictions(preds, window_size=5):
    """
    Applies a rolling majority vote to smooth predictions.
    Removes impossible transitions like NREM -> REM (4s) -> NREM.
    """
    n = len(preds)
    smoothed = np.zeros_like(preds)
    half_window = window_size // 2

    for i in range(n):
        start = max(0, i - half_window)
        end = min(n, i + half_window + 1)
        neighbors = preds[start:end]
        vote = mode(neighbors, keepdims=True)[0][0]
        smoothed[i] = vote

    return smoothed

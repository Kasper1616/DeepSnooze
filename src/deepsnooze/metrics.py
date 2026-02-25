import numpy as np
from sklearn.metrics import classification_report


def custom_classification_report(y_true, y_prob, target_names=None, n_bins=10):
    """
    Classification report extended with probabilistic calibration metrics.

    Args:
        y_true:       (N,) integer true labels
        y_prob:       (N, C) predicted probabilities (softmax outputs)
        target_names: list of class name strings
        n_bins:       number of bins for ECE computation

    Returns:
        Formatted string report.
    """
    y_pred = np.argmax(y_prob, axis=1)

    base = classification_report(
        y_true, y_pred, target_names=target_names, zero_division=0
    )

    # --- NLL and LPD ---
    # Clip for numerical stability
    p_correct = y_prob[np.arange(len(y_true)), y_true].clip(1e-12, 1.0)
    nll = -np.mean(np.log(p_correct))
    lpd = np.mean(np.log(p_correct))

    # --- Brier Score (multiclass) ---
    n_classes = y_prob.shape[1]
    y_one_hot = np.eye(n_classes)[y_true]
    brier = np.mean(np.sum((y_prob - y_one_hot) ** 2, axis=1))

    # --- ECE (Expected Calibration Error) ---
    confidences = np.max(y_prob, axis=1)
    correct = (y_pred == y_true).astype(float)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (confidences > lo) & (confidences <= hi)
        if mask.sum() == 0:
            continue
        acc_bin = correct[mask].mean()
        conf_bin = confidences[mask].mean()
        ece += mask.sum() * abs(acc_bin - conf_bin)
    ece /= len(y_true)

    calibration = (
        f"\n"
        f"{'NLL':>12}  {nll:8.4f}\n"
        f"{'LPD':>12}  {lpd:8.4f}\n"
        f"{'ECE':>12}  {ece:8.4f}\n"
        f"{'Brier':>12}  {brier:8.4f}\n"
    )

    return str(base) + calibration
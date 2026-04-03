from __future__ import annotations

from typing import Dict

import numpy as np


def _safe_pearson(x: np.ndarray, y: np.ndarray) -> float:
    if x.size == 0 or y.size == 0:
        return 0.0
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def concordance_correlation_coefficient(
    y_true: np.ndarray, y_pred: np.ndarray
) -> float:
    if y_true.size == 0 or y_pred.size == 0:
        return 0.0
    true_mean = np.mean(y_true)
    pred_mean = np.mean(y_pred)
    true_var = np.var(y_true)
    pred_var = np.var(y_pred)
    covariance = np.mean((y_true - true_mean) * (y_pred - pred_mean))
    denominator = true_var + pred_var + (true_mean - pred_mean) ** 2
    if abs(denominator) < 1e-12:
        return 0.0
    return float((2.0 * covariance) / denominator)


def compute_regression_metrics(
    predictions: np.ndarray, labels: np.ndarray
) -> Dict[str, float]:
    predictions = np.asarray(predictions, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.float32)

    if predictions.ndim != 2 or labels.ndim != 2 or predictions.shape[1] != 2:
        raise ValueError("Predictions and labels must have shape [batch, 2].")

    mae_v = float(np.mean(np.abs(predictions[:, 0] - labels[:, 0])))
    mae_a = float(np.mean(np.abs(predictions[:, 1] - labels[:, 1])))
    pearson_v = _safe_pearson(predictions[:, 0], labels[:, 0])
    pearson_a = _safe_pearson(predictions[:, 1], labels[:, 1])
    ccc_v = concordance_correlation_coefficient(labels[:, 0], predictions[:, 0])
    ccc_a = concordance_correlation_coefficient(labels[:, 1], predictions[:, 1])

    return {
        "mae": float((mae_v + mae_a) / 2.0),
        "mae_v": mae_v,
        "mae_a": mae_a,
        "pearson": float((pearson_v + pearson_a) / 2.0),
        "pearson_v": pearson_v,
        "pearson_a": pearson_a,
        "ccc": float((ccc_v + ccc_a) / 2.0),
        "ccc_v": ccc_v,
        "ccc_a": ccc_a,
    }

from typing import Dict, Iterable, List

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, f1_score


def classification_metrics(y_true: Iterable[int], y_pred: Iterable[int]) -> Dict[str, float]:
    y_true = list(y_true)
    y_pred = list(y_pred)
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }
    return metrics


def render_report(labels: List[str], y_true: Iterable[int], y_pred: Iterable[int]) -> str:
    return classification_report(
        y_true,
        y_pred,
        target_names=labels,
        digits=4,
        zero_division=0,
    )

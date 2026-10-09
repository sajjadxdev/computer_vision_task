import numpy as np
from typing import Dict, Any, List, Tuple
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    confusion_matrix,
    roc_curve
)


def compute_metrics(y_true: List[int], y_pred: List[int], y_prob: List[float], class_names: List[str]) -> Dict[str, Any]:
    """
    Computes comprehensive quantitative metrics for model evaluation.
    """
    y_true_np = np.array(y_true)
    y_pred_np = np.array(y_pred)
    y_prob_np = np.array(y_prob)

    acc = accuracy_score(y_true_np, y_pred_np)
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(y_true_np, y_pred_np, average="macro", zero_division=0)
    per_class_p, per_class_r, per_class_f1, per_class_supp = precision_recall_fscore_support(
        y_true_np, y_pred_np, average=None, zero_division=0
    )

    try:
        auc = roc_auc_score(y_true_np, y_prob_np)
        fpr, tpr, thresholds = roc_curve(y_true_np, y_prob_np)
        roc_data = {"fpr": fpr.tolist(), "tpr": tpr.tolist()}
    except Exception:
        auc = 0.5
        roc_data = {"fpr": [0, 1], "tpr": [0, 1]}

    cm = confusion_matrix(y_true_np, y_pred_np)

    per_class_metrics = {}
    for idx, cname in enumerate(class_names):
        per_class_metrics[cname] = {
            "precision": float(per_class_p[idx]),
            "recall": float(per_class_r[idx]),
            "f1_score": float(per_class_f1[idx]),
            "support": int(per_class_supp[idx])
        }

    return {
        "accuracy": float(acc),
        "macro_precision": float(macro_p),
        "macro_recall": float(macro_r),
        "macro_f1_score": float(macro_f1),
        "roc_auc": float(auc),
        "confusion_matrix": cm.tolist(),
        "per_class": per_class_metrics,
        "roc_curve": roc_data
    }

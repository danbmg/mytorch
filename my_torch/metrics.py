"""Classification metrics computed from class indices.

Accuracy alone can hide a weak class when the classes are unbalanced: the
per-class precision / recall / F1 and the confusion matrix show it.
"""

import numpy as np


def accuracy(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return float((y_true == y_pred).mean()) if len(y_true) else float("nan")


def confusion_matrix(y_true, y_pred, n_classes):
    """cm[i, j] = number of samples of true class i predicted as class j."""
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    np.add.at(cm, (np.asarray(y_true, dtype=np.int64), np.asarray(y_pred, dtype=np.int64)), 1)
    return cm


def per_class(cm):
    """Precision, recall, F1 and support of every class, from a confusion matrix."""
    tp = np.diag(cm).astype(float)
    predicted = cm.sum(axis=0)
    support = cm.sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        precision = np.where(predicted > 0, tp / predicted, 0.0)
        recall = np.where(support > 0, tp / support, 0.0)
        f1 = np.where(precision + recall > 0, 2 * precision * recall / (precision + recall), 0.0)
    return {"precision": precision, "recall": recall, "f1": f1, "support": support}


def macro_f1(cm):
    """Mean of the per-class F1: every class weighs the same, whatever its size."""
    return float(per_class(cm)["f1"].mean())


def balanced_accuracy(cm):
    """Mean of the per-class recall."""
    support = cm.sum(axis=1)
    recall = per_class(cm)["recall"]
    return float(recall[support > 0].mean())


def report(cm, labels=None):
    """Text table: precision / recall / F1 / support of each class, then the confusion matrix."""
    labels = labels or [str(i) for i in range(len(cm))]
    stats = per_class(cm)
    width = max(len(label) for label in labels) + 2
    lines = [f"{'':<{width}}precision  recall      f1  support"]
    for i, label in enumerate(labels):
        lines.append(f"{label:<{width}}{stats['precision'][i]:9.4f}{stats['recall'][i]:8.4f}"
                     f"{stats['f1'][i]:8.4f}{stats['support'][i]:9d}")
    total = cm.sum()
    lines.append(f"{'accuracy':<{width}}{np.trace(cm) / total if total else float('nan'):25.4f}{total:9d}")
    lines.append(f"{'macro f1':<{width}}{macro_f1(cm):25.4f}")
    lines.append("")
    lines.append("confusion matrix (rows: true class, columns: predicted)")
    lines.append(" " * width + "".join(f"{i:>9}" for i in range(len(cm))))
    for i, label in enumerate(labels):
        lines.append(f"{label:<{width}}" + "".join(f"{v:9d}" for v in cm[i]))
    return "\n".join(lines)

"""Classification metrics. Accuracy alone hides a weak class when the classes are
unbalanced (41.6 % of the chess boards are "Nothing"): the per-class precision,
recall, F1 and the confusion matrix show it."""

import numpy as np


def confusion_matrix(y_true, y_pred, n_classes):
    """cm[i, j] = number of samples of true class i predicted as class j."""
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    np.add.at(cm, (np.asarray(y_true, dtype=np.int64), np.asarray(y_pred, dtype=np.int64)), 1)
    return cm


def per_class(cm):
    tp = np.diag(cm).astype(float)  # true positives of each class
    predicted = cm.sum(axis=0)  # samples predicted as each class
    support = cm.sum(axis=1)  # samples of each class
    with np.errstate(divide="ignore", invalid="ignore"):
        precision = np.where(predicted > 0, tp / predicted, 0.0)
        recall = np.where(support > 0, tp / support, 0.0)
        f1 = np.where(precision + recall > 0, 2 * precision * recall / (precision + recall), 0.0)
    return {"precision": precision, "recall": recall, "f1": f1, "support": support}


def macro_f1(cm):
    """Mean of the per-class F1: every class weighs the same, whatever its size."""
    return float(per_class(cm)["f1"].mean())


def report(cm, labels):
    """Precision / recall / F1 / support of each class, accuracy, macro F1, then the confusion matrix."""
    s = per_class(cm)
    width = max(len(label) for label in labels) + 2
    total = cm.sum()
    lines = [f"{'':<{width}}precision  recall      f1  support"]
    lines += [f"{label:<{width}}{s['precision'][i]:9.4f}{s['recall'][i]:8.4f}{s['f1'][i]:8.4f}{s['support'][i]:9d}"
              for i, label in enumerate(labels)]
    lines += [f"{'accuracy':<{width}}{np.trace(cm) / total if total else float('nan'):25.4f}{total:9d}",
              f"{'macro f1':<{width}}{macro_f1(cm):25.4f}", "",
              "confusion matrix (rows: true class, columns: predicted)",
              " " * width + "".join(f"{i:>9}" for i in range(len(cm)))]
    lines += [f"{label:<{width}}" + "".join(f"{v:9d}" for v in cm[i]) for i, label in enumerate(labels)]
    return "\n".join(lines)

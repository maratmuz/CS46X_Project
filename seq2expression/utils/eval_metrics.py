import numpy as np
from sklearn.metrics import r2_score, roc_auc_score, precision_score, recall_score, f1_score, accuracy_score, confusion_matrix as sk_confusion_matrix


def compute_brackets(y_true: np.ndarray, n_brackets: int) -> np.ndarray:
    percentiles = np.linspace(0, 100, n_brackets + 1)
    thresholds  = np.percentile(y_true, percentiles, axis=0)
    brackets    = np.zeros_like(y_true, dtype=int)
    for b in range(1, n_brackets + 1):
        brackets[y_true >= thresholds[b]] = b
    return np.clip(brackets, 0, n_brackets - 1)


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return r2_score(y_true, y_pred, multioutput="uniform_average")


def auroc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    scores = []
    for t in range(y_true.shape[1]):
        y_bin = (y_true[:, t] >= np.median(y_true[:, t])).astype(int)
        if len(np.unique(y_bin)) < 2:
            continue
        scores.append(roc_auc_score(y_bin, y_pred[:, t]))
    return float(np.mean(scores))


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, n_brackets: int) -> dict:
    true_brackets = compute_brackets(y_true, n_brackets)
    pred_brackets = compute_brackets(y_pred, n_brackets)

    t = true_brackets.flatten()
    p = pred_brackets.flatten()

    kwargs = dict(average="macro", zero_division=0)
    cm     = sk_confusion_matrix(t, p)
    tp     = np.diag(cm).tolist()
    fp     = (cm.sum(axis=0) - np.diag(cm)).tolist()
    fn     = (cm.sum(axis=1) - np.diag(cm)).tolist()

    return {
        "accuracy":         accuracy_score(t, p),
        "precision":        precision_score(t, p, **kwargs),
        "recall":           recall_score(t, p, **kwargs),
        "f1":               f1_score(t, p, **kwargs),
        "confusion_matrix": cm.tolist(),
        "tp":               tp,
        "fp":               fp,
        "fn":               fn,
    }
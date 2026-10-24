#Метрики, матрицы ошибок и оценка качества

import numpy as np
from sklearn.metrics import (
    f1_score,
    confusion_matrix,
    roc_auc_score,
    log_loss,
    brier_score_loss,
    average_precision_score,
)
from scipy.stats import beta


# Метрики по вероятностям (не зависят от порога)
def probabilistic_metrics(y_true, y_prob):
    return {
        "log_loss": float(log_loss(y_true, y_prob, labels=[0, 1])),
        "brier": float(brier_score_loss(y_true, y_prob)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
    }


# Оценка при фиксированном пороге: все метрики одним словарём
def evaluate_at_threshold(y_true, y_prob, threshold):
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    return {
        "threshold": float(threshold),
        "accuracy": float((tp + tn) / len(y_true)),
        "sensitivity": float(tp / (tp + fn)) if (tp + fn) else float("nan"),
        "specificity": float(tn / (tn + fp)) if (tn + fp) else float("nan"),
        "precision": float(tp / (tp + fp)) if (tp + fp) else float("nan"),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "auprc": float(average_precision_score(y_true, y_prob)),
        "log_loss": float(log_loss(y_true, y_prob, labels=[0, 1])),
        "brier": float(brier_score_loss(y_true, y_prob)),
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
    }


# Точный доверительный интервал Клоппера–Пирсона для доли k / n
def clopper_pearson(successes, total, alpha=0.05):
    if total == 0:
        return float("nan"), float("nan")

    lower = 0.0 if successes == 0 else float(
        beta.ppf(alpha / 2, successes, total - successes + 1)
    )
    upper = 1.0 if successes == total else float(
        beta.ppf(1 - alpha / 2, successes + 1, total - successes)
    )

    return lower, upper


# Доверительные интервалы для всех метрик оценки на тесте:
# доли (accuracy, чувствительность, специфичность, precision) — Клоппер–Пирсон,
# остальные метрики — стратифицированный bootstrap (перцентильный интервал)
def confidence_intervals(y_true, y_prob, threshold, n_boot=2000,
                         alpha=0.05, random_state=42):
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)

    point = evaluate_at_threshold(y_true, y_prob, threshold)

    intervals = {
        "accuracy": clopper_pearson(point["TP"] + point["TN"], len(y_true), alpha),
        "sensitivity": clopper_pearson(point["TP"], point["TP"] + point["FN"], alpha),
        "specificity": clopper_pearson(point["TN"], point["TN"] + point["FP"], alpha),
        "precision": clopper_pearson(point["TP"], point["TP"] + point["FP"], alpha),
    }

    # Стратифицированный bootstrap: число объектов каждого класса фиксировано
    rng = np.random.default_rng(random_state)
    pos_idx = np.flatnonzero(y_true == 1)
    neg_idx = np.flatnonzero(y_true == 0)
    boot_metrics = ("f1", "roc_auc", "auprc", "log_loss", "brier")
    samples = {name: [] for name in boot_metrics}

    for _ in range(n_boot):
        idx = np.concatenate([
            rng.choice(pos_idx, size=len(pos_idx), replace=True),
            rng.choice(neg_idx, size=len(neg_idx), replace=True),
        ])
        m = evaluate_at_threshold(y_true[idx], y_prob[idx], threshold)
        for name in boot_metrics:
            samples[name].append(m[name])

    for name in boot_metrics:
        low, high = np.percentile(samples[name], [100 * alpha / 2, 100 * (1 - alpha / 2)])
        intervals[name] = (float(low), float(high))

    return point, intervals

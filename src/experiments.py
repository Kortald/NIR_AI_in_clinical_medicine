
# Подбор силы L2-регуляризации и порога классификации
import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler
from src.models import fit, predict_proba
from src.evaluation import probabilistic_metrics


# Сетка l2 из значений C (l2 = 1 / (C * n), как в sklearn)
def make_l2_grid(n_samples, c_values=None):
    if c_values is None:
        c_values = np.logspace(-2, 0.5, 11)

    return [1.0 / (c * n_samples) for c in c_values]


# Повторная стратифицированная CV по сетке l2
def cross_validate_l2(
    X,
    y,
    l2_values,
    n_splits=5,
    n_repeats=10,
    learning_rate=0.5,
    epochs=30000,
    tol=1e-6,
    random_state=42,
):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=int)

    cv = RepeatedStratifiedKFold(
        n_splits=n_splits,
        n_repeats=n_repeats,
        random_state=random_state,
    )

    rows = []

    # Out-of-fold вероятности: по одному вектору на каждый повтор
    oof = {l2: np.zeros((n_repeats, len(y))) for l2 in l2_values}

    with tqdm(
        total=n_splits * n_repeats,
        desc="Повторная CV",
        unit="фолд",
        dynamic_ncols=True,
    ) as progress:

        for fold_id, (train_idx, val_idx) in enumerate(cv.split(X, y)):
            repeat = fold_id // n_splits

            # Стандартизатор обучается только на тренировочной части фолда
            scaler = StandardScaler().fit(X[train_idx])
            X_train = scaler.transform(X[train_idx])
            X_val = scaler.transform(X[val_idx])

            for l2 in l2_values:
                weights, bias, history = fit(
                    X_train,
                    y[train_idx],
                    learning_rate=learning_rate,
                    epochs=epochs,
                    l2=l2,
                    tol=tol,
                    return_history=True,
                )

                prob = predict_proba(X_val, weights, bias)

                rows.append({
                    "l2": l2,
                    "repeat": repeat,
                    "fold": fold_id,
                    "epochs_used": len(history["loss"]),
                    "converged": history["grad_norm"][-1] < tol,
                    **probabilistic_metrics(y[val_idx], prob),
                })

                oof[l2][repeat, val_idx] = prob

            progress.update(1)

    return pd.DataFrame(rows), oof


# Среднее и σ по фолдам для каждого l2
def summarize_by_l2(fold_table):
    grouped = fold_table.groupby("l2")

    out = grouped[["log_loss", "brier", "roc_auc"]].agg(["mean", "std"])
    out.columns = [f"{metric}_{stat}" for metric, stat in out.columns]

    out["converged_share"] = grouped["converged"].mean()
    out["epochs_mean"] = grouped["epochs_used"].mean()

    return out.reset_index()


# Правило одной стандартной ошибки: самая сильная регуляризация,
# чей средний log-loss не хуже лучшего более чем на σ / sqrt(n_splits)
def select_l2_one_se(summary, n_splits=5, metric="log_loss"):
    best = summary.loc[summary[f"{metric}_mean"].idxmin()]

    limit = best[f"{metric}_mean"] + best[f"{metric}_std"] / np.sqrt(n_splits)

    candidates = summary[summary[f"{metric}_mean"] <= limit]

    return float(candidates["l2"].max())


# Порог: максимум специфичности при чувствительности >= min_sensitivity
def choose_threshold(y_true, y_prob, min_sensitivity=0.95, step=0.01):
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)

    positives = y_true == 1
    negatives = y_true == 0

    best = None

    for threshold in np.round(np.arange(0.0, 1.0 + step / 2, step), 10):
        predicted = y_prob >= threshold

        sensitivity = np.mean(predicted[positives])
        specificity = np.mean(~predicted[negatives])

        if sensitivity < min_sensitivity:
            continue

        # При равной специфичности берём большую чувствительность
        key = (specificity, sensitivity)

        if best is None or key > best[0]:
            best = (key, float(threshold), float(sensitivity), float(specificity))

    return {
        "threshold": best[1],
        "sensitivity": best[2],
        "specificity": best[3],
    }

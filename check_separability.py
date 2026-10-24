# Диагностика линейной разделимости development-выборки (для раздела 2.4 текста).
# Используется ТОЛЬКО development; тестовая выборка не загружается в расчёты.
# Результат печатается и пишется в results/separability_check.txt. Конвейер (main.py) не затрагивается.

import numpy as np
from scipy.optimize import linprog
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.preprocessing import StandardScaler

from src.data_loader import load_data
from src.preprocessing import split_development_test


def is_separable(Z: np.ndarray, y: np.ndarray) -> bool:
    """LP-проверка: существуют ли (w, b), для которых s_i (<w, z_i> + b) >= 1 при всех i."""
    s = 2 * y - 1
    n, p = Z.shape
    A = -(s[:, None] * np.hstack([Z, np.ones((n, 1))]))
    res = linprog(
        c=np.zeros(p + 1), A_ub=A, b_ub=-np.ones(n),
        bounds=[(None, None)] * (p + 1), method="highs",
    )
    return res.status == 0


def main() -> None:
    features, target = load_data()
    X_dev, _, y_dev, _ = split_development_test(features, target)  # тест не используется
    y = y_dev.to_numpy()
    Z = StandardScaler().fit_transform(X_dev.to_numpy())

    lines = [
        f"development: n={len(y)}, злокачественных={int(y.sum())}, признаков={Z.shape[1]}",
        f"линейно разделима (все 30 признаков): {is_separable(Z, y)}",
    ]
    for k in (2, 3, 5, 10):
        lines.append(f"линейно разделима (первые {k} признаков): {is_separable(Z[:, :k], y)}")

    lines.append("")
    lines.append("sklearn LogisticRegression на development (стандартизованные признаки):")
    lines.append("C            max|w|       log-loss (обучение)")
    for C in (1e2, 1e4, 1e6, np.inf):
        model = LogisticRegression(C=C, max_iter=100000, tol=1e-12).fit(Z, y)
        loss = log_loss(y, model.predict_proba(Z)[:, 1])
        lines.append(f"{C:<12.0e} {np.abs(model.coef_).max():<12.4g} {loss:.3e}")

    text = "\n".join(lines)
    print(text)
    with open("results/separability_check.txt", "w", encoding="utf-8") as f:
        f.write(text + "\n")


if __name__ == "__main__":
    main()

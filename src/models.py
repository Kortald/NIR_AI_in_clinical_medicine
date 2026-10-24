
# Собственная логистическая регрессия и её вычисления

import numpy as np


# Преобразование линейной оценки в вероятность (численно устойчивая версия)
def sigmoid(scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=float)
    out = np.empty_like(scores)

    # Для z >= 0 обычная формула не переполняется
    pos = scores >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-scores[pos]))

    # Для z < 0 используем e^z / (1 + e^z): экспонента от отрицательного числа безопасна
    exp_s = np.exp(scores[~pos])
    out[~pos] = exp_s / (1.0 + exp_s)

    return out


# Расчёт линейной оценки по признакам, весам и смещению
def linear_score(
    X: np.ndarray,
    weights: np.ndarray,
    bias: float,
) -> np.ndarray:
    return bias + X @ weights


# Расчёт вероятностей принадлежности к классу 1
def predict_proba(
    X: np.ndarray,
    weights: np.ndarray,
    bias: float,
) -> np.ndarray:
    scores = linear_score(X, weights, bias)
    return sigmoid(scores)


# Расчёт взвешенной бинарной кросс-энтропии
def binary_cross_entropy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    pos_weight: float = 1.0,
) -> float:
    y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)

    losses = -(
        pos_weight * y_true * np.log(y_pred)
        + (1 - y_true) * np.log(1 - y_pred)
    )

    return float(np.mean(losses))


# L2-штраф: (l2 / 2) * ||w||^2. Смещение не штрафуется
def l2_penalty(weights: np.ndarray, l2: float = 0.0) -> float:
    return float(0.5 * l2 * np.sum(weights ** 2))


# Расчёт градиента по смещению
def gradient_bias(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    pos_weight: float = 1.0,
) -> float:
    errors = np.where(
        y_true == 1,
        pos_weight * (y_pred - y_true),
        y_pred - y_true,
    )

    return float(np.mean(errors))


# Расчёт градиентов по весам
def gradient_weights(
    X: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    pos_weight: float = 1.0,
    weights: np.ndarray | None = None,
    l2: float = 0.0,
) -> np.ndarray:
    errors = np.where(
        y_true == 1,
        pos_weight * (y_pred - y_true),
        y_pred - y_true,
    )

    grad = X.T @ errors / X.shape[0]

    # Градиент штрафа (l2 / 2) * ||w||^2 по весам равен l2 * w
    if l2 > 0.0 and weights is not None:
        grad = grad + l2 * weights

    return grad


# Евклидова норма полного градиента (веса и смещение вместе)
def gradient_norm(
    grad_weights: np.ndarray,
    grad_bias: float,
) -> float:
    return float(np.sqrt(np.sum(grad_weights ** 2) + grad_bias ** 2))


# Обучение модели градиентным спуском
def fit(
    X: np.ndarray,
    y: np.ndarray,
    learning_rate: float = 0.01,
    epochs: int = 1000,
    pos_weight: float = 1.0,
    l2: float = 0.0,
    tol: float | None = None,
    return_history: bool = False,
):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)

    weights = np.zeros(X.shape[1])
    bias = 0.0

    # История обучения: потеря и норма градиента на каждой эпохе
    history = {"loss": [], "grad_norm": []}

    for _ in range(epochs):
        # Получаем предсказания
        y_pred = predict_proba(X, weights, bias)

        # Вычисляем градиенты
        grad_weights = gradient_weights(
            X, y, y_pred, pos_weight, weights=weights, l2=l2
        )
        grad_bias = gradient_bias(y, y_pred, pos_weight)

        history["loss"].append(
            binary_cross_entropy(y, y_pred, pos_weight)
            + l2_penalty(weights, l2)
        )
        history["grad_norm"].append(
            gradient_norm(grad_weights, grad_bias)
        )

        # Остановка по норме градиента (если задан порог tol)
        if tol is not None and history["grad_norm"][-1] < tol:
            break

        # Обновляем веса и смещение
        weights -= learning_rate * grad_weights
        bias -= learning_rate * grad_bias

    if return_history:
        return weights, bias, history

    return weights, bias

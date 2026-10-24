import numpy as np
import pytest
import warnings
import numpy as np
from sklearn.linear_model import LogisticRegression
from src.models import (
    sigmoid,
    linear_score,
    predict_proba,
    binary_cross_entropy,
    gradient_bias,
    gradient_weights,
    gradient_norm,
    l2_penalty,
    fit,
)


def test_sig():
    values = np.array([-2.0, 0.0, 2.0])

    result = sigmoid(values)

    assert result.shape == values.shape
    assert np.all(result >= 0)
    assert np.all(result <= 1)
    assert result[0] < result[1] < result[2]

def test_linear():
    X = np.array([
        [1, 2],
        [3, 4],
        [5, 6]
    ])
    weights = np.array([2, 3])
    bias = 1

    result = linear_score(X, weights, bias)

    expected = np.array([9, 19, 29])

    assert np.allclose(result, expected)

def test_proba():
    X = np.array([
        [-100.0],
        [0.0],
        [100.0]
    ])
    weights = np.array([1.0])
    bias = 0.0

    result = predict_proba(X, weights, bias)

    assert np.all(result >= 0)
    assert np.all(result <= 1)

def test_bce():
    y_true = np.array([1, 0])
    y_pred = np.array([0.8, 0.3])
    pos_weight = 3.0

    result = binary_cross_entropy(y_true, y_pred, pos_weight)

    expected = 0.5130527989406808

    assert result == pytest.approx(expected, abs=1e-10)

def test_gradient_bias():
    y_true = np.array([1, 0])
    y_pred = np.array([0.8, 0.3])
    pos_weight = 3.0

    result = gradient_bias(y_true, y_pred, pos_weight)

    assert result == pytest.approx(-0.15, abs= 1e-5)

def test_gradient_weights():

    X = np.array([
        [1, 2],
        [3, 4]
    ])
    y_true = np.array([1, 0])
    y_pred = np.array([0.8, 0.3])
    pos_weight = 3.0

    expected = np.array([0.15, 0.0])

    result = gradient_weights(X, y_true, y_pred, pos_weight)

    assert np.allclose(result, expected)

def test_fit():
    X = np.array([
        [-2.0],
        [-1.0],
        [1.0],
        [2.0]
    ])
    y = np.array([0, 0, 1, 1])

    weights_normal, bias_normal = fit(
        X, y,
        learning_rate=0.1,
        epochs=200,
        pos_weight=1.0
    )

    weights_weighted, bias_weighted = fit(
        X, y,
        learning_rate=0.1,
        epochs=200,
        pos_weight=3.0
    )

    prob_normal = predict_proba(
        X, weights_normal, bias_normal
    )
    prob_weighted = predict_proba(
        X, weights_weighted, bias_weighted
    )

    assert not np.allclose(prob_normal, prob_weighted)
    assert not np.allclose(weights_normal, weights_weighted)
    assert not np.isclose(bias_normal, bias_weighted)

def test_sigmoid_known_values_and_symmetry():
    values = np.array([-3.0, 0.0, 3.0])

    result = sigmoid(values)

    assert result[1] == pytest.approx(0.5, abs=1e-12)
    # sigmoid(-z) = 1 - sigmoid(z)
    assert np.allclose(result[0], 1 - result[2])
    assert result[2] == pytest.approx(0.9525741268224334, abs=1e-12)


def test_sigmoid_extreme_values_no_warnings():
    values = np.array([-1000.0, 1000.0])

    # Любое предупреждение (например, overflow) считаем ошибкой
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = sigmoid(values)

    assert np.all(np.isfinite(result))
    assert result[0] == pytest.approx(0.0, abs=1e-12)
    assert result[1] == pytest.approx(1.0, abs=1e-12)


def test_l2_penalty_value():
    weights = np.array([3.0, 4.0])

    # 0.5 * 0.1 * (9 + 16) = 1.25
    assert l2_penalty(weights, 0.1) == pytest.approx(1.25, abs=1e-12)
    assert l2_penalty(weights, 0.0) == 0.0


def test_gradient_norm_value():
    # sqrt(3^2 + 4^2 + 12^2) = 13
    result = gradient_norm(np.array([3.0, 4.0]), 12.0)

    assert result == pytest.approx(13.0, abs=1e-12)


def test_gradient_weights_without_l2_unchanged():
    X = np.array([[1.0, 2.0], [3.0, 4.0]])
    y_true = np.array([1, 0])
    y_pred = np.array([0.8, 0.3])
    weights = np.array([0.5, -0.5])

    without = gradient_weights(X, y_true, y_pred, 3.0)
    with_zero_l2 = gradient_weights(
        X, y_true, y_pred, 3.0, weights=weights, l2=0.0
    )

    assert np.allclose(without, with_zero_l2)


def test_gradient_matches_finite_differences_with_l2():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(30, 3))
    y = (rng.random(30) > 0.5).astype(float)
    weights = rng.normal(size=3) * 0.3
    bias = 0.1
    l2 = 0.05

    def total_loss(w):
        return binary_cross_entropy(
            y, predict_proba(X, w, bias)
        ) + l2_penalty(w, l2)

    analytic = gradient_weights(
        X, y, predict_proba(X, weights, bias),
        weights=weights, l2=l2,
    )

    # Центральные конечные разности по каждой координате
    eps = 1e-6
    numeric = np.array([
        (total_loss(weights + eps * e) - total_loss(weights - eps * e))
        / (2 * eps)
        for e in np.eye(3)
    ])

    assert np.allclose(analytic, numeric, atol=1e-7)


def test_fit_returns_pair_by_default_and_triple_with_history():
    X = np.array([[-2.0], [-1.0], [1.0], [2.0]])
    y = np.array([0, 0, 1, 1])

    default_result = fit(X, y, epochs=5)
    history_result = fit(X, y, epochs=5, return_history=True)

    assert len(default_result) == 2
    assert len(history_result) == 3
    assert set(history_result[2]) == {"loss", "grad_norm"}


def test_fit_history_starts_at_log2_and_loss_decreases():
    X = np.array([[-2.0], [-1.0], [1.0], [2.0]])
    y = np.array([0, 0, 1, 1])

    _, _, history = fit(
        X, y, learning_rate=0.1, epochs=100, return_history=True
    )
    loss = np.array(history["loss"])

    # При нулевых весах p = 0.5 для всех объектов, потеря = ln 2
    assert loss[0] == pytest.approx(np.log(2), abs=1e-12)
    assert len(loss) == 100
    # Для выпуклой потери и малого шага потеря не растёт
    assert np.all(np.diff(loss) <= 1e-12)


def test_fit_stops_early_by_tolerance():
    X = np.array([[-2.0], [-1.0], [1.0], [2.0]])
    y = np.array([0, 0, 1, 1])

    _, _, history = fit(
        X, y,
        learning_rate=0.1,
        epochs=10_000,
        l2=0.1,
        tol=1e-6,
        return_history=True,
    )

    assert len(history["loss"]) < 10_000
    assert history["grad_norm"][-1] < 1e-6


def test_fit_separates_simple_classes():
    X = np.array([[-2.0], [-1.0], [1.0], [2.0]])
    y = np.array([0, 0, 1, 1])

    weights, bias = fit(X, y, learning_rate=0.5, epochs=500)
    proba = predict_proba(X, weights, bias)

    assert weights[0] > 0
    assert np.array_equal((proba >= 0.5).astype(int), y)


def test_fit_l2_shrinks_weights():
    # Разделимые данные: без штрафа веса растут, со штрафом ограничены
    X = np.array([[-2.0, 0.5], [-1.0, -0.5], [1.0, 0.5], [2.0, -0.5]])
    y = np.array([0, 0, 1, 1])

    w_free, _ = fit(X, y, learning_rate=0.5, epochs=2000, l2=0.0)
    w_reg, _ = fit(X, y, learning_rate=0.5, epochs=2000, l2=0.1)

    assert np.linalg.norm(w_reg) < np.linalg.norm(w_free)


def test_fit_matches_sklearn_with_l2():
    # Неразделимые данные, поэтому оптимум существует
    rng = np.random.default_rng(1)
    n = 200
    X = rng.normal(size=(n, 2))
    logits = 1.5 * X[:, 0] - 1.0 * X[:, 1] + 0.3
    y = (rng.random(n) < 1 / (1 + np.exp(-logits))).astype(int)

    C = 1.0
    l2 = 1 / (C * n)  # соответствие нашего l2 параметру C в sklearn

    ref = LogisticRegression(C=C, tol=1e-12, max_iter=10_000).fit(X, y)

    weights, bias = fit(
        X, y,
        learning_rate=0.5,
        epochs=20_000,
        l2=l2,
        tol=1e-10,
    )

    assert np.allclose(weights, ref.coef_[0], atol=1e-4)
    assert bias == pytest.approx(ref.intercept_[0], abs=1e-4)

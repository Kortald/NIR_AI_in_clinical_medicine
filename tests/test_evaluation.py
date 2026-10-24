import numpy as np
import pytest

from src.evaluation import (
    evaluate_at_threshold,
    clopper_pearson,
    confidence_intervals,
)


# ---------- Вспомогательные данные ----------

def make_scores(n, seed=0):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.4).astype(int)
    prob = np.clip(0.35 * y + rng.normal(0.35, 0.2, n), 0.01, 0.99)
    return y, prob


# ---------- evaluate_at_threshold ----------

def test_evaluate_at_threshold_hand_values():
    y_true = np.array([1, 1, 0, 0])
    y_prob = np.array([0.9, 0.4, 0.6, 0.1])

    result = evaluate_at_threshold(y_true, y_prob, 0.5)

    # Предсказания [1, 0, 1, 0]: по одному объекту в каждой клетке матрицы
    assert (result["TN"], result["FP"], result["FN"], result["TP"]) == (1, 1, 1, 1)
    for name in ("accuracy", "sensitivity", "specificity", "precision", "f1"):
        assert result[name] == pytest.approx(0.5, abs=1e-12)

    # ROC-AUC: три из четырёх пар «злокачественный выше доброкачественного»
    assert result["roc_auc"] == pytest.approx(0.75, abs=1e-12)
    # AUPRC (average precision): (1 + 2/3) / 2
    assert result["auprc"] == pytest.approx(5 / 6, abs=1e-12)
    assert result["log_loss"] == pytest.approx(
        -(2 * np.log(0.9) + 2 * np.log(0.4)) / 4, abs=1e-12
    )
    assert result["brier"] == pytest.approx(0.185, abs=1e-12)


def test_evaluate_at_threshold_zero_threshold_predicts_everything_positive():
    y_true = np.array([1, 1, 0, 0])
    y_prob = np.array([0.9, 0.4, 0.6, 0.1])

    result = evaluate_at_threshold(y_true, y_prob, 0.0)

    assert result["sensitivity"] == pytest.approx(1.0, abs=1e-12)
    assert result["specificity"] == pytest.approx(0.0, abs=1e-12)


def test_evaluate_at_threshold_no_positive_predictions():
    y_true = np.array([1, 1, 0, 0])
    y_prob = np.array([0.9, 0.4, 0.6, 0.1])

    result = evaluate_at_threshold(y_true, y_prob, 0.95)

    assert result["sensitivity"] == pytest.approx(0.0, abs=1e-12)
    assert result["specificity"] == pytest.approx(1.0, abs=1e-12)
    # Precision не определена, когда положительных предсказаний нет
    assert np.isnan(result["precision"])


# ---------- clopper_pearson ----------

def test_clopper_pearson_boundaries_have_closed_form():
    alpha = 0.05

    low_0, high_0 = clopper_pearson(0, 10, alpha)
    low_n, high_n = clopper_pearson(10, 10, alpha)

    # При k = 0 верхняя граница равна 1 - (alpha / 2) ** (1 / n)
    assert low_0 == 0.0
    assert high_0 == pytest.approx(1 - (alpha / 2) ** (1 / 10), abs=1e-9)
    # При k = n нижняя граница равна (alpha / 2) ** (1 / n)
    assert high_n == 1.0
    assert low_n == pytest.approx((alpha / 2) ** (1 / 10), abs=1e-9)


def test_clopper_pearson_reference_value():
    # Табличное значение для 5 из 10: (0.1871, 0.8129)
    low, high = clopper_pearson(5, 10, 0.05)

    assert low == pytest.approx(0.1871, abs=1e-4)
    assert high == pytest.approx(0.8129, abs=1e-4)


def test_clopper_pearson_empty_denominator_gives_nan():
    low, high = clopper_pearson(0, 0)

    assert np.isnan(low) and np.isnan(high)


# ---------- confidence_intervals ----------

def test_confidence_intervals_contain_point_estimates():
    y, prob = make_scores(300)

    point, intervals = confidence_intervals(y, prob, 0.5, n_boot=200)

    expected_keys = {
        "accuracy", "sensitivity", "specificity", "precision",
        "f1", "roc_auc", "auprc", "log_loss", "brier",
    }
    assert set(intervals) == expected_keys

    for name, (low, high) in intervals.items():
        assert low <= point[name] <= high, name


def test_confidence_intervals_are_reproducible():
    y, prob = make_scores(150)

    _, first = confidence_intervals(y, prob, 0.5, n_boot=100, random_state=1)
    _, second = confidence_intervals(y, prob, 0.5, n_boot=100, random_state=1)

    assert first == second


def test_confidence_intervals_narrow_with_more_data():
    y_small, prob_small = make_scores(100, seed=2)
    y_large, prob_large = make_scores(1000, seed=2)

    _, small = confidence_intervals(y_small, prob_small, 0.5, n_boot=200)
    _, large = confidence_intervals(y_large, prob_large, 0.5, n_boot=200)

    for name in ("sensitivity", "specificity", "roc_auc"):
        width_small = small[name][1] - small[name][0]
        width_large = large[name][1] - large[name][0]
        assert width_large < width_small, name

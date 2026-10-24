from unittest import mock

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler

from src.evaluation import probabilistic_metrics
from src.experiments import (
    make_l2_grid,
    cross_validate_l2,
    summarize_by_l2,
    select_l2_one_se,
    choose_threshold,
)


# ---------- Вспомогательные данные ----------

def make_small_dataset(n=120, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 3))
    logits = 1.2 * X[:, 0] - 0.8 * X[:, 1]
    y = (rng.random(n) < 1 / (1 + np.exp(-logits))).astype(int)
    return X, y


# ---------- probabilistic_metrics ----------

def test_probabilistic_metrics_hand_values():
    y_true = np.array([1, 0])
    y_prob = np.array([0.8, 0.3])

    result = probabilistic_metrics(y_true, y_prob)

    expected_log_loss = -(np.log(0.8) + np.log(0.7)) / 2
    expected_brier = (0.2 ** 2 + 0.3 ** 2) / 2

    assert result["log_loss"] == pytest.approx(expected_log_loss, abs=1e-12)
    assert result["brier"] == pytest.approx(expected_brier, abs=1e-12)
    assert result["roc_auc"] == pytest.approx(1.0, abs=1e-12)


def test_log_loss_sees_confidence_but_auc_does_not():
    y_true = np.array([1, 0])

    mild = probabilistic_metrics(y_true, np.array([0.6, 0.4]))
    confident = probabilistic_metrics(y_true, np.array([0.99, 0.01]))

    # Порядок объектов одинаковый, поэтому ROC-AUC равен
    assert mild["roc_auc"] == pytest.approx(confident["roc_auc"], abs=1e-12)
    # А вот log-loss и Brier у уверенной правильной модели меньше
    assert confident["log_loss"] < mild["log_loss"]
    assert confident["brier"] < mild["brier"]


# ---------- make_l2_grid ----------

def test_make_l2_grid_formula():
    # l2 = 1 / (C * n)
    result = make_l2_grid(100, c_values=[1.0, 0.1])

    assert result == pytest.approx([0.01, 0.1], abs=1e-12)


def test_make_l2_grid_default_values():
    n = 455

    grid = make_l2_grid(n)

    assert len(grid) == 11
    # C растёт, поэтому l2 строго убывает
    assert np.all(np.diff(grid) < 0)
    # В сетке есть C = 1, то есть l2 = 1 / n
    assert np.any(np.isclose(grid, 1 / n))


# ---------- summarize_by_l2 ----------

def test_summarize_by_l2_mean_std_and_convergence():
    fold_table = pd.DataFrame({
        "l2": [0.1, 0.1, 0.01, 0.01],
        "log_loss": [0.2, 0.4, 0.1, 0.1],
        "brier": [0.05, 0.07, 0.02, 0.02],
        "roc_auc": [0.99, 0.97, 0.995, 0.995],
        "converged": [True, False, True, True],
        "epochs_used": [100, 300, 1000, 2000],
    })

    summary = summarize_by_l2(fold_table).set_index("l2")

    assert summary.loc[0.1, "log_loss_mean"] == pytest.approx(0.3, abs=1e-12)
    # Выборочное σ (ddof=1) для значений 0.2 и 0.4
    assert summary.loc[0.1, "log_loss_std"] == pytest.approx(
        np.sqrt(0.02), abs=1e-12
    )
    assert summary.loc[0.1, "converged_share"] == pytest.approx(0.5, abs=1e-12)
    assert summary.loc[0.01, "epochs_mean"] == pytest.approx(1500, abs=1e-9)


# ---------- select_l2_one_se ----------

def test_select_l2_one_se_picks_strongest_within_tolerance():
    summary = pd.DataFrame({
        "l2": [0.001, 0.01, 0.1, 1.0],
        "log_loss_mean": [0.10, 0.09, 0.095, 0.20],
        "log_loss_std": [0.05, 0.05, 0.05, 0.05],
    })

    # Лучший: l2=0.01 (0.09). Допуск: 0.05 / sqrt(5) ≈ 0.0224,
    # то есть предел ≈ 0.1124. В допуске l2 = 0.001, 0.01, 0.1.
    # Берём самый сильный штраф среди них.
    assert select_l2_one_se(summary, n_splits=5) == pytest.approx(0.1)


def test_select_l2_one_se_returns_best_when_others_are_clearly_worse():
    summary = pd.DataFrame({
        "l2": [0.001, 0.01, 0.1],
        "log_loss_mean": [0.30, 0.10, 0.40],
        "log_loss_std": [0.01, 0.01, 0.01],
    })

    assert select_l2_one_se(summary, n_splits=5) == pytest.approx(0.01)


# ---------- choose_threshold ----------

def test_choose_threshold_known_answer():
    y_true = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    y_prob = np.array([0.9, 0.8, 0.7, 0.4, 0.5, 0.3, 0.2, 0.1])

    result = choose_threshold(y_true, y_prob, min_sensitivity=0.75)

    # Чувствительность 3/4 достигается при пороге <= 0.7. Ложных
    # срабатываний нет при пороге > 0.5. Из равных вариантов
    # берётся наименьший порог (0.51).
    assert result["threshold"] == pytest.approx(0.51, abs=1e-9)
    assert result["sensitivity"] == pytest.approx(0.75, abs=1e-12)
    assert result["specificity"] == pytest.approx(1.0, abs=1e-12)


def test_choose_threshold_full_sensitivity_costs_specificity():
    y_true = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    y_prob = np.array([0.9, 0.8, 0.7, 0.4, 0.5, 0.3, 0.2, 0.1])

    result = choose_threshold(y_true, y_prob, min_sensitivity=1.0)

    # Чтобы поймать объект с вероятностью 0.4, порог не выше 0.4,
    # при этом ложным срабатыванием становится объект с 0.5.
    assert result["sensitivity"] == pytest.approx(1.0, abs=1e-12)
    assert result["specificity"] == pytest.approx(0.75, abs=1e-12)
    assert result["threshold"] == pytest.approx(0.31, abs=1e-9)


def test_choose_threshold_always_meets_sensitivity_requirement():
    rng = np.random.default_rng(3)
    y_true = (rng.random(200) < 0.4).astype(int)
    y_prob = np.clip(0.3 * y_true + rng.normal(0.4, 0.2, 200), 0, 1)

    result = choose_threshold(y_true, y_prob, min_sensitivity=0.95)

    assert result["sensitivity"] >= 0.95


# ---------- cross_validate_l2 ----------

def test_cross_validate_l2_table_and_oof_shapes():
    X, y = make_small_dataset()
    l2_values = [0.1, 0.01]

    table, oof = cross_validate_l2(
        X, y, l2_values,
        n_splits=3, n_repeats=2, epochs=300, tol=1e-6,
    )

    # 3 фолда x 2 повтора x 2 значения l2
    assert len(table) == 12
    assert set(table["l2"]) == set(l2_values)
    for column in ("log_loss", "brier", "roc_auc", "converged", "epochs_used"):
        assert column in table.columns

    for l2 in l2_values:
        assert oof[l2].shape == (2, len(y))
        # Каждый объект получил out-of-fold вероятность в каждом повторе
        assert np.all(oof[l2] > 0)
        assert np.all(oof[l2] < 1)


def test_cross_validate_l2_scaler_sees_only_training_rows():
    X, y = make_small_dataset(n=120)
    fit_sizes = []

    class SpyScaler(StandardScaler):
        def fit(self, X, y=None, **kwargs):
            fit_sizes.append(len(X))
            return super().fit(X, y, **kwargs)

    with mock.patch("src.experiments.StandardScaler", SpyScaler):
        cross_validate_l2(
            X, y, [0.1],
            n_splits=3, n_repeats=2, epochs=50,
        )

    # 6 фолдов, и в каждом стандартизатор обучается на 2/3 данных
    assert len(fit_sizes) == 6
    assert all(size == 80 for size in fit_sizes)


def test_cross_validate_l2_is_reproducible():
    X, y = make_small_dataset()

    table_1, _ = cross_validate_l2(
        X, y, [0.05], n_splits=3, n_repeats=1, epochs=100, random_state=7,
    )
    table_2, _ = cross_validate_l2(
        X, y, [0.05], n_splits=3, n_repeats=1, epochs=100, random_state=7,
    )

    assert np.allclose(table_1["log_loss"], table_2["log_loss"])

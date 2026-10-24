import json

import numpy as np
import pandas as pd
import pytest

import make_figures
from src import visualisation as viz


# ---------- Вспомогательные данные ----------

def make_scores(n=200, seed=0):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.4).astype(int)
    prob = np.clip(0.4 * y + rng.normal(0.3, 0.2, n), 0.01, 0.99)
    return y, prob


def make_summary():
    c = np.logspace(-2, 0.5, 11)
    loss = 0.08 + 0.02 * (np.log10(c) + 0.3) ** 2
    return pd.DataFrame({
        "C": c,
        "l2": 1 / (c * 455),
        "log_loss_mean": loss,
        "log_loss_std": 0.03,
        "roc_auc_mean": 0.99 - 0.001 * (np.log10(c) + 0.3) ** 2,
        "roc_auc_std": 0.005,
    })


def assert_files_created(paths):
    assert len(paths) == 2
    for path in paths:
        assert path.exists()
        assert path.stat().st_size > 1000


# ---------- Названия признаков ----------

def test_russian_feature_name_known_patterns():
    assert viz.russian_feature_name("worst texture") == "текстура (наихудшее)"
    assert viz.russian_feature_name("mean concave points") == \
        "число вогнутых точек (среднее)"
    assert viz.russian_feature_name("radius error") == "радиус (ошибка)"
    assert viz.russian_feature_name("fractal dimension error") == \
        "фрактальная размерность (ошибка)"


def test_russian_feature_name_unknown_name_is_kept():
    assert viz.russian_feature_name("(intercept)") == "(intercept)"


def test_all_wdbc_feature_names_are_translated():
    from src.data_loader import load_data

    features, _ = load_data()

    for name in features.columns:
        translated = viz.russian_feature_name(name)
        # Все английские слова должны быть заменены
        assert not any(ch.isascii() and ch.isalpha() for ch in translated), name


# ---------- Функции рисунков ----------

def test_plot_logloss_vs_c_creates_files(tmp_path):
    paths = viz.plot_logloss_vs_c(
        make_summary(), 0.18, 5, tmp_path / "fig1", note="тест"
    )

    assert_files_created(paths)


def test_plot_roc_pr_creates_files(tmp_path):
    y, prob = make_scores()

    paths = viz.plot_roc_pr(y, prob, 0.44, tmp_path / "fig2")

    assert_files_created(paths)


def test_plot_calibration_creates_files(tmp_path):
    y, prob = make_scores()

    paths = viz.plot_calibration(y, prob, tmp_path / "fig3", n_bins=8)

    assert_files_created(paths)


def test_plot_threshold_curves_creates_files(tmp_path):
    y, prob = make_scores()

    paths = viz.plot_threshold_curves(y, prob, 0.44, 0.95, tmp_path / "fig4")

    assert_files_created(paths)


def test_plot_coefficients_handles_mixed_signs(tmp_path):
    table = pd.DataFrame({
        "feature": ["(intercept)", "worst texture", "mean radius",
                    "area error", "worst area"],
        "coef_standardized": [-0.4, 0.7, -0.5, 0.2, -0.1],
        "coef_original_units": [-20.0, 0.1, -0.2, 0.01, -0.001],
    })

    paths = viz.plot_coefficients(table, tmp_path / "fig5", top_k=4)

    assert_files_created(paths)


def test_plot_convergence_creates_files(tmp_path):
    history = pd.DataFrame({
        "loss": np.linspace(0.7, 0.1, 100),
        "grad_norm": np.logspace(0, -6, 100),
    })

    paths = viz.plot_convergence(history, 1e-6, tmp_path / "fig6")

    assert_files_created(paths)


# ---------- Сборка из сохранённых результатов ----------

def write_minimal_results(folder, with_test):
    y, prob = make_scores(120)
    prefix = make_figures.PREFIX

    make_summary().to_csv(folder / f"{prefix}cv_summary.csv", index=False)

    oof = pd.DataFrame({"y_true": y}, index=np.arange(len(y)))
    for repeat in range(3):
        oof[f"y_prob_oof_r{repeat}"] = np.clip(prob + 0.01 * repeat, 0.01, 0.99)
    oof.to_csv(folder / f"{prefix}oof_predictions.csv")

    pd.DataFrame({
        "feature": ["(intercept)", "worst texture", "mean radius"],
        "coef_standardized": [-0.4, 0.7, -0.5],
        "coef_original_units": [-20.0, 0.1, -0.2],
    }).to_csv(folder / f"{prefix}coefficients.csv", index=False)

    pd.DataFrame({
        "loss": np.linspace(0.7, 0.1, 50),
        "grad_norm": np.logspace(0, -6, 50),
    }).to_csv(folder / f"{prefix}final_history.csv", index=False)

    selection = {
        "C": 0.18, "l2": 0.012, "threshold": 0.44, "min_sensitivity": 0.95,
        "n_splits": 5, "tol": 1e-6, "final_epochs": 50, "learning_rate": 0.5,
    }
    with open(folder / f"{prefix}selection.json", "w", encoding="utf-8") as f:
        json.dump(selection, f)

    if with_test:
        pd.DataFrame({"y_true": y[:60], "y_prob": prob[:60]}).to_csv(
            folder / f"{prefix}test_predictions.csv"
        )


def test_load_pooled_oof_tiles_labels_and_stacks_repeats():
    table = pd.DataFrame({
        "y_true": [0, 1, 1],
        "y_prob_oof_r0": [0.1, 0.8, 0.7],
        "y_prob_oof_r1": [0.2, 0.9, 0.6],
    })

    y_pool, p_pool, columns = make_figures.load_pooled_oof(table)

    assert columns == ["y_prob_oof_r0", "y_prob_oof_r1"]
    assert list(y_pool) == [0, 1, 1, 0, 1, 1]
    assert list(p_pool) == [0.1, 0.8, 0.7, 0.2, 0.9, 0.6]


def test_make_figures_without_test_file(tmp_path):
    write_minimal_results(tmp_path, with_test=False)

    make_figures.main(tmp_path)

    names = {p.name for p in (tmp_path / "figures").iterdir()}
    assert "fig1_logloss_vs_c.png" in names
    assert "fig4_threshold.pdf" in names
    assert "fig2_roc_pr_oof.png" in names
    # Рисунков по тесту нет, пока нет файла с предсказаниями на тесте
    assert not any("test" in name for name in names)


def test_make_figures_with_test_file(tmp_path):
    write_minimal_results(tmp_path, with_test=True)

    make_figures.main(tmp_path)

    names = {p.name for p in (tmp_path / "figures").iterdir()}
    assert "fig2_roc_pr_test.png" in names
    assert "fig3_calibration_test.pdf" in names


# Сборка всех рисунков из сохранённых результатов (обучение не выполняется).
#
# Запуск:
#   python make_figures.py                       рисунки из results/
#   python make_figures.py --dir results/quick   рисунки из быстрого прогона
#
# Рисунки по тесту строятся, только если есть файл test_predictions.csv.

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from compare_models import CMP_PREFIX
from main import PREFIX, RESULTS_DIR
from src import visualisation as viz


def load_pooled_oof(oof_table):
    # Все повторы CV вместе: ровно так же, как выбирался порог
    columns = [c for c in oof_table.columns if c.startswith("y_prob_oof_r")]
    y_true = np.tile(oof_table["y_true"].to_numpy(), len(columns))
    y_prob = np.concatenate([oof_table[c].to_numpy() for c in columns])
    return y_true, y_prob, columns


# Все повторы вместе для столбцов вида <prefix>_r0, <prefix>_r1, ...
def pool_prefix(table, prefix):
    columns = [c for c in table.columns if c.startswith(f"{prefix}_r")]
    y_true = np.tile(table["y_true"].to_numpy(), len(columns))
    y_prob = np.concatenate([table[c].to_numpy() for c in columns])
    return y_true, y_prob


def main(results_dir):
    results_dir = Path(results_dir)
    figures_dir = results_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    def path(name):
        return results_dir / f"{PREFIX}{name}"

    with open(path("selection.json"), encoding="utf-8") as file:
        selection = json.load(file)

    summary = pd.read_csv(path("cv_summary.csv"))
    oof_table = pd.read_csv(path("oof_predictions.csv"), index_col=0)
    coefficients = pd.read_csv(path("coefficients.csv"))
    history = pd.read_csv(path("final_history.csv"))

    y_pool, p_pool, oof_columns = load_pooled_oof(oof_table)
    n_dev = len(oof_table)
    repeats = len(oof_columns)
    oof_note = (
        f"Данные: out-of-fold предсказания на обучающей выборке "
        f"(n = {n_dev}, повторная CV {selection['n_splits']}×{repeats})"
    )

    made = []

    made += viz.plot_logloss_vs_c(
        summary, selection["C"], selection["n_splits"],
        figures_dir / "fig1_logloss_vs_c",
        note=f"Данные: повторная CV {selection['n_splits']}×{repeats} "
             f"на обучающей выборке (n = {n_dev})",
    )

    made += viz.plot_threshold_curves(
        y_pool, p_pool, selection["threshold"], selection["min_sensitivity"],
        figures_dir / "fig4_threshold", note=oof_note,
    )

    made += viz.plot_coefficients(
        coefficients, figures_dir / "fig5_coefficients",
        note="Модель обучена на всей обучающей выборке; признаки стандартизованы.\n"
             "Признаки сильно коррелируют, поэтому вес отдельного признака "
             "нестабилен и не равен его вкладу.",
    )

    made += viz.plot_convergence(
        history, selection["tol"], figures_dir / "fig6_convergence",
        note=f"Итоговая модель: {selection['final_epochs']} эпох, "
             f"lr = {selection['learning_rate']}",
    )

    # ROC, PR и калибровка по out-of-fold (первый повтор: по одному
    # предсказанию на объект)
    y_dev = oof_table["y_true"].to_numpy()
    p_first = oof_table[oof_columns[0]].to_numpy()

    made += viz.plot_roc_pr(
        y_dev, p_first, selection["threshold"],
        figures_dir / "fig2_roc_pr_oof", note=oof_note + ", первый повтор",
    )
    made += viz.plot_calibration(
        y_dev, p_first, figures_dir / "fig3_calibration_oof",
        n_bins=10, note=oof_note + ", первый повтор",
    )

    # Рисунки по независимому тесту, если оценка на тесте уже выполнялась
    test_path = path("test_predictions.csv")
    if test_path.exists():
        test = pd.read_csv(test_path, index_col=0)
        test_note = f"Данные: независимая тестовая выборка (n = {len(test)})"

        made += viz.plot_roc_pr(
            test["y_true"], test["y_prob"], selection["threshold"],
            figures_dir / "fig2_roc_pr_test", note=test_note,
        )
        made += viz.plot_calibration(
            test["y_true"], test["y_prob"],
            figures_dir / "fig3_calibration_test", n_bins=5, note=test_note,
        )
    else:
        print("Файла test_predictions.csv нет: рисунки по тесту пропущены.")

    # Рисунок 7: сравнение моделей, если запускался compare_models.py
    cmp_oof_path = results_dir / f"{CMP_PREFIX}oof_predictions.csv"
    if cmp_oof_path.exists():
        cmp_oof = pd.read_csv(cmp_oof_path, index_col=0)
        y_cmp, p_own = pool_prefix(oof_table, "y_prob_oof")
        probs = {
            "own_lr": p_own,
            "sklearn_lr": pool_prefix(cmp_oof, "y_prob_sklearn_lr")[1],
            "random_forest": pool_prefix(cmp_oof, "y_prob_random_forest")[1],
        }
        made += viz.plot_model_comparison(
            y_cmp, probs, figures_dir / "fig7_comparison_oof", n_bins=10,
            note=oof_note,
        )
    else:
        print("Файла cmp_oof_predictions.csv нет: рисунок сравнения пропущен.")

    cmp_test_path = results_dir / f"{CMP_PREFIX}test_predictions.csv"
    if cmp_test_path.exists():
        cmp_test = pd.read_csv(cmp_test_path, index_col=0)
        probs = {m: cmp_test[f"y_prob_{m}"] for m in viz.MODEL_LABELS}
        made += viz.plot_model_comparison(
            cmp_test["y_true"], probs, figures_dir / "fig7_comparison_test",
            n_bins=5,
            note=f"Данные: независимая тестовая выборка (n = {len(cmp_test)})",
        )

    print(f"Построено файлов: {len(made)}. Папка: {figures_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default=str(RESULTS_DIR),
                        help="папка с результатами main.py")
    args = parser.parse_args()

    main(args.dir)

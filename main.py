
# Основной сценарий исследования (схема с L2-регуляризацией)
#
# 1. Данные делятся на development (80%) и независимый тест (20%).
# 2. На development повторной кросс-валидацией подбирается сила
#    регуляризации l2 (критерий: log-loss, правило одной стандартной ошибки).
# 3. Порог выбирается отдельно по out-of-fold вероятностям.
# 4. Итоговая модель обучается на всём development и ОДИН раз
#    оценивается на тесте.
#
# Запуск:
#   python main.py                  полный прогон (повторная CV 5x10)
#   python main.py --quick          быстрая проверка (5x2), результаты в results/quick/
#   python main.py --skip-test      без обращения к тестовой выборке

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.data_loader import load_data
from src.evaluation import confidence_intervals
from src.experiments import (
    choose_threshold,
    cross_validate_l2,
    make_l2_grid,
    select_l2_one_se,
    summarize_by_l2,
)
from src.models import fit, predict_proba
from src.preprocessing import split_development_test

# Параметры сценария
RANDOM_STATE = 42
TEST_SIZE = 0.2
N_SPLITS = 5
N_REPEATS = 10
LEARNING_RATE = 0.5
EPOCHS = 30000
TOL = 1e-6
MIN_SENSITIVITY = 0.95

# Папка и префикс файлов новой схемы (старые отчёты остаются как есть)
RESULTS_DIR = Path("results")
PREFIX = "l2_"

METRIC_ORDER = (
    "accuracy", "sensitivity", "specificity", "precision", "f1",
    "roc_auc", "auprc", "log_loss", "brier",
)
PROPORTION_METRICS = {"accuracy", "sensitivity", "specificity", "precision"}


# Подбор l2 и порога на development-наборе
def run_development(X_dev, y_dev, n_repeats):
    l2_grid = make_l2_grid(len(y_dev))

    fold_table, oof = cross_validate_l2(
        X_dev,
        y_dev,
        l2_grid,
        n_splits=N_SPLITS,
        n_repeats=n_repeats,
        learning_rate=LEARNING_RATE,
        epochs=EPOCHS,
        tol=TOL,
        random_state=RANDOM_STATE,
    )

    summary = summarize_by_l2(fold_table)
    summary.insert(0, "C", 1.0 / (summary["l2"] * len(y_dev)))

    best_l2 = select_l2_one_se(summary, n_splits=N_SPLITS)

    # Порог выбираем по out-of-fold вероятностям выбранной модели
    # (все повторы вместе)
    threshold_info = choose_threshold(
        np.tile(np.asarray(y_dev), n_repeats),
        oof[best_l2].ravel(),
        min_sensitivity=MIN_SENSITIVITY,
    )

    return {
        "fold_table": fold_table,
        "summary": summary,
        "oof": oof,
        "l2": best_l2,
        "C": 1.0 / (best_l2 * len(y_dev)),
        "threshold_info": threshold_info,
    }


# Итоговая модель: стандартизатор и веса по всему development-набору
def train_final_model(X_dev, y_dev, l2):
    scaler = StandardScaler().fit(X_dev)

    weights, bias, history = fit(
        scaler.transform(X_dev),
        np.asarray(y_dev),
        learning_rate=LEARNING_RATE,
        epochs=EPOCHS,
        l2=l2,
        tol=TOL,
        return_history=True,
    )

    return scaler, weights, bias, history


# Единственная оценка на независимом тесте
def evaluate_on_test(scaler, weights, bias, X_test, y_test, threshold):
    prob = predict_proba(scaler.transform(X_test), weights, bias)
    point, intervals = confidence_intervals(y_test, prob, threshold)

    return prob, point, intervals


# Таблица метрик на тесте: значение и 95%-й интервал
def metrics_table(point, intervals):
    rows = []

    for name in METRIC_ORDER:
        low, high = intervals[name]
        rows.append({
            "metric": name,
            "value": point[name],
            "ci_low": low,
            "ci_high": high,
            "ci_method": (
                "Clopper-Pearson" if name in PROPORTION_METRICS else "bootstrap"
            ),
        })

    return pd.DataFrame(rows)


# Коэффициенты: в стандартизованных и в исходных единицах признаков
def coefficient_table(feature_names, scaler, weights, bias):
    weights_original = weights / scaler.scale_
    bias_original = bias - np.sum(weights * scaler.mean_ / scaler.scale_)

    table = pd.DataFrame({
        "feature": list(feature_names),
        "coef_standardized": weights,
        "coef_original_units": weights_original,
    })

    order = table["coef_standardized"].abs().sort_values(ascending=False).index
    table = table.loc[order].reset_index(drop=True)

    intercept = pd.DataFrame({
        "feature": ["(intercept)"],
        "coef_standardized": [bias],
        "coef_original_units": [bias_original],
    })

    return pd.concat([intercept, table], ignore_index=True)


def save_csv(table, output_dir, name, index=False):
    path = output_dir / f"{PREFIX}{name}"
    table.to_csv(path, index=index, encoding="utf-8")
    return path


def main(quick=False, skip_test=False):
    started = time.perf_counter()

    n_repeats = 2 if quick else N_REPEATS
    output_dir = RESULTS_DIR / "quick" if quick else RESULTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Данные и независимый тест
    features, target = load_data()
    X_dev, X_test, y_dev, y_test = split_development_test(
        features,
        target,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    print(
        f"Development: {len(y_dev)} объектов ({int(y_dev.sum())} злокачественных); "
        f"тест: {len(y_test)} объектов ({int(y_test.sum())} злокачественных)"
    )

    # 2-3. Подбор l2 и порога (только development)
    dev = run_development(X_dev, y_dev, n_repeats)

    fold_table = dev["fold_table"]
    summary = dev["summary"]
    threshold_info = dev["threshold_info"]

    save_csv(fold_table, output_dir, "cv_folds.csv")
    save_csv(summary, output_dir, "cv_summary.csv")

    print("\nСводка кросс-валидации (среднее по фолдам):")
    print(
        summary[["C", "l2", "log_loss_mean", "log_loss_std",
                 "brier_mean", "roc_auc_mean", "converged_share"]]
        .round(5)
        .to_string(index=False)
    )

    print(
        f"\nВыбрано по правилу одной стандартной ошибки: "
        f"C = {dev['C']:.4f}, l2 = {dev['l2']:.6f}"
    )
    print(
        f"Порог (чувствительность >= {MIN_SENSITIVITY} на out-of-fold): "
        f"{threshold_info['threshold']:.2f} "
        f"(чувствительность {threshold_info['sensitivity']:.4f}, "
        f"специфичность {threshold_info['specificity']:.4f})"
    )
    print(
        f"Обучений остановлено по tol: {fold_table['converged'].mean():.1%} "
        f"из {len(fold_table)}"
    )

    # 4. Итоговая модель на всём development
    scaler, weights, bias, history = train_final_model(X_dev, y_dev, dev["l2"])

    final_converged = bool(history["grad_norm"][-1] < TOL)

    save_csv(
        coefficient_table(features.columns, scaler, weights, bias),
        output_dir,
        "coefficients.csv",
    )
    save_csv(pd.DataFrame(history), output_dir, "final_history.csv")

    # Out-of-fold вероятности выбранной модели: отдельный столбец на каждый
    # повтор CV (графики строятся ровно так же, как выбирался порог)
    oof_table = pd.DataFrame({"y_true": np.asarray(y_dev)}, index=X_dev.index)
    for repeat, row in enumerate(dev["oof"][dev["l2"]]):
        oof_table[f"y_prob_oof_r{repeat}"] = row

    save_csv(oof_table, output_dir, "oof_predictions.csv", index=True)

    selection = {
        "C": dev["C"],
        "l2": dev["l2"],
        "threshold": threshold_info["threshold"],
        "oof_sensitivity": threshold_info["sensitivity"],
        "oof_specificity": threshold_info["specificity"],
        "min_sensitivity": MIN_SENSITIVITY,
        "n_dev": int(len(y_dev)),
        "n_test": int(len(y_test)),
        "n_splits": N_SPLITS,
        "n_repeats": n_repeats,
        "learning_rate": LEARNING_RATE,
        "epochs_cap": EPOCHS,
        "tol": TOL,
        "cv_converged_share": float(fold_table["converged"].mean()),
        "final_epochs": len(history["loss"]),
        "final_converged": final_converged,
        "random_state": RANDOM_STATE,
        "test_evaluated": not skip_test,
    }

    # 5. Оценка на тесте: один раз, с доверительными интервалами
    if skip_test:
        print("\nОценка на тесте пропущена (--skip-test).")
    else:
        prob, point, intervals = evaluate_on_test(
            scaler, weights, bias, X_test, y_test, threshold_info["threshold"]
        )

        results = metrics_table(point, intervals)
        save_csv(results, output_dir, "test_metrics.csv")
        save_csv(
            pd.DataFrame({"y_true": np.asarray(y_test), "y_prob": prob},
                         index=X_test.index),
            output_dir,
            "test_predictions.csv",
            index=True,
        )

        print("\nИтоговая оценка на независимой тестовой выборке (95% ДИ):")
        for _, row in results.iterrows():
            print(
                f"{row['metric']:12s} {row['value']:.4f}  "
                f"[{row['ci_low']:.4f}, {row['ci_high']:.4f}]  ({row['ci_method']})"
            )

        print(
            "\nМатрица ошибок:"
            f"\nTN: {point['TN']}, FP: {point['FP']}"
            f"\nFN: {point['FN']}, TP: {point['TP']}"
        )

    selection["elapsed_seconds"] = round(time.perf_counter() - started, 1)

    with open(output_dir / f"{PREFIX}selection.json", "w", encoding="utf-8") as file:
        json.dump(selection, file, ensure_ascii=False, indent=2)

    print(f"\nГотово за {selection['elapsed_seconds']} с. Файлы: {output_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true",
                        help="повторная CV 5x2 вместо 5x10, результаты в results/quick/")
    parser.add_argument("--skip-test", action="store_true",
                        help="не обращаться к тестовой выборке")
    args = parser.parse_args()

    main(quick=args.quick, skip_test=args.skip_test)

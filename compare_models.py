# Сравнение трёх моделей на одних и тех же данных и по одним правилам.
#
#   1. собственная логистическая регрессия (градиентный спуск, L2): результаты
#      берутся из файлов main.py, здесь не пересчитываются;
#   2. sklearn.LogisticRegression с тем же C: проверка корректности нашей
#      реализации (метрики и коэффициенты должны совпасть);
#   3. случайный лес (sklearn): модель другого семейства.
#
# Правила те же, что у основной модели: гиперпараметры подбираются только на
# development (та же повторная CV с теми же фолдами), порог выбирается по
# out-of-fold вероятностям (чувствительность >= MIN_SENSITIVITY), на тест
# каждая модель попадает один раз, доверительные интервалы считаются теми же
# функциями.
#
# Порядок запуска: сначала main.py, затем этот скрипт.
#   python compare_models.py               полное сравнение
#   python compare_models.py --quick       быстрая проверка (после main.py --quick)
#   python compare_models.py --skip-test   без обращения к тестовой выборке

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from main import (
    METRIC_ORDER,
    MIN_SENSITIVITY,
    PREFIX,
    RESULTS_DIR,
    TEST_SIZE,
    metrics_table,
)
from src.data_loader import load_data
from src.evaluation import confidence_intervals, probabilistic_metrics
from src.experiments import choose_threshold
from src.preprocessing import split_development_test

CMP_PREFIX = "cmp_"

OWN = "own_lr"
SKLEARN_LR = "sklearn_lr"
FOREST = "random_forest"
MODEL_ORDER = (OWN, SKLEARN_LR, FOREST)

N_TREES = 500
LEAF_GRID = (1, 2, 3, 5, 8)  # min_samples_leaf: больше значение = проще модель


# ---------- Вспомогательные функции ----------

def make_sklearn_lr(l2, random_state=None, n_train=None, solver="lbfgs"):
    # Штраф sklearn: 1/(2C)·||w||² к сумме потерь. Наш l2 стоит при средней
    # потере, поэтому C = 1 / (l2 · n_train); n_train зависит от размера
    # обучающей части (на фолде 364 объекта, на всём development 455)
    return LogisticRegression(
        C=1.0 / (l2 * n_train), solver=solver, tol=1e-10, max_iter=10000
    )


def make_forest(min_samples_leaf, random_state=0, n_train=None,
                n_estimators=N_TREES):
    return RandomForestClassifier(
        n_estimators=n_estimators,
        min_samples_leaf=min_samples_leaf,
        max_features="sqrt",
        n_jobs=-1,
        random_state=random_state,
    )


# Повторная стратифицированная CV для набора конфигураций одной модели.
# Фолды совпадают с фолдами cross_validate_l2 (тот же random_state).
def cross_validate_model(
    X, y, make_model, configs, scale,
    n_splits=5, n_repeats=10, random_state=42, desc="CV",
):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=int)

    cv = RepeatedStratifiedKFold(
        n_splits=n_splits, n_repeats=n_repeats, random_state=random_state
    )

    rows = []
    oof = {config: np.zeros((n_repeats, len(y))) for config in configs}

    with tqdm(total=n_splits * n_repeats, desc=desc, unit="фолд",
              dynamic_ncols=True) as progress:
        for fold_id, (train_idx, val_idx) in enumerate(cv.split(X, y)):
            repeat = fold_id // n_splits

            X_train, X_val = X[train_idx], X[val_idx]
            if scale:
                scaler = StandardScaler().fit(X_train)
                X_train = scaler.transform(X_train)
                X_val = scaler.transform(X_val)

            for config, params in configs.items():
                model = make_model(params, random_state + fold_id, len(train_idx))
                model.fit(X_train, y[train_idx])
                prob = model.predict_proba(X_val)[:, 1]

                rows.append({
                    "config": config,
                    "repeat": repeat,
                    "fold": fold_id,
                    **probabilistic_metrics(y[val_idx], prob),
                })
                oof[config][repeat, val_idx] = prob

            progress.update(1)

    return pd.DataFrame(rows), oof


# Среднее и σ по фолдам для каждой конфигурации
def summarize_configs(fold_table):
    out = fold_table.groupby("config")[["log_loss", "brier", "roc_auc"]].agg(
        ["mean", "std"]
    )
    out.columns = [f"{metric}_{stat}" for metric, stat in out.columns]
    return out.reset_index()


# Выбор конфигурации леса: минимальный средний log-loss. Правило одной
# стандартной ошибки здесь не применяется: у леса нет однозначного
# направления «проще», как у силы L2-штрафа
def select_best(summary):
    return summary.loc[summary["log_loss_mean"].idxmin(), "config"]


# Коэффициенты sklearn при двух решателях против коэффициентов нашей модели
# (в стандартизованных признаках)
def solver_check(X, y, l2, own_coefficients, feature_names, solvers=None):
    if solvers is None:
        solvers = ("lbfgs", "newton-cholesky")

    X = np.asarray(X, dtype=float)
    X_scaled = StandardScaler().fit_transform(X)
    own = own_coefficients.set_index("feature")["coef_standardized"]

    rows = []
    for solver in solvers:
        model = make_sklearn_lr(l2, n_train=len(X), solver=solver).fit(
            X_scaled, np.asarray(y)
        )
        weights = pd.Series(model.coef_[0], index=list(feature_names))

        rows.append({
            "solver": solver,
            "n_iter": int(np.max(model.n_iter_)),
            "max_abs_diff_weights": float(
                np.max(np.abs(weights - own[weights.index]))
            ),
            "abs_diff_intercept": float(abs(model.intercept_[0] - own["(intercept)"])),
        })

    return pd.DataFrame(rows)


def summary_row(model, config_text, summary_line, threshold_info):
    return {
        "model": model,
        "config": config_text,
        "log_loss_mean": summary_line["log_loss_mean"],
        "log_loss_std": summary_line["log_loss_std"],
        "brier_mean": summary_line["brier_mean"],
        "brier_std": summary_line["brier_std"],
        "roc_auc_mean": summary_line["roc_auc_mean"],
        "roc_auc_std": summary_line["roc_auc_std"],
        "threshold": threshold_info["threshold"],
        "oof_sensitivity": threshold_info["sensitivity"],
        "oof_specificity": threshold_info["specificity"],
    }


def pooled(y_dev, oof_matrix):
    return np.tile(np.asarray(y_dev), oof_matrix.shape[0]), oof_matrix.ravel()


# ---------- Основной сценарий ----------

def main(quick=False, skip_test=False, results_dir=None):
    started = time.perf_counter()

    if results_dir is None:
        results_dir = RESULTS_DIR / "quick" if quick else RESULTS_DIR
    results_dir = Path(results_dir)

    def path(name, prefix=PREFIX):
        return results_dir / f"{prefix}{name}"

    if not path("selection.json").exists():
        raise FileNotFoundError(
            f"Нет {path('selection.json')}. Сначала запустите main.py"
            + (" --quick" if quick else "")
        )

    with open(path("selection.json"), encoding="utf-8") as file:
        selection = json.load(file)

    n_splits = selection["n_splits"]
    n_repeats = selection["n_repeats"]
    random_state = selection["random_state"]
    c_own = selection["C"]
    l2_own = selection["l2"]

    # 1. Данные: то же разбиение, что в main.py
    features, target = load_data()
    X_dev, X_test, y_dev, y_test = split_development_test(
        features, target, test_size=TEST_SIZE, random_state=random_state
    )

    # 2. Собственная модель: готовые результаты main.py
    own_oof = pd.read_csv(path("oof_predictions.csv"), index_col=0)
    own_columns = [c for c in own_oof.columns if c.startswith("y_prob_oof_r")]
    own_oof_matrix = own_oof[own_columns].to_numpy().T
    own_summary = pd.read_csv(path("cv_summary.csv"))
    own_line = own_summary.iloc[(own_summary["C"] - c_own).abs().argmin()]
    own_threshold = {
        "threshold": selection["threshold"],
        "sensitivity": selection["oof_sensitivity"],
        "specificity": selection["oof_specificity"],
    }

    # 3. sklearn.LogisticRegression с тем же C
    fold_table, oof_lr = cross_validate_model(
        X_dev, y_dev, make_sklearn_lr, {c_own: l2_own}, scale=True,
        n_splits=n_splits, n_repeats=n_repeats, random_state=random_state,
        desc="sklearn LogisticRegression",
    )
    lr_summary = summarize_configs(fold_table).iloc[0]
    lr_oof = oof_lr[c_own]
    lr_threshold = choose_threshold(*pooled(y_dev, lr_oof),
                                    min_sensitivity=MIN_SENSITIVITY)

    # 4. Случайный лес: подбор min_samples_leaf только по development
    fold_table, oof_rf = cross_validate_model(
        X_dev, y_dev, make_forest, {leaf: leaf for leaf in LEAF_GRID},
        scale=False,
        n_splits=n_splits, n_repeats=n_repeats, random_state=random_state,
        desc="Случайный лес",
    )
    rf_grid = summarize_configs(fold_table)
    leaf = int(select_best(rf_grid))
    rf_line = rf_grid[rf_grid["config"] == leaf].iloc[0]
    rf_oof = oof_rf[leaf]
    rf_threshold = choose_threshold(*pooled(y_dev, rf_oof),
                                    min_sensitivity=MIN_SENSITIVITY)

    results_dir.mkdir(parents=True, exist_ok=True)

    # 5. Сводка по development
    cv_summary = pd.DataFrame([
        summary_row(OWN, f"C = {c_own:.4g}", own_line, own_threshold),
        summary_row(SKLEARN_LR, f"C = {c_own:.4g}", lr_summary, lr_threshold),
        summary_row(FOREST, f"{N_TREES} деревьев, min_samples_leaf = {leaf}",
                    rf_line, rf_threshold),
    ])
    cv_summary.to_csv(path("cv_summary.csv", CMP_PREFIX), index=False)
    rf_grid.rename(columns={"config": "min_samples_leaf"}).to_csv(
        path("rf_grid.csv", CMP_PREFIX), index=False
    )

    oof_table = pd.DataFrame({"y_true": np.asarray(y_dev)}, index=X_dev.index)
    for repeat in range(n_repeats):
        oof_table[f"y_prob_{SKLEARN_LR}_r{repeat}"] = lr_oof[repeat]
    for repeat in range(n_repeats):
        oof_table[f"y_prob_{FOREST}_r{repeat}"] = rf_oof[repeat]
    oof_table.to_csv(path("oof_predictions.csv", CMP_PREFIX))

    # 6. Совпадение решателей с собственной моделью (коэффициенты)
    own_coefficients = pd.read_csv(path("coefficients.csv"))
    check = solver_check(X_dev, y_dev, l2_own, own_coefficients, features.columns)
    check.to_csv(path("solver_check.csv", CMP_PREFIX), index=False)

    print("\nСравнение на development (повторная CV, среднее по фолдам):")
    print(cv_summary.drop(columns=["config"]).round(4).to_string(index=False))
    print("\nСлучайный лес, сетка min_samples_leaf:")
    print(rf_grid.round(4).to_string(index=False))
    print("\nСовпадение коэффициентов sklearn с собственной моделью:")
    print(check.to_string(index=False))

    comparison = {
        "rf_min_samples_leaf": leaf,
        "rf_n_trees": N_TREES,
        "rf_leaf_grid": list(LEAF_GRID),
        "thresholds": {
            OWN: own_threshold["threshold"],
            SKLEARN_LR: lr_threshold["threshold"],
            FOREST: rf_threshold["threshold"],
        },
        "test_evaluated": not skip_test,
    }

    # 7. Единственная оценка всех трёх моделей на тесте
    if skip_test:
        print("\nОценка на тесте пропущена (--skip-test).")
    else:
        if not path("test_predictions.csv").exists():
            raise FileNotFoundError(
                f"Нет {path('test_predictions.csv')}. Сначала запустите main.py "
                "без --skip-test"
            )

        own_test = pd.read_csv(path("test_predictions.csv"), index_col=0)
        if not own_test.index.equals(X_test.index):
            raise ValueError("Тестовая выборка не совпадает с выборкой main.py")

        scaler = StandardScaler().fit(X_dev)
        lr_model = make_sklearn_lr(l2_own, n_train=len(X_dev)).fit(
            scaler.transform(X_dev), np.asarray(y_dev)
        )
        rf_model = make_forest(leaf, random_state, len(X_dev)).fit(
            X_dev.to_numpy(), np.asarray(y_dev)
        )

        test_probs = {
            OWN: own_test["y_prob"].to_numpy(),
            SKLEARN_LR: lr_model.predict_proba(scaler.transform(X_test))[:, 1],
            FOREST: rf_model.predict_proba(X_test.to_numpy())[:, 1],
        }
        thresholds = comparison["thresholds"]

        tables = []
        points = {}
        for model in MODEL_ORDER:
            point, intervals = confidence_intervals(
                y_test, test_probs[model], thresholds[model]
            )
            points[model] = point
            table = metrics_table(point, intervals)
            table.insert(0, "model", model)
            tables.append(table)

        pd.concat(tables, ignore_index=True).to_csv(
            path("test_metrics.csv", CMP_PREFIX), index=False
        )

        predictions = pd.DataFrame({"y_true": np.asarray(y_test)},
                                   index=X_test.index)
        for model in MODEL_ORDER:
            predictions[f"y_prob_{model}"] = test_probs[model]
        predictions.to_csv(path("test_predictions.csv", CMP_PREFIX))

        comparison["max_abs_prob_diff_own_vs_sklearn_test"] = float(
            np.max(np.abs(test_probs[OWN] - test_probs[SKLEARN_LR]))
        )

        print("\nСравнение на независимом тесте (значения при своём пороге):")
        wide = pd.DataFrame(
            {model: {m: points[model][m] for m in METRIC_ORDER}
             for model in MODEL_ORDER}
        )
        print(wide.round(4).to_string())
        print(
            "\nМаксимальная разница вероятностей: собственная модель против "
            f"sklearn: {comparison['max_abs_prob_diff_own_vs_sklearn_test']:.2e}"
        )

    comparison["elapsed_seconds"] = round(time.perf_counter() - started, 1)
    with open(path("selection.json", CMP_PREFIX), "w", encoding="utf-8") as file:
        json.dump(comparison, file, ensure_ascii=False, indent=2)

    print(f"\nГотово за {comparison['elapsed_seconds']} с. Файлы: {results_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true",
                        help="после main.py --quick, результаты в results/quick/")
    parser.add_argument("--skip-test", action="store_true",
                        help="не обращаться к тестовой выборке")
    args = parser.parse_args()

    main(quick=args.quick, skip_test=args.skip_test)

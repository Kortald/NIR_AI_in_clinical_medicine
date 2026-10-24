import numpy as np
import pandas as pd
import pytest

from src.data_loader import load_data
from src.preprocessing import split_development_test
import main as pipeline


# ---------- data_loader ----------

def test_load_data_shape_and_labels():
    features, target = load_data()

    assert features.shape == (569, 30)
    assert len(target) == 569
    assert set(target.unique()) == {0, 1}


def test_load_data_label_one_is_malignant():
    _, target = load_data()

    # В WDBC 212 злокачественных и 357 доброкачественных опухолей
    assert int(target.sum()) == 212
    assert int((target == 0).sum()) == 357


def test_load_data_has_no_missing_values():
    features, target = load_data()

    assert not features.isna().any().any()
    assert not target.isna().any()


# ---------- preprocessing ----------

def test_split_development_test_sizes_and_no_overlap():
    features, target = load_data()

    X_dev, X_test, y_dev, y_test = split_development_test(features, target)

    assert len(X_dev) == 455
    assert len(X_test) == 114
    assert len(X_dev) + len(X_test) == len(features)
    assert set(X_dev.index).isdisjoint(set(X_test.index))
    assert X_dev.index.equals(y_dev.index)
    assert X_test.index.equals(y_test.index)


def test_split_development_test_is_stratified():
    features, target = load_data()

    _, _, y_dev, y_test = split_development_test(features, target)

    overall = target.mean()
    assert y_dev.mean() == pytest.approx(overall, abs=0.01)
    assert y_test.mean() == pytest.approx(overall, abs=0.01)


def test_split_development_test_is_reproducible():
    features, target = load_data()

    first = split_development_test(features, target)
    second = split_development_test(features, target)

    assert first[1].index.equals(second[1].index)


# ---------- main.py: вспомогательные функции ----------

def test_coefficient_table_original_units_give_same_probabilities():
    features, target = load_data()
    X_dev, _, y_dev, _ = split_development_test(features, target)

    scaler, weights, bias, _ = pipeline.train_final_model(X_dev, y_dev, 0.01)
    table = pipeline.coefficient_table(features.columns, scaler, weights, bias)
    table = table.set_index("feature")

    weights_original = table.loc[features.columns, "coef_original_units"].to_numpy()
    bias_original = table.loc["(intercept)", "coef_original_units"]

    from src.models import predict_proba

    standardized = predict_proba(scaler.transform(X_dev), weights, bias)
    original = predict_proba(X_dev.to_numpy(), weights_original, bias_original)

    assert np.allclose(standardized, original, atol=1e-10)


def test_coefficient_table_sorted_by_abs_standardized_value():
    features, target = load_data()
    X_dev, _, y_dev, _ = split_development_test(features, target)

    scaler, weights, bias, _ = pipeline.train_final_model(X_dev, y_dev, 0.05)
    table = pipeline.coefficient_table(features.columns, scaler, weights, bias)

    assert table.loc[0, "feature"] == "(intercept)"
    sorted_part = table.loc[1:, "coef_standardized"].abs().to_numpy()
    assert np.all(np.diff(sorted_part) <= 0)


def test_metrics_table_structure():
    point = {name: 0.5 for name in pipeline.METRIC_ORDER}
    intervals = {name: (0.4, 0.6) for name in pipeline.METRIC_ORDER}

    table = pipeline.metrics_table(point, intervals)

    assert list(table["metric"]) == list(pipeline.METRIC_ORDER)
    assert set(table.columns) == {"metric", "value", "ci_low", "ci_high", "ci_method"}

    methods = dict(zip(table["metric"], table["ci_method"]))
    assert methods["sensitivity"] == "Clopper-Pearson"
    assert methods["roc_auc"] == "bootstrap"


def test_evaluate_on_test_returns_probabilities_and_intervals():
    features, target = load_data()
    X_dev, _, y_dev, _ = split_development_test(features, target)

    # Проверяем механику на части development, а не на настоящем тесте
    half = len(X_dev) // 2
    scaler, weights, bias, _ = pipeline.train_final_model(
        X_dev.iloc[:half], y_dev.iloc[:half], 0.01
    )

    prob, point, intervals = pipeline.evaluate_on_test(
        scaler, weights, bias,
        X_dev.iloc[half:], y_dev.iloc[half:], 0.5,
    )

    assert len(prob) == len(X_dev) - half
    assert np.all((prob > 0) & (prob < 1))
    for name in pipeline.METRIC_ORDER:
        low, high = intervals[name]
        assert low <= point[name] <= high, name

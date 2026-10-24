#Загрузка датасета, описание признаков и классов
import pandas as pd
from sklearn.datasets import load_breast_cancer


def load_data() -> tuple[pd.DataFrame, pd.Series]:
    """Загружает WDBC и возвращает признаки и целевые метки."""
    dataset = load_breast_cancer()

    features = pd.DataFrame(
        dataset.data,
        columns=dataset.feature_names,
    )

    # В sklearn: 0 — malignant, 1 — benign.
    target = pd.Series(
        1 - dataset.target,
        name="diagnosis",
    )

    return features, target
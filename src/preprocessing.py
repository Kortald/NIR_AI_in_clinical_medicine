
# Разбиение данных

from sklearn.model_selection import train_test_split


def split_development_test(features, target, test_size=0.2, random_state=42):
    """Делит данные на development (для подбора и обучения) и независимый тест."""

    X_dev, X_test, y_dev, y_test = train_test_split(
        features,
        target,
        test_size=test_size,
        random_state=random_state,
        stratify=target,
    )

    return X_dev, X_test, y_dev, y_test

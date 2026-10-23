import pandas as pd

from mlproject.data import (
    CLASSIFICATION_TARGET,
    REGRESSION_TARGET,
    generate_classification_data,
    generate_regression_data,
)


def test_regression_data_is_deterministic() -> None:
    first = generate_regression_data(200, seed=7)
    second = generate_regression_data(200, seed=7)

    pd.testing.assert_frame_equal(first, second)
    assert len(first) == 200
    assert REGRESSION_TARGET in first.columns
    assert (first[REGRESSION_TARGET] > 0).all()
    assert first["property_type"].nunique() == 3


def test_classification_data_has_both_classes() -> None:
    data = generate_classification_data(500, seed=7)

    assert len(data) == 500
    assert CLASSIFICATION_TARGET in data.columns
    assert set(data[CLASSIFICATION_TARGET].unique()) == {0, 1}
    assert 0.1 < data[CLASSIFICATION_TARGET].mean() < 0.9

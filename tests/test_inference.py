import numpy as np

from src.feature_engineering import FEATURES
from src.inference import SEQ_LENGTH, recursive_forecast


class IdentityScaler:
    def transform(self, x):
        return np.asarray(x, dtype=float)

    def inverse_transform(self, x):
        return np.asarray(x, dtype=float)


class ConstantModel:
    def __init__(self, value):
        self.value = value
        self.calls = 0

    def predict(self, X, verbose=0):
        assert X.shape == (1, SEQ_LENGTH, len(FEATURES))
        self.calls += 1
        return np.array([[self.value]])


def make_window(close=100.0):
    return np.full((SEQ_LENGTH, len(FEATURES)), close)


def test_recursive_forecast_returns_one_price_per_day():
    model = ConstantModel(101.0)
    preds = recursive_forecast(make_window(), IdentityScaler(), IdentityScaler(), model, 10)
    assert preds == [101.0] * 10
    assert model.calls == 10


def test_recursive_forecast_clamps_unrealistic_predictions():
    scaler = IdentityScaler()
    assert recursive_forecast(make_window(), scaler, scaler, ConstantModel(-50.0), 1) == [33.0]
    assert recursive_forecast(make_window(), scaler, scaler, ConstantModel(1e6), 1) == [300.0]


def test_recursive_forecast_does_not_modify_input():
    window = make_window()
    recursive_forecast(window, IdentityScaler(), IdentityScaler(), ConstantModel(120.0), 3)
    assert (window == 100.0).all()

import numpy as np
import pytest

from src import inference
from src.feature_engineering import FEATURES
from src.inference import SEQ_LENGTH, model_for, recursive_forecast
from tests.test_feature_engineering import make_prices


class ConstantModel:
    """Always predicts the same scaled log return."""

    def __init__(self, value):
        self.value = value
        self.calls = 0

    def predict(self, X, verbose=0):
        assert X.shape[1:] == (SEQ_LENGTH, len(FEATURES))
        self.calls += 1
        return np.full((len(X), 1), self.value)


def test_recursive_forecast_returns_one_price_per_day():
    raw = make_prices()
    model = ConstantModel(0.1)  # 1% a day once divided by the target scale
    preds = recursive_forecast(raw, model, 10, target_scale=10.0)
    last = raw["Close"].iloc[-1]
    assert len(preds) == 10 and model.calls == 10
    assert preds == pytest.approx([last * np.exp(0.01 * (i + 1)) for i in range(10)])


def test_recursive_forecast_clamps_unrealistic_predictions():
    raw = make_prices()
    last = raw["Close"].iloc[-1]
    up = recursive_forecast(raw, ConstantModel(1e6), 20)
    assert up[0] == pytest.approx(last * np.exp(0.10))  # one day is capped at +10%
    assert max(up) == pytest.approx(last * 3.0)          # the path is capped at 3×
    down = recursive_forecast(raw, ConstantModel(-1e6), 20)
    assert min(down) == pytest.approx(last * 0.33)


def test_recursive_forecast_does_not_modify_input():
    raw = make_prices()
    before = raw.copy()
    recursive_forecast(raw, ConstantModel(0.2), 3)
    assert raw.equals(before)


def test_model_for_prefers_ticker_model(tmp_path, monkeypatch):
    monkeypatch.setattr(inference, "MODEL_DIR", str(tmp_path))
    assert model_for("AAPL") is None
    (tmp_path / "universal.npz").write_bytes(b"")
    assert model_for("AAPL") == "universal"
    (tmp_path / "AAPL.npz").write_bytes(b"")
    assert model_for("AAPL") == "AAPL"
    assert model_for("MSFT") == "universal"

import numpy as np
import pytest

from src.feature_engineering import FEATURES, compute_indicators
from src.nn import NumpyLSTM, load_bundle
from src.preprocessing import SEQ_LENGTH, build_sequences, normalize_window
from tests.test_feature_engineering import make_prices


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def test_lstm_matches_hand_computed_step():
    # One unit, one input, one time step: gates in Keras order i, f, c, o
    kernel    = np.array([[0.5, -0.3, 0.8, 0.2]])
    recurrent = np.zeros((1, 4))
    bias      = np.array([0.1, 0.0, -0.2, 0.3])
    model = NumpyLSTM([(kernel, recurrent, bias)], [(np.array([[2.0]]), np.array([1.0]), "linear")])

    x = 0.7
    i, g, o = sigmoid(0.5 * x + 0.1), np.tanh(0.8 * x - 0.2), sigmoid(0.2 * x + 0.3)
    h = o * np.tanh(i * g)  # c starts at zero, so the forget gate drops out
    assert model.predict(np.array([[[x]]]))[0, 0] == pytest.approx(2 * h + 1)


def test_lstm_handles_batches_and_stacks():
    rng = np.random.default_rng(0)
    layers = [(rng.normal(size=(3, 8)), rng.normal(size=(2, 8)), rng.normal(size=8)),
              (rng.normal(size=(2, 4)), rng.normal(size=(1, 4)), rng.normal(size=4))]
    model = NumpyLSTM(layers, [(rng.normal(size=(1, 1)), np.zeros(1), "relu")])
    X = rng.normal(size=(5, 7, 3))
    out = model.predict(X)
    assert out.shape == (5, 1) and (out >= 0).all()
    # Each sample is independent of the rest of the batch
    assert model.predict(X[2:3])[0] == pytest.approx(out[2])


def test_load_bundle_round_trip(tmp_path):
    rng = np.random.default_rng(1)
    path = tmp_path / "m.npz"
    np.savez(path, n_lstm=1, n_dense=1, target_scale=10.0, tickers="AAPL,MSFT",
             trained_on="2026-01-01", test_mape=0.012,
             lstm0_kernel=rng.normal(size=(11, 8)), lstm0_recurrent=rng.normal(size=(2, 8)),
             lstm0_bias=np.zeros(8), dense0_kernel=np.ones((2, 1)), dense0_bias=np.zeros(1),
             dense0_activation="linear")
    model, meta = load_bundle(path)
    assert meta["tickers"] == ["AAPL", "MSFT"] and meta["target_scale"] == 10.0
    assert meta["test_mape"] == pytest.approx(0.012) and meta["naive_mape"] is None
    assert model.predict(np.zeros((1, SEQ_LENGTH, 11))).shape == (1, 1)


def test_normalize_window_is_scale_free():
    feats = compute_indicators(make_prices()).dropna()[FEATURES].values[-SEQ_LENGTH:]
    a = normalize_window(feats)
    scaled = feats.copy()
    price_like = [FEATURES.index(c) for c in ("Close", "High", "Low", "Open", "MA50", "MA200", "MACD", "Signal")]
    scaled[:, price_like] *= 7.0  # the same chart at a different price level
    assert a[-1, 0] == 0.0
    np.testing.assert_allclose(normalize_window(scaled), a, atol=1e-12)


def test_build_sequences_targets_next_day_log_return():
    feats = compute_indicators(make_prices()).dropna()[FEATURES].values
    X, y, prev = build_sequences(feats)
    assert X.shape == (len(feats) - SEQ_LENGTH, SEQ_LENGTH, len(FEATURES))
    assert y[0] == pytest.approx(np.log(feats[SEQ_LENGTH, 0] / feats[SEQ_LENGTH - 1, 0]) * 10)
    assert prev[0] == feats[SEQ_LENGTH - 1, 0]

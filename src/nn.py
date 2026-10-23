# src/nn.py
"""
Dependency-light inference for the trained LSTMs.

TensorFlow is far too large for serverless hosts such as Vercel, so
`scripts/export_models.py` copies each trained Keras model into
`models/<name>.npz`, and this module replays the forward pass with NumPy.
It exposes the same `predict` method as a Keras model.
"""

import numpy as np


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


class NumpyLSTM:
    """
    Stacked Keras LSTM layers followed by Dense layers.

    `lstm` is a list of (kernel, recurrent_kernel, bias) with Keras' gate order
    (input, forget, cell, output); every LSTM but the last returns sequences.
    `dense` is a list of (kernel, bias, activation).
    """

    def __init__(self, lstm, dense):
        self.lstm  = lstm
        self.dense = dense

    @staticmethod
    def _run_lstm(x, kernel, recurrent, bias):
        batch, steps, _ = x.shape
        units = recurrent.shape[0]
        h = np.zeros((batch, units))
        c = np.zeros((batch, units))
        xw = x @ kernel + bias          # input projections for every step at once
        out = np.empty((batch, steps, units))
        for t in range(steps):
            z = xw[:, t] + h @ recurrent
            i = _sigmoid(z[:, :units])
            f = _sigmoid(z[:, units:2 * units])
            g = np.tanh(z[:, 2 * units:3 * units])
            o = _sigmoid(z[:, 3 * units:])
            c = f * c + i * g
            h = o * np.tanh(c)
            out[:, t] = h
        return out

    def predict(self, X, verbose=0):
        x = np.asarray(X, dtype=float)
        for kernel, recurrent, bias in self.lstm:
            x = self._run_lstm(x, kernel, recurrent, bias)
        x = x[:, -1]
        for kernel, bias, activation in self.dense:
            x = x @ kernel + bias
            if activation == "relu":
                x = np.maximum(x, 0)
        return x


def load_bundle(path):
    """Load a `models/<name>.npz` written by scripts/export_models.py."""
    with np.load(path, allow_pickle=False) as z:
        n_lstm  = int(z["n_lstm"])
        n_dense = int(z["n_dense"])
        lstm = [(z[f"lstm{i}_kernel"], z[f"lstm{i}_recurrent"], z[f"lstm{i}_bias"])
                for i in range(n_lstm)]
        dense = [(z[f"dense{i}_kernel"], z[f"dense{i}_bias"], str(z[f"dense{i}_activation"]))
                 for i in range(n_dense)]
        meta = {"target_scale": float(z["target_scale"])}
        for key in ("trained_on", "tickers"):
            meta[key] = str(z[key]) if key in z else None
        for key in ("test_mape", "naive_mape", "test_direction_acc", "test_samples"):
            meta[key] = float(z[key]) if key in z else None
    meta["tickers"] = meta["tickers"].split(",") if meta["tickers"] else []
    return NumpyLSTM(lstm, dense), meta

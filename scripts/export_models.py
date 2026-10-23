#!/usr/bin/env python3
# scripts/export_models.py

"""
Convert trained Keras models in models/ (lstm_<name>.h5) into the NumPy
bundles the web app loads (models/<name>.npz). Run after training; needs h5py
from requirements-train.txt, but not TensorFlow.

If data/processed/X_test_<name>.npy exists, the held-out test split is also
scored so the app can show how accurate each model is, next to a naive
"tomorrow = today" baseline.

    python scripts/export_models.py              # every model in models/
    python scripts/export_models.py universal    # just this one
"""

import json
import os
import sys
from datetime import datetime, timezone

import h5py
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.insert(0, ROOT)

from src.nn import NumpyLSTM  # noqa: E402
from src.preprocessing import TARGET_SCALE, UNIVERSE  # noqa: E402

MODEL_DIR = os.path.join(ROOT, "models")
DATA_DIR  = os.path.join(ROOT, "data", "processed")


def _layer_weights(group):
    """Collect a layer's weight arrays by variable name, whatever the nesting."""
    found = {}

    def collect(name, obj):
        if isinstance(obj, h5py.Dataset):
            found[name.rsplit("/", 1)[-1]] = obj[()]
        # returning None keeps visititems walking

    group.visititems(collect)
    return found


def read_keras_h5(path):
    with h5py.File(path, "r") as f:
        config = json.loads(f.attrs["model_config"])
        weights = f["model_weights"]
        lstm, dense = [], []
        for layer in config["config"]["layers"]:
            kind, cfg = layer["class_name"], layer["config"]
            if kind == "LSTM":
                w = _layer_weights(weights[cfg["name"]])
                lstm.append((w["kernel"], w["recurrent_kernel"], w["bias"]))
            elif kind == "Dense":
                w = _layer_weights(weights[cfg["name"]])
                dense.append((w["kernel"], w["bias"], cfg.get("activation", "linear")))
    return lstm, dense


def evaluate(model, name):
    """Next-day price MAPE (model and naive) and direction hit rate on the test split."""
    paths = [os.path.join(DATA_DIR, f"{part}_test_{name}.npy") for part in ("X", "y", "prev")]
    if not all(os.path.exists(p) for p in paths):
        return {}
    X, y, prev = (np.load(p) for p in paths)
    pred  = model.predict(X).ravel() / TARGET_SCALE
    true  = y / TARGET_SCALE
    price_true = prev * np.exp(true)
    price_pred = prev * np.exp(pred)
    moved = true != 0
    return {
        "test_mape":  float(np.mean(np.abs(price_pred - price_true) / price_true)),
        "naive_mape": float(np.mean(np.abs(prev - price_true) / price_true)),
        "test_direction_acc": float(np.mean(np.sign(pred[moved]) == np.sign(true[moved]))),
        "test_samples": float(len(X)),
    }


def export(name):
    h5 = os.path.join(MODEL_DIR, f"lstm_{name}.h5")
    lstm, dense = read_keras_h5(h5)
    scores = evaluate(NumpyLSTM(lstm, dense), name)

    arrays = {
        "n_lstm": np.array(len(lstm)),
        "n_dense": np.array(len(dense)),
        "target_scale": np.array(TARGET_SCALE),
        "tickers": np.array(",".join(UNIVERSE if name == "universal" else [name])),
        "trained_on": np.array(datetime.fromtimestamp(os.path.getmtime(h5), timezone.utc)
                               .date().isoformat()),
        **{k: np.array(v) for k, v in scores.items()},
    }
    for i, (k, r, b) in enumerate(lstm):
        arrays.update({f"lstm{i}_kernel": k, f"lstm{i}_recurrent": r, f"lstm{i}_bias": b})
    for i, (k, b, act) in enumerate(dense):
        arrays.update({f"dense{i}_kernel": k, f"dense{i}_bias": b,
                       f"dense{i}_activation": np.array(act)})

    out = os.path.join(MODEL_DIR, f"{name}.npz")
    np.savez_compressed(out, **arrays)
    summary = (f"test MAPE {scores['test_mape']:.2%} vs naive {scores['naive_mape']:.2%}, "
               f"direction {scores['test_direction_acc']:.1%}") if scores else "no test split"
    print(f"{name}: wrote {os.path.relpath(out, ROOT)} ({summary})")


def main(argv):
    names = argv or sorted(
        fn[len("lstm_"):-len(".h5")] for fn in os.listdir(MODEL_DIR)
        if fn.startswith("lstm_") and fn.endswith(".h5")
    )
    for n in names:
        export(n if n == "universal" else n.upper())


if __name__ == "__main__":
    main(sys.argv[1:])

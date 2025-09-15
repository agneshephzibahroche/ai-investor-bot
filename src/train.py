import os
import argparse
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

ROOT_DIR   = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
DATA_DIR   = os.path.join(ROOT_DIR, "data", "processed")
MODEL_DIR  = os.path.join(ROOT_DIR, "models")

os.makedirs(MODEL_DIR, exist_ok=True)

def build_model(input_shape):
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(32),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")
    return model

def train_ticker(ticker: str):
    # Load datasets
    X_train = np.load(os.path.join(DATA_DIR, f"X_train_{ticker}.npy"))
    y_train = np.load(os.path.join(DATA_DIR, f"y_train_{ticker}.npy"))
    X_val   = np.load(os.path.join(DATA_DIR, f"X_val_{ticker}.npy"))
    y_val   = np.load(os.path.join(DATA_DIR, f"y_val_{ticker}.npy"))

    model = build_model((X_train.shape[1], X_train.shape[2]))

    checkpoint = ModelCheckpoint(
        os.path.join(MODEL_DIR, f"lstm_{ticker}.h5"),
        save_best_only=True, monitor="val_loss", mode="min"
    )
    earlystop = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=30,
        batch_size=32,
        callbacks=[checkpoint, earlystop],
        verbose=1
    )

    print(f"✅ Training complete for {ticker}, best model saved!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", required=True, help="Ticker symbol (e.g. AAPL)")
    args = parser.parse_args()
    train_ticker(args.ticker)

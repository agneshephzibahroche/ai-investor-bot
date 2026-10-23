import os
import argparse
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

ROOT_DIR   = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
DATA_DIR   = os.path.join(ROOT_DIR, "data", "processed")
MODEL_DIR  = os.path.join(ROOT_DIR, "models")

os.makedirs(MODEL_DIR, exist_ok=True)

def build_model(input_shape):
    model = Sequential([
        Input(shape=input_shape),
        LSTM(64, return_sequences=True),
        Dropout(0.2),
        LSTM(32),
        Dropout(0.2),
        Dense(1)
    ])
    # Huber loss: daily returns have fat tails that would dominate plain MSE
    model.compile(optimizer="adam", loss="huber")
    return model

def train(name: str, epochs: int):
    # Load datasets written by src.preprocessing
    X_train = np.load(os.path.join(DATA_DIR, f"X_train_{name}.npy"))
    y_train = np.load(os.path.join(DATA_DIR, f"y_train_{name}.npy"))
    X_val   = np.load(os.path.join(DATA_DIR, f"X_val_{name}.npy"))
    y_val   = np.load(os.path.join(DATA_DIR, f"y_val_{name}.npy"))

    model = build_model((X_train.shape[1], X_train.shape[2]))

    checkpoint = ModelCheckpoint(
        os.path.join(MODEL_DIR, f"lstm_{name}.h5"),
        save_best_only=True, monitor="val_loss", mode="min"
    )
    earlystop = EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True)

    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=128,
        shuffle=True,
        callbacks=[checkpoint, earlystop],
        verbose=2
    )

    print(f"Training complete for {name}; best model saved to models/lstm_{name}.h5")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", help="Single-ticker model (default: the universal model)")
    parser.add_argument("--epochs", type=int, default=25)
    args = parser.parse_args()
    train(args.ticker.upper() if args.ticker else "universal", args.epochs)

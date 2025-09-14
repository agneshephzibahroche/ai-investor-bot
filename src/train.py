# src/train.py

import os
import argparse
import numpy as np
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from src.model import build_model
import tensorflow as tf

def main():
    p = argparse.ArgumentParser(
        description="Train LSTM model for a given ticker"
    )
    p.add_argument('--ticker', default='^GSPC')
    args = p.parse_args()
    ticker = args.ticker.upper()

    # paths
    ROOT      = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    PROC_DIR  = os.path.join(ROOT, 'data', 'processed')
    MODEL_DIR = os.path.join(ROOT, 'models')
    os.makedirs(MODEL_DIR, exist_ok=True)

    # npy files
    X_train = np.load(os.path.join(PROC_DIR, f'X_train_{ticker}.npy'))
    y_train = np.load(os.path.join(PROC_DIR, f'y_train_{ticker}.npy'))
    X_val   = np.load(os.path.join(PROC_DIR, f'X_val_{ticker}.npy'))
    y_val   = np.load(os.path.join(PROC_DIR, f'y_val_{ticker}.npy'))

    # build & compile model
    seq_len, n_feat = X_train.shape[1], X_train.shape[2]
    model = build_model(input_shape=(seq_len, n_feat))

    # callbacks
    model_path = os.path.join(MODEL_DIR, f'lstm_{ticker}.h5')
    es = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    mc = ModelCheckpoint(filepath=model_path, monitor='val_loss', save_best_only=True)

    # train
    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=32,
        callbacks=[es, mc],
        verbose=2
    )

    model.save(model_path)
    print(f"Model for {ticker} saved to {model_path}")


if __name__ == '__main__':
    main()

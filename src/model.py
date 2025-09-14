# src/model.py

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

def build_model(input_shape):
    """
    Builds and returns an LSTM model.
    
    Args:
        input_shape (tuple): (SEQ_LENGTH, n_features)
        
    Returns:
        tf.keras.Model: compiled LSTM model
    """
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(64),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(1, activation='linear')  # predicting scaled Close price
    ])
    
    model.compile(
        loss='mean_squared_error',
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        metrics=['mae']
    )
    return model

import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import joblib

# 1. Download S&P 500 data
ticker = '^GSPC'
start_date = '2010-01-01'
end_date = '2025-01-01'
df = yf.download(ticker, start=start_date, end=end_date)

# 2. Feature Engineering
# 2.1 Returns
df['Return'] = df['Close'].pct_change()

# 2.2 Moving Averages
df['MA50'] = df['Close'].rolling(window=50).mean()
df['MA200'] = df['Close'].rolling(window=200).mean()

# 2.3 MACD
df['EMA12'] = df['Close'].ewm(span=12, adjust=False).mean()
df['EMA26'] = df['Close'].ewm(span=26, adjust=False).mean()
df['MACD'] = df['EMA12'] - df['EMA26']
df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

# 2.4 RSI (14-day)
delta = df['Close'].diff()
gain = delta.clip(lower=0)
loss = -delta.clip(upper=0)
avg_gain = gain.rolling(window=14).mean()
avg_loss = loss.rolling(window=14).mean()
rs = avg_gain / avg_loss
df['RSI'] = 100 - (100 / (1 + rs))

# 2.5 Clean up
df.drop(columns=['Open', 'High', 'Low', 'Adj Close', 'Volume', 'EMA12', 'EMA26'], inplace=True)
df.dropna(inplace=True)

# 3. Scaling
feature_cols = ['Close', 'Return', 'MA50', 'MA200', 'MACD', 'Signal', 'RSI']
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(df[feature_cols])
joblib.dump(scaler, 'scaler_sp500.save')

# 4. Create sequences
window_size = 60
X, y = [], []
for i in range(window_size, len(data_scaled)):
    X.append(data_scaled[i-window_size:i])
    y.append(data_scaled[i, feature_cols.index('Close')])
X, y = np.array(X), np.array(y)

# 5. Train/Val/Test split (70/15/15)
total = len(X)
train_end = int(total * 0.7)
val_end = train_end + int(total * 0.15)
X_train, y_train = X[:train_end], y[:train_end]
X_val, y_val = X[train_end:val_end], y[train_end:val_end]
X_test, y_test = X[val_end:], y[val_end:]

# 6. Build LSTM model
model = Sequential([    
    LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),    
    Dropout(0.2),
    LSTM(50),
    Dropout(0.2),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse')

# 7. Train
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=50,
    batch_size=32,
    callbacks=[early_stop]
)

# 8. Save model
model.save('lstm_sp500_model.h5')

print("Preprocessing and training complete. Model saved to 'lstm_sp500_model.h5'.")
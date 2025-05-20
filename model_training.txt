import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, GRU, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.callbacks import EarlyStopping
from scipy.signal import savgol_filter
import joblib
import os

# ---------------------- Step 1: Generate Simulated Data ----------------------
np.random.seed(42)
BASE_INDEX = 23000
t = np.arange(0, 10000)
wave = np.sin(t / 30)
trend = 0.0005 * t
noise = np.random.normal(0, 0.02, len(t))
price_series = (wave + noise + trend) * 100 + BASE_INDEX

# ---------------------- Step 2: Create Sliding Windows ----------------------
SEQ_LEN = 30
PRED_HORIZON = 300

def create_dataset(data, seq_len, horizon):
    X, y = [], []
    for i in range(len(data) - seq_len - horizon):
        window = data[i:i+seq_len]
        window_denoised = savgol_filter(window, window_length=7, polyorder=2)

        future_window = data[i+seq_len:i+seq_len+horizon]
        future_max = np.max(future_window)
        future_median = np.median(future_window)
        current_price = data[i + seq_len - 1]

        reward = (current_price - future_median) / (future_max - future_median + 1e-6)
        reward = np.clip(reward, 0, 1)

        X.append(window_denoised)
        y.append(reward)
    return np.array(X), np.array(y)

X, y = create_dataset(price_series, SEQ_LEN, PRED_HORIZON)

# ---------------------- Step 3: Normalize ----------------------
scaler = MinMaxScaler()
X_scaled = np.array([scaler.fit_transform(x.reshape(-1, 1)).flatten() for x in X])
X_scaled = X_scaled[..., np.newaxis]

# ---------------------- Step 4: Build RNN Model ----------------------
def build_model(input_shape):
    inp = Input(shape=input_shape)
    x = GRU(64, return_sequences=False)(inp)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.3)(x)
    out = Dense(1, activation='sigmoid')(x)
    return Model(inputs=inp, outputs=out)

model = build_model((SEQ_LEN, 1))
model.compile(optimizer=Adam(1e-3), loss=MeanSquaredError(), metrics=['mae'])

# ---------------------- Step 5: Train Model ----------------------
callbacks = [EarlyStopping(patience=5, restore_best_weights=True)]
model.fit(X_scaled, y, epochs=20, batch_size=32, validation_split=0.2, callbacks=callbacks)

# ---------------------- Step 6: Save Model and Scaler ----------------------
save_dir = "trained_model"
os.makedirs(save_dir, exist_ok=True)
model.save(os.path.join(save_dir, "gru_predictor.keras"))
joblib.dump(scaler, os.path.join(save_dir, "gru_scaler.gz"))

print(f"Model and scaler saved to: {save_dir}")

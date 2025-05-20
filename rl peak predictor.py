
import numpy as np
import time
import joblib
from tensorflow.keras.models import load_model

# ---------------------- Load Model and Scaler ----------------------
model = load_model("hangseng_model/peak_model_continuous.keras")
scaler = joblib.load("hangseng_model/scaler_continuous.gz")

# ---------------------- Configuration ----------------------
SEQ_LEN = 60
MAX_INTERVALS = 300  # 10 minutes = 300 intervals of 2 seconds

# ---------------------- State ----------------------
price_window = []
status_counter = 0
in_sell_window = False

# ---------------------- Simulated Streaming Source ----------------------
def get_new_data():
    price = np.random.randn() + 1000 + np.sin(time.time() / 30) * 10  # Simulated price pattern
    status = "T"
    return price, status

def evaluate_sell_score(price_sequence):
    normed = scaler.fit_transform(np.array(price_sequence).reshape(-1, 1)).reshape(1, SEQ_LEN, 1)
    score = model.predict(normed, verbose=0)[0][0]
    return score

print("Starting live continuous prediction... (Ctrl+C to stop)")
try:
    while True:
        price, status = get_new_data()
        print(f"New price: {price:.2f} | Status: {status}")

        if status == "T":
            if not in_sell_window:
                print("\nðŸŸ¢ New 10-minute window started.")
                in_sell_window = True
                status_counter = 1
                price_window = []
            else:
                status_counter += 1

            price_window.append(price)
            if len(price_window) > SEQ_LEN:
                price_window = price_window[-SEQ_LEN:]

            if len(price_window) == SEQ_LEN:
                score = evaluate_sell_score(price_window)
                if score > 0.85:
                    print(f"âœ… SELL NOW (score: {score:.2f})")
                elif score > 0.65:
                    print(f"âš ï¸  CONSIDER SELLING (score: {score:.2f})")
                else:
                    print(f"â³ HOLD (score: {score:.2f})")

            if status_counter >= MAX_INTERVALS:
                print("â¹ï¸  10-minute window complete.\n")
                in_sell_window = False
                status_counter = 0
                price_window = []

        time.sleep(2)

except KeyboardInterrupt:
    print("\nLive prediction stopped.")

import requests
import time
import numpy as np
from scipy.signal import savgol_filter
from stable_baselines3 import PPO


# ---------------------- Load PPO Model ----------------------
model = PPO.load("trained_model/ppo_hangseng_agent")

# ---------------------- Configuration ----------------------
SEQ_LEN = 30
API_URL = "http://localhost:8000/latest"
INVENTORY = 100
MAX_STEPS = 300  # 10 minutes at 2-second intervals
price_window = []
cash = 0.0
step = 0

print("🔄 Starting PPO-based consumer...")

try:
    while INVENTORY > 0 and step < MAX_STEPS:
        # 1. Get latest price
        try:
            response = requests.get(API_URL, timeout=2)
            if response.status_code != 200:
                time.sleep(2)
                continue
            data = response.json()
            current_price = data["price"]
            status = data["status"]
        except:
            print("⚠️ Failed to fetch price.")
            time.sleep(2)
            continue

        if status != "T":
            time.sleep(2)
            continue

        print(f"📈 Price @ {step * 2}s: {current_price:.2f}")
        price_window.append(current_price)
        if len(price_window) > SEQ_LEN:
            price_window = price_window[-SEQ_LEN:]

        if len(price_window) == SEQ_LEN:
            # 2. Denoise + normalize + build state
            denoised = savgol_filter(price_window, 7, 2)
            normed = (denoised - np.min(denoised)) / (np.max(denoised) - np.min(denoised) + 1e-6)
            inventory_norm = INVENTORY / 100
            time_norm = step / MAX_STEPS
            obs = np.concatenate([normed, [inventory_norm], [time_norm]]).reshape(1, -1).astype(np.float32)

            # 3. Predict action
            action, _ = model.predict(obs, deterministic=True)
            units_to_sell = min(action, INVENTORY)

            # 4. Apply decision
            reward = units_to_sell * current_price
            cash += reward
            INVENTORY -= units_to_sell

            print(f"🤖 Action: Sell {units_to_sell} units | 💰 +{reward:.2f} | 🧾 Inventory left: {INVENTORY} | 💼 Cash: {cash:.2f}")

        step += 1
        time.sleep(2)

    print(f"\n✅ Simulation complete. Total cash: {cash:.2f} | Units sold: {100 - INVENTORY}")

except KeyboardInterrupt:
    print("\n⏹️ Simulation manually stopped.")

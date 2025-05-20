import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
import os
from gymnasium import Env
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv

# ---------------------- Step 1: Generate and Save Simulated Data ----------------------
np.random.seed(42)
t = np.arange(0, 20000)
BASE_INDEX = 23000

# Slow exponential trend (e.g., 0.5% per 1000 steps)
trend = 1 + 0.000005 * t  # ~10% growth over 20,000 steps

# Gradually increasing sine wave amplitude (volatility)
amplitude_growth = 1 + 0.00005 * t  # starts at 1, ends ~2

# Slight frequency shift over time (optional)
frequency_modulation = t / (30 + 0.0005 * t)

# Base sine wave
wave = np.sin(frequency_modulation) * amplitude_growth

# Market-style noise (constant scale)
noise = np.random.normal(0, 0.02, len(t))

# Final price series
price_series = (wave + trend + noise) * 100 + BASE_INDEX

df = pd.DataFrame({"current": price_series})
os.makedirs("data", exist_ok=True)
csv_path = "data/hangseng_simulated.csv"
df.to_csv(csv_path, index=False)
print(f"Simulated data saved to {csv_path}")

# ---------------------- Step 2: Define Custom Environment ----------------------
class HangSengSellEnv(Env):
    def __init__(self, price_series, seq_len=30, max_inventory=100, max_steps=300):
        super().__init__()
        self.price_series = price_series
        self.seq_len = seq_len
        self.max_inventory = max_inventory
        self.max_steps = max_steps

        self.action_space = spaces.Discrete(11)  # 0 to 10 units
        self.observation_space = spaces.Box(low=0, high=1, shape=(seq_len + 2,), dtype=np.float32)
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.pointer = np.random.randint(0, len(self.price_series) - self.max_steps - self.seq_len - 300)
        self.current_step = 0
        self.inventory = self.max_inventory
        self.total_cash = 0.0
        self.done = False
        return self._get_obs(), {}

    def _get_obs(self):
        window = self.price_series[self.pointer + self.current_step:
                                   self.pointer + self.current_step + self.seq_len]
        denoised = savgol_filter(window, window_length=7, polyorder=2)
        normed = (denoised - np.min(denoised)) / (np.max(denoised) - np.min(denoised) + 1e-6)
        obs = np.concatenate([normed, [self.inventory / self.max_inventory], [self.current_step / self.max_steps]])
        return obs.astype(np.float32)

    def step(self, action):
        units_to_sell = min(action, self.inventory)
        current_idx = self.pointer + self.current_step + self.seq_len - 1
        current_price = self.price_series[current_idx]

        future_window = self.price_series[current_idx + 1:current_idx + 300]
        future_max = np.max(future_window) if len(future_window) > 0 else current_price
        proximity = (current_price / future_max) if future_max > 0 else 1.0

        reward = units_to_sell * current_price/23000 * proximity

        self.inventory -= units_to_sell
        self.total_cash += reward
        self.current_step += 1
        self.done = self.inventory <= 0 or self.current_step >= self.max_steps

        return self._get_obs(), reward, self.done, False, {}

# ---------------------- Step 3: Create and Validate Environment ----------------------
price_series = df["current"].values
env = HangSengSellEnv(price_series=price_series)
check_env(env, warn=True)

# ---------------------- Step 4: Train PPO Agent ----------------------
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=100000)

# ---------------------- Step 5: Save PPO Model ----------------------
os.makedirs("trained_model", exist_ok=True)
model.save("trained_model/ppo_hangseng_agent")
print("✅ PPO model saved to trained_model/ppo_hangseng_agent.zip")

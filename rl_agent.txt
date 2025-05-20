
import numpy as np
import gymnasium
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from gym_env import HangSengSellEnv

# ---------------------- Step 1: Simulate Hang Seng Index Data ----------------------
np.random.seed(42)
t = np.arange(0, 20000)
BASE_INDEX = 23000
trend = 0.0005 * t
noise = np.random.normal(0, 0.02, len(t))
wave = np.sin(t / 30)
price_series = (wave + trend + noise) * 100 + BASE_INDEX

# ---------------------- Step 2: Create Environment ----------------------
env = HangSengSellEnv(price_series=price_series)
check_env(env, warn=True)

# ---------------------- Step 3: Train PPO Agent ----------------------
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=50000)

# ---------------------- Step 4: Save Model ----------------------
model.save("ppo_hangseng_agent")
print("PPO agent saved as ppo_hangseng_agent.zip")

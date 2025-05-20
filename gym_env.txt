
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from scipy.signal import savgol_filter

class HangSengSellEnv(gym.Env):
    def __init__(self, price_series, seq_len=30, max_inventory=100, max_steps=300):
        super(HangSengSellEnv, self).__init__()
        self.price_series = price_series
        self.seq_len = seq_len
        self.max_inventory = max_inventory
        self.max_steps = max_steps

        self.action_space = spaces.Discrete(11)  # Sell 0 to 10 units
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(seq_len + 2,), dtype=np.float32
        )

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.pointer = np.random.randint(0, len(self.price_series) - self.max_steps - self.seq_len)
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
        current_price = self.price_series[self.pointer + self.current_step + self.seq_len - 1]
        reward = units_to_sell * current_price

        self.inventory -= units_to_sell
        self.total_cash += reward
        self.current_step += 1
        self.done = self.inventory <= 0 or self.current_step >= self.max_steps

        obs = self._get_obs()
        return obs, reward, self.done, False, {}

    def render(self, mode='human'):
        print(f"Step: {self.current_step}, Inventory: {self.inventory}, Cash: {self.total_cash}")

import safe_simple_gymnasium
import gymnasium as gym


from stable_baselines3 import SPMA
from stable_baselines3.common.callbacks import CostEvalCallback
from stable_baselines3.common.utils import set_random_seed

from stable_baselines3.common.env_util import DummyVecEnv, make_vec_env


seed = 0
set_random_seed(seed, using_cuda=True)

env = make_vec_env("SafeCartPole-Arushi-v0", seed=seed)
eval_env = make_vec_env("SafeCartPole-Arushi-v0", seed=seed + 1337)
callback = CostEvalCallback(eval_env, eval_freq=2048 + 1, log_path="test_callback", deterministic=False)
model = SPMA("MlpPolicy", env, verbose=1, batch_size=2048, eta=0.3, n_epochs=10)
model.learn(total_timesteps=2048 * 10 + 1, callback=callback, progress_bar=True, log_interval=None)

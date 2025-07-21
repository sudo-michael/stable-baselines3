import safe_simple_gymnasium
import gymnasium as gym


from stable_baselines3 import SPMAPD, SPMAALM
from stable_baselines3.common.callbacks import CostEvalCallback
from stable_baselines3.common.utils import set_random_seed

from stable_baselines3.common.env_util import make_vec_env

samples_per_epoch = 2084
total_epochs = 100
num_inner_updates = 5


seed = 1
set_random_seed(seed, using_cuda=True)

# env = make_vec_env("SafeCartPole-ArushiModify-v0", seed=seed)
# eval_env = make_vec_env("SafeCartPole-ArushiModify-v0", seed=seed + 1337)

# callback = CostEvalCallback(
#     eval_env, eval_freq=samples_per_epoch * num_inner_updates + 1, log_path="spma_alm_2", deterministic=False, verbose=0
# )
# # model = SPMAPD("MlpPolicy", env, verbose=1, normalize_advantage=True, cost_limit=160, eta=0.1)
# model = SPMAALM(
#     "MlpPolicy",
#     env,
#     verbose=1,
#     normalize_advantage=True,
#     cost_limit=25,
#     eta=0.1,
#     tau=0.1,
#     n_epochs=10,
#     batch_size=2048,
#     n_steps=2048,
#     stats_window_size=1000,
# )
# model.learn(
#     total_timesteps=samples_per_epoch * total_epochs + 1,
#     num_inner_updates=num_inner_updates,
#     callback=callback,
#     progress_bar=True,
#     log_interval=None
# )
# model.save('spma_alm_model_2')
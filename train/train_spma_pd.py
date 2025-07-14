import safe_simple_gymnasium
import gymnasium as gym

from stable_baselines3 import SPMAPD
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecVideoRecorder

# SPMA grid search
etas = [0.3, 0.5, 0.7, 0.9, 1.0]
# Parallel environments
n_steps = 2048
n_envs = 4
rollout_buffer_size = n_steps * n_envs
n_inner_updates = 1
n_updates = 100
# total number of samples collected
total_timesteps = rollout_buffer_size * n_updates
# number fo samples collected and used for each inner-optimiztion loop
inner_timesteps = total_timesteps // n_inner_updates

for eta in etas:
    vec_env = make_vec_env(
        "SafeCartPole-v0",
        n_envs=n_envs,
        monitor_dir=f"spma_eta_{eta}",
    )
    model = SPMAPD(
        "MlpPolicy",
        vec_env,
        verbose=1,
        n_steps=n_steps,
        batch_size=n_steps,
        eta=eta,
        n_epochs=5,
        use_armijo_actor=True,
        use_armijo_critic=True,
    )
    model.learn(total_timesteps=total_timesteps, inner_timesteps=inner_timesteps)

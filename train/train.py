import safe_simple_gymnasium
import gymnasium as gym

from stable_baselines3 import SPMA
from stable_baselines3.common.env_util import make_vec_env

# Parallel environments
vec_env = make_vec_env("SafeCartPole-v0", n_envs=4)

model = SPMA("MlpPolicy", vec_env, verbose=1)
model.learn(total_timesteps=25000)
# model.save("spma_cartpole")
# del model # remove to demonstrate saving and loading
#
# model = SPMA.load("spma_cartpole")
#
# obs = vec_env.reset()
# while True:
#     action, _states = model.predict(obs)
#     obs, rewards, dones, info = vec_env.step(action)
#     vec_env.render("human")
